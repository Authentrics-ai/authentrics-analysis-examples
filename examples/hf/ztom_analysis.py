"""Minimal example of ZTOM (Zero Train Optimization and Maintenance) analysis.

ZTOM optimizes scaling factors over training deltas between consecutive checkpoints
to minimize a user-defined loss on the model output. The project must have at least
two checkpoints (order matters: they define the sequence of training deltas).
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from peft import PeftConfig
from sentence_transformers import SentenceTransformer
from transformers.models.auto import AutoConfig
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    WeightBias,
    ZtomOptimizationOptions,
    use_backend,
)


def _get_input_data() -> Any:
    stimuli_dir = Path("/stimuli/medical-advice-prompts")
    stimuli = sorted(stimuli_dir.glob("*.json"))
    return [json.load(open(p)) for p in stimuli]


def _get_expected_output() -> list[str]:
    stimuli_dir = Path("/stimuli/medical-advice-prompts/medical-advice-responses.txt")
    stimuli = open(stimuli_dir).readlines()
    return [line.strip() for line in stimuli]


class SimpleHFModel(ModelInterface):
    def __init__(
        self, inference_config: dict[str, Any] | None = None, batch_size: int = 1
    ):
        super().__init__()
        self._module = None
        self._input_data = None
        self._inference_config = inference_config
        self._batch_size = batch_size
        self._input_data = _get_input_data()
        self.expected_output = _get_expected_output()

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        config_filepaths = list(checkpoint_path.rglob("*config.json"))
        model_path = config_filepaths[0].parent

        try:
            peft_config = PeftConfig.from_pretrained(str(model_path))
            config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
        except Exception:
            config = AutoConfig.from_pretrained(str(model_path))

        if hasattr(config, "torch_dtype"):
            torch_dtype = config.torch_dtype
        else:
            torch_dtype = None

        # Load model with proper device placement and quantization handling
        # Use device_map="sequential" to handle quantization and multi-device models
        # If that doesn't work, use device_map="auto" or "balanced_low_0"
        self._module: TextGenerationPipeline = pipeline(  # type: ignore
            "text-generation",
            model=str(model_path),
            device_map="sequential",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # Set pad_token_id for batching support
        if (
            self._module.tokenizer is not None
            and self._module.tokenizer.pad_token_id is None
        ):
            if (
                hasattr(self._module.model.config, "eos_token_id")
                and self._module.model.config.eos_token_id is not None
            ):
                self._module.tokenizer.pad_token_id = (
                    self._module.model.config.eos_token_id
                )
            else:
                # Fallback: set pad_token_id to 0 if eos_token_id is not available
                self._module.tokenizer.pad_token_id = 0

    def get_weight_bias(
        self,
        weight_names: Optional[list[str]] = None,
        bias_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        for name, param in self._module.model.named_parameters():
            last_part = name.rsplit(".", 1)[-1]
            if last_part == "weight":
                if weight_names is None or name in weight_names:
                    weights[name] = param.detach().cpu()
            elif last_part == "bias":
                if bias_names is None or name in bias_names:
                    biases[name] = param.detach().cpu()
        return weights, biases

    def perform_inference(
        self,
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:

        max_new_tokens = int(self._inference_config.pop("max_new_tokens", 50))
        chat_template = self._inference_config.pop("chat_template", None)

        # Perform inference
        assert self._module.tokenizer is not None
        if chat_template is not None:
            self._module.tokenizer.chat_template = chat_template

        result = self._module(
            text_inputs=self._input_data,
            max_new_tokens=max_new_tokens,
            batch_size=self._batch_size,
            chat_template=chat_template,
            **self._inference_config,
        )

        return InferenceResult(output=result)

    def set_weight_bias(self, weight_bias: WeightBias) -> None:
        state = {n: p for n, p in self._module.model.named_parameters()}
        for name, tensor in weight_bias.weights.items():
            if name in state:
                state[name].data.copy_(tensor)
        for name, tensor in weight_bias.biases.items():
            if name in state:
                state[name].data.copy_(tensor)

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._module.save_pretrained(path)


def _sentence_similarity_loss_fn(
    expected_output: list[str],
    output: list[list[dict[str, str]]],
) -> float:
    similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generated_embeddings = similarity_model.encode(
        [o[0]["generated_text"] for o in output]
    )
    expected_embeddings = similarity_model.encode(expected_output)
    score = (
        similarity_model.similarity_pairwise(generated_embeddings, expected_embeddings)
        .mean()
        .item()
    )
    return 1.0 - score


if __name__ == "__main__":
    # Set the backend to torch
    use_backend("torch")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleHFModel(inference_config={"max_new_tokens": 50}, batch_size=1)
    session.model = model

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MedicalChatbot/iteration_{i}") for i in range(12)
    ]

    new_checkpoint_path = Path("./checkpoint_optimized")
    if new_checkpoint_path.exists():
        shutil.rmtree(new_checkpoint_path)

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_hf_analysis_project")
    if not project_path.exists() or not (project_path / ".authentrics.json").exists():
        project = session.init_project(
            project_path,
            "example_hf_project_" + str(uuid.uuid4()),
            "Example Hugging Face project for static_analysis",
        )
    else:
        project = session.load_project(project_path)

    if not project.checkpoints:
        # Register all checkpoints, then select the ones needed for the analysis
        project = session.add_checkpoints(project, *checkpoint_paths)

    # Optional: tune optimization (defaults are often sufficient)
    options = ZtomOptimizationOptions(
        max_evaluations=50,
        xtol_rel=1e-6,
        ftol_rel=1e-6,
        lower_bound=-1.0,
        upper_bound=1.0,
        minimize=True,
    )

    # loss_function: callable(model_output) -> float
    # can be passed directly or as LossFunction(loss_function)
    result = session.ztom_analysis(
        project,
        lambda y_pred: _sentence_similarity_loss_fn(model.expected_output, y_pred),
        new_checkpoint_path,
        options,
    )

    print("ZTOM analysis completed.")
    print(f"Optimized checkpoint saved to: {result.optimized_checkpoint_path}")
    print(
        f"Original loss: {result.original_loss:.6f}, best loss: {result.best_loss:.6f}"
    )
    print(f"Scaling factors: {result.scaling_factors}")
    print(f"Number of inferences: {result.number_of_inferences}")
