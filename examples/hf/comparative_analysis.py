import json
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftConfig
from transformers.models.auto import AutoConfig
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    WeightBias,
    use_backend,
)


def _make_capture_hook(layer_name: str, storage: dict[str, torch.Tensor]) -> object:
    """Return a forward hook that stores this layer's output in storage (torch, CPU)."""

    def hook(
        _module: torch.nn.Module, _input: object, output: torch.Tensor | tuple
    ) -> None:
        out = output[0] if isinstance(output, tuple) else output
        storage[layer_name] = out.detach().clone()

    return hook


def _register_per_layer_output_hooks(
    model: TextGenerationPipeline,
    storage: dict[str, torch.Tensor],
    layer_names: list[str] | None = None,
) -> list[object]:
    """Register forward hooks on the given layers; capture outputs into storage.
    Returns handle list for removal.
    """
    layer_modules = {
        name: mod
        for name, mod in model.model.named_modules()
        if name != "" and name in layer_names
    }
    handles = []
    for name, module in layer_modules.items():
        if hasattr(module, "register_forward_hook"):
            h = module.register_forward_hook(_make_capture_hook(name, storage))
            handles.append(h)
    return handles


def _get_input_data() -> Any:
    stimuli_dir = Path("/stimuli/medical-advice-prompts")
    stimuli = sorted(stimuli_dir.glob("*.json"))
    return [json.load(open(p)) for p in stimuli]


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
        self._parameters = {}

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

    def get_weight_bias_from_layer_names(
        self,
        layer_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        layer_set = set(layer_names) if layer_names else None

        for name, layer in self._module.model.named_modules():
            if layer_set is not None and name in layer_set:
                if hasattr(layer, "weight") and layer.weight is not None:
                    weights[name + ".weight"] = layer.weight
                if hasattr(layer, "bias") and layer.bias is not None:
                    biases[name + ".bias"] = layer.bias

        return weights, biases

    def perform_inference(
        self,
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:
        intermediate_outputs: dict[str, torch.Tensor] = {}
        handles: list[object] = []
        if return_intermediate_outputs:
            handles = _register_per_layer_output_hooks(
                self._module, intermediate_outputs, layer_names=layer_names
            )

        max_new_tokens = int(self._inference_config.pop("max_new_tokens", 50))
        chat_template = self._inference_config.pop("chat_template", None)

        # Perform inference
        assert self._module.tokenizer is not None
        if chat_template is not None:
            self._module.tokenizer.chat_template = chat_template
        try:
            result = self._module(
                text_inputs=self._input_data,
                max_new_tokens=max_new_tokens,
                batch_size=self._batch_size,
                chat_template=chat_template,
                **self._inference_config,
            )
        finally:
            for h in handles:
                h.remove()
        return InferenceResult(
            output=result,
            intermediate_outputs=intermediate_outputs,
        )


if __name__ == "__main__":
    # Set the backend to torch
    use_backend("torch")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleHFModel(inference_config={"max_new_tokens": 50}, batch_size=1)
    session.model = model

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MedicalChatbot/iteration_{i}") for i in range(1, 3)
    ]

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

    # Run comparative_analysis (project and Checkpoint objects from project.checkpoints)
    # layer_names: optional list of layer names to analyze (None = all layers)
    result = session.comparative_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        ["model.layers.10.mlp.down_proj.lora_A.default", "model.norm"],
    )

    print("Comparative analysis completed.")
    print(result)
