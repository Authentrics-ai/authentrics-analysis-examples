import uuid
from pathlib import Path
from typing import Optional

from peft import PeftConfig
from transformers.models.auto import AutoConfig
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from authentrics import (
    AuthentricsSession,
    ModelInterface,
    WeightBias,
    use_backend,
)


class SimpleHFModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self._module = None

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
                    weights[name] = param
            elif last_part == "bias":
                if bias_names is None or name in bias_names:
                    biases[name] = param
        return weights, biases

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


if __name__ == "__main__":
    # Set the backend to torch
    use_backend("torch")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleHFModel()
    session.model = model

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MedicalChatbot/iteration_{i}") for i in range(1, 8)
    ]

    new_checkpoint_path = Path("./checkpoint_excluded")

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_hf_analysis_project")
    if not project_path.exists() or not (project_path / ".authentrics.json").exists():
        project_path.mkdir(parents=True, exist_ok=True)
        project = session.init_project(
            project_path,
            "example_hf_project_" + str(uuid.uuid4()),
            "Example Hugging Face project for exclude_training",
        )
    else:
        project = session.load_project(project_path)

    if not project.checkpoints:
        # Register all checkpoints, then select the ones needed for the analysis
        project = session.add_checkpoints(project, *checkpoint_paths)

    # Run exclude_training (project, list of (Checkpoint, Checkpoint), latest Checkpoint, new path)
    # Example: exclude training from checkpoint 3->4, apply to latest checkpoint
    # Indices: checkpoint_paths[3] and [4] are the pair; checkpoint_paths[-1] is latest
    result = session.exclude_training(
        project,
        [(project.checkpoints[3], project.checkpoints[4])],
        project.checkpoints[-1],
        new_checkpoint_path,
    )

    print(
        f"Exclude training completed. New checkpoint saved to: {result.new_checkpoint_path}"
    )
