import uuid
from pathlib import Path
from typing import Optional

from peft import PeftConfig
from transformers.models.auto import AutoConfig
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from authentrics import AuthentricsSession, ModelInterface, WeightBias, use_backend


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


if __name__ == "__main__":
    # Set the backend to torch
    use_backend("torch")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleHFModel()
    session.model = model

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MedicalChatbot/iteration_{i}") for i in range(1, 3)
    ]

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_hf_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_hf_project_" + str(uuid.uuid4()),
        "Example Hugging Face project for static_analysis",
    )

    # Register all checkpoints, then select the ones needed for the analysis
    project = session.add_checkpoints(project, *checkpoint_paths)

    # Run static_analysis: project, then previous and chosen Checkpoint (two separate args).
    # Optionally pass weight_names=[...] and/or bias_names=[...] to restrict to specific layers;
    # omit both to analyze all weights and biases.
    result = session.static_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        # weight_names=[
        #     "base_model.model.model.layers.10.mlp.down_proj.lora_A.weight"
        # ],  # Omit to analyze all weights
    )

    print("Static analysis completed.")
    print(f"Result: {result}")
