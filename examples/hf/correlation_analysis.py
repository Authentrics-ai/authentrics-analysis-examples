import json
from pathlib import Path
from typing import Any

from authentrics import AuthentricsSession, use_backend

from authentrics_examples.models.hf import SimpleHFModel
from authentrics_examples.config import get_example_data_path


def _get_input_data(stimuli: Path) -> Any:
    if not stimuli.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli}")
    with open(stimuli, "r") as f:
        return [json.loads(line) for line in f]


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(["stimuli.jsonl", "iteration_0"])
    input_data = _get_input_data(example_data_path / "stimuli.jsonl")

    # Create a session and a minimal model
    model = SimpleHFModel(inference_config={"max_new_tokens": 50}, batch_size=1)
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"iteration_{i}" for i in range(4)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MedicalChatbotExample",
        "Example project for Hugging Face-based LLM",
    )

    # Run correlation_analysis (project id, chosen checkpoint, input data, list of layer names, reference layer name)
    # layer_names: list of layer names to analyze
    result = session.correlation_analysis(
        project.id,
        checkpoint_paths[1],
        input_data,
        layer_names=[
            "model.layers.10.mlp.down_proj.lora_A.default",
            "model.layers.10.mlp.down_proj.lora_B.default",
        ],
        reference_layer_name="model.norm",
    )

    print("Correlation analysis completed.")
    print(result.correlation_scores)
    print(result.correlation_histograms)


if __name__ == "__main__":
    main()
