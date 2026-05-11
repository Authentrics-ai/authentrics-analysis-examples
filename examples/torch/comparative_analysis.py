from pathlib import Path
from typing import Any

import numpy as np

from authentrics import AuthentricsSession, use_backend

from authentrics_examples.models.torch import SimpleModel, preprocess_image
from authentrics_examples.config import get_example_data_path


def _get_input_data(stimuli: Path) -> Any:
    if not stimuli.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli}")

    arrays = []
    for stimulus in sorted(stimuli.glob("*.jpg")):
        arrays.append(preprocess_image(stimulus))

    return np.stack(arrays, axis=0)


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(["stimuli", "checkpoint_1.pt"])
    input_data = _get_input_data(example_data_path / "stimuli")

    # Create a session and a minimal model
    model = SimpleModel()
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"checkpoint_{i}.pt" for i in range(1, 8)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MilAirClassificationExample:Torch",
        "Example project for Torch-based CNN model",
    )

    # Run activation_analysis (project and Checkpoint objects from checkpoint_paths)
    # layer_names: optional list of layer names to analyze (None = all layers)
    result = session.activation_analysis(
        project.id,
        checkpoint_paths[0],
        checkpoint_paths[1],
        input_data,
        [
            "squeeze_edit_model.features.6.1.block.2.fc2",
            "squeeze_edit_model.classifier.1",
        ],
    )

    print("Activation analysis completed.")
    print(result.cosine_similarities)
    print(result.l2_distances)


if __name__ == "__main__":
    main()
