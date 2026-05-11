"""Minimal example of ZTOM (Zero Train Optimization and Maintenance) analysis.

ZTOM optimizes scaling factors over training deltas between consecutive checkpoints
to minimize a user-defined loss on the model output. The project must have at least
two checkpoints (order matters: they define the sequence of training deltas).
"""

from pathlib import Path

import numpy as np
import torch

from typing import Any


from authentrics import AuthentricsSession, ZtomOptimizationOptions, use_backend

from authentrics_examples.models.torch import SimpleModel, preprocess_image
from authentrics_examples.config import get_example_data_path


def _get_input_data(stimuli: Path) -> Any:
    if not stimuli.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli}")

    arrays = []
    for stimulus in sorted(stimuli.glob("*.jpg")):
        arrays.append(preprocess_image(stimulus))

    return torch.stack(arrays, dim=0)


def _get_expected_output(output_path: Path) -> np.ndarray:
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    outputs = np.loadtxt(output_path, delimiter=",")
    return torch.as_tensor(outputs).argmax(dim=1)


def _milair_loss_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the categorical cross-entropy loss between y_true and y_pred.

    y_true: tensor of class indices (ground truth labels).
    y_pred: tensor of predicted logits.

    Returns:
        scalar float loss.
    """
    return float(
        torch.nn.functional.cross_entropy(
            y_pred,
            y_true.to(y_pred.device).long(),
        ).item()
    )


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(
        ["stimuli", "expected_outputs.csv", "checkpoint_1.pt"]
    )

    input_data = _get_input_data(example_data_path / "stimuli")
    expected_output = _get_expected_output(example_data_path / "expected_outputs.csv")

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

    # Optional: tune optimization (defaults are often sufficient)
    options = ZtomOptimizationOptions()

    # loss_function: callable(model_output) -> float
    # can be passed directly as a lambda or normal function
    result = session.ztom_analysis(
        project.id,
        checkpoint_paths,
        input_data,
        loss_function=lambda y_pred: _milair_loss_function(expected_output, y_pred),
        new_checkpoint_path=example_data_path / "checkpoint_optimized.pt",
        optimization_options=options,
    )

    print("ZTOM analysis completed.")
    print(f"Optimized checkpoint saved to: {result.new_checkpoint_path}")
    print(
        f"Original loss: {result.original_loss:.6f}, best loss: {result.best_loss:.6f}"
    )
    print(f"Scaling factors: {result.scaling_factors}")
    print(f"Number of inferences: {result.number_of_inferences}")


if __name__ == "__main__":
    main()
