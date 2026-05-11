from pathlib import Path
from typing import Any

import numpy as np

from authentrics import AuthentricsSession, ZtomOptimizationOptions

from authentrics_examples.models.onnx import SimpleModel, preprocess_image
from authentrics_examples.config import get_example_data_path


def _get_input_data(stimuli: Path) -> Any:
    if not stimuli.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli}")

    arrays = []
    for stimulus in sorted(stimuli.glob("*.jpg")):
        arrays.append(preprocess_image(stimulus))

    return np.stack(arrays, axis=0)


def _get_expected_output(output_path: Path) -> np.ndarray:
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    outputs = np.loadtxt(output_path, delimiter=",")
    return outputs.argmax(axis=1)


def _milair_loss_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the categorical cross-entropy loss between y_true and y_pred.

    y_true: numpy array of class indices (ground truth labels), shape (samples,).
    y_pred: numpy array of predicted logits, shape (samples, num_classes).

    Returns:
        scalar float loss (mean over samples).
    """
    logits = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.intp).ravel()

    # Numerically stable log-softmax: log_softmax_i = logits_i - logsumexp(logits)
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True)) + logits_max
    )
    log_softmax = logits - log_sum_exp

    # NLL: -log_softmax at the true class index
    n_samples = logits.shape[0]
    nll = -log_softmax[np.arange(n_samples), y_true]

    return float(np.mean(nll))


def main():
    example_data_path = get_example_data_path(
        ["stimuli", "expected_outputs.csv", "checkpoint_1.onnx"]
    )

    input_data = _get_input_data(example_data_path / "stimuli")
    expected_output = _get_expected_output(example_data_path / "expected_outputs.csv")

    # Create a session and a minimal model
    model = SimpleModel()
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"checkpoint_{i}.onnx" for i in range(1, 8)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MilAirClassificationExample:ONNX",
        "Example project for ONNX-based CNN model",
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
        new_checkpoint_path=example_data_path / "checkpoint_optimized.onnx",
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
