"""Minimal example of ZTOM (Zero Train Optimization and Maintenance) analysis.

ZTOM optimizes scaling factors over training deltas between consecutive checkpoints
to minimize a user-defined loss on the model output. The project must have at least
two checkpoints (order matters: they define the sequence of training deltas).
"""

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    WeightBias,
    ZtomOptimizationOptions,
    use_backend,
)

from models.MilAirClass import MilAirModel


def preprocess_image(image_file: Path) -> torch.Tensor:
    """Load and preprocess an image to CHW float32 tensor in [0, 1]."""
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    arr = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
    return arr / 255.0


def _get_stimuli_tensor() -> torch.Tensor:
    """Load stimuli from /stimuli/F*.jpg."""
    stimuli_dir = Path("/stimuli")
    stimuli = sorted(stimuli_dir.glob("F*.jpg"))
    return torch.stack([preprocess_image(p) for p in stimuli])


class SimpleModel(ModelInterface):
    """PyTorch model that loads/saves state dicts and exposes weight/bias for ZTOM."""

    def __init__(self) -> None:
        super().__init__()
        self._module = MilAirModel()
        self._device = "cpu"

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        self._input_data = _get_stimuli_tensor().to(device=self._device)

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._module.load_state_dict(state["model"], strict=True)

        self._module.to(device=self._device)
        self._module.eval()

    def get_weight_bias(
        self,
        weight_names: Optional[list[str]] = None,
        bias_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        for name, param in self._module.named_parameters():
            last_part = name.rsplit(".", 1)[-1]
            if last_part == "weight":
                if weight_names is None or name in weight_names:
                    weights[name] = param
            elif last_part == "bias":
                if bias_names is None or name in bias_names:
                    biases[name] = param
        return weights, biases

    def perform_inference(
        self,
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:
        with torch.no_grad():
            output = self._module(self._input_data)
        return InferenceResult(output, {})

    def set_weight_bias(self, weight_bias: WeightBias) -> None:
        state = {n: p for n, p in self._module.named_parameters()}
        for name, tensor in weight_bias.weights.items():
            if name in state:
                state[name].data.copy_(tensor)
        for name, tensor in weight_bias.biases.items():
            if name in state:
                state[name].data.copy_(tensor)

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._module.state_dict()}, path)


def milair_loss_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the categorical cross-entropy loss between y_true and y_pred.

    y_true: tensor of class indices (ground truth labels).
    y_pred: tensor of predicted logits.

    Returns:
        scalar float loss.
    """
    return float(
        nn.functional.cross_entropy(
            y_pred,
            y_true.to(y_pred.device).long(),
        ).item()
    )


if __name__ == "__main__":
    use_backend("torch")
    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleModel()
    session.model = model

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_ztom_project")
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.pt") for i in range(1, 4)
    ]

    if not project_path.exists() or not (project_path / ".authentrics.json").exists():
        project = session.create_project(
            project_path,
            "ztom_example_" + str(uuid.uuid4()),
            "Example project for ZTOM analysis",
        )
    else:
        project = session.load_project(project_path)

    # Register checkpoints with the project
    if not project.checkpoints:
        project = session.add_checkpoints(project, *checkpoint_paths)

    new_checkpoint_path = project_path / "checkpoint_optimized.pt"

    # Optional: tune optimization (defaults are often sufficient)
    options = ZtomOptimizationOptions(
        max_evaluations=50,
        xtol_rel=1e-4,
        ftol_rel=1e-4,
        lower_bound=-1.0,
        upper_bound=1.0,
        minimize=True,
    )

    y_true = torch.tensor([9, 9, 9, 10, 13, 13, 13], dtype=torch.long)

    # loss_function: callable(model_output) -> float
    # can be passed directly or as LossFunction(loss_function)
    result = session.ztom_analysis(
        project,
        lambda y_pred: milair_loss_function(y_true, y_pred),
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
