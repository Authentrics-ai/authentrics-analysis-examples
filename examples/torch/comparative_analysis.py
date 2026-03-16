import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    WeightBias,
    use_backend,
)

from .models.MilAirClass import MilAirModel


def _make_capture_hook(layer_name: str, storage: dict[str, torch.Tensor]) -> object:
    """Return a forward hook that stores this layer's output in storage (torch, CPU)."""

    def hook(
        _module: torch.nn.Module, _input: object, output: torch.Tensor | tuple
    ) -> None:
        out = output[0] if isinstance(output, tuple) else output
        storage[layer_name] = out.detach().clone()

    return hook


def _register_per_layer_output_hooks(
    model: torch.nn.Module,
    storage: dict[str, torch.Tensor],
    layer_names: list[str] | None = None,
) -> list[object]:
    """Register forward hooks on the given layers; capture outputs into storage.
    Returns handle list for removal.
    """
    layer_modules = {
        name: mod for name, mod in model.named_modules() if name and name in layer_names
    }
    handles = []
    for name, module in layer_modules.items():
        if hasattr(module, "register_forward_hook"):
            h = module.register_forward_hook(_make_capture_hook(name, storage))
            handles.append(h)
    return handles


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
        try:
            with torch.no_grad():
                output = self._module(self._input_data)
        finally:
            for h in handles:
                h.remove()
        return InferenceResult(output, intermediate_outputs)

    def get_weight_bias_from_layer_names(
        self,
        layer_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        layer_set = set(layer_names) if layer_names else None

        for name, layer in self._module.named_modules():
            if layer_set is not None and name in layer_set:
                if hasattr(layer, "weight"):
                    weights[name + ".weight"] = layer.weight
                if hasattr(layer, "bias"):
                    biases[name + ".bias"] = layer.bias

        return weights, biases


if __name__ == "__main__":
    use_backend("torch")
    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.pt") for i in range(1, 4)
    ]
    # Create a session
    session = AuthentricsSession()
    model = SimpleModel()
    session.model = model

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_project_" + str(uuid.uuid4()),
        "Example project for comparative_analysis",
    )

    # Register all checkpoints, then select the ones needed for the analysis
    project = session.add_checkpoints(project, *checkpoint_paths)

    # Run comparative_analysis (project and Checkpoint objects from project.checkpoints)
    # layer_names: optional list of layer names to analyze (None = all layers)
    result = session.comparative_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        [
            "squeeze_edit_model.features.6.0.block.2.fc2",
            "squeeze_edit_model.features.6.0.block.2.fc1",
        ],
    )

    print("Comparative analysis completed.")
    print(result)
