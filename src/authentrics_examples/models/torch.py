from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from authentrics import (
    InferenceResult,
    ModelInterface,
    TensorDict,
)

classes = [
    "A10",
    "AH64",
    "B1",
    "B52",
    "C130",
    "C17",
    "C2",
    "EF2000",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "F4",
    "J10",
    "J20",
    "JAS39",
    "Rafale",
    "US2",
    "V22",
]

Handle = torch.utils.hooks.RemovableHandle
Hook = Callable[[torch.nn.Module, object, object], None]


class MilAirModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.num_classes = len(classes)
        self.squeeze_edit_model = torch.hub.load(
            "pytorch/vision:v0.20.1",
            "efficientnet_b3",
        )

        self.squeeze_edit_model.classifier[1] = torch.nn.Linear(1536, 256)
        self.squeeze_edit_model.classifier.extend(
            [
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.45),
                torch.nn.Linear(256, self.num_classes),
            ]
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze_edit_model(x)


def preprocess_image(image_file: Path) -> torch.Tensor:
    """Load and preprocess an image to CHW float32 tensor in [0, 1]."""
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    arr = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
    return arr / 255.0


def _make_capture_hook(layer_name: str, storage: dict[str, torch.Tensor]) -> Hook:
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
) -> list[Handle]:
    """Register forward hooks on the given layers; capture outputs into storage.
    Returns handle list for removal.
    """
    layer_modules = {
        name: mod for name, mod in model.named_modules() if name and name in layer_names
    }
    handles: list[Handle] = []
    for name, module in layer_modules.items():
        if hasattr(module, "register_forward_hook"):
            h: Handle = module.register_forward_hook(_make_capture_hook(name, storage))
            handles.append(h)
    return handles


class SimpleModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self._module = MilAirModel()
        self._device = "cpu"

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._module.load_state_dict(state["model"], strict=True)

        self._module.to(device=self._device)
        self._module.eval()

    def get_parameters(
        self,
        parameter_names: Optional[list[str]] = None,
    ) -> TensorDict:
        parameters = TensorDict()
        for name, param in self._module.named_parameters():
            if parameter_names is None or name in parameter_names:
                parameters[name] = param
        return parameters

    def perform_inference(
        self,
        input_data: list[Path] | list[str],
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:
        input_data = [Path(path) for path in input_data]
        input_data_tensor = torch.stack(
            [preprocess_image(stimulus) for stimulus in input_data],
            axis=0,
        ).to(device=self._device)

        captured: dict[str, torch.Tensor] = {}
        handles: list[Handle] = []
        if return_intermediate_outputs:
            handles = _register_per_layer_output_hooks(
                self._module, captured, layer_names=layer_names
            )

        try:
            with torch.no_grad():
                output = self._module(input_data_tensor)
        finally:
            for h in handles:
                h.remove()

        inter = TensorDict(captured) if captured else {}
        return InferenceResult(output, inter)

    def set_parameters(self, parameters: TensorDict) -> None:
        state = {n: p for n, p in self._module.named_parameters()}
        for name, tensor in parameters.items():
            if name in state:
                state[name].data.copy_(tensor)

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._module.state_dict()}, path)
