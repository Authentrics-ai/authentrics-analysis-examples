import uuid
from pathlib import Path
from typing import Optional

import torch

from authentrics import AuthentricsSession, ModelInterface, WeightBias, use_backend

from models.MilAirClass import MilAirModel


class SimpleModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self._module = MilAirModel()
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        self._module.to(device=self._device)
        self._module.eval()

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


if __name__ == "__main__":
    # Set the backend to torch
    use_backend("torch")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleModel()
    session.model = model

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.pt") for i in range(1, 3)
    ]

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_project_" + str(uuid.uuid4()),
        "Example project for static_analysis",
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
        #     "squeeze_edit_model.features.3.2.block.2.fc2.weight",
        #     "squeeze_edit_model.features.6.4.block.0.0.weight",
        # ],  # Omit to analyze all weights
        # bias_names=[
        #     "squeeze_edit_model.features.3.1.block.2.fc2.bias",
        #     "squeeze_edit_model.features.6.3.block.1.1.bias",
        # ],  # Omit to analyze all biases
    )

    print("Static analysis completed.")
    print(f"Result: {result}")
