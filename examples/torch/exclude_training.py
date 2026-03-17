import uuid
from pathlib import Path
from typing import Optional

import torch

from authentrics import AuthentricsSession, ModelInterface, WeightBias, use_backend

from models.MilAirClass import MilAirModel


class SimpleModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self._module = MilAirModel()
        self._device = "cpu"

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
                    weights[name] = param.data.clone()
            elif last_part == "bias":
                if bias_names is None or name in bias_names:
                    biases[name] = param.data.clone()
        return weights, biases

    def set_weight_bias(self, weight_bias: WeightBias) -> None:
        for name, tensor in self._module.named_parameters():
            if name in weight_bias.weights:
                tensor.data.copy_(weight_bias.weights[name])
            elif name in weight_bias.biases:
                tensor.data.copy_(weight_bias.biases[name])

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._module.state_dict()}, path)


if __name__ == "__main__":
    use_backend("torch")
    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.pt") for i in range(1, 8)
    ]
    new_checkpoint_path = Path("./checkpoint_excluded.pt")

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
        "Example project for exclude_training",
    )

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
