import uuid
from pathlib import Path

import onnx

from authentrics import (
    AuthentricsSession,
    ModelInterface,
    WeightBias,
)


class SimpleModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        self.model = onnx.load(checkpoint_path)

    def get_weight_bias(
        self, weight_names: list[str] | None = None, bias_names: list[str] | None = None
    ) -> WeightBias:
        weights = {}
        biases = {}
        for initializer in self.model.graph.initializer:
            if weight_names is not None and initializer.name not in weight_names:
                continue
            if bias_names is not None and initializer.name not in bias_names:
                continue

            last_part = initializer.name.rsplit(".", 1)[-1]
            if "bias" == last_part:
                biases[initializer.name] = onnx.numpy_helper.to_array(initializer)
            elif "weight" == last_part:
                weights[initializer.name] = onnx.numpy_helper.to_array(initializer)

        return weights, biases

    def set_weight_bias(self, weight_bias: WeightBias) -> None:
        for initializer in self.model.graph.initializer:
            if initializer.name in weight_bias.weights:
                weight = weight_bias.weights[initializer.name]
                initializer.CopyFrom(
                    onnx.numpy_helper.from_array(weight, initializer.name)
                )
            elif initializer.name in weight_bias.biases:
                bias = weight_bias.biases[initializer.name]
                initializer.CopyFrom(
                    onnx.numpy_helper.from_array(bias, initializer.name)
                )

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        onnx.save(self.model, checkpoint_path)


if __name__ == "__main__":
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.onnx") for i in range(1, 8)
    ]
    new_checkpoint_path = Path("./checkpoint_excluded.onnx")

    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleModel()

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_project_" + str(uuid.uuid4()),
        "Example project for exclude_training",
    )

    session.model = model

    # Register checkpoints with the project
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
