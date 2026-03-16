import uuid
from pathlib import Path
from typing import Optional

import onnx
import onnx.tools.update_model_dims
from onnx.numpy_helper import to_array

from authentrics import AuthentricsSession, ModelInterface, Parameters, WeightBias


class RealisticModel(ModelInterface):
    def load(self, model_path: str | Path):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        if not model_path.is_file():
            raise NotADirectoryError(f"Model path {model_path} is not a file")

        self.model = onnx.load(model_path.as_posix())

    def get_weight_bias(
        self,
        weight_names: Optional[list[str]] = None,
        bias_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        for initializer in self.model.graph.initializer:
            if "bias" in initializer.name:
                biases[initializer.name] = to_array(initializer).copy()
            else:
                weights[initializer.name] = to_array(initializer).copy()
        return WeightBias(Parameters(weights), Parameters(biases))


if __name__ == "__main__":
    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.onnx") for i in range(1, 3)
    ]

    # Create a session
    session = AuthentricsSession()

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_project_" + str(uuid.uuid4()),
        "Example project for static_analysis",
    )

    # Set a model on the session (required for analysis operations)
    model = RealisticModel()
    model.load(checkpoint_paths[0])
    session.model = model

    # Register all checkpoints, then select the ones needed for the analysis
    project = session.add_checkpoints(project, *checkpoint_paths)

    # Run static_analysis: project, then previous and chosen Checkpoint (two separate args).
    # Optionally pass weight_names=[...] and/or bias_names=[...] to restrict to specific layers;
    # omit both to analyze all weights and biases.
    result = session.static_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        # weight_names=["node_Gemm_844", "node_Gemm_850"],  # Omit to analyze all weights
        # bias_names=["node_Bias_844", "node_Bias_850"],  # Omit to analyze all biases
    )

    print("Static analysis completed.")
    print(f"Result: {result}")
