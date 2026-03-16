import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx.tools.update_model_dims
import onnxruntime as ort
from PIL import Image

from authentrics import AuthentricsSession, InferenceResult, ModelInterface, Parameters


class RealisticModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_data = load_images(
            list(Path("/stimuli/milair/validate/A10").iterdir())[:10]
        )

    def load(self, checkpoint_path: str):
        self.model = onnx.load(checkpoint_path.as_posix())

    def _prepare_model(self, model_input_name: str, model_output_name: str):
        input_dims: list[int | str] = [
            dim.dim_value
            for dim in self.model.graph.input[0].type.tensor_type.shape.dim
        ]
        output_dims: list[int | str] = [
            dim.dim_value
            for dim in self.model.graph.output[0].type.tensor_type.shape.dim
        ]
        input_dims[0] = "batch_size"
        output_dims[0] = "batch_size"
        onnx.tools.update_model_dims.update_inputs_outputs_dims(
            self.model, {model_input_name: input_dims}, {model_output_name: output_dims}
        )

    def perform_inference(
        self,
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:
        model_output_name = "output"
        model_intput_name = "inputs"

        self._prepare_model(model_intput_name, model_output_name)

        if return_intermediate_outputs:
            # Add the outputs for the layers of interest
            for node in self.model.graph.node:
                if layer_names is None or node.name in layer_names:
                    for output in node.output:
                        self.model.graph.output.append(onnx.ValueInfoProto(name=output))

        # Intermediate outputs are collected in the outputs
        session = ort.InferenceSession(self.model.SerializeToString())
        output_names: list[str] = [x.name for x in session.get_outputs()]

        ort_outputs = session.run(
            output_names=output_names,
            input_feed={model_intput_name: self.input_data},
        )

        output_parameters = {
            output_name: output.copy()
            for output_name, output in zip(output_names, ort_outputs, strict=True)
        }
        output = output_parameters[model_output_name]
        intermediate_outputs = (
            output_parameters if return_intermediate_outputs else None
        )

        return InferenceResult(
            output=output,
            intermediate_outputs=Parameters(intermediate_outputs),
        )


def load_images(image_paths: list[str] | list[Path]) -> np.ndarray:
    if isinstance(image_paths[0], str):
        image_paths = [Path(path) for path in image_paths]
    images = [Image.open(path).resize((224, 224)) for path in image_paths]
    image_array: np.ndarray = (
        np.stack(images, axis=0, dtype=np.uint8).astype(np.float32) / 255.0
    )
    return image_array.transpose(0, 3, 1, 2)


if __name__ == "__main__":
    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_1_path = Path("/models/MilAirClassification/checkpoint_1.onnx")
    checkpoint_2_path = Path("/models/MilAirClassification/checkpoint_2.onnx")

    # Create a session
    session = AuthentricsSession()

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_analysis_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "example_project_" + str(uuid.uuid4()),
        "Example project for comparative_analysis",
    )

    # Set a model on the session (required for analysis operations)
    # The model is used to perform inference and extract layer outputs
    model = RealisticModel()
    session.model = model

    # Register checkpoints with the project (required for analysis)
    project = session.add_checkpoints(project, checkpoint_1_path, checkpoint_2_path)

    # Run correlation_analysis (project and Checkpoint objects from project.checkpoints)
    # layer_names: list of layer names to analyze (required)
    result = session.correlation_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        [
            "node_conv2d_2",
            "node_conv2d_87",
        ],
        "output",
    )

    print("Correlation analysis completed.")
    print(result)
