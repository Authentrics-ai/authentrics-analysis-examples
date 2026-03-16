import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx.tools.update_model_dims
import onnxruntime as ort
from onnx.numpy_helper import to_array
from PIL import Image

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    Parameters,
    WeightBias,
)


class RealisticModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_data = load_images(
            list(Path("/stimuli/milair/validate/A10").iterdir())[:10]
        )

    def load(self, checkpoint_path: str):
        self.model = onnx.load(checkpoint_path.as_posix())

    def get_weight_bias_from_layer_names(
        self,
        layer_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        initializer_names = [
            initializer.name for initializer in self.model.graph.initializer
        ]
        layer_set = set(layer_names) if layer_names else None

        # Include params whose names are in layer_names (node names or initializer names)
        for node in self.model.graph.node:
            if layer_set is not None and node.name not in layer_set:
                continue
            for input_name in node.input:
                try:
                    initializer_index = initializer_names.index(input_name)
                except ValueError:
                    continue
                arr = np.asarray(
                    to_array(self.model.graph.initializer[initializer_index]).copy(),
                    dtype=np.float64,
                )
                if "bias" in input_name:
                    biases[input_name] = arr
                else:
                    weights[input_name] = arr

        # Also include any layer_names that are initializer names (weight/bias param names)
        if layer_set is not None:
            for init in self.model.graph.initializer:
                if init.name not in layer_set:
                    continue
                arr = np.asarray(to_array(init).copy(), dtype=np.float64)
                if "bias" in init.name:
                    biases[init.name] = arr
                else:
                    weights[init.name] = arr

        return WeightBias(Parameters(weights), Parameters(biases))

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

    # Run comparative_analysis (project and Checkpoint objects from project.checkpoints).
    # layer_names: optional list of ONNX node names and/or initializer names (weight/bias
    # parameter names); None = all layers. This model accepts initializer names as above.
    result = session.comparative_analysis(
        project,
        project.checkpoints[0],
        project.checkpoints[1],
        [
            "squeeze_edit_model.features.6.0.block.2.fc2.weight",
            "squeeze_edit_model.features.6.0.block.2.fc1.bias",
        ],
    )

    print("Comparative analysis completed.")
    print(result)
