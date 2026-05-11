from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from authentrics import (
    InferenceResult,
    ModelInterface,
    TensorDict,
)


def preprocess_image(image_file: Path) -> np.ndarray:
    image = np.array(
        Image.open(image_file).convert("RGB").resize((224, 224)),
        dtype=np.float32,
    ).transpose(2, 0, 1)
    return image / 255.0


class SimpleModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        self.model = onnx.load(checkpoint_path)

    def get_parameters(self, parameter_names: list[str] | None = None) -> TensorDict:
        parameters = TensorDict()
        for initializer in self.model.graph.initializer:
            if parameter_names is not None and initializer.name not in parameter_names:
                continue

            parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

        return parameters

    def perform_inference(
        self,
        input_data: np.ndarray,
        return_intermediate_outputs: bool = False,
        layer_names: list[str] | None = None,
    ) -> InferenceResult:
        output_names: list[str] = [x.name for x in self.model.graph.output]
        input_name = self.model.graph.input[0].name

        intermediate_output_names: list[str] = []
        if return_intermediate_outputs:
            for node in self.model.graph.node:
                if layer_names is None or node.name in layer_names:
                    intermediate_output_names.append(node.output[0])
                    self.model.graph.output.append(
                        onnx.ValueInfoProto(name=node.output[0])
                    )

        all_output_names = output_names + intermediate_output_names

        session = ort.InferenceSession(self.model.SerializeToString())
        ort_outputs = session.run(
            output_names=all_output_names,
            input_feed={input_name: input_data},
        )

        output = np.asarray(ort_outputs[0])
        intermediate_outputs = TensorDict(
            {
                name: np.asarray(ort_outputs[i])
                for i, name in enumerate(all_output_names)
                if name in intermediate_output_names
            }
        )

        return InferenceResult(output, intermediate_outputs)

    def set_parameters(self, parameters: TensorDict) -> None:
        for initializer in self.model.graph.initializer:
            if initializer.name in parameters:
                initializer.CopyFrom(
                    onnx.numpy_helper.from_array(
                        parameters[initializer.name], initializer.name
                    )
                )

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        onnx.save(self.model, checkpoint_path)
