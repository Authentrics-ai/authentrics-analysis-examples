from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from authentrics import (
    InferenceResult,
    ModelInterface,
    Parameters,
    WeightBias,
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

        return WeightBias(Parameters(weights), Parameters(biases))

    def perform_inference(
        self,
        input_data: list[Path] | list[str],
        return_intermediate_outputs: bool = False,
        layer_names: list[str] | None = None,
    ) -> InferenceResult:
        input_data = [Path(path) for path in input_data]
        input_data_array = np.stack(
            [preprocess_image(stimulus) for stimulus in input_data],
            axis=0,
        )

        session = ort.InferenceSession(self.model.SerializeToString())
        output_names: list[str] = [x.name for x in session.get_outputs()]
        input_name = self.model.graph.input[0].name

        intermediate_output_names: list[str] = []
        if return_intermediate_outputs:
            for node in self.model.graph.node:
                if layer_names is None or node.name in layer_names:
                    intermediate_output_names.append(node.output[0])

        all_output_names = output_names + intermediate_output_names

        ort_outputs = session.run(
            output_names=all_output_names,
            input_feed={input_name: input_data_array},
        )
        output = np.asarray(ort_outputs[0])
        intermediate_outputs = {
            name: np.asarray(ort_outputs[i])
            for i, name in enumerate(all_output_names)
            if name in intermediate_output_names
        }
        return InferenceResult(output, Parameters(intermediate_outputs))

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
