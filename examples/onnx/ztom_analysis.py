"""Minimal example of ZTOM (Zero Train Optimization and Maintenance) analysis.

ZTOM optimizes scaling factors over training deltas between consecutive checkpoints
to minimize a user-defined loss on the model output. The project must have at least
two checkpoints (order matters: they define the sequence of training deltas).
"""

import uuid
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from authentrics import (
    AuthentricsSession,
    InferenceResult,
    ModelInterface,
    Parameters,
    WeightBias,
    ZtomOptimizationOptions,
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
        self.stimuli = sorted(Path("/stimuli").glob("F*.jpg"))
        if len(self.stimuli) == 0:
            raise RuntimeError("No stimuli found")
        self.input_data = np.stack(
            [preprocess_image(stimulus) for stimulus in self.stimuli],
            axis=0,
        )

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
        return_intermediate_outputs: bool = False,
        layer_names: list[str] | None = None,
    ) -> InferenceResult:
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
            input_feed={input_name: self.input_data},
        )
        output = np.asarray(ort_outputs[0])
        intermediate_outputs = {
            name: np.asarray(ort_outputs[i])
            for i, name in enumerate(all_output_names)
            if name in intermediate_output_names
        }
        return InferenceResult(output, Parameters(intermediate_outputs))

    def set_weight_bias(self, weight_bias: WeightBias) -> None:
        for weight_name, weight in weight_bias.weights.items():
            for initializer in self.model.graph.initializer:
                if initializer.name == weight_name:
                    initializer.CopyFrom(
                        onnx.numpy_helper.from_array(weight, weight_name)
                    )
                    break
        for bias_name, bias in weight_bias.biases.items():
            for initializer in self.model.graph.initializer:
                if initializer.name == bias_name:
                    initializer.CopyFrom(onnx.numpy_helper.from_array(bias, bias_name))
                    break

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        onnx.save(self.model, checkpoint_path)


def milair_loss_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the categorical cross-entropy loss between y_true and y_pred.

    y_true: numpy array of class indices (ground truth labels), shape (batch,).
    y_pred: numpy array of predicted logits, shape (batch, num_classes).

    Returns:
        scalar float loss (mean over batch).
    """
    logits = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.intp).ravel()
    # Numerically stable log-softmax: log_softmax_i = logits_i - logsumexp(logits)
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True)) + logits_max
    )
    log_softmax = logits - log_sum_exp
    # NLL: -log_softmax at the true class index
    n_samples = logits.shape[0]
    nll = -log_softmax[np.arange(n_samples), y_true]
    return float(np.mean(nll))


if __name__ == "__main__":
    # Create a session and a minimal model
    session = AuthentricsSession()
    model = SimpleModel()

    # Initialize a project (required for analysis operations)
    project_path = Path("./my_ztom_project")
    project_path.mkdir(parents=True, exist_ok=True)
    project = session.init_project(
        project_path,
        "ztom_example_" + str(uuid.uuid4()),
        "Example project for ZTOM analysis",
    )
    checkpoint_paths = [
        Path(f"/models/MilAirClassification/checkpoint_{i}.onnx") for i in range(1, 4)
    ]
    new_checkpoint_path = Path("./checkpoint_optimized.onnx")

    session.model = model

    # Register checkpoints with the project
    project = session.add_checkpoints(project, *checkpoint_paths)

    # Optional: tune optimization (defaults are often sufficient)
    options = ZtomOptimizationOptions(
        max_evaluations=50,
        xtol_rel=1e-4,
        ftol_rel=1e-4,
        lower_bound=-1.0,
        upper_bound=1.0,
        minimize=True,
    )

    y_true = np.array([9, 9, 9, 10, 13, 13, 13], dtype=np.int32)

    # loss_function: callable(model_output) -> float
    # can be passed directly or as LossFunction(loss_function)
    result = session.ztom_analysis(
        project,
        lambda y_pred: milair_loss_function(y_true, y_pred),
        new_checkpoint_path,
        options,
    )

    print("ZTOM analysis completed.")
    print(f"Optimized checkpoint saved to: {result.optimized_checkpoint_path}")
    print(
        f"Original loss: {result.original_loss:.6f}, best loss: {result.best_loss:.6f}"
    )
    print(f"Scaling factors: {result.scaling_factors}")
    print(f"Number of inferences: {result.number_of_inferences}")
