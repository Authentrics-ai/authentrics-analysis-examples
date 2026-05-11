from pathlib import Path
from typing import Any, Callable, Optional

import torch
from peft import PeftConfig
from transformers.models.auto import AutoConfig
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from authentrics import (
    InferenceResult,
    ModelInterface,
    WeightBias,
)

Handle = torch.utils.hooks.RemovableHandle
Hook = Callable[[torch.nn.Module, object, object], None]


def _make_capture_hook(layer_name: str, storage: dict[str, torch.Tensor]) -> Hook:
    """Return a forward hook that stores this layer's output in storage (torch, CPU)."""

    def hook(
        _module: torch.nn.Module, _input: object, output: torch.Tensor | tuple
    ) -> None:
        out = output[0] if isinstance(output, tuple) else output
        storage[layer_name] = out.detach().clone()

    return hook


def _register_per_layer_output_hooks(
    model: TextGenerationPipeline,
    storage: dict[str, torch.Tensor],
    layer_names: list[str] | None = None,
) -> list[Handle]:
    """Register forward hooks on the given layers; capture outputs into storage.
    Returns handle list for removal.
    """
    layer_modules = {
        name: mod
        for name, mod in model.model.named_modules()
        if name != "" and name in layer_names
    }
    handles: list[Handle] = []
    for name, module in layer_modules.items():
        if hasattr(module, "register_forward_hook"):
            h: Handle = module.register_forward_hook(_make_capture_hook(name, storage))
            handles.append(h)
    return handles


class SimpleHFModel(ModelInterface):
    def __init__(
        self, inference_config: dict[str, Any] | None = None, batch_size: int = 1
    ):
        super().__init__()
        self._module = None
        self._inference_config = inference_config
        self._batch_size = batch_size

    def load(self, checkpoint_path: Path | str | bytes) -> None:
        config_filepaths = list(checkpoint_path.rglob("*config.json"))
        model_path = config_filepaths[0].parent

        try:
            peft_config = PeftConfig.from_pretrained(str(model_path))
            config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
        except Exception:
            config = AutoConfig.from_pretrained(str(model_path))

        if hasattr(config, "torch_dtype"):
            torch_dtype = config.torch_dtype
        else:
            torch_dtype = None

        # Load model with proper device placement and quantization handling
        # Use device_map="sequential" to handle quantization and multi-device models
        # If that doesn't work, use device_map="auto" or "balanced_low_0"
        self._module: TextGenerationPipeline = pipeline(  # type: ignore
            "text-generation",
            model=str(model_path),
            device_map="sequential",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # Set pad_token_id for batching support
        if (
            self._module.tokenizer is not None
            and self._module.tokenizer.pad_token_id is None
        ):
            if (
                hasattr(self._module.model.config, "eos_token_id")
                and self._module.model.config.eos_token_id is not None
            ):
                self._module.tokenizer.pad_token_id = (
                    self._module.model.config.eos_token_id
                )
            else:
                # Fallback: set pad_token_id to 0 if eos_token_id is not available
                self._module.tokenizer.pad_token_id = 0

    def get_parameters(
        self,
        weight_names: Optional[list[str]] = None,
        bias_names: Optional[list[str]] = None,
    ) -> WeightBias:
        weights = {}
        biases = {}
        for name, param in self._module.model.named_parameters():
            last_part = name.rsplit(".", 1)[-1]
            if last_part == "weight":
                if weight_names is None or name in weight_names:
                    weights[name] = param.detach().cpu()
            elif last_part == "bias":
                if bias_names is None or name in bias_names:
                    biases[name] = param.detach().cpu()
        return weights, biases

    def perform_inference(
        self,
        input_data: list[str],
        return_intermediate_outputs: bool = False,
        layer_names: Optional[list[str]] = None,
    ) -> InferenceResult:

        intermediate_outputs: dict[str, torch.Tensor] = {}
        handles: list[Handle] = []
        if return_intermediate_outputs:
            handles = _register_per_layer_output_hooks(
                self._module, intermediate_outputs, layer_names=layer_names
            )

        max_new_tokens = int(self._inference_config.pop("max_new_tokens", 50))
        chat_template = self._inference_config.pop("chat_template", None)

        # Perform inference
        assert self._module.tokenizer is not None
        if chat_template is not None:
            self._module.tokenizer.chat_template = chat_template

        try:
            result = self._module(
                text_inputs=input_data,
                max_new_tokens=max_new_tokens,
                batch_size=self._batch_size,
                chat_template=chat_template,
                **self._inference_config,
            )
        finally:
            for h in handles:
                h.remove()

        return InferenceResult(
            output=result,
            intermediate_outputs=intermediate_outputs,
        )

    def set_parameters(self, weight_bias: WeightBias) -> None:
        state = {n: p for n, p in self._module.model.named_parameters()}
        for name, tensor in weight_bias.weights.items():
            if name in state:
                state[name].data.copy_(tensor)
        for name, tensor in weight_bias.biases.items():
            if name in state:
                state[name].data.copy_(tensor)

    def save(self, checkpoint_path: Path | str | bytes) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._module.save_pretrained(path)
