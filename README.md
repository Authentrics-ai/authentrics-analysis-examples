# Authentrics Examples

This repository contains examples of using the Authentrics Python Library.

Each notebook shows a different model:

- [A medical advice chatbot](./hf_medical_chatbot.ipynb) — a LoRA-tuned LLM, walking through each analysis type across its fine-tuning checkpoints
- [An example of task interference on an LLM](./hf_legal_understanding.ipynb) — detecting and fixing regressions caused by continued fine-tuning, without retraining
- [A CNN classifying military aircraft](./torch_military_aircraft.ipynb) — tracing a mislabeled-data checkpoint back to its cause and impact
- [The same CNN in ONNX format](./onnx_military_aircraft.ipynb) — the identical scenario, run against ONNX-exported checkpoints

The notebooks themselves exemplify an implementation of a complete `ModelInterface` followed by each type of
analysis we offer. The model checkpoints and sample data are pulled from a secondary Git LFS repo,
[Authentrics-ai/authentrics-analysis-example-models](https://github.com/Authentrics-ai/authentrics-analysis-example-models).

## Reference Hardware & Configuration

The notebooks were developed and run on the following setup. You don't need to match it exactly, but it's a
useful reference if you hit performance differences or version mismatches.

- **OS:** Ubuntu 24.04.4 LTS, kernel 7.0.0-1011-gcp
- **CPU:** Intel Xeon @ 2.20GHz, 12 cores / 24 threads (1 socket)
- **RAM:** 94 GiB
- **GPU:** 2x NVIDIA L4 (24 GB each), driver 580.126.20, CUDA 12.8
- **Python:** 3.12.3

`authentrics` was `0.38.0` and `torch` was pinned to `2.10.0` across every notebook that uses it. `transformers`
is **not** shared between the two HF notebooks — each notebook installs its own pins in its first cell:

| Notebook | `transformers` | Other notable pins |
| --- | --- | --- |
| `hf_medical_chatbot.ipynb` | `~=4.53.3` | `torchvision`, `sentence-transformers`, `peft==0.18.1` |
| `hf_legal_understanding.ipynb` | `==5.2.0` | `datasets==4.8.4`, `peft==0.18.1`, `rapidfuzz==3.14.3` |
| `torch_military_aircraft.ipynb` | n/a | `torchvision`, `pillow` |
| `onnx_military_aircraft.ipynb` | n/a | `onnx` (unpinned, resolved to `1.23.2`), `onnxruntime` (unpinned, resolved to `1.31.1`) |
> **Note:** these notebooks are currently tested against `authentrics` SDK version `0.38.0`.
> Other versions may require API adjustments.

## Running Notebooks

First, please follow the instructions on our [website](https://app.authentrics.ai/docs/setup) to sign up and receive
a download link for the library (a wheel file).

Once you have the wheel file, generate an API key on our website, install the wheel:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install path/to/wheelfile/authentrics-*.whl
```

and register your machine useing the API key:

```bash
$ authrx init
Enter your API key: <paste it here>
```

Then you can open the notebook in your preferred runtime (using the same virtual environment) and enjoy!

## Known issues

- **Platform support:** the published wheel only targets Python 3.12 on Linux x86_64. There is no
  Windows or macOS wheel, so those platforms currently need a Linux container or VM.
- **`transformers` version:** `hf_legal_understanding.ipynb` pins `transformers==5.2.0`, while
  `hf_medical_chatbot.ipynb` pins `transformers~=4.53.3`. These two notebooks are not meant to
  share one virtualenv — each notebook installs its own pinned dependencies in its first cell, so
  run them in separate environments if you want to run more than one.
