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

Key package versions:

| Package | Version |
| --- | --- |
| `authentrics` | 0.38.0 |
| `torch` | 2.10.0 |
| `torchvision` | 0.25.0 |
| `transformers` | 4.53.3 |
| `sentence-transformers` | 5.7.0 |
| `onnx` | 1.23.2 |
| `onnxruntime` | 1.31.1 |
| `scikit-learn` | 1.9.1 |
| `numpy` | 2.4.3 |

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
