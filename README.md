# Authentrics Examples

This repository contains examples of using the Authentrics SDK.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pip install -e '.[all]' --index-url https://us-central1-python.pkg.dev/authentrics/authentrics/simple
```

If you want to try only a specific set of examples, you can install the optional dependencies:

```bash
pip install -e '.[torch]' --index-url https://us-central1-python.pkg.dev/authentrics/authentrics/simple
pip install -e '.[hf]' --index-url https://us-central1-python.pkg.dev/authentrics/authentrics/simple
pip install -e '.[onnx]' --index-url https://us-central1-python.pkg.dev/authentrics/authentrics/simple
```

## Running the examples

The examples use checkpoints located in the `/models` directory and stimulus files located in the `/stimuli` directory.

To run the examples, you need to have the checkpoints and stimulus files available. In the future, we will provide a way to specify paths to the checkpoints and stimulus files.

```bash
python examples/torch/static_analysis.py
python examples/hf/static_analysis.py
python examples/onnx/static_analysis.py
```

To run all the examples as a stress test, you can execute the script `run_all_examples.sh`.
