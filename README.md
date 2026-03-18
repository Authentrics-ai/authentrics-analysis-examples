# Authentrics Examples

This repository contains examples of using the Authentrics SDK.

## Installation

```bash
# Create a virtual environment
python3 -m venv .venv

# Set the extra index url for the virtual environment to download the Authentrics SDK from the Google Cloud Package Registry
echo -e "[global]\nextra-index-url = https://us-central1-python.pkg.dev/authentrics/authentrics/simple\n" > .venv/pip.conf

# Activate the virtual environment
source .venv/bin/activate

# Install the Google Cloud Package Registry authentication library
pip install keyrings.google-artifactregistry-auth

# Install the Authentrics SDK and all the dependencies
pip install -e '.[all]'
```

If you want to try only a specific set of examples, you can install the optional dependencies:

```bash
pip install -e '.[torch]'
pip install -e '.[hf]'
pip install -e '.[onnx]'
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
