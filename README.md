# Authentrics Examples

This repository contains examples of using the Authentrics Python Library.

Each notebook shows a different model:

- [A medical advice chatbot](./hf_medical_chatbot.ipynb)
- [An example of task interference on an LLM](./hf_legal_understanding.ipynb)
- [A CNN classifying military aircraft](./torch_military_aircraft.ipynb)
- [The same CNN in ONNX format](./onnx_military_aircraft.ipynb)

The notebooks themselves exemplify an implementation of a complete `ModelInterface` followed by each type of
analysis we offer. The model checkpoints and sample data are pulled from a secondary Git LFS repo,
[Authentrics-ai/authentrics-analysis-example-models](https://github.com/Authentrics-ai/authentrics-analysis-example-models).

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
