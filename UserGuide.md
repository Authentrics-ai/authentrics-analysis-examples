# User Guide for Authentrics Python Library

## Introduction

Authentrics is a model analysis and maintenance tool for neural networks with built-in checkpoint tracking.

It helps you:

- **Run local analysis on your checkpoints** without sending model weights to a remote service. Only project metadata (names, descriptions) is exchanged with the Authentrics API gateway.
- **Plug in any model backend** by implementing the `ModelInterface` as a thin wrapper around your model.
- **Work with multiple tensor libraries**: NumPy is supported by default, with PyTorch as an optional backend.

At a high level you:

1. Implement `ModelInterface` for your model (reusable across runs).
2. Create an `AuthentricsSession`, point it at your gateway, and log in.
3. Create a `Project` on the gateway via `session.create_project(...)`.
4. Run analyses (static, activation, correlation, ZTOM, exclude training) against checkpoint files on disk.

> **Architecture note:** project + checkpoint metadata lives on the gateway (in MongoDB). Analyses run locally on the customer's machine — model weights never leave it.

---

## Installation

> **Note:** This will be changing in the near future for greater flexibility.

The Authentrics wheel is published for **Linux x86_64** and requires **Python 3.12**.

```bash
pip install authentrics \
  --index-url https://us-central1-python.pkg.dev/authentrics/authentrics-repo/simple/
```

Authentication to the registry uses your Google Application Default Credentials. If you haven't already:

```bash
gcloud auth application-default login
pip install keyrings.google-artifactregistry-auth
```

To pin a specific version:

```bash
pip install 'authentrics==<x.y.z>' \
  --index-url https://us-central1-python.pkg.dev/authentrics/authentrics-repo/simple/
```

Or, if you have the wheel file (e.g., `authentrics-0.27.0-py3-none-manylinux_2_39_x86_64.whl`):

```bash
pip install ./authentrics-0.27.0-py3-none-manylinux_2_39_x86_64.whl
```

> **macOS users:** see the §"Running on macOS" section at the end of this guide.

### Optional: PyTorch backend

The PyTorch backend is opt-in. Install `torch` separately if you intend to use it:

```bash
pip install torch
```

---

## Configuring the gateway URL

The library talks to an Authentrics API gateway. Set the URL programmatically:

```python
import authentrics as ax

session = ax.AuthentricsSession()
session.set_base_url("http://your-gateway:8080")
print(session.get_base_url())
```

Or via the CLI (persisted to user config):

```bash
authrx config set base_url http://your-gateway:8080
authrx config get base_url
authrx config list
```

The supported config keys are `base_url` and `log_level`.

---

## Login

You authenticate with credentials against the gateway. The login endpoint accepts your **email address** in the `username` field.

```python
import authentrics as ax

session = ax.AuthentricsSession()
session.set_base_url("http://your-gateway:8080")

# With explicit credentials
session.login(username="john@example.com", password="my-password")

# Or read from environment variables (AAI_USERNAME / AAI_PASSWORD)
session.login()
```

Or via the CLI:

```bash
# Interactive (prompts for both)
authrx login

# Non-interactive
authrx login -u john@example.com -p my-password
```

### Provisioning a regular user

The default admin account is for user management. To create projects and run analyses, log in as a regular user. An administrator provisions one through the gateway's admin endpoint, for example:

```bash
TOKEN=$(curl -s -X POST $BASE_URL/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin@example.com","password":"<admin-pass>"}')

curl -s -X POST $BASE_URL/api/v2/auth/admin/user \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
        "username":"john",
        "firstName":"John",
        "lastName":"Doe",
        "emailAddress":"john@example.com",
        "password":"a-strong-password"
      }'
```

Once provisioned, log in as that user from the SDK or CLI.

---

## Minimal example

```python
import authentrics as ax

session = ax.AuthentricsSession()
session.set_base_url("http://your-gateway:8080")
session.login(username="john@example.com", password="my-password")

project = session.create_project(
    name="My Experiment",
    description="ResNet-50 baseline",
)

# Provide your ModelInterface implementation (see "Model Analysis & Maintenance" below)
session.model = MyModelInterface()

# Run analysis against checkpoint files on disk
result = session.static_analysis(
    project.id,
    "checkpoints/epoch_1.pt",
    "checkpoints/epoch_3.pt",
)
print(result.weight_summary_score, result.bias_summary_score)
```

---

## Tensor backends

Authentrics works with multiple tensor libraries. NumPy is always available; PyTorch is opt-in. JAX and Tensorflow are planned for the future.

| Function                                 | What it does                             |
| ---------------------------------------- | ---------------------------------------- |
| `authentrics.use_backend(name)`          | Switch to backend `"numpy"` or `"torch"` |
| `authentrics.get_backend()`              | Return the active backend                |
| `authentrics.list_available_backends()`  | Backends installed and ready to use      |
| `authentrics.is_backend_available(name)` | Boolean: is this backend installed?      |

```python
import authentrics as ax

print("Available:", ax.list_available_backends())   # ['numpy', 'torch']

if ax.is_backend_available("torch"):
    ax.use_backend("torch")

print("Current:", ax.get_backend())   # 'torch'
```

After switching backends, `TensorDict` and `InferenceResult` accept tensors from the active backend:

| Backend   | Tensor type     |
| --------- | --------------- |
| `"numpy"` | `numpy.ndarray` |
| `"torch"` | `torch.Tensor`  |

---

## Project management

`Project` objects are returned by session methods. They have the following fields:

| Field         | Type                         |
| ------------- | ---------------------------- |
| `id`          | `str` (server-assigned)      |
| `name`        | `str` (must be unique)       |
| `description` | `str`                        |
| `created_at`  | `datetime` (server-assigned) |
| `checkpoints` | `list[Checkpoint]`           |

Project state lives on the gateway. There is no local state file.

### Creating a project

```python
project = session.create_project(
    name="My Project",
    description="Experiment tracking for ResNet",
)
```

`name` is required; `description` is optional.

### Loading an existing project

```python
project = session.get_project_by_id("69f3b4a5f42cb15537d012e0")
project = session.get_project_by_name("My Project")

# Or with kwargs
project = session.get_project(project_id="...")
project = session.get_project(name="My Project")

# All projects you have access to
projects = session.get_projects()
```

### Updating a project

```python
project.description = "Updated description"
project = session.update_project(project)
```

### Deleting a project

```python
session.delete_project(project)
```

This removes the project from the gateway. The customer's local checkpoint files on disk are not touched — the SDK only ever held metadata, not the files themselves.

---

## Checkpoint management

```python
project = session.add_checkpoints(
    project,
    Checkpoint("Epoch 1", "checkpoints/epoch_1.pt"),
    Checkpoint("Epoch 2", "checkpoints/epoch_2.pt"),
)

checkpoint = checkpoint_paths[0]
checkpoint.name = "After fine-tuning"
project = session.update_checkpoint(
    project,
    checkpoint=checkpoint,
)

project = session.delete_checkpoints(project, checkpoint)
```

---

## Command-line interface (CLI)

The shipped CLI is `authrx`. The available subcommands are:

```
authrx login                  # authenticate (interactive or via -u/-p flags)
authrx config get/set/list    # CLI / session configuration
authrx sys-info               # Python, platform, authentrics, backend info
```

For project and checkpoint management, use the Python SDK directly.

### CLI login

```bash
authrx login
authrx login -u john@example.com -p my-password
```

### CLI configuration

```bash
authrx config set base_url http://gateway.example.com:8080
authrx config get base_url
authrx config list
```

Supported keys: `base_url`, `log_level`.

---

## Model analysis & maintenance

### `ModelInterface`

`ModelInterface` is a thin wrapper around your model. The library calls it to load and save checkpoints, run inference, and read or write the model's parameters. You implement the six abstract methods below.

```python
import authentrics as ax
import torch
import torch.nn as nn

ax.use_backend("torch")


class MyMLP(ax.ModelInterface):
    """A working ModelInterface for a simple PyTorch MLP."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    # 1 — load weights from disk into self.model
    def load(self, checkpoint_path):
        state = torch.load(str(checkpoint_path), weights_only=True)
        self.model.load_state_dict(state)

    # 2 — write the current model state to disk
    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), str(checkpoint_path))

    # 3 — forward pass; optionally collect intermediate layer outputs
    def perform_inference(self, input_data, return_intermediate_outputs, layer_names) -> ax.InferenceResult:
        x = (
            input_data
            if isinstance(input_data, torch.Tensor)
            else torch.as_tensor(input_data)
        )

        intermediates = ax.TensorDict()
        if return_intermediate_outputs:
            handles = []
            for name in layer_names or []:
                module = dict(self.model.named_modules()).get(name)
                if module is None:
                    raise ValueError(f"unknown layer: {name!r}")

                def _hook(_m, _i, out, n=name):
                    intermediates[n] = out.detach()

                handles.append(module.register_forward_hook(_hook))
            try:
                with torch.no_grad():
                    output = self.model(x)
            finally:
                for h in handles:
                    h.remove()
        else:
            with torch.no_grad():
                output = self.model(x)

        return ax.InferenceResult(output, intermediates, None)

    # 4 — return the named weight + bias tensors as a TensorDict.
    # When parameter_names is None, return ALL weights / biases.
    def get_parameters(self, parameter_names=None) -> ax.TensorDict:
        sd = self.model.state_dict()
        params = ax.TensorDict()

        if parameter_names is None:
            for k, v in sd.items():
                params[k] = v
        else:
            for n in parameter_names:
                if n in sd:
                    params[n] = sd[n]

        return params

    # 5 — same as get_parameters, expanding "<layer>" to
    # "<layer>.weight" + "<layer>.bias" (PyTorch convention)
    def get_parameters_from_layer_names(self, layer_names=None) -> ax.TensorDict:
        if layer_names is None:
            return self.get_parameters()

        parameter_names = [f"{n}.weight" for n in layer_names]
        parameter_names.extend([f"{n}.bias" for n in layer_names])
        return self.get_parameters(parameter_names)

    # 6 — apply the given TensorDict back onto self.model
    def set_parameters(self, parameters: ax.TensorDict):
        sd = dict(self.model.state_dict())
        for name, tensor in parameters.items():
            sd[name] = tensor
        self.model.load_state_dict(sd)
```

#### Key types

| Type                                                                                  | Shape                                                                                            |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `ax.TensorDict`                                                                       | dict-like (`items`, `keys`, `values`, `pop`) — maps name → tensor                                |
| `ax.InferenceResult(output, intermediate_outputs=TensorDict(), additional_data=None)` | result of `perform_inference`; access via `.output`, `.intermediate_outputs`, `.additional_data` |

#### Required `ModelInterface` methods per analysis

| Analysis         | `get_parameters` | `get_parameters_from_layer_names` | `perform_inference` | `set_parameters` | `save` |
| ---------------- | ---------------- | --------------------------------- | ------------------- | ---------------- | ------ |
| Static           | ✓                |                                   |                     |                  |        |
| Activation       |                  | ✓                                 | ✓                   |                  |        |
| Correlation      |                  |                                   | ✓                   |                  |        |
| ZTOM             | ✓                |                                   | ✓                   | ✓                | ✓      |
| Exclude training | ✓                |                                   |                     | ✓                | ✓      |

All five analyses additionally require `load`.

### Static analysis

Compares the weights and biases of two checkpoints, returning per-parameter scores of change between the two,
a summary score of the whole model, and histograms of the changes of each parameter for easy visual analysis.

```python
import matplotlib.pyplot as plt
import numpy as np

result = session.static_analysis(
    project.id,
    previous_checkpoint_path="checkpoints/epoch_1.pt",
    chosen_checkpoint_path="checkpoints/epoch_3.pt",
    # parameter_names=["fc1.weight", "fc1.bias"]   # optional
)
print(
    result.summary_score,
    result.parameter_names,
    result.parameter_scores,
    result.parameter_histograms,
)

plt.figure("Scores")
plt.plot(result.parameter_names, result.parameter_scores, 'ro-')

for name in result.parameter_names:
    histogram = result.get_parameter_histogram(name)
    bin_edges = np.array(histogram.bin_edges)
    counts = np.array(histogram.bin_counts)
    bin_width = bin_edges[1] - bin_edges[0]
    centers = (bin_edges[1:] - bin_edges[:-1]) / 2

    plt.figure(name)
    plt.bar(centers, counts, width=bin_width)

plt.show()
```

If `parameter_names` is omitted, the user's implementation of `get_parameters` decides the behavior, but we suggest that all parameters are returned.

### Activation analysis

Computes both the similarity (cosine similarity) and difference (L2 distance) between the activation outputs of the given layers. This indicates the degree to which a model's changes affect each layer's output while maintaining the same input.

```python
import torch
input_data = torch.randn(8, 10)

result = session.activation_analysis(
    project.id,
    previous_checkpoint_path="checkpoints/epoch_1.pt",
    chosen_checkpoint_path="checkpoints/epoch_3.pt",
    input_data=input_data,
    layer_names=["fc1", "fc2"],
)

print(
    result.layer_names,
    result.cosine_similarities,
    result.l2_distances,
)
```

### Correlation analysis

Analyzes the statistical correlation between intermediate layer outputs and a reference layer, typically the output layer. Both intermediate and reference layers must produce tensors whose first dimension is the batch size.
A layer with a high correlation score strongly influences the output (either positively or negatively), while one with a correlation score close to 0 has little influence for the input provided.

```python
result = session.correlation_analysis(
    project.id,
    checkpoint_path="checkpoints/epoch_1.pt",
    input_data=input_data,
    layer_names=["fc1"],
    reference_layer_name="fc2",
)

print(
    result.layer_names,
    result.correlation_scores,
    result.correlation_histograms,
)
```

### Exclude training

Removes the effect of training for a given transition between checkpoints by walking back through the checkpoint sequence.

> Transitions are indexed by the "to-checkpoint"; i.e., for the example below, index 1 would indicate that the transition from `epoch_1.pt` to `epoch_2.pt` should be excluded, and index 2 would indicate the transition between `epoch_2.pt` and `epoch_3.pt`. **Index 0 is not allowed,** as the transition to `epoch_1.pt` is not recorded.

```python
result = session.exclude_training(
    project.id,
    checkpoint_sequence=[
        "checkpoints/epoch_1.pt",
        "checkpoints/epoch_2.pt",
        "checkpoints/epoch_3.pt",
    ],
    checkpoint_indices=[2],   # exclude training between epoch_2.pt and epoch_3.pt
    new_checkpoint_path="excluded.pt",
)
print(result.new_checkpoint_path)  # The resulting file should perform identically to "checkpoints/epoch_2.pt"
```

### ZTOM (Zero-Train Optimization & Maintenance)

This maintenance scales the transitions between consecutive checkpoints to minimize (or maximize) a user-defined loss. The optimization uses NLopt's COBYLA algorithm under the hood.

```python
# Default options, override as needed
opts = ax.ZtomOptimizationOptions(
    max_evaluations=50,
    xtol_rel=1e-4,
    ftol_rel=1e-4,
    lower_bound=-1.0,
    upper_bound=1.0,
    minimize=True,
)

def loss_fn(y_true, y_pred):
    ...

input_data = ...
expected_outputs = ...

result = session.ztom_analysis(
    project.id,
    checkpoint_sequence=[
        "checkpoints/epoch_1.pt",
        "checkpoints/epoch_2.pt",
        "checkpoints/epoch_3.pt",
    ],
    input_data=input_data,
    loss_function=lambda y_pred: loss_fn(expected_outputs, y_pred),
    new_checkpoint_path="ztom.pt",
    optimization_options=opts,
)

print(result.original_loss, "→", result.best_loss)
print("scaling factors:", result.scaling_factors)
print("inferences:", result.number_of_inferences)
```

The loss function returns a Python float, not a tensor.

### Result types

| Analysis         | Result type                 | Key attributes                                                                                       |
| ---------------- | --------------------------- | ---------------------------------------------------------------------------------------------------- |
| Static           | `StaticAnalysisResult`      | `summary_score`, `parameter_scores`                                                                  |
| Activation       | `ActivationAnalysisResult`  | `l2_distances`, `cosine_similarities`                                                                |
| Correlation      | `CorrelationAnalysisResult` | `correlation_scores`                                                                                 |
| Exclude training | `ExcludeTrainingResult`     | `new_checkpoint_path`                                                                                |
| ZTOM             | `ZtomAnalysisResult`        | `original_loss`, `best_loss`, `scaling_factors`, `number_of_inferences`, `optimized_checkpoint_path` |

---

## Result serialization

Most analysis results can be serialised to HDF5 with `.to_hdf5(path)` and re-loaded with the corresponding class's `from_hdf5(path)` static method.

```python
result = session.static_analysis(project.id, "ckpt_a.pt", "ckpt_b.pt")
result.to_hdf5("static_result.h5")

# Later
result2 = ax.StaticAnalysisResult.from_hdf5("static_result.h5")
print(result2.weight_summary_score, result2.bias_summary_score)
```

| Result type                 | `to_hdf5` / `from_hdf5`                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| `StaticAnalysisResult`      | ✓                                                                              |
| `ActivationAnalysisResult`  | ✓                                                                              |
| `CorrelationAnalysisResult` | ✓                                                                              |
| `ZtomAnalysisResult`        | ✓                                                                              |
| `ExcludeTrainingResult`     | ✗ — its only output is the new checkpoint file at `result.new_checkpoint_path` |

> **bfloat16 caveat:** if you save a result whose tensors are `bfloat16` under the `'torch'` backend and later load it under `'numpy'`, the load will error — NumPy has no `bfloat16` dtype. Standard `float32` / `float64` performs round-trip serializations cleanly across backends.

---

## Running on macOS

Authentrics wheels are currently published only for Linux x86_64. macOS users (both Intel and Apple Silicon) cannot `pip install` directly:

```
ERROR: Could not find a version that satisfies the requirement authentrics
ERROR: No matching distribution found for authentrics
```

> Both macOS and Windows support are planned for the future.

### Recommended setup: run inside a Linux container

Run the SDK and your analysis code inside a Linux x86_64 container. Apple Silicon hosts will use Rosetta emulation transparently.

A reference setup is below. Save as `Dockerfile`:

```dockerfile
FROM --platform=linux/amd64 python:3.12-slim

ARG GOOGLE_OAUTH_TOKEN

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
        --index-url "https://oauth2accesstoken:${GOOGLE_OAUTH_TOKEN}@us-central1-python.pkg.dev/authentrics/authentrics-repo/simple/" \
        --extra-index-url "https://pypi.org/simple/" \
        authentrics

# Sanity check at build time so wheel/Python ABI mismatches fail fast
RUN python -c "import authentrics; print('authentrics', authentrics.__version__)"
```

Build and run:

```bash
export GOOGLE_OAUTH_TOKEN=$(gcloud auth application-default print-access-token)

docker build --build-arg GOOGLE_OAUTH_TOKEN -t authrx-client .

docker run --rm -it \
  -v "$(pwd):/workspace" \
  authrx-client \
  bash
```

Inside the container, your workspace is mounted at `/workspace` and the SDK is installed at the requested version.

### Network connectivity

If your gateway also runs in Docker, attach the client container to the same Docker network so it can reach the gateway by service name:

```bash
docker run --rm -it \
  --platform linux/amd64 \
  --network <gateway-docker-network> \
  -v "$(pwd):/workspace" \
  -e AAI_BASE_URL=http://<gateway-service-name>:8080 \
  authrx-client \
  bash
```

---

## Support

If you have questions or feedback, contact us at [info@authentrics.ai](mailto:info@authentrics.ai).

When reporting an issue, include the output of `authentrics.get_system_info()` (or `authrx sys-info` from the CLI).
