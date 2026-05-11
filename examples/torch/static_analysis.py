from authentrics import AuthentricsSession, use_backend

from authentrics_examples.models.torch import SimpleModel
from authentrics_examples.config import get_example_data_path


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(["checkpoint_1.pt"])

    # Create a session and a minimal model
    model = SimpleModel()
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"checkpoint_{i}.pt" for i in range(1, 8)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MilAirClassificationExample:Torch",
        "Example project for Torch-based CNN model",
    )

    # Run static_analysis: project, then previous and chosen checkpoint (two separate args).
    # parameter_names: list of parameter names to analyze
    result = session.static_analysis(
        project.id,
        checkpoint_paths[0],
        checkpoint_paths[1],
        parameter_names=[
            "squeeze_edit_model.features.0.0.weight",
            "squeeze_edit_model.classifier.1.weight",
            "squeeze_edit_model.classifier.4.weight",
            "squeeze_edit_model.classifier.4.bias",
        ],
    )

    print("Static analysis completed.")
    print(result.summary_score)
    print(result.parameter_changes)
    print(result.parameter_histograms)

    # Save the result to a file
    result.to_hdf5(example_data_path / "static_analysis.h5")


if __name__ == "__main__":
    main()
