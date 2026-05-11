from authentrics import AuthentricsSession

from authentrics_examples.models.onnx import SimpleModel
from authentrics_examples.config import get_example_data_path


def main():
    example_data_path = get_example_data_path(["checkpoint_1.onnx"])

    # Create a session and a minimal model
    model = SimpleModel()
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"checkpoint_{i}.onnx" for i in range(1, 8)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MilAirClassificationExample:ONNX",
        "Example project for ONNX-based CNN model",
    )

    # Run exclude_training (project, list of checkpoint paths, list of checkpoint indices, new checkpoint path)
    result = session.exclude_training(
        project.id,
        checkpoint_paths,
        [3],
        example_data_path / "checkpoint_excluded.onnx",
    )

    print("Exclude training completed.")
    print(result.new_checkpoint_path)


if __name__ == "__main__":
    main()
