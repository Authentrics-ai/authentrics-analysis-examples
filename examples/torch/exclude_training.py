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

    # Run exclude_training (project, list of checkpoint paths, list of checkpoint indices, new checkpoint path)
    result = session.exclude_training(
        project.id,
        checkpoint_paths,
        [3],
        example_data_path / "checkpoint_excluded.pt",
    )

    print("Exclude training completed.")
    print(result.new_checkpoint_path)


if __name__ == "__main__":
    main()
