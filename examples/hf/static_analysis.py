from authentrics import AuthentricsSession, use_backend

from authentrics_examples.models.hf import SimpleHFModel
from authentrics_examples.config import get_example_data_path


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(["iteration_0"])

    # Create a session and a minimal model
    model = SimpleHFModel()
    session = AuthentricsSession(model)

    # Example checkpoint paths - update these to match your actual checkpoint files
    checkpoint_paths = [example_data_path / f"iteration_{i}" for i in range(4)]
    if any(not checkpoint.exists() for checkpoint in checkpoint_paths):
        raise FileNotFoundError(f"Checkpoint files not found: {checkpoint_paths}")

    # Initialize a project (required for analysis operations)
    project = session.get_or_create_project(
        "MedicalChatbotExample",
        "Example project for Hugging Face-based LLM",
    )

    # Run static_analysis: project, then previous and chosen checkpoint (two separate args).
    # parameter_names: list of parameter names to analyze
    result = session.static_analysis(
        project.id,
        checkpoint_paths[0],
        checkpoint_paths[1],
        parameter_names=[
            "base_model.model.model.layers.10.mlp.down_proj.lora_A.weight"
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
