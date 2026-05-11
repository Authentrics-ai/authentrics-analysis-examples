import json
from pathlib import Path
from typing import Any

from authentrics import AuthentricsSession, ZtomOptimizationOptions, use_backend
from sentence_transformers import SentenceTransformer

from authentrics_examples.models.hf import SimpleHFModel
from authentrics_examples.config import get_example_data_path


def _get_input_data(stimuli: Path) -> Any:
    if not stimuli.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli}")
    with open(stimuli, "r") as f:
        return [json.loads(line) for line in f]


def _get_expected_output(output_path: Path) -> list[str]:
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    outputs = open(output_path).readlines()
    return [line.strip() for line in outputs]


def _sentence_similarity_loss_fn(
    expected_output: list[str],
    output: list[list[dict[str, str]]],
) -> float:
    similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generated_embeddings = similarity_model.encode(
        [o[0]["generated_text"] for o in output]
    )
    expected_embeddings = similarity_model.encode(expected_output)
    score = (
        similarity_model.similarity_pairwise(generated_embeddings, expected_embeddings)
        .mean()
        .item()
    )
    return score


def main():
    # Set the backend to torch
    use_backend("torch")

    example_data_path = get_example_data_path(
        ["stimuli.jsonl", "expected_outputs.txt", "iteration_0"]
    )

    input_data = _get_input_data(example_data_path / "stimuli.jsonl")
    expected_output = _get_expected_output(example_data_path / "expected_outputs.txt")

    # Create a session and a minimal model
    model = SimpleHFModel(inference_config={"max_new_tokens": 50}, batch_size=1)
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

    # Optional: tune optimization (defaults are often sufficient)
    # In this case, we want to maximize the similarity between the generated and expected outputs.
    options = ZtomOptimizationOptions(minimize=False)

    # loss_function: callable(model_output) -> float
    # can be passed directly as a lambda or normal function
    result = session.ztom_analysis(
        project.id,
        checkpoint_paths,
        input_data,
        loss_function=lambda y_pred: _sentence_similarity_loss_fn(
            expected_output, y_pred
        ),
        new_checkpoint_path=example_data_path / "checkpoint_optimized",
        optimization_options=options,
    )

    print("ZTOM analysis completed.")
    print(f"Optimized checkpoint saved to: {result.new_checkpoint_path}")
    print(
        f"Original loss: {result.original_loss:.6f}, best loss: {result.best_loss:.6f}"
    )
    print(f"Scaling factors: {result.scaling_factors}")
    print(f"Number of inferences: {result.number_of_inferences}")


if __name__ == "__main__":
    main()
