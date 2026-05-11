from pathlib import Path


def get_example_data_path(should_contain: list[str]) -> Path:
    path = input(
        f"The example data directory should contain: {should_contain}\n"
        "Enter the path to the example data directory (default: `.`): "
    )
    return Path(path) if path else Path()
