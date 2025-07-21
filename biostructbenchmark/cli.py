"""CLI scripts called from __main__.py"""

import argparse
import os
import ast
from pathlib import Path


def validate_file_path(input_path: str) -> Path:
    """Validate file_path and readability"""
    file_path = Path(input_path)
    checks = [
        (Path.exists(file_path), "Path does not exist"),
        (Path.is_file(file_path), "Not a valid file"),
        (os.access(file_path, os.R_OK), "No read permission"),
        (os.path.getsize(file_path) > 0, "File is empty"),
    ]
    for condition, error_message in checks:
        if not condition:
            raise ValueError(f"File Validation Error: {error_message}")
    return file_path


# TODO: improve by loading package info
def get_version() -> str:
    """Get version from __init__.py"""
    for line in open("biostructbenchmark/__init__.py"):
        if line.startswith("__version__ = "):
            return ast.literal_eval(line.split("=")[1].strip())
    return "Undefined version"


def arg_parser() -> argparse.Namespace:
    """Assemble command-line argument processing"""
    parser = argparse.ArgumentParser()

    # Version argument
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="0.0.1",
        help="View BioStructBenchmark version number",
    )

    # File arguments
    parser.add_argument(
        "file_path_observed",
        type=validate_file_path,
        help="Path to observed structure file",
    )
    parser.add_argument(
        "file_path_predicted",
        type=validate_file_path,
        help="Path to predicted structure file",
    )

    # Parse the command line arguments
    return parser.parse_args()
