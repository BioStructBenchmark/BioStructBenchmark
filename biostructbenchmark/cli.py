"""CLI scripts called from __main__.py"""

import argparse
import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


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


def get_version() -> str:
    """Get version from package metadata"""
    try:
        return version("BioStructBenchmark")
    except PackageNotFoundError:
        return "0.0.1"  # Fallback for development


def arg_parser() -> argparse.Namespace:
    """Assemble command-line argument processing"""
    parser = argparse.ArgumentParser()

    # Version argument
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_version(),
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

    # Output options
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Directory to save aligned structures (default: current directory)",
    )
    parser.add_argument(
        "--save-structures",
        action="store_true",
        help="Save aligned structures to output files",
    )

    # Parse the command line arguments
    return parser.parse_args()
