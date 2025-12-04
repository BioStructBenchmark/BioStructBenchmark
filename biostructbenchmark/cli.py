"""CLI scripts called from __main__.py"""

import argparse
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def setup_logging(verbose: int = 0) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
    """
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:  # verbose >= 2
        level = logging.DEBUG

    # Configure root logger for biostructbenchmark
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Set level specifically for our package
    logging.getLogger("biostructbenchmark").setLevel(level)


def validate_file_path(input_path: str) -> Path:
    """Validate file_path and readability"""
    file_path = Path(input_path)
    checks = [
        (file_path.exists(), "Path does not exist"),
        (file_path.is_file(), "Not a valid file"),
        (os.access(file_path, os.R_OK), "No read permission"),
        (file_path.stat().st_size > 0, "File is empty"),
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
        "-V",
        "--version",
        action="version",
        version=get_version(),
        help="View BioStructBenchmark version number",
    )

    # Verbosity argument
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v for INFO, -vv for DEBUG/trace)",
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
        "-o",
        "--output-dir",
        type=str,
        help="Directory to save aligned structures (default: current directory)",
    )
    parser.add_argument(
        "--save-structures",
        action="store_true",
        help="Save aligned structures to output files",
    )

    # Analysis options
    parser.add_argument(
        "-b",
        "--analyze-bfactor",
        action="store_true",
        help="Perform B-factor/pLDDT confidence analysis",
    )
    parser.add_argument(
        "--bfactor-output",
        type=str,
        help="Output path for B-factor CSV (default: <output_dir>/analysis/bfactor_comparison.csv)",
    )

    # Parse the command line arguments
    return parser.parse_args()
