"""CLI scripts called from __main__.py"""

import argparse
import os
import ast


def validate_file_path(file_path: str) -> str:
    """Validate file_path input"""
    if os.path.isfile(file_path):
        return file_path
    else:
        raise argparse.ArgumentTypeError(f"{file_path} is not a valid file path")


def file_type(file_type: str) -> str:
    """Ensure input file_type is mmcif or pdb"""
    if file_type.lower() == "mmcif":
        return "MMCIF"
    if file_type.lower() == "pdb":
        return "PDB"
    else:
        raise argparse.ArgumentTypeError(
            "file_type argument must be either MMCIF or PDB"
        )


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
        "file_path", type=validate_file_path, help="Path to input file(s)"
    )
    parser.add_argument(
        "file_type", type=file_type, help="Specify the type of file(s) to be read"
    )

    # Parse the command line arguments
    return parser.parse_args()
