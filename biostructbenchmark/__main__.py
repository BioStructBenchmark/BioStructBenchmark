"""Single-file validation"""

import argparse
import os
from Bio.PDB import *
from typing import Optional
import ast
from cli import arg_parser


def validate_file(file_path: str, file_type: str) -> None:
    """Validate a single file with a specified type"""
    parser_type = {"MMCIF": MMCIFParser(), "PDB": PDBParser()}
    file_parser = parser_type[file_type]
    try:
        structure = file_parser.get_structure("xxx", file_path)
    except:
        raise argparse.ArgumentTypeError(
            f"{file_path} is not a valid {file_type.lower()} file"
        )


def main():
    args = arg_parser()
    validate_file(args.file_path, args.file_type)


if __name__ == "__main__":
    main()
