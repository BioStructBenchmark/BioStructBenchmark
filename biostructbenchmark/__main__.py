#!/usr/bin/env python3

"""Single-file validation"""

from Bio.PDB import MMCIFParser, PDBParser
from cli import arg_parser


def validate_file(file_path: str, file_type: str) -> None:
    """Validate a single file with a specified type"""
    parser_type = {"MMCIF": MMCIFParser(), "PDB": PDBParser()}
    file_parser = parser_type[file_type]
    try:
        file_parser.get_structure("xxx", file_path)
    except:
        raise TypeError(f"{file_path} is not a valid {file_type.lower()} file")


def main() -> None:
    args = arg_parser()
    validate_file(args.file_path, args.file_type)


if __name__ == "__main__":
    main()
