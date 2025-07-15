#!/usr/bin/env python3

"""Entry point for biostructbenchmark"""

from biostructbenchmark.core.io import validate_file
from biostructbenchmark.cli import arg_parser  # change to parser


def main() -> None:
    args = arg_parser()
    print(validate_file(args.file_path))


if __name__ == "__main__":
    main()
