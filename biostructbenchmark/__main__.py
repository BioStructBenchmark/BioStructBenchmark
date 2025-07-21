#!/usr/bin/env python3

"""Entry point for biostructbenchmark"""

from biostructbenchmark.core.alignment import compare_structures
from biostructbenchmark.cli import arg_parser


def main() -> None:
    """Main entry point for structure comparison."""
    args = arg_parser()

    rmsd = compare_structures(args.file_path_observed, args.file_path_predicted)

    if rmsd is not None:
        print(f"RMSD: {rmsd:.3f} Å")
    else:
        print("Error: Unable to compare structures")


if __name__ == "__main__":
    main()
