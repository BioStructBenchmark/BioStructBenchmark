import argparse
import os


def check_file_exists(file_path: str):
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"{file_path} is not a valid file path")
    return file_path


def get_args():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Version argument
    # TODO autodetect version from pyproject.toml or __init__.py
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="0.0.1",
        help="View BioStructBenchmark version number",
    )

    # File argument
    parser.add_argument("file", type=check_file_exists, help="Open a file to read")


    # Parse the command line arguments
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.file, "r") as file:
        print(file.read())
    print('hello, world')


if __name__ == '__main__':
    main()