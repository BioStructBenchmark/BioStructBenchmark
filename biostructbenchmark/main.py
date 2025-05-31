import argparse
import os

def check_file_exists(file_path: str):
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"{file_path} is not a valid file path")
    return file_path

# Create the parser
parser = argparse.ArgumentParser()

# File argument
parser.add_argument("file", type=check_file_exists, help="Open a file to read")

# Version argument
parser.add_argument("--version", help="View BioStructBenchmark version number")

# Parse the command line arguments
args = parser.parse_args()


with open(args.file, 'r') as file:
    print(file.read())

