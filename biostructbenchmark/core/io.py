"""Structure file handling"""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

from typing import Optional
from Bio.PDB import MMCIFParser, PDBParser, Structure

from pathlib import Path


parser = None
parser_type = {".cif": MMCIFParser(), ".pdb": PDBParser()}


def file_type(file_path: Path) -> str:
    return str(file_path.suffix).lower()


def file_parser(file_path: Path) -> MMCIFParser | PDBParser:
    try:
        return parser_type[file_type(file_path)]
    except KeyError:
        raise


def validate_file(file_path: Path) -> bool:
    """Validate a single file with a specified type"""
    file_type = str(file_path.suffix).lower()
    try:
        parser = file_parser(file_path)
    # Unknown filetypes are to be ignored so that mixed-type folders will be handled gracefully
    except:
        return False

    try:
        structure = parser.get_structure("foo", file_path)  # Works iff valid file
        next(structure.get_models())  # Works iff a model can be extracted
    except ValueError as e:
        print(
            f"Error: {file_path} could not be parsed as {file_type} file. Reason:\n{e}"
        )
        return False
    except StopIteration:
        print(f"Error: no valid model can be extracted from {file_type}")
        return False
    else:
        return True


def get_structure(file_path: Path) -> Optional[Structure.Structure]:
    """Load and return structure from file, or None if invalid."""
    if not validate_file(file_path):
        return None

    try:
        parser = file_parser(file_path)
        structure = parser.get_structure("structure", file_path)
        return structure
    except Exception as e:
        print(f"Error loading structure from {file_path}: {e}")
        return None
