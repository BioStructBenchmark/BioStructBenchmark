"""Structure file handling and validation."""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

import logging
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, Structure

logger = logging.getLogger(__name__)

parser_type = {".cif": MMCIFParser(), ".pdb": PDBParser()}


def file_type(file_path: Path) -> str:
    """Get the file extension in lowercase."""
    return str(file_path.suffix).lower()


def file_parser(file_path: Path) -> MMCIFParser | PDBParser:
    """Get the appropriate parser for the file type."""
    try:
        return parser_type[file_type(file_path)]
    except KeyError:
        raise


def validate_file(file_path: Path) -> bool:
    """Validate a single file with a specified type."""
    ftype = str(file_path.suffix).lower()
    try:
        parser = file_parser(file_path)
    # Unknown filetypes are to be ignored so that mixed-type folders will be handled gracefully
    except KeyError:
        return False

    try:
        structure = parser.get_structure("foo", file_path)  # Works iff valid file
        next(structure.get_models())  # Works iff a model can be extracted
    except ValueError as e:
        logger.warning("File %s could not be parsed as %s file: %s", file_path, ftype, e)
        return False
    except StopIteration:
        logger.warning("No valid model can be extracted from %s", ftype)
        return False
    else:
        return True


def get_structure(file_path: Path) -> Structure.Structure | None:
    """Load and return structure from file, or None if invalid."""
    if not validate_file(file_path):
        return None

    try:
        parser = file_parser(file_path)
        structure = parser.get_structure("structure", file_path)
        return structure
    except Exception:
        logger.exception("Error loading structure from %s", file_path)
        return None
