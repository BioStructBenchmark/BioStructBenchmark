"""Structure file handling and validation."""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

import logging
from pathlib import Path

import gemmi

logger = logging.getLogger(__name__)


def file_type(file_path: Path) -> str:
    """Get the file extension in lowercase."""
    return str(file_path.suffix).lower()


def validate_file(file_path: Path) -> bool:
    """Validate a single file with a specified type."""
    ftype = file_type(file_path)

    # GEMMI supports .pdb, .cif, .ent (PDB), .mmcif, .cif.gz, .pdb.gz
    supported_formats = {".pdb", ".cif", ".ent", ".mmcif"}
    if ftype not in supported_formats:
        return False

    try:
        structure = gemmi.read_structure(str(file_path))
        # Verify structure has at least one model
        if len(structure) == 0:
            logger.warning("No valid model can be extracted from %s", file_path)
            return False
        return True
    except (RuntimeError, ValueError) as e:
        logger.warning("File %s could not be parsed as %s file: %s", file_path, ftype, e)
        return False
    except Exception:
        logger.exception("Unexpected error validating %s", file_path)
        return False


def get_structure(file_path: Path) -> gemmi.Structure | None:
    """Load and return structure from file, or None if invalid."""
    if not validate_file(file_path):
        return None

    try:
        structure = gemmi.read_structure(str(file_path))
        return structure
    except Exception:
        logger.exception("Error loading structure from %s", file_path)
        return None
