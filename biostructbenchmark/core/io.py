"""Structure file handling and validation."""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

import logging
from pathlib import Path

import gemmi

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".cif", ".pdb"}


def file_type(file_path: Path) -> str:
    """Get the file extension in lowercase."""
    return str(file_path.suffix).lower()


def validate_file(file_path: Path) -> bool:
    """Validate a single file can be parsed."""
    return get_structure(file_path) is not None


def get_structure(file_path: Path) -> gemmi.Structure | None:
    """Load and return structure from file, or None if invalid.

    Uses GEMMI for fast parsing of PDB and mmCIF files.
    Returns None if the file cannot be parsed or has no useful content
    (no models, no chains, or no residues).
    """
    ftype = file_type(file_path)
    if ftype not in SUPPORTED_EXTENSIONS:
        return None

    try:
        structure = gemmi.read_structure(str(file_path))
        if not structure or len(structure) == 0:
            logger.warning("No models in %s", file_path)
            return None

        # Check that there's at least one chain with residues
        model = structure[0]
        has_content = False
        for chain in model:
            if len(chain) > 0:
                has_content = True
                break

        if not has_content:
            logger.warning("No chains with residues in %s", file_path)
            return None

        return structure
    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path, e)
        return None
