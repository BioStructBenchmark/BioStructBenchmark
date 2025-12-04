"""Structure file handling and validation."""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

import logging
import time
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
    start_time = time.perf_counter()
    logger.debug("Loading structure from %s", file_path)

    ftype = file_type(file_path)
    if ftype not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Unsupported file extension: %s (supported: %s)", ftype, SUPPORTED_EXTENSIONS
        )
        return None

    try:
        structure = gemmi.read_structure(str(file_path))
        parse_time = (time.perf_counter() - start_time) * 1000

        if not structure or len(structure) == 0:
            logger.warning("No models in %s", file_path)
            return None

        # Check that there's at least one chain with residues
        model = structure[0]
        has_content = False
        chain_count = 0
        residue_count = 0
        atom_count = 0
        for chain in model:
            chain_count += 1
            for residue in chain:
                residue_count += 1
                atom_count += len(list(residue))
            if len(chain) > 0:
                has_content = True

        if not has_content:
            logger.warning("No chains with residues in %s", file_path)
            return None

        logger.debug(
            "Parsed %s: %d chains, %d residues, %d atoms in %.2f ms",
            file_path.name,
            chain_count,
            residue_count,
            atom_count,
            parse_time,
        )
        return structure
    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path, e)
        return None
