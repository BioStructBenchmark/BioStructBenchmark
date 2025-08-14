"""Structure file handling"""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

from typing import Optional
from Bio.PDB import MMCIFParser, PDBParser, Structure
from pathlib import Path

# Cache parser instances (they're reusable)
_parsers = {}


def file_type(file_path: Path) -> str:
    """Get file extension in lowercase."""
    return str(file_path.suffix).lower()


def file_parser(file_path: Path) -> Optional[MMCIFParser | PDBParser]:
    """Get or create appropriate parser for file type."""
    suffix = file_type(file_path)
    
    if suffix not in _parsers:
        if suffix in ['.cif', '.mmcif']:
            _parsers[suffix] = MMCIFParser(QUIET=True)
        elif suffix == '.pdb':
            _parsers[suffix] = PDBParser(QUIET=True)
        else:
            return None
    
    return _parsers[suffix]

# Validates that file exists
def validate_file(file_path: Path) -> bool:
    """Validate a single file with a specified type"""
    parser = file_parser(file_path)
    if not parser:
        print(f"Error: Unknown file type for {file_path}")
        return False
    
    try:
        structure = parser.get_structure("foo", file_path)  # Works iff valid file
        next(structure.get_models())  # Works iff a model can be extracted
    except ValueError as e:
        print(
            f"Error: {file_path} could not be parsed as {file_type(file_path)} file. Reason:\n{e}"
        )
        return False

    except FileNotFoundError:
        print(f"Erorr: File not found - {file_path}")
        return False

    except StopIteration:
        print(f"Error: no valid model can be extracted from {file_type(file_path)}")
        return False
    else:
        return True


def get_structure(file_path: Path) -> Optional[Structure.Structure]:
    """Load and return structure from file, or None if invalid."""
    if not validate_file(file_path):
        return None

    try:
        parser = file_parser(file_path)
        if not parser:
            print(f"Error: Unknown file type for {file_path}")
            return None
            
        structure = parser.get_structure("structure", file_path)
        return structure
    except Exception as e:
        print(f"Error loading structure from {file_path}: {e}")
        return None
