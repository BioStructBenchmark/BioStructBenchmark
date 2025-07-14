"""cif/pdb file handling"""

from Bio.PDB import MMCIFParser, PDBParser, Structure
from ..cli import arg_parser
from pathlib import Path


def validate_file(file_path: Path, file_type: str) -> bool:
    """Validate a single file with a specified type"""
    parser_type = {"MMCIF": MMCIFParser(), "PDB": PDBParser()}
    file_parser = parser_type[file_type]
    try:
        file_parser.get_structure("xxx", file_path)  # Works iff valid file
        print(file_parser.get_structure("xxx", file_path))  # Works iff valid file

        return True
    except:
        raise TypeError(f"{file_path} is not a valid {file_type.lower()} file")
