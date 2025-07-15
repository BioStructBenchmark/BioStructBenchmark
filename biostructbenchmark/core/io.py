"""cif/pdb file handling"""

# TODO: handle batches of files
# TODO: handle missing atoms/residues

from Bio.PDB import MMCIFParser, PDBParser
from pathlib import Path


def validate_file(file_path: Path) -> bool:
    """Validate a single file with a specified type"""
    file_type = str(file_path.suffix).lower()
    parser_type = {".cif": MMCIFParser(), ".pdb": PDBParser()}

    # Unknown filetypes are to be ignored so that mixed-type folders will be handled gracefully
    if file_type not in parser_type:
        return False
    file_parser = parser_type[file_type]

    try:
        structure = file_parser.get_structure("foo", file_path)  # Works iff valid file
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
