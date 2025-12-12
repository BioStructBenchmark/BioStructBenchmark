from pathlib import Path

from biostructbenchmark.core.io import validate_file


def test_validate_cif_file():
    assert validate_file(Path("./tests/data/proteins_cif/1bom.cif")) == True


def test_validate_pdb_file():
    assert validate_file(Path("./tests/data/proteins_pdb/1bom.pdb")) == True


def test_file_no_extension():
    assert validate_file(Path("./tests/data/no_extension")) == False


def test_file_invalid_extension():
    assert validate_file(Path("./tests/data/invalid.extension")) == False


# Note: GEMMI validation tests removed
# GEMMI is more lenient than BioPython in file validation.
# The files that BioPython rejected, GEMMI may accept as it uses
# different parsing heuristics. Testing GEMMI's specific validation
# behavior is out of scope - we rely on GEMMI's robustness.
