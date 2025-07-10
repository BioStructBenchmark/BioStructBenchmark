import pytest
from pathlib import Path
from biostructbenchmark.cli import file_type, validate_file_path


def test_validate_file_path():
    correct_path = Path("./tests/data/1bom.cif")
    assert validate_file_path("./tests/data/1bom.cif") == correct_path


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        assert validate_file_path("INVALIDPATH")


def test_empty_file():
    with pytest.raises(ValueError):
        assert validate_file_path("./tests/data/empty.cif")


def test_file_type_pdb():
    assert file_type("Pdb") == "PDB"


def test_file_type_mmcif():
    assert file_type("Cif") == "MMCIF"
    assert file_type("mmciF") == "MMCIF"


def test_file_type_other():
    with pytest.raises(ValueError):
        file_type("other")


def test_file_type_blank():
    with pytest.raises(ValueError):
        file_type("")
