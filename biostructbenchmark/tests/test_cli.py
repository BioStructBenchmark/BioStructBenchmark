import pytest
from pathlib import Path
from biostructbenchmark.cli import validate_file_path


def test_validate_file_path():
    correct_path = Path("./tests/data/proteins_cif/1bom.cif")
    assert validate_file_path("./tests/data/proteins_cif/1bom.cif") == correct_path


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        assert validate_file_path("INVALIDPATH")


def test_empty_file():
    with pytest.raises(ValueError):
        assert validate_file_path("./tests/data/empty.cif")
