import pytest
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


def test_invalid_cif_file():
    assert validate_file(Path("./tests/data/invalid.cif")) == False


def test_invalid_pdb_file():
    assert validate_file(Path("./tests/data/invalid.pdb")) == False
