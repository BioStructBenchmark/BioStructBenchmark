import pytest
import argparse
from pathlib import Path
from biostructbenchmark.cli import validate_file_path


def test_validate_file_path():
    correct_path = Path("./tests/data/proteins_cif/1bom.cif")
    assert validate_file_path("./tests/data/proteins_cif/1bom.cif") == correct_path


def test_invalid_file():
    with pytest.raises(argparse.ArgumentTypeError):
        assert validate_file_path("INVALIDPATH")


def test_empty_file(tmp_path):  # <-- Add tmp_path parameter
    """Test that empty file is handled properly"""
    # Create an empty file using the tmp_path fixture
    empty_file = tmp_path / "empty.pdb"
    empty_file.write_text("")
    
    # validate_file_path should succeed for existing files (even if empty)
    # The validation of content happens later in the pipeline
    result = validate_file_path(str(empty_file))
    assert result == empty_file
