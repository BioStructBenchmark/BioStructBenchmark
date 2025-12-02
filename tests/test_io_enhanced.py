"""Enhanced tests for I/O functionality with modern testing practices."""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume

from biostructbenchmark.core.io import (
    validate_file,
    file_type,
    file_parser,
    get_structure
)


class TestFileTypeDetection:
    """Test file type detection with parametrized cases."""
    
    @pytest.mark.parametrize("file_path,expected_type", [
        ("test.cif", ".cif"),
        ("test.pdb", ".pdb"),
        ("test.CIF", ".cif"),
        ("test.PDB", ".pdb"),
        ("complex_name.cif", ".cif"),
        ("path/to/file.pdb", ".pdb"),
        ("file.with.dots.cif", ".cif"),
    ])
    def test_file_type_detection(self, file_path, expected_type):
        """Test file type detection for various path formats."""
        assert file_type(Path(file_path)) == expected_type
    
    @pytest.mark.parametrize("invalid_extension", [
        ".txt", ".xyz", ".mol", "", ".unknown"
    ])
    def test_invalid_file_extensions(self, invalid_extension):
        """Test handling of invalid file extensions."""
        test_path = Path(f"test{invalid_extension}")
        with pytest.raises(KeyError):
            file_parser(test_path)


class TestFileValidation:
    """Test file validation with comprehensive edge cases."""
    
    def test_validate_existing_cif_file(self):
        """Test validation of existing CIF files."""
        assert validate_file(Path("./tests/data/proteins_cif/1bom.cif")) is True
    
    def test_validate_existing_pdb_file(self):
        """Test validation of existing PDB files."""
        assert validate_file(Path("./tests/data/proteins_pdb/1bom.pdb")) is True
    
    @pytest.mark.parametrize("invalid_file", [
        "./tests/data/no_extension",
        "./tests/data/invalid.extension", 
        "./tests/data/invalid.cif",
        "./tests/data/invalid.pdb"
    ])
    def test_validate_invalid_files(self, invalid_file):
        """Test validation rejects invalid files."""
        assert validate_file(Path(invalid_file)) is False
    
    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent files."""
        assert validate_file(Path("./nonexistent.cif")) is False
    
    @given(st.text(min_size=1, max_size=50))
    def test_validate_random_filenames(self, filename):
        """Property-based test for random filenames."""
        assume(not filename.endswith(('.cif', '.pdb')))
        assume('/' not in filename and '\\' not in filename)
        
        test_path = Path(f"{filename}.unknown")
        result = validate_file(test_path)
        assert result is False


class TestFileParser:
    """Test file parser selection and error handling."""
    
    def test_parser_selection_cif(self):
        """Test CIF parser selection."""
        parser = file_parser(Path("test.cif"))
        assert parser.__class__.__name__ == "MMCIFParser"
    
    def test_parser_selection_pdb(self):
        """Test PDB parser selection."""
        parser = file_parser(Path("test.pdb"))
        assert parser.__class__.__name__ == "PDBParser"
    
    def test_parser_error_handling(self):
        """Test parser error handling for unknown types."""
        with pytest.raises(KeyError):
            file_parser(Path("test.unknown"))


class TestStructureLoading:
    """Test structure loading functionality."""
    
    def test_get_structure_cif(self):
        """Test loading CIF structure."""
        structure = get_structure(Path("./tests/data/proteins_cif/1bom.cif"))
        assert structure is not None
        assert len(list(structure.get_models())) > 0
    
    def test_get_structure_pdb(self):
        """Test loading PDB structure."""
        structure = get_structure(Path("./tests/data/proteins_pdb/1bom.pdb"))
        assert structure is not None
        assert len(list(structure.get_models())) > 0
    
    def test_get_structure_complex_data(self):
        """Test loading complex protein-DNA structures."""
        structure = get_structure(Path("./tests/data/complexes/experimental_9ny8.cif"))
        assert structure is not None
        
        models = list(structure.get_models())
        assert len(models) > 0
        
        chains = list(models[0].get_chains())
        assert len(chains) >= 2  # Should have protein and DNA chains


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_file_handling(self):
        """Test handling of empty files."""
        assert validate_file(Path("./tests/data/empty.cif")) is False
    
    @patch('biostructbenchmark.core.io.parser_type')
    def test_parser_exception_handling(self, mock_parser_type):
        """Test exception handling in parser selection."""
        mock_parser_type.__getitem__.side_effect = Exception("Parser error")
        
        result = validate_file(Path("test.cif"))
        assert result is False
    
    def test_malformed_structure_handling(self):
        """Test handling of malformed structure files."""
        with tempfile.NamedTemporaryFile(suffix='.cif', mode='w', delete=False) as f:
            f.write("invalid_structure_data")
            temp_path = Path(f.name)
        
        try:
            result = validate_file(temp_path)
            assert result is False
        finally:
            temp_path.unlink()


@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for I/O operations."""
    
    def test_file_validation_performance(self, benchmark):
        """Benchmark file validation performance."""
        test_file = Path("./tests/data/proteins_cif/1bom.cif")
        result = benchmark(validate_file, test_file)
        assert result is True
    
    def test_structure_loading_performance(self, benchmark):
        """Benchmark structure loading performance."""
        test_file = Path("./tests/data/complexes/experimental_9ny8.cif")
        result = benchmark(get_structure, test_file)
        assert result is not None


@pytest.mark.integration
class TestIntegration:
    """Integration tests for I/O with real data."""
    
    def test_full_io_pipeline(self):
        """Test complete I/O pipeline from validation to structure loading."""
        test_files = [
            "./tests/data/proteins_cif/1bom.cif",
            "./tests/data/proteins_pdb/1bom.pdb", 
            "./tests/data/complexes/experimental_9ny8.cif"
        ]
        
        for file_path in test_files:
            path = Path(file_path)
            
            # Validate file
            assert validate_file(path) is True
            
            # Load structure
            structure = get_structure(path)
            assert structure is not None
            
            # Verify structure has models
            models = list(structure.get_models())
            assert len(models) > 0
            
            # Verify structure has chains
            chains = list(models[0].get_chains())
            assert len(chains) > 0