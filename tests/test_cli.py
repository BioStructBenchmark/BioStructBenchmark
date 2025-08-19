import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from biostructbenchmark.cli import validate_file_path, arg_parser


def test_validate_file_path():
    correct_path = Path("./tests/data/proteins_cif/1bom.cif")
    assert validate_file_path("./tests/data/proteins_cif/1bom.cif") == correct_path


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        assert validate_file_path("INVALIDPATH")


def test_empty_file():
    with pytest.raises(ValueError):
        assert validate_file_path("./tests/data/empty.cif")


class TestCLIOutputOptions:
    """Test CLI parsing for output-related options"""
    
    def test_arg_parser_save_structures_flag(self):
        """Test parsing --save-structures flag"""
        test_args = [
            "./tests/data/complexes/experimental_9ny8.cif",
            "./tests/data/complexes/predicted_9ny8.cif",
            "--save-structures"
        ]
        
        with patch('sys.argv', ['biostructbenchmark'] + test_args):
            args = arg_parser()
            
            assert args.save_structures is True
            assert args.output_dir is None
    
    def test_arg_parser_output_dir_long(self):
        """Test parsing --output-dir option"""
        test_args = [
            "./tests/data/complexes/experimental_9ny8.cif",
            "./tests/data/complexes/predicted_9ny8.cif",
            "--output-dir", "/tmp/output"
        ]
        
        with patch('sys.argv', ['biostructbenchmark'] + test_args):
            args = arg_parser()
            
            assert args.output_dir == "/tmp/output"
            assert args.save_structures is False
    
    def test_arg_parser_combined_options(self):
        """Test parsing both save-structures and output-dir together"""
        test_args = [
            "./tests/data/complexes/experimental_9ny8.cif",
            "./tests/data/complexes/predicted_9ny8.cif",
            "--save-structures",
            "--output-dir", "/tmp/output"
        ]
        
        with patch('sys.argv', ['biostructbenchmark'] + test_args):
            args = arg_parser()
            
            assert args.save_structures is True
            assert args.output_dir == "/tmp/output"


class TestCLIIntegration:
    """Test CLI integration with the main application"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('biostructbenchmark.__main__.align_protein_dna_complex')
    @patch('biostructbenchmark.__main__.get_structure')
    def test_main_passes_output_options_to_alignment(self, mock_get_structure, mock_align):
        """Test that main() passes CLI output options to alignment function"""
        from biostructbenchmark.__main__ import main
        
        # Mock structures
        mock_exp_structure = Mock()
        mock_comp_structure = Mock()
        mock_get_structure.side_effect = [mock_exp_structure, mock_comp_structure]
        
        # Mock alignment result
        mock_result = Mock()
        mock_result.structural_rmsd = 1.0
        mock_result.protein_rmsd = 1.0
        mock_result.dna_rmsd = 1.0
        mock_result.interface_rmsd = 1.0
        mock_result.orientation_error = 0.0
        mock_result.translational_error = 0.0
        mock_result.protein_chains = ["A"]
        mock_result.dna_chains = ["B"]
        mock_result.sequence_mapping = {"A:1": "A:1"}
        mock_result.interface_residues = {"A": ["1"]}
        mock_result.per_residue_rmsd = {"A:1": 1.0}
        mock_result.output_files = None
        mock_align.return_value = mock_result
        
        # Test with save-structures and output-dir
        test_args = [
            "biostructbenchmark",
            "./tests/data/complexes/experimental_9ny8.cif",
            "./tests/data/complexes/predicted_9ny8.cif",
            "--save-structures",
            "--output-dir", str(self.temp_path)
        ]
        
        with patch('sys.argv', test_args):
            main()
            
            # Verify alignment was called with correct output options
            mock_align.assert_called_once()
            call_kwargs = mock_align.call_args[1]
            assert call_kwargs['save_structures'] is True
            assert call_kwargs['output_dir'] == str(self.temp_path)
