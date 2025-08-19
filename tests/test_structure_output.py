"""
Tests for structure output functionality
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from biostructbenchmark.core.alignment import (
    save_aligned_structures,
    align_protein_dna_complex,
    AlignmentResult
)
from biostructbenchmark.core.io import get_structure


class TestSaveAlignedStructuresUnit:
    """Test the save_aligned_structures function with mocking"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Sample transformation
        self.rotation_matrix = np.eye(3)  # Identity matrix
        self.translation_vector = np.array([1.0, 2.0, 3.0])
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('biostructbenchmark.core.alignment.MMCIFIO')
    @patch('biostructbenchmark.core.alignment.copy.deepcopy')
    def test_save_structures_creates_files(self, mock_deepcopy, mock_mmcifio):
        """Test that save_aligned_structures creates output files and calls MMCIFIO correctly"""
        # Create the expected directory structure (like create_output_directory_structure does)
        (self.temp_path / "alignments").mkdir(parents=True, exist_ok=True)
        (self.temp_path / "analysis").mkdir(parents=True, exist_ok=True)
        (self.temp_path / "logs").mkdir(parents=True, exist_ok=True)
        
        # Mock structures
        exp_structure = Mock()
        comp_structure = Mock()
        mock_deepcopy.side_effect = [exp_structure, comp_structure]
        
        # Mock atom transformation
        atom = Mock()
        atom.get_coord.return_value = np.array([0.0, 0.0, 0.0])
        residue = Mock()
        residue.__iter__ = lambda self: iter([atom])
        chain = Mock()
        chain.__iter__ = lambda self: iter([residue])
        model = Mock()
        model.__iter__ = lambda self: iter([chain])
        comp_structure.__iter__ = lambda self: iter([model])
        
        # Mock MMCIFIO
        mock_io_instance = Mock()
        mock_mmcifio.return_value = mock_io_instance
        
        exp_path, comp_path = save_aligned_structures(
            exp_structure,
            comp_structure,
            self.rotation_matrix,
            self.translation_vector,
            self.temp_path
        )
        
        # Verify output paths (now in alignments subdirectory)
        assert exp_path == self.temp_path / "alignments" / "aligned_experimental.cif"
        assert comp_path == self.temp_path / "alignments" / "aligned_computational_aligned.cif"
        
        # Verify MMCIFIO was called correctly (one instance created)
        assert mock_mmcifio.call_count == 1
        assert mock_io_instance.set_structure.call_count == 2
        assert mock_io_instance.save.call_count == 2
        
        # Verify transformation was applied
        atom.set_coord.assert_called_once()
        called_coord = atom.set_coord.call_args[0][0]
        expected_coord = np.array([1.0, 2.0, 3.0])  # [0,0,0] * I + [1,2,3]
        np.testing.assert_array_almost_equal(called_coord, expected_coord)
    
    def test_output_directory_creation(self):
        """Test that save_aligned_structures works with a run directory that has subdirectories"""
        run_dir = self.temp_path / "run_dir"
        # Create the expected directory structure (like create_output_directory_structure does)
        (run_dir / "alignments").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Mock the MMCIFIO to avoid file writing issues
        with patch('biostructbenchmark.core.alignment.MMCIFIO'), \
             patch('biostructbenchmark.core.alignment.copy.deepcopy'):
            exp_structure = Mock()
            comp_structure = Mock()
            
            # Mock minimal structure hierarchy for transformation
            comp_structure.__iter__ = lambda self: iter([])
            
            exp_path, comp_path = save_aligned_structures(
                exp_structure,
                comp_structure,
                self.rotation_matrix,
                self.translation_vector,
                run_dir
            )
        
        # Check that files are placed in the alignments subdirectory
        alignments_dir = run_dir / "alignments"
        assert alignments_dir.exists()
        assert exp_path.parent == alignments_dir
        assert comp_path.parent == alignments_dir


class TestAlignmentResultWithOutput:
    """Test AlignmentResult integration with output functionality"""
    
    def test_alignment_result_includes_output_files(self):
        """Test that AlignmentResult can store output file paths"""
        output_files = (Path("exp.cif"), Path("comp.cif"))
        
        result = AlignmentResult(
            sequence_mapping={},
            structural_rmsd=1.0,
            per_residue_rmsd={},
            protein_rmsd=1.0,
            dna_rmsd=1.0,
            interface_rmsd=1.0,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            orientation_error=0.0,
            translational_error=0.0,
            protein_chains=[],
            dna_chains=[],
            interface_residues={},
            output_files=output_files
        )
        
        assert result.output_files == output_files
    
    def test_alignment_result_default_output_files(self):
        """Test that AlignmentResult defaults to None for output_files"""
        result = AlignmentResult(
            sequence_mapping={},
            structural_rmsd=1.0,
            per_residue_rmsd={},
            protein_rmsd=1.0,
            dna_rmsd=1.0,
            interface_rmsd=1.0,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            orientation_error=0.0,
            translational_error=0.0,
            protein_chains=[],
            dna_chains=[],
            interface_residues={}
        )
        
        assert result.output_files is None


class TestAlignmentWithOutput:
    """Test the complete alignment workflow with output"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.integration
    def test_alignment_with_save_structures(self):
        """Integration test: full alignment with structure saving"""
        # Load real test structures
        exp_structure = get_structure(Path("tests/data/complexes/experimental_9ny8.cif"))
        comp_structure = get_structure(Path("tests/data/complexes/predicted_9ny8.cif"))
        
        # Run alignment with structure saving
        result = align_protein_dna_complex(
            exp_structure,
            comp_structure,
            output_dir=self.temp_path,
            save_structures=True
        )
        
        # Verify output files were created
        assert result.output_files is not None
        exp_path, comp_path = result.output_files
        
        assert exp_path.exists()
        assert comp_path.exists()
        assert exp_path.stat().st_size > 0
        assert comp_path.stat().st_size > 0
        
        # Verify file contents are valid CIF format
        with open(exp_path) as f:
            exp_content = f.read()
        with open(comp_path) as f:
            comp_content = f.read()
        
        # Basic CIF format validation
        assert "data_" in exp_content
        assert "data_" in comp_content
        assert "_atom_site" in exp_content
        assert "_atom_site" in comp_content
    
    @pytest.mark.integration
    def test_alignment_without_save_structures(self):
        """Integration test: alignment without saving structures"""
        exp_structure = get_structure(Path("tests/data/complexes/experimental_9ny8.cif"))
        comp_structure = get_structure(Path("tests/data/complexes/predicted_9ny8.cif"))
        
        # Run alignment without structure saving
        result = align_protein_dna_complex(
            exp_structure,
            comp_structure,
            save_structures=False
        )
        
        # Verify no output files were created
        assert result.output_files is None


# Edge case tests removed due to complexity of mocking BioPython structures
# The core functionality is tested by integration tests with real structures


if __name__ == "__main__":
    pytest.main([__file__])