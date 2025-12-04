"""
Tests for structure output functionality
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from biostructbenchmark.core.alignment import (
    AlignmentResult,
    align_protein_dna_complex,
    save_aligned_structures,
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

    def test_save_structures_creates_files(self):
        """Test that save_aligned_structures creates output files using GEMMI"""
        # Create the expected directory structure (like create_output_directory_structure does)
        (self.temp_path / "alignments").mkdir(parents=True, exist_ok=True)
        (self.temp_path / "analysis").mkdir(parents=True, exist_ok=True)
        (self.temp_path / "logs").mkdir(parents=True, exist_ok=True)

        # Create mock GEMMI-compatible structures
        exp_structure = Mock()
        comp_structure = Mock()

        # Create mock atom with GEMMI interface
        atom = Mock()
        atom.pos = Mock()
        atom.pos.x = 0.0
        atom.pos.y = 0.0
        atom.pos.z = 0.0

        # Create residue with GEMMI interface
        residue = Mock()
        residue.__iter__ = Mock(return_value=iter([atom]))

        # Create chain with GEMMI interface
        chain = Mock()
        chain.__iter__ = Mock(return_value=iter([residue]))

        # Create model with GEMMI interface
        model = Mock()
        model.__iter__ = Mock(return_value=iter([chain]))

        # Create GEMMI structure mock
        comp_copy = Mock()
        comp_copy.__iter__ = Mock(return_value=iter([model]))

        exp_copy = Mock()
        exp_copy.__iter__ = Mock(return_value=iter([]))

        # Mock the document creation
        mock_doc = Mock()
        mock_doc.write_file = Mock()
        exp_copy.make_mmcif_document = Mock(return_value=mock_doc)
        comp_copy.make_mmcif_document = Mock(return_value=mock_doc)

        with patch("biostructbenchmark.core.alignment.copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.side_effect = [exp_copy, comp_copy]

            exp_path, comp_path = save_aligned_structures(
                exp_structure,
                comp_structure,
                self.rotation_matrix,
                self.translation_vector,
                self.temp_path,
            )

        # Verify output paths (now in alignments subdirectory)
        assert exp_path == self.temp_path / "alignments" / "aligned_experimental.cif"
        assert comp_path == self.temp_path / "alignments" / "aligned_computational_aligned.cif"

        # Verify GEMMI's make_mmcif_document was called
        assert exp_copy.make_mmcif_document.call_count == 1
        assert comp_copy.make_mmcif_document.call_count == 1

        # Verify write_file was called
        assert mock_doc.write_file.call_count == 2

    def test_output_directory_creation(self):
        """Test that save_aligned_structures works with a run directory that has subdirectories"""
        run_dir = self.temp_path / "run_dir"
        # Create the expected directory structure (like create_output_directory_structure does)
        (run_dir / "alignments").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Create mock GEMMI-compatible structures
        exp_structure = Mock()
        comp_structure = Mock()

        exp_copy = Mock()
        exp_copy.__iter__ = Mock(return_value=iter([]))
        comp_copy = Mock()
        comp_copy.__iter__ = Mock(return_value=iter([]))

        # Mock the document creation
        mock_doc = Mock()
        mock_doc.write_file = Mock()
        exp_copy.make_mmcif_document = Mock(return_value=mock_doc)
        comp_copy.make_mmcif_document = Mock(return_value=mock_doc)

        with patch("biostructbenchmark.core.alignment.copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.side_effect = [exp_copy, comp_copy]

            exp_path, comp_path = save_aligned_structures(
                exp_structure,
                comp_structure,
                self.rotation_matrix,
                self.translation_vector,
                run_dir,
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
            output_files=output_files,
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
            interface_residues={},
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
            exp_structure, comp_structure, output_dir=self.temp_path, save_structures=True
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
        result = align_protein_dna_complex(exp_structure, comp_structure, save_structures=False)

        # Verify no output files were created
        assert result.output_files is None


if __name__ == "__main__":
    pytest.main([__file__])
