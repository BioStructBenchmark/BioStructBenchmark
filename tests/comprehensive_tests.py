"""
Comprehensive test suite for BioStructBenchmark
Tests all existing components for functionality and integration
"""
import argparse
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import csv

# ============================================================================
# TEST: biostructbenchmark/core/io.py
# ============================================================================

class TestCoreIO:
    """Test suite for core I/O functionality"""
    
    def test_file_type_detection(self):
        """Test file type detection"""
        from biostructbenchmark.core.io import file_type
        
        assert file_type(Path("test.pdb")) == ".pdb"
        assert file_type(Path("test.cif")) == ".cif"
        assert file_type(Path("test.PDB")) == ".pdb"
        assert file_type(Path("test.mmcif")) == ".mmcif"
    
    def test_parser_creation(self):
        """Test parser instance creation and caching"""
        from biostructbenchmark.core.io import file_parser, _parsers
        
        # Clear cache
        _parsers.clear()
        
        # Test PDB parser
        pdb_path = Path("test.pdb")
        parser1 = file_parser(pdb_path)
        parser2 = file_parser(Path("another.pdb"))
        
        assert parser1 is parser2  # Should be same cached instance
        assert ".pdb" in _parsers
    
    def test_validate_file_missing(self):
        """Test validation of missing file"""
        from biostructbenchmark.core.io import validate_file
        
        missing_file = Path("nonexistent.pdb")
        assert validate_file(missing_file) == False
    
    @pytest.mark.skipif(not Path("tests/test_data/proteins_pdb/1bom.pdb").exists(),
                       reason="Test data not available")
    def test_get_structure_valid(self):
        """Test loading valid structure"""
        from biostructbenchmark.core.io import get_structure
        
        structure = get_structure(Path("tests/test_data/proteins_pdb/1bom.pdb"))
        assert structure is not None
        assert len(list(structure.get_models())) > 0


# ============================================================================
# TEST: biostructbenchmark/core/alignment.py
# ============================================================================

class TestCoreAlignment:
    """Test suite for structure alignment"""
    
    def test_residue_rmsd_dataclass(self):
        """Test ResidueRMSD dataclass"""
        from biostructbenchmark.core.alignment import ResidueRMSD
        
        rmsd = ResidueRMSD(
            residue_id="ALA_10",
            residue_type="ALA",
            chain_id="A",
            position=10,
            rmsd=2.5,
            atom_count=5,
            molecule_type="protein"
        )
        
        assert rmsd.residue_id == "ALA_10"
        assert rmsd.rmsd == 2.5
        assert rmsd.molecule_type == "protein"
    
    def test_alignment_result_dataclass(self):
        """Test AlignmentResult dataclass"""
        from biostructbenchmark.core.alignment import AlignmentResult, ResidueRMSD
        
        result = AlignmentResult(
            overall_rmsd=3.2,
            residue_rmsds=[],
            transformation_matrix=np.eye(4),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            aligned_atom_count=100,
            reference_frame="full"
        )
        
        assert result.overall_rmsd == 3.2
        assert result.aligned_atom_count == 100
        assert result.rotation_matrix.shape == (3, 3)
    
    def test_is_protein_residue(self):
        """Test protein residue identification"""
        from biostructbenchmark.core.alignment import is_protein_residue
        
        # Mock residue
        mock_residue = Mock()
        mock_residue.get_resname.return_value = "ALA"
        assert is_protein_residue(mock_residue) == True
        
        mock_residue.get_resname.return_value = "DA"
        assert is_protein_residue(mock_residue) == False
    
    def test_is_dna_residue(self):
        """Test DNA residue identification"""
        from biostructbenchmark.core.alignment import is_dna_residue
        
        mock_residue = Mock()
        mock_residue.get_resname.return_value = "DA"
        assert is_dna_residue(mock_residue) == True
        
        mock_residue.get_resname.return_value = "ALA"
        assert is_dna_residue(mock_residue) == False
    
    def test_calculate_atom_rmsd(self):
        """Test RMSD calculation between atom sets"""
        from biostructbenchmark.core.alignment import calculate_atom_rmsd
        
        # Mock atoms with coordinates
        atoms1 = [Mock() for _ in range(3)]
        atoms2 = [Mock() for _ in range(3)]
        
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        coords2 = np.array([[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]])
        
        for i, (a1, a2) in enumerate(zip(atoms1, atoms2)):
            a1.get_coord.return_value = coords1[i]
            a2.get_coord.return_value = coords2[i]
        
        rmsd = calculate_atom_rmsd(atoms1, atoms2)
        assert abs(rmsd - 0.5) < 0.01  # All atoms shifted by 0.5 in z
    
    def test_export_residue_rmsd_csv(self, tmp_path):
        """Test CSV export functionality"""
        from biostructbenchmark.core.alignment import export_residue_rmsd_csv, ResidueRMSD
        
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.5, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 3.2, 10, "dna")
        ]
        
        output_file = tmp_path / "test_rmsd.csv"
        export_residue_rmsd_csv(residue_rmsds, output_file)
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['residue_id'] == "ALA_10"
            assert rows[1]['molecule_type'] == "dna"


# ============================================================================
# TEST: biostructbenchmark/core/metrics.py
# ============================================================================

class TestCoreMetrics:
    """Test suite for metrics calculations"""
    
    def test_error_components_dataclass(self):
        """Test ErrorComponents dataclass"""
        from biostructbenchmark.core.metrics import ErrorComponents
        
        error = ErrorComponents(
            translation_error=2.0,
            orientation_error=15.0,
            total_error=3.5,
            translation_vector=np.array([1, 2, 3]),
            rotation_angle=30.0
        )
        
        assert error.translation_error == 2.0
        assert error.rotation_angle == 30.0
    
    def test_calculate_residue_statistics(self):
        """Test statistical calculations"""
        from biostructbenchmark.core.metrics import calculate_residue_statistics
        from biostructbenchmark.core.alignment import ResidueRMSD
        
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
            ResidueRMSD("VAL_12", "VAL", "A", 12, 4.0, 5, "protein")
        ]
        
        stats = calculate_residue_statistics(residue_rmsds)
        
        assert stats['mean_rmsd'] == 3.0
        assert stats['min_rmsd'] == 2.0
        assert stats['max_rmsd'] == 4.0
        assert 'std_rmsd' in stats
    
    def test_identify_outlier_residues(self):
        """Test outlier identification"""
        from biostructbenchmark.core.metrics import identify_outlier_residues
        from biostructbenchmark.core.alignment import ResidueRMSD
        
        residue_rmsds = [
            ResidueRMSD(f"RES_{i}", "ALA", "A", i, float(i), 5, "protein")
            for i in range(1, 11)
        ]
        
        worst, best = identify_outlier_residues(residue_rmsds, n_worst=3, n_best=3)
        
        assert len(worst) == 3
        assert len(best) == 3
        assert worst[0][1] == 10.0  # Highest RMSD
        assert best[0][1] == 1.0    # Lowest RMSD
    
    def test_calculate_molecule_specific_rmsd(self):
        """Test molecule-specific RMSD calculation"""
        from biostructbenchmark.core.metrics import calculate_molecule_specific_rmsd
        from biostructbenchmark.core.alignment import ResidueRMSD
        
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 4.0, 10, "dna"),
            ResidueRMSD("DA_6", "DA", "B", 6, 5.0, 10, "dna")
        ]
        
        protein_rmsd, dna_rmsd = calculate_molecule_specific_rmsd(residue_rmsds)
        
        assert protein_rmsd == 2.5  # (2.0 + 3.0) / 2
        assert dna_rmsd == 4.5      # (4.0 + 5.0) / 2


# ============================================================================
# TEST: biostructbenchmark/cli.py
# ============================================================================

class TestCLI:
    """Test suite for CLI functionality"""
    
    def test_validate_file_path_invalid(self):
        """Test invalid file path validation"""
        from biostructbenchmark.cli import validate_file_path
        
        with pytest.raises(argparse.ArgumentTypeError):
            validate_file_path("nonexistent_file.pdb")
    
    def test_validate_file_path_valid(self, tmp_path):
        """Test valid file path validation"""
        from biostructbenchmark.cli import validate_file_path
        
        # Create a temporary file
        test_file = tmp_path / "test.pdb"
        test_file.write_text("HEADER TEST")
        
        result = validate_file_path(str(test_file))
        assert result == test_file
    
    def test_get_version(self):
        """Test version retrieval"""
        from biostructbenchmark.cli import get_version
        
        with patch('builtins.open', mock_open(read_data='__version__ = "1.0.0"\n')):
            version = get_version()
            assert version == "1.0.0"
    
    @patch('sys.argv', ['biostructbenchmark', '--version'])
    def test_arg_parser_version(self):
        """Test version argument"""
        from biostructbenchmark.cli import arg_parser
        
        with pytest.raises(SystemExit) as exc_info:
            arg_parser()
        assert exc_info.value.code == 0


# ============================================================================
# TEST: biostructbenchmark/visualization/structure.py
# ============================================================================

class TestStructureVisualization:
    """Test suite for structure visualization"""
    
    def test_structure_visualizer_init(self):
        """Test StructureVisualizer initialization"""
        from biostructbenchmark.visualization.structure import StructureVisualizer
        
        viz = StructureVisualizer()
        assert viz.width == 800
        assert viz.height == 600
        assert viz.backend in ['py3dmol', 'matplotlib']
    
    def test_backend_selection(self):
        """Test backend selection logic"""
        from biostructbenchmark.visualization.structure import StructureVisualizer
        
        with patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', False):
            viz = StructureVisualizer()
            assert viz.backend == 'matplotlib'
    
    def test_save_summary(self, tmp_path):
        """Test summary JSON generation"""
        from biostructbenchmark.visualization.structure import StructureVisualizer
        from biostructbenchmark.core.alignment import ResidueRMSD
        
        viz = StructureVisualizer()
        
        rmsd_data = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 3.0, 10, "dna")
        ]
        
        output_file = tmp_path / "summary.json"
        viz._save_summary(rmsd_data, output_file)
        
        assert output_file.exists()
        
        with open(output_file) as f:
            summary = json.load(f)
            assert summary['overall']['count'] == 2
            assert summary['overall']['mean'] == 2.5


# ============================================================================
# TEST: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for component interactions"""
    
    @pytest.mark.integration
    def test_full_pipeline_mock(self, tmp_path):
        """Test full pipeline with mocked structures"""
        from biostructbenchmark.core.alignment import compare_structures
        from biostructbenchmark.visualization.structure import StructureVisualizer
        
        # Create mock PDB files
        pdb_content = """ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLY A  11       3.800   0.000   0.000  1.00 20.00           C
END"""
        
        obs_file = tmp_path / "observed.pdb"
        pred_file = tmp_path / "predicted.pdb"
        
        obs_file.write_text(pdb_content)
        pred_file.write_text(pdb_content)
        
        # Test alignment
        with patch('biostructbenchmark.core.io.get_structure') as mock_get:
            mock_struct = MagicMock()
            mock_get.return_value = mock_struct
            
            result = compare_structures(obs_file, pred_file)
            
            if result:
                # Test visualization
                viz = StructureVisualizer()
                output_dir = tmp_path / "results"
                viz.create_report(obs_file, pred_file, result.residue_rmsds, output_dir)
                
                assert output_dir.exists()
    
    @pytest.mark.integration
    def test_cli_to_alignment(self, tmp_path):
        """Test CLI to alignment integration"""
        from biostructbenchmark.cli import validate_file_path
        from biostructbenchmark.core.io import validate_file
        
        # Create test file
        test_file = tmp_path / "test.pdb"
        test_file.write_text("HEADER TEST\nEND")
        
        # Test CLI validation
        validated_path = validate_file_path(str(test_file))
        assert validated_path == test_file
        
        # Test IO validation (will fail without valid PDB content)
        is_valid = validate_file(validated_path)
        assert is_valid == False  # Not a valid PDB structure


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_open():
    """Mock file opening"""
    from unittest.mock import mock_open as _mock_open
    return _mock_open


@pytest.fixture
def sample_rmsd_data():
    """Sample RMSD data for testing"""
    from biostructbenchmark.core.alignment import ResidueRMSD
    
    return [
        ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
        ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
        ResidueRMSD("VAL_12", "VAL", "A", 12, 4.0, 5, "protein"),
        ResidueRMSD("DG_5", "DG", "B", 5, 3.5, 10, "dna"),
        ResidueRMSD("DA_6", "DA", "B", 6, 4.5, 10, "dna")
    ]


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
