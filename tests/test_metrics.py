import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Import the actual functions and classes from metrics.py
from biostructbenchmark.core.metrics import (
    decompose_structural_error,
    calculate_residue_statistics,
    identify_outlier_residues,
    calculate_molecule_specific_rmsd,
    generate_comprehensive_metrics,
    ErrorComponents,
    StructureMetrics
)
from biostructbenchmark.core.alignment import ResidueRMSD, AlignmentResult


class TestErrorComponents:
    """Test ErrorComponents dataclass and error decomposition"""
    
    def test_error_components_creation(self):
        """Test creating ErrorComponents dataclass"""
        error = ErrorComponents(
            translation_error=2.5,
            orientation_error=15.0,
            total_error=3.0,
            translation_vector=np.array([1.0, 2.0, 3.0]),
            rotation_angle=45.0
        )
        
        assert error.translation_error == 2.5
        assert error.orientation_error == 15.0
        assert error.total_error == 3.0
        assert error.rotation_angle == 45.0
        assert np.array_equal(error.translation_vector, np.array([1.0, 2.0, 3.0]))
    
    def test_decompose_structural_error(self):
        """Test error decomposition into translation/orientation components"""
        # Create mock AlignmentResult
        alignment_result = AlignmentResult(
            overall_rmsd=5.0,
            residue_rmsds=[],
            transformation_matrix=np.eye(4),
            rotation_matrix=np.eye(3),  # Identity = no rotation
            translation_vector=np.array([1.0, 2.0, 2.0]),  # Translation of magnitude 3
            aligned_atom_count=100,
            reference_frame='full'
        )
        
        error_components = decompose_structural_error(alignment_result)
        
        assert error_components.translation_error == 3.0  # sqrt(1^2 + 2^2 + 2^2)
        assert error_components.rotation_angle == 0.0  # No rotation
        assert isinstance(error_components.translation_vector, np.ndarray)
    
    def test_decompose_with_rotation(self):
        """Test error decomposition with rotation"""
        # Create rotation matrix for 90 degrees around z-axis
        angle = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        alignment_result = AlignmentResult(
            overall_rmsd=5.0,
            residue_rmsds=[],
            transformation_matrix=np.eye(4),
            rotation_matrix=rotation_matrix,
            translation_vector=np.array([0.0, 0.0, 0.0]),
            aligned_atom_count=100,
            reference_frame='full'
        )
        
        error_components = decompose_structural_error(alignment_result)
        
        assert error_components.translation_error == 0.0
        assert abs(error_components.rotation_angle - 90.0) < 1e-5  # 90 degrees


class TestStatistics:
    """Test statistical calculations"""
    
    def test_calculate_residue_statistics(self):
        """Test residue statistics calculation"""
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
            ResidueRMSD("VAL_12", "VAL", "A", 12, 4.0, 5, "protein"),
            ResidueRMSD("LEU_13", "LEU", "A", 13, 5.0, 5, "protein")
        ]
        
        stats = calculate_residue_statistics(residue_rmsds)
        
        assert stats['mean_rmsd'] == 3.5
        assert stats['median_rmsd'] == 3.5
        assert stats['min_rmsd'] == 2.0
        assert stats['max_rmsd'] == 5.0
        assert stats['rmsd_range'] == 3.0
        assert 'std_rmsd' in stats
        assert 'q25_rmsd' in stats
        assert 'q75_rmsd' in stats
    
    def test_empty_statistics(self):
        """Test statistics with empty input"""
        stats = calculate_residue_statistics([])
        assert stats == {}


class TestOutlierIdentification:
    """Test outlier residue identification"""
    
    def test_identify_outlier_residues(self):
        """Test finding best and worst residues"""
        residue_rmsds = [
            ResidueRMSD(f"RES_{i}", "ALA", "A", i, float(i), 5, "protein")
            for i in range(1, 11)  # RMSDs from 1.0 to 10.0
        ]
        
        worst, best = identify_outlier_residues(residue_rmsds, n_worst=3, n_best=3)
        
        # Check worst residues
        assert len(worst) == 3
        assert worst[0] == ("RES_10", 10.0)
        assert worst[1] == ("RES_9", 9.0)
        assert worst[2] == ("RES_8", 8.0)
        
        # Check best residues
        assert len(best) == 3
        assert best[0] == ("RES_1", 1.0)
        assert best[1] == ("RES_2", 2.0)
        assert best[2] == ("RES_3", 3.0)
    
    def test_outliers_with_fewer_residues(self):
        """Test outlier identification with fewer residues than requested"""
        residue_rmsds = [
            ResidueRMSD("RES_1", "ALA", "A", 1, 1.0, 5, "protein"),
            ResidueRMSD("RES_2", "GLY", "A", 2, 2.0, 5, "protein")
        ]
        
        worst, best = identify_outlier_residues(residue_rmsds, n_worst=5, n_best=5)
        
        assert len(worst) == 2
        assert len(best) == 2


class TestMoleculeSpecific:
    """Test molecule-specific RMSD calculations"""
    
    def test_calculate_molecule_specific_rmsd(self):
        """Test separate protein and DNA RMSD calculation"""
        residue_rmsds = [
            # Protein residues
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 4.0, 5, "protein"),
            # DNA residues
            ResidueRMSD("DG_5", "DG", "B", 5, 3.0, 10, "dna"),
            ResidueRMSD("DA_6", "DA", "B", 6, 5.0, 10, "dna"),
            ResidueRMSD("DT_7", "DT", "B", 7, 7.0, 10, "dna")
        ]
        
        protein_rmsd, dna_rmsd = calculate_molecule_specific_rmsd(residue_rmsds)
        
        assert protein_rmsd == 3.0  # (2.0 + 4.0) / 2
        assert dna_rmsd == 5.0  # (3.0 + 5.0 + 7.0) / 3
    
    def test_protein_only(self):
        """Test with only protein residues"""
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein")
        ]
        
        protein_rmsd, dna_rmsd = calculate_molecule_specific_rmsd(residue_rmsds)
        
        assert protein_rmsd == 2.5
        assert dna_rmsd == 0.0
    
    def test_dna_only(self):
        """Test with only DNA residues"""
        residue_rmsds = [
            ResidueRMSD("DG_5", "DG", "B", 5, 4.0, 10, "dna"),
            ResidueRMSD("DA_6", "DA", "B", 6, 6.0, 10, "dna")
        ]
        
        protein_rmsd, dna_rmsd = calculate_molecule_specific_rmsd(residue_rmsds)
        
        assert protein_rmsd == 0.0
        assert dna_rmsd == 5.0


class TestComprehensiveMetrics:
    """Test comprehensive metrics generation"""
    
    def test_generate_comprehensive_metrics(self):
        """Test full metrics generation from alignment result"""
        # Create sample residue RMSDs
        residue_rmsds = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 4.0, 10, "dna")
        ]
        
        # Create mock alignment result
        alignment_result = AlignmentResult(
            overall_rmsd=3.0,
            residue_rmsds=residue_rmsds,
            transformation_matrix=np.eye(4),
            rotation_matrix=np.eye(3),
            translation_vector=np.array([1.0, 1.0, 1.0]),
            aligned_atom_count=20,
            reference_frame='full'
        )
        
        # Generate metrics
        metrics = generate_comprehensive_metrics(alignment_result)
        
        # Verify metrics
        assert metrics.overall_rmsd == 3.0
        assert metrics.protein_rmsd == 2.5  # (2.0 + 3.0) / 2
        assert metrics.dna_rmsd == 4.0
        assert isinstance(metrics.error_components, ErrorComponents)
        assert isinstance(metrics.residue_error_stats, dict)
        assert len(metrics.worst_residues) > 0
        assert len(metrics.best_residues) > 0
    
    def test_structure_metrics_dataclass(self):
        """Test StructureMetrics dataclass"""
        error_comp = ErrorComponents(
            translation_error=1.0,
            orientation_error=2.0,
            total_error=3.0,
            translation_vector=np.zeros(3),
            rotation_angle=15.0
        )
        
        metrics = StructureMetrics(
            overall_rmsd=3.5,
            protein_rmsd=3.0,
            dna_rmsd=4.0,
            error_components=error_comp,
            residue_error_stats={'mean': 3.5},
            worst_residues=[("RES_1", 5.0)],
            best_residues=[("RES_2", 1.0)]
        )
        
        assert metrics.overall_rmsd == 3.5
        assert metrics.protein_rmsd == 3.0
        assert metrics.dna_rmsd == 4.0
        assert metrics.error_components.translation_error == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
