"""Tests for B-factor analysis functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import Mock

from biostructbenchmark.analysis.bfactor import (
    BFactorAnalyzer,
    BFactorComparison,
    BFactorStatistics,
)
from biostructbenchmark.core.io import get_structure


class TestBFactorDataClasses:
    """Test B-factor dataclass creation and attributes."""

    def test_bfactor_comparison_creation(self):
        """Test BFactorComparison dataclass creation."""
        comparison = BFactorComparison(
            residue_id="A_42",
            chain_id="A",
            position=42,
            experimental_bfactor=25.5,
            predicted_confidence=85.3,
            difference=59.8,
            normalized_bfactor=1.2
        )

        assert comparison.residue_id == "A_42"
        assert comparison.chain_id == "A"
        assert comparison.position == 42
        assert comparison.experimental_bfactor == 25.5
        assert comparison.predicted_confidence == 85.3
        assert comparison.difference == 59.8
        assert comparison.normalized_bfactor == 1.2

    def test_bfactor_statistics_creation(self):
        """Test BFactorStatistics dataclass creation."""
        stats = BFactorStatistics(
            mean_experimental=30.5,
            mean_predicted=75.2,
            correlation=0.65,
            rmsd=15.3,
            high_confidence_accuracy=10.5,
            low_confidence_accuracy=22.8
        )

        assert stats.mean_experimental == 30.5
        assert stats.mean_predicted == 75.2
        assert stats.correlation == 0.65
        assert stats.rmsd == 15.3
        assert stats.high_confidence_accuracy == 10.5
        assert stats.low_confidence_accuracy == 22.8


class TestBFactorAnalyzer:
    """Test BFactorAnalyzer initialization."""

    def test_analyzer_initialization(self):
        """Test BFactorAnalyzer initialization with pLDDT thresholds."""
        analyzer = BFactorAnalyzer()

        assert analyzer.plddt_thresholds['very_high'] == 90
        assert analyzer.plddt_thresholds['confident'] == 70
        assert analyzer.plddt_thresholds['low'] == 50


class TestBFactorExtraction:
    """Test B-factor extraction from structure files."""

    def test_extract_bfactors_from_cif(self):
        """Test B-factor extraction from CIF file."""
        analyzer = BFactorAnalyzer()
        test_file = Path("./tests/data/complexes/experimental_9chl.cif")

        bfactors = analyzer.extract_bfactors(test_file)

        assert isinstance(bfactors, dict)
        assert len(bfactors) > 0
        # Check that keys are in expected format (chain_position)
        for key in list(bfactors.keys())[:5]:
            assert '_' in key
            chain_id, position = key.split('_')
            assert isinstance(chain_id, str)
            assert position.lstrip('-').isdigit()

    def test_extract_bfactors_from_predicted(self):
        """Test pLDDT extraction from predicted structure."""
        analyzer = BFactorAnalyzer()
        test_file = Path("./tests/data/complexes/predicted_9chl.cif")

        bfactors = analyzer.extract_bfactors(test_file)

        assert isinstance(bfactors, dict)
        assert len(bfactors) > 0
        # pLDDT values should be between 0 and 100
        for value in list(bfactors.values())[:10]:
            assert 0 <= value <= 100

    def test_extract_bfactors_invalid_file(self):
        """Test extraction from invalid file returns empty dict."""
        analyzer = BFactorAnalyzer()
        test_file = Path("./tests/data/invalid.cif")

        bfactors = analyzer.extract_bfactors(test_file)

        assert bfactors == {}


class TestStructureValidation:
    """Test B-factor validation functionality."""

    def test_validate_valid_structure(self):
        """Test validation passes for structure with valid B-factors."""
        analyzer = BFactorAnalyzer()
        structure = get_structure(Path("./tests/data/complexes/experimental_9chl.cif"))

        is_valid = analyzer.validate_bfactors(structure)

        assert is_valid is True

    def test_validate_none_structure(self):
        """Test validation fails for None structure."""
        analyzer = BFactorAnalyzer()

        is_valid = analyzer.validate_bfactors(None)

        assert is_valid is False


class TestStructureAnalysis:
    """Test structure comparison and analysis."""

    def test_analyze_structures_basic(self):
        """Test basic structure analysis with real data."""
        analyzer = BFactorAnalyzer()
        exp_structure = get_structure(Path("./tests/data/complexes/experimental_9chl.cif"))
        pred_structure = get_structure(Path("./tests/data/complexes/predicted_9chl.cif"))

        comparisons, stats = analyzer.analyze_structures(exp_structure, pred_structure)

        # Check comparisons
        assert isinstance(comparisons, list)
        assert len(comparisons) > 0
        assert all(isinstance(c, BFactorComparison) for c in comparisons)

        # Check statistics
        assert isinstance(stats, BFactorStatistics)
        assert stats.mean_experimental > 0
        assert stats.mean_predicted > 0
        assert -1 <= stats.correlation <= 1
        assert stats.rmsd >= 0

    def test_analyze_structures_9ny8(self):
        """Test structure analysis with 9ny8 complex."""
        analyzer = BFactorAnalyzer()
        exp_structure = get_structure(Path("./tests/data/complexes/experimental_9ny8.cif"))
        pred_structure = get_structure(Path("./tests/data/complexes/predicted_9ny8.cif"))

        comparisons, stats = analyzer.analyze_structures(exp_structure, pred_structure)

        assert len(comparisons) > 0
        assert stats.mean_experimental > 0
        assert stats.mean_predicted > 0

    def test_analyze_structures_none_input(self):
        """Test that None structures raise ValueError."""
        analyzer = BFactorAnalyzer()

        with pytest.raises(ValueError, match="Both structures must be valid"):
            analyzer.analyze_structures(None, None)

    def test_compare_bfactors_from_files(self):
        """Test compare_bfactors convenience method."""
        analyzer = BFactorAnalyzer()
        exp_path = Path("./tests/data/complexes/experimental_9chl.cif")
        pred_path = Path("./tests/data/complexes/predicted_9chl.cif")

        comparisons, stats = analyzer.compare_bfactors(exp_path, pred_path)

        assert len(comparisons) > 0
        assert isinstance(stats, BFactorStatistics)


class TestStatisticsCalculation:
    """Test statistics calculation from comparisons."""

    def test_calculate_statistics_basic(self):
        """Test basic statistics calculation."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 20.0, 80.0, 60.0, 0.0),
            BFactorComparison("A_2", "A", 2, 30.0, 90.0, 60.0, 0.0),
            BFactorComparison("A_3", "A", 3, 40.0, 85.0, 45.0, 0.0),
            BFactorComparison("A_4", "A", 4, 25.0, 65.0, 40.0, 0.0),
            BFactorComparison("A_5", "A", 5, 35.0, 70.0, 35.0, 0.0),
        ]

        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics(comparisons)

        assert stats.mean_experimental == 30.0
        assert stats.mean_predicted == 78.0
        assert stats.rmsd > 0
        assert stats.high_confidence_accuracy >= 0
        assert stats.low_confidence_accuracy >= 0

    def test_calculate_statistics_empty_comparisons(self):
        """Test statistics calculation with empty comparisons."""
        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics([])

        assert stats.mean_experimental == 0.0
        assert stats.mean_predicted == 0.0
        assert stats.correlation == 0.0
        assert stats.rmsd == 0.0

    def test_confidence_stratification(self):
        """Test that high/low confidence regions are separated correctly."""
        # Create comparisons with known pLDDT values
        high_conf = [
            BFactorComparison("A_1", "A", 1, 20.0, 85.0, 65.0, 0.0),
            BFactorComparison("A_2", "A", 2, 25.0, 90.0, 65.0, 0.0),
        ]
        low_conf = [
            BFactorComparison("A_3", "A", 3, 30.0, 60.0, 30.0, 0.0),
            BFactorComparison("A_4", "A", 4, 35.0, 65.0, 30.0, 0.0),
        ]
        all_comparisons = high_conf + low_conf

        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics(all_comparisons)

        # High confidence (>70) should have difference of 65.0
        assert abs(stats.high_confidence_accuracy - 65.0) < 0.1
        # Low confidence (<=70) should have difference of 30.0
        assert abs(stats.low_confidence_accuracy - 30.0) < 0.1

    def test_normalized_bfactors_calculated(self):
        """Test that normalized B-factors are calculated during analysis."""
        analyzer = BFactorAnalyzer()
        exp_structure = get_structure(Path("./tests/data/complexes/experimental_9chl.cif"))
        pred_structure = get_structure(Path("./tests/data/complexes/predicted_9chl.cif"))

        comparisons, _ = analyzer.analyze_structures(exp_structure, pred_structure)

        # Check that at least some normalized values are non-zero
        normalized_values = [c.normalized_bfactor for c in comparisons]
        assert any(abs(v) > 0.01 for v in normalized_values)
        # Check that normalization is approximately mean=0, std=1
        assert abs(np.mean(normalized_values)) < 0.01
        assert abs(np.std(normalized_values) - 1.0) < 0.1


class TestDataFrameConversion:
    """Test DataFrame and CSV export functionality."""

    def test_to_dataframe_basic(self):
        """Test conversion of comparisons to DataFrame."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 20.0, 80.0, 60.0, -0.5),
            BFactorComparison("A_2", "A", 2, 30.0, 90.0, 60.0, 0.5),
        ]

        analyzer = BFactorAnalyzer()
        df = analyzer.to_dataframe(comparisons)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            'residue_id', 'chain_id', 'position',
            'experimental_bfactor', 'predicted_confidence',
            'difference', 'normalized_bfactor'
        ]

    def test_to_dataframe_empty(self):
        """Test DataFrame creation with empty comparisons."""
        analyzer = BFactorAnalyzer()
        df = analyzer.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestCSVExport:
    """Test CSV export functionality."""

    def test_save_to_csv_basic(self):
        """Test saving comparisons to CSV file."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 20.0, 80.0, 60.0, -0.5),
            BFactorComparison("A_2", "A", 2, 30.0, 90.0, 60.0, 0.5),
        ]

        analyzer = BFactorAnalyzer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bfactor_test.csv"
            analyzer.save_to_csv(comparisons, output_path)

            # Verify file exists
            assert output_path.exists()

            # Verify content
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert df['residue_id'].tolist() == ["A_1", "A_2"]
            assert df['position'].tolist() == [1, 2]

    def test_save_to_csv_creates_directories(self):
        """Test that save_to_csv creates parent directories."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 20.0, 80.0, 60.0, 0.0),
        ]

        analyzer = BFactorAnalyzer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "analysis" / "subfolder" / "bfactor.csv"
            analyzer.save_to_csv(comparisons, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_bfactors_no_nan(self):
        """Test that constant B-factors don't produce NaN correlation."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 50.0, 80.0, 30.0, 0.0),
            BFactorComparison("A_2", "A", 2, 50.0, 90.0, 40.0, 0.0),
            BFactorComparison("A_3", "A", 3, 50.0, 85.0, 35.0, 0.0),
        ]

        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics(comparisons)

        # Correlation should be 0.0, not NaN
        assert not np.isnan(stats.correlation)
        assert stats.correlation == 0.0
        # Other stats should still be valid
        assert stats.mean_experimental == 50.0
        assert stats.rmsd > 0

    def test_constant_plddt_no_nan(self):
        """Test that constant pLDDT values don't produce NaN correlation."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 20.0, 95.0, 75.0, 0.0),
            BFactorComparison("A_2", "A", 2, 30.0, 95.0, 65.0, 0.0),
            BFactorComparison("A_3", "A", 3, 40.0, 95.0, 55.0, 0.0),
        ]

        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics(comparisons)

        # Correlation should be 0.0, not NaN
        assert not np.isnan(stats.correlation)
        assert stats.correlation == 0.0

    def test_validate_empty_structure(self):
        """Test validation of structure with no atoms."""
        analyzer = BFactorAnalyzer()

        # Create mock empty structure
        empty_structure = Mock()
        empty_model = Mock()
        empty_model.__iter__ = Mock(return_value=iter([]))
        empty_structure.__iter__ = Mock(return_value=iter([empty_model]))

        # Should return False, not crash with division by zero
        result = analyzer.validate_bfactors(empty_structure)
        assert result is False

    def test_single_residue_comparison(self):
        """Test that single residue works correctly."""
        comparisons = [
            BFactorComparison("A_1", "A", 1, 25.0, 85.0, 60.0, 0.0)
        ]

        analyzer = BFactorAnalyzer()
        stats = analyzer.calculate_statistics(comparisons)

        # Should not crash, correlation should be 0 (insufficient data)
        assert stats.correlation == 0.0
        assert stats.mean_experimental == 25.0
        assert stats.mean_predicted == 85.0

    def test_insertion_code_in_residue_key(self):
        """Test that insertion codes are preserved in residue keys."""
        from biostructbenchmark.core.io import get_structure
        from pathlib import Path

        analyzer = BFactorAnalyzer()

        # Create a mock structure with insertion codes
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "A"

        # Create residues with insertion codes
        residues = []
        for i, icode in [(42, ' '), (42, 'A'), (42, 'B'), (43, ' ')]:
            residue = Mock()
            residue.get_id.return_value = (' ', i, icode)
            atom = Mock()
            atom.get_bfactor.return_value = 50.0 + len(residues)
            residue.__iter__ = Mock(return_value=iter([atom]))
            residues.append(residue)

        chain.__iter__ = Mock(return_value=iter(residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        model.get_models = Mock(return_value=iter([model]))
        structure.get_models = Mock(return_value=iter([model]))

        bfactors = analyzer._extract_bfactors_from_structure(structure, filter_heteroatoms=False)

        # Should have 4 distinct entries
        assert len(bfactors) == 4
        assert "A_42" in bfactors  # Regular residue 42
        assert "A_42A" in bfactors  # Residue 42 with insertion code A
        assert "A_42B" in bfactors  # Residue 42 with insertion code B
        assert "A_43" in bfactors  # Regular residue 43

    def test_heteroatom_filtering(self):
        """Test that water and ligands are filtered out."""
        analyzer = BFactorAnalyzer()

        # Create mix of standard residues and heteroatoms
        residues = []

        # Standard residue (hetflag = ' ')
        std_res = Mock()
        std_res.get_id.return_value = (' ', 1, ' ')
        atom1 = Mock()
        atom1.get_bfactor.return_value = 30.0
        std_res.__iter__ = Mock(return_value=iter([atom1]))
        residues.append(std_res)

        # Water molecule (hetflag = 'W')
        water = Mock()
        water.get_id.return_value = ('W', 100, ' ')
        atom2 = Mock()
        atom2.get_bfactor.return_value = 50.0
        water.__iter__ = Mock(return_value=iter([atom2]))
        residues.append(water)

        # Ligand (hetflag = 'H_LIG')
        ligand = Mock()
        ligand.get_id.return_value = ('H_LIG', 200, ' ')
        atom3 = Mock()
        atom3.get_bfactor.return_value = 40.0
        ligand.__iter__ = Mock(return_value=iter([atom3]))
        residues.append(ligand)

        # Build structure properly
        chain = Mock()
        chain.get_id.return_value = "A"
        chain.__iter__ = Mock(return_value=iter(residues))

        model = Mock()
        model.__iter__ = Mock(return_value=iter([chain]))

        structure = Mock()
        structure.get_models.return_value = [model]

        # With filtering (default)
        bfactors_filtered = analyzer._extract_bfactors_from_structure(structure, filter_heteroatoms=True)
        assert len(bfactors_filtered) == 1  # Only standard residue
        assert "A_1" in bfactors_filtered

        # Without filtering - need to rebuild residues list since we consumed it
        residues2 = []
        std_res2 = Mock()
        std_res2.get_id.return_value = (' ', 1, ' ')
        atom1b = Mock()
        atom1b.get_bfactor.return_value = 30.0
        std_res2.__iter__ = Mock(return_value=iter([atom1b]))
        residues2.append(std_res2)

        water2 = Mock()
        water2.get_id.return_value = ('W', 100, ' ')
        atom2b = Mock()
        atom2b.get_bfactor.return_value = 50.0
        water2.__iter__ = Mock(return_value=iter([atom2b]))
        residues2.append(water2)

        ligand2 = Mock()
        ligand2.get_id.return_value = ('H_LIG', 200, ' ')
        atom3b = Mock()
        atom3b.get_bfactor.return_value = 40.0
        ligand2.__iter__ = Mock(return_value=iter([atom3b]))
        residues2.append(ligand2)

        chain2 = Mock()
        chain2.get_id.return_value = "A"
        chain2.__iter__ = Mock(return_value=iter(residues2))

        model2 = Mock()
        model2.__iter__ = Mock(return_value=iter([chain2]))

        structure2 = Mock()
        structure2.get_models.return_value = [model2]

        bfactors_all = analyzer._extract_bfactors_from_structure(structure2, filter_heteroatoms=False)
        assert len(bfactors_all) == 3  # All residues


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real structure data."""

    def test_full_workflow_9chl(self):
        """Test complete workflow from loading to CSV export."""
        analyzer = BFactorAnalyzer()

        # Load structures
        exp_structure = get_structure(Path("./tests/data/complexes/experimental_9chl.cif"))
        pred_structure = get_structure(Path("./tests/data/complexes/predicted_9chl.cif"))

        # Validate
        assert analyzer.validate_bfactors(exp_structure)
        assert analyzer.validate_bfactors(pred_structure)

        # Analyze
        comparisons, stats = analyzer.analyze_structures(exp_structure, pred_structure)

        # Verify results
        assert len(comparisons) > 0
        assert stats.mean_experimental > 0
        assert stats.mean_predicted > 0
        assert stats.correlation != 0

        # Export to CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "integration_test.csv"
            analyzer.save_to_csv(comparisons, output_path)

            # Verify CSV content
            df = pd.read_csv(output_path)
            assert len(df) == len(comparisons)
            assert all(col in df.columns for col in [
                'residue_id', 'chain_id', 'position',
                'experimental_bfactor', 'predicted_confidence'
            ])

    def test_full_workflow_9ny8(self):
        """Test complete workflow with 9ny8 structure."""
        analyzer = BFactorAnalyzer()
        exp_path = Path("./tests/data/complexes/experimental_9ny8.cif")
        pred_path = Path("./tests/data/complexes/predicted_9ny8.cif")

        comparisons, stats = analyzer.compare_bfactors(exp_path, pred_path)

        assert len(comparisons) > 0
        assert stats.mean_experimental > 0
        assert stats.mean_predicted > 0

        # Verify correlation is within reasonable range
        assert -1 <= stats.correlation <= 1

        # Verify RMSD is positive
        assert stats.rmsd > 0
