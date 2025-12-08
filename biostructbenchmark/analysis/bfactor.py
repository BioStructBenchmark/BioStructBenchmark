"""
B-factor and pLDDT confidence analysis for protein structures.

This module provides tools to analyze B-factors from experimental structures
and compare them with pLDDT confidence scores from AlphaFold predictions.

B-factors represent thermal motion/disorder in experimental structures, while
pLDDT (predicted Local Distance Difference Test) represents AlphaFold's
confidence in its predictions (0-100 scale, stored in B-factor field).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import Structure

from biostructbenchmark.core.io import get_structure

logger = logging.getLogger(__name__)


@dataclass
class BFactorComparison:
    """
    Container for individual residue B-factor analysis results.

    Attributes:
        residue_id: Unique identifier (e.g., "A_42" for chain A position 42)
        chain_id: Chain identifier
        position: Residue position number
        experimental_bfactor: B-factor from experimental structure
        predicted_confidence: pLDDT value from AlphaFold prediction (0-100)
        difference: predicted_confidence - experimental_bfactor
        normalized_bfactor: Z-score normalized experimental B-factor
    """

    residue_id: str
    chain_id: str
    position: int
    experimental_bfactor: float
    predicted_confidence: float
    difference: float
    normalized_bfactor: float


@dataclass
class BFactorStatistics:
    """
    Summary statistics for B-factor analysis.

    Attributes:
        mean_experimental: Mean experimental B-factor across all residues
        mean_predicted: Mean predicted confidence (pLDDT) across all residues
        correlation: Pearson correlation between normalized B-factors and pLDDT
        rmsd: Root mean square difference
        high_confidence_accuracy: Mean absolute difference for pLDDT > 70
        low_confidence_accuracy: Mean absolute difference for pLDDT ≤ 70
    """

    mean_experimental: float
    mean_predicted: float
    correlation: float
    rmsd: float
    high_confidence_accuracy: float
    low_confidence_accuracy: float


class BFactorAnalyzer:
    """
    Analyze B-factors and pLDDT confidence metrics between structures.

    This class provides methods to extract B-factors from experimental structures,
    extract pLDDT values from AlphaFold predictions, compare them, and export
    results to CSV format.

    pLDDT confidence thresholds used:
        - Very high confidence: > 90
        - Confident: > 70
        - Low confidence: > 50
        - Very low: ≤ 50

    Example:
        >>> analyzer = BFactorAnalyzer()
        >>> exp_structure = get_structure(Path("experimental.cif"))
        >>> pred_structure = get_structure(Path("predicted.cif"))
        >>> comparisons, stats = analyzer.analyze_structures(exp_structure, pred_structure)
        >>> analyzer.save_to_csv(comparisons, Path("bfactor_comparison.csv"))
    """

    def __init__(self) -> None:
        """Initialize the B-factor analyzer with pLDDT thresholds."""
        self.plddt_thresholds = {"very_high": 90, "confident": 70, "low": 50}

    def validate_bfactors(self, structure: Structure.Structure) -> bool:
        """
        Check if structure contains valid B-factor values.

        Args:
            structure: BioPython Structure object to validate

        Returns:
            True if structure has at least some non-zero B-factors, False otherwise
        """
        if not structure:
            return False

        try:
            bfactor_count = 0
            nonzero_count = 0

            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            bfactor_count += 1
                            if atom.get_bfactor() > 0:
                                nonzero_count += 1

            # At least 50% of atoms should have non-zero B-factors
            # Check bfactor_count > 0 to prevent division by zero
            return bfactor_count > 0 and nonzero_count > 0 and (nonzero_count / bfactor_count) > 0.5
        except Exception as e:
            logger.warning(f"Failed to validate B-factors: {e}")
            return False

    def _extract_bfactors_from_structure(
        self, structure: Structure.Structure, filter_heteroatoms: bool = True
    ) -> dict[str, float]:
        """
        Extract average B-factors per residue from a structure object.

        Args:
            structure: BioPython Structure object
            filter_heteroatoms: If True, skip water molecules, ligands, and other heteroatoms

        Returns:
            Dictionary mapping residue IDs to average B-factors
            Residue ID format: "chain_position" or "chain_positionInsCode" if insertion code exists
        """
        bfactors = {}

        try:
            # Use only the first model (standard for X-ray structures)
            # For NMR structures with multiple models, this uses model 0
            models = list(structure.get_models())
            if not models:
                return {}

            model = models[0]

            for chain in model:
                chain_id = chain.get_id()

                for residue in chain:
                    res_id = residue.get_id()
                    hetflag, resseq, icode = res_id

                    # Skip heteroatoms if filtering enabled
                    # hetflag=' ' means standard residue
                    # hetflag='W' means water
                    # hetflag='H_*' means other heteroatoms
                    if filter_heteroatoms and hetflag != " ":
                        continue

                    # Calculate average B-factor across all atoms in residue
                    residue_bfactors = [atom.get_bfactor() for atom in residue]
                    if not residue_bfactors:
                        continue

                    avg_bfactor = np.mean(residue_bfactors)

                    # Include insertion code in key if present
                    # This ensures residues like 42A and 42B are distinct
                    if icode.strip():
                        residue_key = f"{chain_id}_{resseq}{icode}"
                    else:
                        residue_key = f"{chain_id}_{resseq}"

                    bfactors[residue_key] = float(avg_bfactor)

        except Exception as e:
            logger.warning(
                f"Error extracting B-factors (returning {len(bfactors)} partial results): {e}"
            )
            return bfactors

        return bfactors

    def extract_bfactors(self, structure_path: Path) -> dict[str, float]:
        """
        Extract average B-factors per residue from a structure file.

        Filters out water molecules and other heteroatoms by default.
        Handles insertion codes properly.

        Args:
            structure_path: Path to structure file (.pdb or .cif)

        Returns:
            Dictionary mapping residue IDs (e.g., "A_42" or "A_42A") to average B-factors
        """
        structure = get_structure(structure_path)
        if not structure:
            return {}

        try:
            return self._extract_bfactors_from_structure(structure, filter_heteroatoms=True)
        except Exception:
            logger.exception("Error extracting B-factors from %s", structure_path)
            return {}

    def analyze_structures(
        self, observed_structure: Structure.Structure, predicted_structure: Structure.Structure
    ) -> tuple[list[BFactorComparison], BFactorStatistics]:
        """
        Analyze B-factors between experimental and predicted structures.

        This is the main analysis method. It extracts B-factors from both structures,
        finds common residues, calculates comparisons, and computes statistics.

        Filters out water molecules and heteroatoms automatically.
        Handles insertion codes properly.
        Uses first model for multi-model structures.

        Args:
            observed_structure: Experimental structure (Bio.PDB Structure object)
            predicted_structure: Predicted structure (Bio.PDB Structure object)

        Returns:
            Tuple of (list of BFactorComparison objects, BFactorStatistics object)

        Raises:
            ValueError: If structures are None or have no common residues
        """
        if not observed_structure or not predicted_structure:
            raise ValueError("Both structures must be valid Structure objects")

        # Extract B-factors from both structures using helper method
        # This automatically filters heteroatoms and handles insertion codes
        try:
            obs_bfactors = self._extract_bfactors_from_structure(
                observed_structure, filter_heteroatoms=True
            )
            pred_bfactors = self._extract_bfactors_from_structure(
                predicted_structure, filter_heteroatoms=True
            )
        except Exception as e:
            raise ValueError(f"Error extracting B-factors from structures: {e}") from e

        # Find common residues
        common_residues = set(obs_bfactors.keys()) & set(pred_bfactors.keys())

        if not common_residues:
            raise ValueError("No common residues found between structures")

        # Create comparisons for common residues
        comparisons = []
        exp_values = []

        for res_key in sorted(common_residues):
            # Parse residue key (handles insertion codes like "A_42A")
            # Use rsplit to handle potential underscores in chain IDs
            parts = res_key.rsplit("_", 1)
            if len(parts) == 2:
                chain_id, position_str = parts
                # Extract numeric position (may have insertion code suffix)
                # Find where non-digits start
                position = int("".join(c for c in position_str if c.isdigit() or c == "-"))
            else:
                # Fallback for malformed keys
                chain_id = res_key
                position = 0

            exp_value = obs_bfactors[res_key]
            pred_value = pred_bfactors[res_key]

            exp_values.append(exp_value)

            comparison = BFactorComparison(
                residue_id=res_key,
                chain_id=chain_id,
                position=position,
                experimental_bfactor=exp_value,
                predicted_confidence=pred_value,
                difference=pred_value - exp_value,
                normalized_bfactor=0.0,  # Will be calculated below
            )
            comparisons.append(comparison)

        # Calculate Z-score normalization for experimental B-factors
        if len(exp_values) > 1:
            mean_exp = float(np.mean(exp_values))
            std_exp = float(np.std(exp_values))
            if std_exp > 0:
                for comp in comparisons:
                    comp.normalized_bfactor = (comp.experimental_bfactor - mean_exp) / std_exp

        # Calculate statistics
        stats = self._calculate_statistics(comparisons)

        return comparisons, stats

    def compare_bfactors(
        self, experimental_path: Path, predicted_path: Path
    ) -> tuple[list[BFactorComparison], BFactorStatistics]:
        """
        Compare B-factors between experimental and predicted structure files.

        This is a convenience method that loads structures from files and
        calls analyze_structures().

        Args:
            experimental_path: Path to experimental structure file
            predicted_path: Path to predicted structure file

        Returns:
            Tuple of (list of BFactorComparison objects, BFactorStatistics object)
        """
        exp_structure = get_structure(experimental_path)
        pred_structure = get_structure(predicted_path)

        if not exp_structure:
            raise ValueError(f"Could not load experimental structure from {experimental_path}")
        if not pred_structure:
            raise ValueError(f"Could not load predicted structure from {predicted_path}")

        return self.analyze_structures(exp_structure, pred_structure)

    def _calculate_statistics(self, comparisons: list[BFactorComparison]) -> BFactorStatistics:
        """
        Calculate summary statistics from B-factor comparisons.

        Args:
            comparisons: List of BFactorComparison objects

        Returns:
            BFactorStatistics object with summary metrics
        """
        if not comparisons:
            return BFactorStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        exp_values = np.array([c.experimental_bfactor for c in comparisons])
        pred_values = np.array([c.predicted_confidence for c in comparisons])

        # Normalize B-factors for correlation
        # Check that we have enough data and non-constant values
        if len(exp_values) > 1:
            exp_std = np.std(exp_values)
            pred_std = np.std(pred_values)

            # Only calculate correlation if both have variation
            # If either is constant (std=0), correlation is undefined
            if exp_std > 0 and pred_std > 0:
                exp_normalized = (exp_values - np.mean(exp_values)) / exp_std
                pred_normalized = (pred_values - np.mean(pred_values)) / pred_std
                correlation = float(np.corrcoef(exp_normalized, pred_normalized)[0, 1])
            else:
                # One or both sets are constant - no correlation
                correlation = 0.0
        else:
            correlation = 0.0

        # Calculate RMSD
        differences = np.array([c.difference for c in comparisons])
        rmsd = float(np.sqrt(np.mean(differences**2)))

        # Accuracy by confidence regions (using pLDDT threshold of 70)
        high_conf = [c for c in comparisons if c.predicted_confidence > 70]
        low_conf = [c for c in comparisons if c.predicted_confidence <= 70]

        high_acc = float(np.mean([abs(c.difference) for c in high_conf])) if high_conf else 0.0
        low_acc = float(np.mean([abs(c.difference) for c in low_conf])) if low_conf else 0.0

        return BFactorStatistics(
            mean_experimental=float(np.mean(exp_values)),
            mean_predicted=float(np.mean(pred_values)),
            correlation=correlation,
            rmsd=rmsd,
            high_confidence_accuracy=high_acc,
            low_confidence_accuracy=low_acc,
        )

    def calculate_statistics(self, comparisons: list[BFactorComparison]) -> BFactorStatistics:
        """
        Calculate statistics from B-factor comparisons.

        Public wrapper for _calculate_statistics.

        Args:
            comparisons: List of BFactorComparison objects

        Returns:
            BFactorStatistics object
        """
        return self._calculate_statistics(comparisons)

    def to_dataframe(self, comparisons: list[BFactorComparison]) -> pd.DataFrame:
        """
        Convert B-factor comparisons to pandas DataFrame.

        Args:
            comparisons: List of BFactorComparison objects

        Returns:
            DataFrame with columns: residue_id, chain_id, position,
                                   experimental_bfactor, predicted_confidence,
                                   difference, normalized_bfactor
        """
        data = [
            {
                "residue_id": comp.residue_id,
                "chain_id": comp.chain_id,
                "position": comp.position,
                "experimental_bfactor": comp.experimental_bfactor,
                "predicted_confidence": comp.predicted_confidence,
                "difference": comp.difference,
                "normalized_bfactor": comp.normalized_bfactor,
            }
            for comp in comparisons
        ]
        return pd.DataFrame(data)

    def save_to_csv(self, comparisons: list[BFactorComparison], output_path: Path) -> None:
        """
        Export B-factor comparisons to CSV file.

        Creates parent directories if they don't exist.

        Args:
            comparisons: List of BFactorComparison objects
            output_path: Path where CSV file should be saved

        Raises:
            IOError: If file cannot be written
        """
        try:
            df = self.to_dataframe(comparisons)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        except Exception as e:
            raise OSError(f"Failed to save CSV to {output_path}: {e}") from e
