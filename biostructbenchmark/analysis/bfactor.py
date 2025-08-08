"""
biostructbenchmark/analysis/bfactor.py
B-factor and pLDDT confidence analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.io import get_structure

@dataclass
class BFactorComparison:
    """Container for B-factor analysis results"""
    residue_id: str
    chain_id: str
    position: int
    experimental_bfactor: float
    predicted_confidence: float  # pLDDT for AlphaFold
    difference: float
    normalized_bfactor: float  # Z-score normalized
    
@dataclass 
class BFactorStatistics:
    """Summary statistics for B-factor analysis"""
    mean_experimental: float
    mean_predicted: float
    correlation: float
    rmsd: float
    high_confidence_accuracy: float  # Accuracy in high pLDDT regions
    low_confidence_accuracy: float   # Accuracy in low pLDDT regions

class BFactorAnalyzer:
    """Analyze B-factors and confidence metrics"""
    
    def __init__(self):
        self.plddt_thresholds = {
            'very_high': 90,
            'confident': 70,
            'low': 50
        }
    
    def extract_bfactors(self, structure_path: Path) -> Dict[str, List[float]]:
        """Extract B-factors from structure"""
        structure = get_structure(structure_path)
        if not structure:
            return {}
        
        bfactors = {}
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                chain_bfactors = []
                
                for residue in chain:
                    # Get average B-factor for residue
                    residue_bfactors = [atom.get_bfactor() for atom in residue]
                    avg_bfactor = np.mean(residue_bfactors)
                    
                    residue_key = f"{chain_id}_{residue.get_id()[1]}"
                    bfactors[residue_key] = avg_bfactor
                    
        return bfactors
    
    def compare_bfactors(self, experimental_path: Path, 
                        predicted_path: Path) -> Tuple[List[BFactorComparison], BFactorStatistics]:
        """Compare B-factors between experimental and predicted structures"""
        
        exp_bfactors = self.extract_bfactors(experimental_path)
        pred_bfactors = self.extract_bfactors(predicted_path)  # These are pLDDT values for AF
        
        comparisons = []
        
        # Find common residues
        common_residues = set(exp_bfactors.keys()) & set(pred_bfactors.keys())
        
        for res_key in common_residues:
            chain_id, position = res_key.split('_')
            
            comparison = BFactorComparison(
                residue_id=res_key,
                chain_id=chain_id,
                position=int(position),
                experimental_bfactor=exp_bfactors[res_key],
                predicted_confidence=pred_bfactors[res_key],
                difference=pred_bfactors[res_key] - exp_bfactors[res_key],
                normalized_bfactor=0.0  # Will calculate after
            )
            comparisons.append(comparison)
        
        # Calculate statistics
        stats = self._calculate_statistics(comparisons)
        
        return comparisons, stats
    
    def _calculate_statistics(self, comparisons: List[BFactorComparison]) -> BFactorStatistics:
        """Calculate summary statistics"""
        if not comparisons:
            return BFactorStatistics(0, 0, 0, 0, 0, 0)
        
        exp_values = [c.experimental_bfactor for c in comparisons]
        pred_values = [c.predicted_confidence for c in comparisons]
        
        # Normalize B-factors for correlation
        exp_normalized = (exp_values - np.mean(exp_values)) / np.std(exp_values)
        pred_normalized = (pred_values - np.mean(pred_values)) / np.std(pred_values)
        
        correlation = np.corrcoef(exp_normalized, pred_normalized)[0, 1]
        
        # Calculate RMSD
        differences = [c.difference for c in comparisons]
        rmsd = np.sqrt(np.mean(np.array(differences) ** 2))
        
        # Accuracy by confidence regions
        high_conf = [c for c in comparisons if c.predicted_confidence > 70]
        low_conf = [c for c in comparisons if c.predicted_confidence <= 70]
        
        high_acc = np.mean([abs(c.difference) for c in high_conf]) if high_conf else 0
        low_acc = np.mean([abs(c.difference) for c in low_conf]) if low_conf else 0
        
        return BFactorStatistics(
            mean_experimental=np.mean(exp_values),
            mean_predicted=np.mean(pred_values),
            correlation=correlation,
            rmsd=rmsd,
            high_confidence_accuracy=high_acc,
            low_confidence_accuracy=low_acc
        )