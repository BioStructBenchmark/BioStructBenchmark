# Create the file with the consensus analyzer code
"""
biostructbenchmark/analysis/consensus.py
Consensus error mapping across multiple structure comparisons
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..core.alignment import ResidueRMSD


@dataclass
class ConsensusError:
    """Container for consensus error at a position"""
    position: str  # e.g., "A_10" for chain A position 10
    mean_rmsd: float
    std_rmsd: float
    frequency: int  # Number of structures with this position
    confidence: float  # Statistical confidence of the error
    residue_types: List[str]  # Residue types seen at this position


class ConsensusAnalyzer:
    """Identify consistently mispredicted regions across structures"""
    
    def __init__(self, rmsd_threshold: float = 3.0):
        """
        Initialize consensus analyzer
        
        Args:
            rmsd_threshold: RMSD threshold for considering a position as mispredicted
        """
        self.rmsd_threshold = rmsd_threshold
        self.position_errors = defaultdict(list)
    
    def add_structure_comparison(self, residue_rmsds: List[ResidueRMSD], 
                               structure_id: str) -> None:
        """
        Add RMSD data from one structure comparison
        
        Args:
            residue_rmsds: Per-residue RMSD data from alignment
            structure_id: Identifier for this structure pair
        """
        for rmsd_data in residue_rmsds:
            position_key = f"{rmsd_data.chain_id}_{rmsd_data.position}"
            self.position_errors[position_key].append({
                'rmsd': rmsd_data.rmsd,
                'residue_type': rmsd_data.residue_type,
                'structure_id': structure_id
            })
    
    def calculate_consensus(self, min_structures: int = 3) -> List[ConsensusError]:
        """
        Calculate consensus errors across all added structures
        
        Args:
            min_structures: Minimum number of structures required for consensus
            
        Returns:
            List of consensus errors sorted by mean RMSD
        """
        consensus_errors = []
        
        for position, errors in self.position_errors.items():
            if len(errors) < min_structures:
                continue
            
            rmsds = [e['rmsd'] for e in errors]
            residue_types = list(set(e['residue_type'] for e in errors))
            
            mean_rmsd = np.mean(rmsds)
            std_rmsd = np.std(rmsds)
            
            # Calculate confidence based on consistency
            # High confidence if low std and high mean RMSD
            if mean_rmsd > 0:
                confidence = (1 - std_rmsd/mean_rmsd) * min(1.0, mean_rmsd/self.rmsd_threshold)
            else:
                confidence = 0.0
            
            consensus_error = ConsensusError(
                position=position,
                mean_rmsd=mean_rmsd,
                std_rmsd=std_rmsd,
                frequency=len(errors),
                confidence=max(0, min(1, confidence)),
                residue_types=residue_types
            )
            
            consensus_errors.append(consensus_error)
        
        # Sort by mean RMSD (worst first)
        consensus_errors.sort(key=lambda x: x.mean_rmsd, reverse=True)
        
        return consensus_errors
    
    def identify_problem_regions(self, consensus_errors: List[ConsensusError],
                                rmsd_cutoff: float = None) -> Dict[str, List[ConsensusError]]:
        """
        Identify problematic regions based on consensus errors
        
        Args:
            consensus_errors: List of consensus errors
            rmsd_cutoff: RMSD threshold (uses self.rmsd_threshold if None)
            
        Returns:
            Dictionary categorizing errors by severity
        """
        if rmsd_cutoff is None:
            rmsd_cutoff = self.rmsd_threshold
        
        regions = {
            'critical': [],  # Very high RMSD, high confidence
            'problematic': [],  # High RMSD, medium confidence
            'moderate': [],  # Moderate RMSD
            'acceptable': []  # Low RMSD
        }
        
        for error in consensus_errors:
            if error.mean_rmsd > rmsd_cutoff * 1.5 and error.confidence > 0.7:
                regions['critical'].append(error)
            elif error.mean_rmsd > rmsd_cutoff and error.confidence > 0.5:
                regions['problematic'].append(error)
            elif error.mean_rmsd > rmsd_cutoff * 0.5:
                regions['moderate'].append(error)
            else:
                regions['acceptable'].append(error)
        
        return regions
    
    def export_consensus_map(self, consensus_errors: List[ConsensusError],
                            output_path: Path) -> None:
        """
        Export consensus error map to CSV
        
        Args:
            consensus_errors: List of consensus errors
            output_path: Path for output CSV file
        """
        data = []
        for error in consensus_errors:
            data.append({
                'position': error.position,
                'mean_rmsd': f"{error.mean_rmsd:.3f}",
                'std_rmsd': f"{error.std_rmsd:.3f}",
                'frequency': error.frequency,
                'confidence': f"{error.confidence:.3f}",
                'residue_types': ','.join(error.residue_types)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Consensus error map saved to: {output_path}")
    
    def generate_summary_report(self, consensus_errors: List[ConsensusError]) -> Dict:
        """
        Generate summary statistics of consensus errors
        
        Args:
            consensus_errors: List of consensus errors
            
        Returns:
            Dictionary with summary statistics
        """
        if not consensus_errors:
            return {'total_positions': 0}
        
        rmsds = [e.mean_rmsd for e in consensus_errors]
        high_error_positions = [e for e in consensus_errors if e.mean_rmsd > self.rmsd_threshold]
        
        return {
            'total_positions': len(consensus_errors),
            'mean_rmsd_overall': np.mean(rmsds),
            'std_rmsd_overall': np.std(rmsds),
            'max_rmsd': max(rmsds),
            'min_rmsd': min(rmsds),
            'high_error_count': len(high_error_positions),
            'high_error_percentage': 100 * len(high_error_positions) / len(consensus_errors),
            'worst_positions': [
                {'position': e.position, 'rmsd': e.mean_rmsd}
                for e in consensus_errors[:10]
            ]
        }