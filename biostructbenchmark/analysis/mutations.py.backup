# Create the mutations analyzer file
"""
biostructbenchmark/analysis/mutations.py
Mutation detection and impact analysis
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

from ..core.io import get_structure
from ..core.alignment import ResidueRMSD


@dataclass
class Mutation:
    """Container for mutation information"""
    chain_id: str
    position: int
    wild_type: str  # Original residue
    mutant: str  # Mutated residue
    local_rmsd: float  # RMSD at mutation site
    neighbor_rmsd: float  # Average RMSD of neighboring residues
    impact_score: float  # Calculated impact on structure
    mutation_type: str  # 'conservative', 'non-conservative', 'radical'


class MutationAnalyzer:
    """Detect and analyze mutations between structures"""
    
    def __init__(self):
        """Initialize mutation analyzer"""
        self.aa_properties = self._load_aa_properties()
    
    def _load_aa_properties(self) -> Dict[str, Dict]:
        """Load amino acid properties for classification"""
        return {
            # Hydrophobic
            'ALA': {'hydrophobic': True, 'charge': 0, 'size': 'small'},
            'VAL': {'hydrophobic': True, 'charge': 0, 'size': 'medium'},
            'ILE': {'hydrophobic': True, 'charge': 0, 'size': 'medium'},
            'LEU': {'hydrophobic': True, 'charge': 0, 'size': 'medium'},
            'MET': {'hydrophobic': True, 'charge': 0, 'size': 'medium'},
            'PHE': {'hydrophobic': True, 'charge': 0, 'size': 'large'},
            'TRP': {'hydrophobic': True, 'charge': 0, 'size': 'large'},
            'PRO': {'hydrophobic': True, 'charge': 0, 'size': 'medium'},
            
            # Polar
            'SER': {'hydrophobic': False, 'charge': 0, 'size': 'small'},
            'THR': {'hydrophobic': False, 'charge': 0, 'size': 'small'},
            'CYS': {'hydrophobic': False, 'charge': 0, 'size': 'small'},
            'TYR': {'hydrophobic': False, 'charge': 0, 'size': 'large'},
            'ASN': {'hydrophobic': False, 'charge': 0, 'size': 'medium'},
            'GLN': {'hydrophobic': False, 'charge': 0, 'size': 'medium'},
            'GLY': {'hydrophobic': False, 'charge': 0, 'size': 'tiny'},
            
            # Charged
            'ASP': {'hydrophobic': False, 'charge': -1, 'size': 'medium'},
            'GLU': {'hydrophobic': False, 'charge': -1, 'size': 'medium'},
            'LYS': {'hydrophobic': False, 'charge': 1, 'size': 'large'},
            'ARG': {'hydrophobic': False, 'charge': 1, 'size': 'large'},
            'HIS': {'hydrophobic': False, 'charge': 1, 'size': 'medium'},
        }
    
    def detect_mutations(self, observed_path: Path, predicted_path: Path) -> List[Mutation]:
        """
        Detect mutations between two structures
        
        Args:
            observed_path: Path to observed/reference structure
            predicted_path: Path to predicted/mutated structure
            
        Returns:
            List of detected mutations
        """
        obs_struct = get_structure(observed_path)
        pred_struct = get_structure(predicted_path)
        
        if not obs_struct or not pred_struct:
            return []
        
        mutations = []
        
        # Compare residues in each chain
        for obs_chain, pred_chain in zip(obs_struct[0], pred_struct[0]):
            if obs_chain.get_id() != pred_chain.get_id():
                continue
            
            obs_residues = {r.get_id()[1]: r for r in obs_chain}
            pred_residues = {r.get_id()[1]: r for r in pred_chain}
            
            # Find common positions
            common_positions = set(obs_residues.keys()) & set(pred_residues.keys())
            
            for position in common_positions:
                obs_res = obs_residues[position]
                pred_res = pred_residues[position]
                
                obs_name = obs_res.get_resname().strip()
                pred_name = pred_res.get_resname().strip()
                
                # Check if mutation occurred
                if obs_name != pred_name and obs_name in self.aa_properties and pred_name in self.aa_properties:
                    mutation = Mutation(
                        chain_id=obs_chain.get_id(),
                        position=position,
                        wild_type=obs_name,
                        mutant=pred_name,
                        local_rmsd=0.0,  # Will be calculated
                        neighbor_rmsd=0.0,  # Will be calculated
                        impact_score=0.0,  # Will be calculated
                        mutation_type=self._classify_mutation(obs_name, pred_name)
                    )
                    mutations.append(mutation)
        
        return mutations
    
    def _classify_mutation(self, wild_type: str, mutant: str) -> str:
        """
        Classify mutation type based on amino acid properties
        
        Args:
            wild_type: Original amino acid
            mutant: Mutated amino acid
            
        Returns:
            Mutation classification
        """
        if wild_type not in self.aa_properties or mutant not in self.aa_properties:
            return 'unknown'
        
        wt_props = self.aa_properties[wild_type]
        mut_props = self.aa_properties[mutant]
        
        # Check property changes
        hydrophobic_change = wt_props['hydrophobic'] != mut_props['hydrophobic']
        charge_change = wt_props['charge'] != mut_props['charge']
        size_change = wt_props['size'] != mut_props['size']
        
        if charge_change:
            return 'radical'  # Charge change is most disruptive
        elif hydrophobic_change:
            return 'non-conservative'  # Hydrophobicity change
        elif size_change:
            return 'non-conservative'  # Size change
        else:
            return 'conservative'  # Similar properties