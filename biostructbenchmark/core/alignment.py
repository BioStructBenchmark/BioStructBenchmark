"""Handles structure superposition and per-residue/nucleotide RMSD calculations"""

# TODO: Use sequence-based alignment for DNA to handle numbering inconsistencies

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

from biostructbenchmark.core.io import get_structure
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB import Structure


@dataclass
class ResidueRMSD:
    """Container for per-residue RMSD data"""
    residue_id: str
    residue_type: str
    chain_id: str
    position: int
    rmsd: float
    atom_count: int
    molecule_type: str  # 'protein' or 'dna'


@dataclass
class AlignmentResult:
    """Container for alignment results"""
    overall_rmsd: float
    residue_rmsds: List[ResidueRMSD]
    transformation_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    aligned_atom_count: int
    
    
def compare_structures(observed_path: Path, predicted_path: Path) -> Optional[AlignmentResult]:
    """Compare two structures and return detailed alignment results."""
    observed = get_structure(observed_path)
    predicted = get_structure(predicted_path)

    if observed is None or predicted is None:
        return None

    try:
        # Perform alignment and calculate overall RMSD
        alignment_result = align_structures(observed, predicted)
        
        # Calculate per-residue RMSDs
        residue_rmsds = calculate_per_residue_rmsd(observed, predicted, alignment_result)
        
        # Return structured results
        return AlignmentResult(
            overall_rmsd=alignment_result['rmsd'],
            residue_rmsds=residue_rmsds,
            transformation_matrix=alignment_result['transformation'],
            rotation_matrix=alignment_result['rotation'],
            translation_vector=alignment_result['translation'],
            aligned_atom_count=alignment_result['atom_count']
        )

    except Exception as e:
        print(f"Error comparing structures: {e}")
        return None


def align_structures(observed: Structure.Structure, 
                    predicted: Structure.Structure) -> Dict:
    """Perform structure alignment and return transformation details."""
    
    # Get CA atoms for proteins, P atoms for DNA
    obs_atoms = []
    pred_atoms = []
    
    for obs_chain, pred_chain in zip(observed[0], predicted[0]):
        obs_ca = get_backbone_atoms(obs_chain)
        pred_ca = get_backbone_atoms(pred_chain)
        
        # Align by position (could be improved with sequence alignment)
        min_len = min(len(obs_ca), len(pred_ca))
        obs_atoms.extend(obs_ca[:min_len])
        pred_atoms.extend(pred_ca[:min_len])
    
    if not obs_atoms or not pred_atoms:
        raise ValueError("No backbone atoms found for alignment")
    
    # Ensure equal lengths
    min_len = min(len(obs_atoms), len(pred_atoms))
    obs_atoms = obs_atoms[:min_len]
    pred_atoms = pred_atoms[:min_len]

    # Superimpose structures
    superimposer = Superimposer()
    superimposer.set_atoms(obs_atoms, pred_atoms)
    
    # Apply transformation to predicted structure
    superimposer.apply(predicted[0].get_atoms())
    
    return {
        'rmsd': float(superimposer.rms),
        'transformation': superimposer.rotran[1],  # transformation matrix
        'rotation': superimposer.rotran[0],       # rotation matrix  
        'translation': superimposer.rotran[1],    # translation vector
        'atom_count': len(obs_atoms)
    }


def get_backbone_atoms(chain):
    """Get appropriate backbone atoms for protein (CA) or DNA (P)."""
    backbone_atoms = []
    
    for residue in chain:
        if is_protein_residue(residue):
            if 'CA' in residue:
                backbone_atoms.append(residue['CA'])
        elif is_dna_residue(residue):
            if 'P' in residue:
                backbone_atoms.append(residue['P'])
    
    return backbone_atoms


def is_protein_residue(residue) -> bool:
    """Check if residue is a standard amino acid."""
    standard_aa = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.get_resname().strip() in standard_aa


def is_dna_residue(residue) -> bool:
    """Check if residue is a DNA nucleotide."""
    dna_bases = {'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C'}
    return residue.get_resname().strip() in dna_bases


def calculate_per_residue_rmsd(observed: Structure.Structure, 
                              predicted: Structure.Structure,
                              alignment_result: Dict) -> List[ResidueRMSD]:
    """Calculate RMSD for each residue after alignment."""
    residue_rmsds = []
    
    for obs_chain, pred_chain in zip(observed[0], predicted[0]):
        # Get residue pairs (this assumes 1:1 mapping - could be improved)
        obs_residues = list(obs_chain)
        pred_residues = list(pred_chain)
        
        min_residues = min(len(obs_residues), len(pred_residues))
        
        for i in range(min_residues):
            obs_res = obs_residues[i]
            pred_res = pred_residues[i]
            
            # Skip if different residue types
            if obs_res.get_resname() != pred_res.get_resname():
                continue
            
            # Get corresponding atoms
            obs_atoms = []
            pred_atoms = []
            
            atom_names = get_common_atoms(obs_res, pred_res)
            
            for atom_name in atom_names:
                if atom_name in obs_res and atom_name in pred_res:
                    obs_atoms.append(obs_res[atom_name])
                    pred_atoms.append(pred_res[atom_name])
            
            if len(obs_atoms) >= 2:  # Need at least 2 atoms for meaningful RMSD
                rmsd = calculate_atom_rmsd(obs_atoms, pred_atoms)
                
                residue_rmsd = ResidueRMSD(
                    residue_id=f"{obs_res.get_resname()}_{obs_res.get_id()[1]}",
                    residue_type=obs_res.get_resname(),
                    chain_id=obs_chain.get_id(),
                    position=obs_res.get_id()[1],
                    rmsd=rmsd,
                    atom_count=len(obs_atoms),
                    molecule_type='protein' if is_protein_residue(obs_res) else 'dna'
                )
                
                residue_rmsds.append(residue_rmsd)
    
    return residue_rmsds


def get_common_atoms(res1, res2) -> List[str]:
    """Get list of common atom names between two residues."""
    atoms1 = {atom.get_name() for atom in res1}
    atoms2 = {atom.get_name() for atom in res2}
    return list(atoms1 & atoms2)


def calculate_atom_rmsd(atoms1: List, atoms2: List) -> float:
    """Calculate RMSD between two sets of atoms."""
    if len(atoms1) != len(atoms2):
        raise ValueError("Atom lists must be same length")
    
    coords1 = np.array([atom.get_coord() for atom in atoms1])
    coords2 = np.array([atom.get_coord() for atom in atoms2])
    
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def export_residue_rmsd_csv(residue_rmsds: List[ResidueRMSD], output_path: Path) -> None:
    """Export per-residue RMSD data to CSV."""
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['residue_id', 'residue_type', 'chain_id', 'position', 
                     'rmsd', 'atom_count', 'molecule_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for residue_rmsd in residue_rmsds:
            writer.writerow({
                'residue_id': residue_rmsd.residue_id,
                'residue_type': residue_rmsd.residue_type,
                'chain_id': residue_rmsd.chain_id,
                'position': residue_rmsd.position,
                'rmsd': f"{residue_rmsd.rmsd:.3f}",
                'atom_count': residue_rmsd.atom_count,
                'molecule_type': residue_rmsd.molecule_type
            })


# Legacy function for backward compatibility
def compare_structures_legacy(observed_path: Path, predicted_path: Path) -> Optional[float]:
    """Legacy function that returns only overall RMSD."""
    result = compare_structures(observed_path, predicted_path)
    return result.overall_rmsd if result else None