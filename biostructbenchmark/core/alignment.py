"""Handles structure superposition and per-residue/nucleotide RMSD calculations"""

# TODO: Use sequence-based alignment for DNA to handle numbering inconsistencies

from pathlib import Path
from typing import Optional
from biostructbenchmark.core.io import get_structure
from Bio.PDB.Superimposer import Superimposer


def compare_structures(observed_path: Path, predicted_path: Path) -> Optional[float]:
    """Compare two structures and return RMSD, or None if comparison fails."""
    observed = get_structure(observed_path)
    predicted = get_structure(predicted_path)

    if observed is None or predicted is None:
        return None

    try:
        # Get CA atoms from first model of each structure
        obs_atoms = [
            atom for atom in observed[0].get_atoms() if atom.get_name() == "CA"
        ]
        pred_atoms = [
            atom for atom in predicted[0].get_atoms() if atom.get_name() == "CA"
        ]

        # Ensure same number of atoms for comparison
        min_len = min(len(obs_atoms), len(pred_atoms))
        if min_len == 0:
            print("Error: No CA atoms found for comparison")
            return None

        obs_atoms = obs_atoms[:min_len]
        pred_atoms = pred_atoms[:min_len]

        # Superimpose and calculate RMSD
        superimposer = Superimposer()
        superimposer.set_atoms(obs_atoms, pred_atoms)

        return float(superimposer.rms)

    except Exception as e:
        print(f"Error comparing structures: {e}")
        return None
