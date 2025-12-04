"""
Structural alignment and RMSD calculations
"""

from typing import Any

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer


def superimpose_structures(
    exp_coords: np.ndarray, comp_coords: np.ndarray
) -> tuple[Any, np.ndarray, np.ndarray]:
    """
    Perform structural superimposition using SVD.

    Args:
        exp_coords: Experimental structure coordinates
        comp_coords: Computational structure coordinates

    Returns:
        tuple: (rmsd, rotation_matrix, translation_vector)
    """
    superimposer = SVDSuperimposer()
    superimposer.set(exp_coords, comp_coords)
    superimposer.run()

    rmsd = superimposer.get_rms()
    rotation_matrix = superimposer.get_rotran()[0]
    translation_vector = superimposer.get_rotran()[1]

    return rmsd, rotation_matrix, translation_vector


def calculate_per_residue_rmsd(
    exp_atoms: dict[str, list[Any]],
    comp_atoms: dict[str, list[Any]],
    mapping: dict[str, str],
    rotation_matrix: np.ndarray | None = None,
    translation_vector: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate per-residue RMSD for aligned residues.

    Args:
        exp_atoms: Dict mapping residue_id to list of atom coordinates
        comp_atoms: Dict mapping residue_id to list of atom coordinates
        mapping: Sequence alignment mapping
        rotation_matrix: Rotation matrix from superimposition
        translation_vector: Translation vector from superimposition

    Returns:
        Dict mapping residue_id to RMSD
    """
    per_residue_rmsd = {}

    for exp_res_id, comp_res_id in mapping.items():
        if exp_res_id in exp_atoms and comp_res_id in comp_atoms:
            exp_coords = np.array(exp_atoms[exp_res_id])
            comp_coords = np.array(comp_atoms[comp_res_id])

            # Both should have same number of atoms for proper RMSD
            if exp_coords.shape == comp_coords.shape:
                # Apply transformation to computational coordinates if provided
                if rotation_matrix is not None and translation_vector is not None:
                    comp_coords_transformed = (
                        np.dot(comp_coords, rotation_matrix) + translation_vector
                    )
                else:
                    comp_coords_transformed = comp_coords

                # Calculate RMSD: sqrt(mean(squared_distances))
                squared_diffs = np.sum((exp_coords - comp_coords_transformed) ** 2, axis=1)
                rmsd = np.sqrt(np.mean(squared_diffs))
                per_residue_rmsd[exp_res_id] = rmsd

    return per_residue_rmsd


def calculate_orientation_error(rotation_matrix: np.ndarray) -> float:
    """
    Calculate orientation error in degrees from rotation matrix.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Rotation angle in degrees
    """
    # Extract rotation angle from rotation matrix
    # trace(R) = 1 + 2*cos(θ)
    trace = np.trace(rotation_matrix)
    cos_theta = (trace - 1) / 2
    # Clamp to valid range to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta)
    return float(np.degrees(angle_rad))
