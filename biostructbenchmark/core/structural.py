"""
Structural alignment and RMSD calculations

Supports both all-atom and CAPRI-compliant backbone RMSD calculations:
- All-atom RMSD: Uses all common heavy atoms (detailed, non-standard)
- Backbone RMSD: Uses Cα (protein) and P (DNA/RNA) atoms (CAPRI standard)
- i-RMSD: Interface backbone RMSD after superimposing interface
- l-RMSD: Ligand backbone RMSD after superimposing receptor
"""

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import Structure


def superimpose_structures(exp_coords: np.ndarray, comp_coords: np.ndarray) -> tuple:
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
    exp_atoms: dict[str, list],
    comp_atoms: dict[str, list],
    mapping: dict[str, str],
    rotation_matrix: np.ndarray | None = None,
    translation_vector: np.ndarray | None = None
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
                    comp_coords_transformed = np.dot(comp_coords, rotation_matrix) + translation_vector
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
    return np.degrees(angle_rad)


def calculate_rmsd_from_coords(
    coords1: np.ndarray,
    coords2: np.ndarray,
    rotation_matrix: np.ndarray | None = None,
    translation_vector: np.ndarray | None = None
) -> float:
    """
    Calculate RMSD between two sets of coordinates.

    Args:
        coords1: Reference coordinates (N x 3 array)
        coords2: Coordinates to compare (N x 3 array)
        rotation_matrix: Optional rotation to apply to coords2
        translation_vector: Optional translation to apply to coords2

    Returns:
        RMSD value in Angstroms

    Raises:
        ValueError: If coordinate arrays have different shapes or are empty
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate arrays must have same shape. "
            f"Got {coords1.shape} and {coords2.shape}"
        )

    if len(coords1) == 0:
        raise ValueError("Cannot calculate RMSD for empty coordinate arrays")

    # Apply transformation if provided
    if rotation_matrix is not None and translation_vector is not None:
        coords2_transformed = np.dot(coords2, rotation_matrix) + translation_vector
    else:
        coords2_transformed = coords2

    # Calculate RMSD: sqrt(mean(squared_distances))
    squared_dists = np.sum((coords1 - coords2_transformed) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_dists))

    return float(rmsd)


def calculate_interface_rmsd(
    exp_interface_coords: np.ndarray,
    comp_interface_coords: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate interface RMSD (i-RMSD) using CAPRI methodology.

    Superimposes structures based on interface backbone atoms, then
    calculates RMSD of interface backbone atoms. This is the CAPRI
    standard i-RMSD metric.

    Args:
        exp_interface_coords: Experimental interface backbone coordinates (N x 3)
        comp_interface_coords: Computational interface backbone coordinates (N x 3)

    Returns:
        tuple: (i_rmsd, rotation_matrix, translation_vector)
            - i_rmsd: Interface RMSD in Angstroms
            - rotation_matrix: Rotation matrix from superimposition
            - translation_vector: Translation vector from superimposition

    Note:
        CAPRI i-RMSD:
        - Uses backbone atoms (Cα for protein, P for DNA/RNA)
        - Interface defined as residues within 10 Å of binding partner
        - Superimposition performed on interface residues only
        - Quality thresholds: <1 Å (very high), 1-2 Å (high), 2-4 Å (medium)
    """
    if exp_interface_coords.shape != comp_interface_coords.shape:
        raise ValueError(
            f"Interface coordinate arrays must have same shape. "
            f"Got {exp_interface_coords.shape} and {comp_interface_coords.shape}"
        )

    if len(exp_interface_coords) == 0:
        raise ValueError("Cannot calculate i-RMSD: no interface residues found")

    # Superimpose on interface backbone atoms
    i_rmsd, rotation_matrix, translation_vector = superimpose_structures(
        exp_interface_coords,
        comp_interface_coords
    )

    return i_rmsd, rotation_matrix, translation_vector


def calculate_ligand_rmsd(
    exp_receptor_coords: np.ndarray,
    comp_receptor_coords: np.ndarray,
    exp_ligand_coords: np.ndarray,
    comp_ligand_coords: np.ndarray
) -> float:
    """
    Calculate ligand RMSD (l-RMSD) using CAPRI methodology.

    Superimposes structures based on receptor (protein) backbone atoms,
    then calculates RMSD of ligand (DNA/RNA) backbone atoms WITHOUT
    superimposing the ligand. This measures how well the ligand
    position/orientation is predicted.

    Args:
        exp_receptor_coords: Experimental receptor (protein) backbone coordinates
        comp_receptor_coords: Computational receptor (protein) backbone coordinates
        exp_ligand_coords: Experimental ligand (DNA/RNA) backbone coordinates
        comp_ligand_coords: Computational ligand (DNA/RNA) backbone coordinates

    Returns:
        l_rmsd: Ligand RMSD in Angstroms

    Note:
        CAPRI l-RMSD:
        - Superimpose on receptor (protein) backbone only
        - Calculate RMSD of ligand (DNA/RNA) backbone
        - Uses backbone atoms (Cα for protein, P for DNA/RNA)
        - Quality thresholds: <1 Å (very high), 1-2 Å (high), 2-5 Å (medium)
    """
    if exp_receptor_coords.shape != comp_receptor_coords.shape:
        raise ValueError(
            f"Receptor coordinate arrays must have same shape. "
            f"Got {exp_receptor_coords.shape} and {comp_receptor_coords.shape}"
        )

    if exp_ligand_coords.shape != comp_ligand_coords.shape:
        raise ValueError(
            f"Ligand coordinate arrays must have same shape. "
            f"Got {exp_ligand_coords.shape} and {comp_ligand_coords.shape}"
        )

    if len(exp_receptor_coords) == 0:
        raise ValueError("Cannot calculate l-RMSD: no receptor residues found")

    if len(exp_ligand_coords) == 0:
        raise ValueError("Cannot calculate l-RMSD: no ligand residues found")

    # Superimpose on receptor (protein) backbone
    _, rotation_matrix, translation_vector = superimpose_structures(
        exp_receptor_coords,
        comp_receptor_coords
    )

    # Calculate ligand RMSD using the receptor-based transformation
    l_rmsd = calculate_rmsd_from_coords(
        exp_ligand_coords,
        comp_ligand_coords,
        rotation_matrix,
        translation_vector
    )

    return l_rmsd


def calculate_dockq(
    i_rmsd: float,
    l_rmsd: float,
    fnat: float
) -> float:
    """
    Calculate DockQ score - combined CAPRI quality metric.

    DockQ integrates i-RMSD, l-RMSD, and fnat into a single normalized score
    between 0 and 1, providing an overall quality assessment of the prediction.

    Args:
        i_rmsd: Interface backbone RMSD in Angstroms
        l_rmsd: Ligand backbone RMSD in Angstroms
        fnat: Fraction of native contacts (0.0-1.0)

    Returns:
        DockQ score (0.0-1.0)

    Note:
        DockQ quality thresholds:
        - Incorrect: DockQ < 0.23
        - Acceptable: 0.23 ≤ DockQ < 0.49
        - Medium: 0.49 ≤ DockQ < 0.80
        - High/Very High: DockQ ≥ 0.80

    Formula (Basu & Wallner, 2016):
        DockQ = (fnat + 1/(1 + (i_rmsd/1.5)²) + 1/(1 + (l_rmsd/8.5)²)) / 3

    The formula normalizes and combines the three metrics:
    - fnat: Already normalized [0, 1]
    - i_rmsd term: Normalized with characteristic scale 1.5 Å
    - l_rmsd term: Normalized with characteristic scale 8.5 Å

    Reference:
        Basu & Wallner (2016) "DockQ: A Quality Measure for Protein-Protein
        Docking Models" PLOS ONE 11(8): e0161879

    Example:
        If i_rmsd=2.0 Å, l_rmsd=5.0 Å, fnat=0.60:
        - fnat_term = 0.60
        - i_rmsd_term = 1/(1 + (2.0/1.5)²) = 1/(1 + 1.78) = 0.36
        - l_rmsd_term = 1/(1 + (5.0/8.5)²) = 1/(1 + 0.35) = 0.74
        - DockQ = (0.60 + 0.36 + 0.74) / 3 = 0.57 (Medium quality)
    """
    # Handle infinite RMSD values
    if i_rmsd == float('inf') or l_rmsd == float('inf'):
        return 0.0

    # Normalized fnat term (already [0, 1])
    fnat_term = fnat

    # Normalized i-RMSD term with characteristic scale 1.5 Å
    i_rmsd_term = 1.0 / (1.0 + (i_rmsd / 1.5) ** 2)

    # Normalized l-RMSD term with characteristic scale 8.5 Å
    l_rmsd_term = 1.0 / (1.0 + (l_rmsd / 8.5) ** 2)

    # Average of three normalized terms
    dockq = (fnat_term + i_rmsd_term + l_rmsd_term) / 3.0

    return dockq
