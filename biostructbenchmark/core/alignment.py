"""
Streamlined protein-DNA complex alignment module
"""

import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import parasail

logger = logging.getLogger(__name__)

from .interface import INTERFACE_DISTANCE_THRESHOLD, find_interface_residues
from .sequences import (
    AMINO_ACID_MAP,
    AMINO_ACIDS,
    DNA_NUCLEOTIDE_MAP,
    align_specific_dna_chains,
    align_specific_protein_chains,
    classify_chains,
    match_chains_by_similarity,
)
from .structural import (
    calculate_orientation_error,
    calculate_per_residue_rmsd,
    superimpose_structures,
)


def _get_residue_id(residue: gemmi.Residue) -> tuple:
    """Get residue ID tuple compatible with old BioPython format."""
    icode = residue.seqid.icode if residue.seqid.icode else " "
    return (" ", residue.seqid.num, icode)


def create_output_directory_structure(base_output_dir: Path | None = None) -> Path:
    """
    Create standardized output directory structure with timestamped subdirectory.

    Args:
        base_output_dir: Base directory for outputs (default: ./results)

    Returns:
        Path to the timestamped run directory
    """
    if base_output_dir is None:
        base_output_dir = Path.cwd() / "results"
    else:
        base_output_dir = Path(base_output_dir)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"biostructbenchmark_{timestamp}"

    # Create subdirectories
    (run_dir / "alignments").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    return run_dir


def save_aligned_structures(
    experimental_structure: gemmi.Structure,
    computational_structure: gemmi.Structure,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    run_dir: Path,
    prefix: str = "aligned",
) -> tuple[Path, Path]:
    """
    Save aligned structures to output files in the alignments subdirectory.

    Args:
        experimental_structure: Reference structure (unchanged)
        computational_structure: Structure to be transformed and saved
        rotation_matrix: Rotation matrix from superimposition
        translation_vector: Translation vector from superimposition
        run_dir: Timestamped run directory containing subdirectories
        prefix: Prefix for output filenames

    Returns:
        tuple: (experimental_output_path, computational_output_path)
    """
    # Use the alignments subdirectory
    alignments_dir = run_dir / "alignments"

    # Deep copy structures to avoid modifying originals
    exp_copy = copy.deepcopy(experimental_structure)
    comp_copy = copy.deepcopy(computational_structure)

    # Apply transformation to computational structure
    for model in comp_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    # Apply rotation and translation: new_coord = coord * R + t
                    transformed_coord = np.dot(coord, rotation_matrix) + translation_vector
                    atom.pos = gemmi.Position(
                        transformed_coord[0], transformed_coord[1], transformed_coord[2]
                    )

    # Determine output format and filenames
    exp_output_path = alignments_dir / f"{prefix}_experimental.cif"
    comp_output_path = alignments_dir / f"{prefix}_computational_aligned.cif"

    # Save structures using GEMMI's mmCIF writer
    exp_copy.make_mmcif_document().write_file(str(exp_output_path))
    comp_copy.make_mmcif_document().write_file(str(comp_output_path))

    return exp_output_path, comp_output_path


@dataclass
class AlignmentResult:
    """Result of protein-DNA complex alignment"""

    sequence_mapping: dict[str, str]  # exp_residue_id -> comp_residue_id
    structural_rmsd: float
    per_residue_rmsd: dict[str, float]  # residue_id -> RMSD
    protein_rmsd: float
    dna_rmsd: float
    interface_rmsd: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    orientation_error: float  # degrees
    translational_error: float  # Angstroms
    protein_chains: list[str]
    dna_chains: list[str]
    interface_residues: dict[str, list[str]]  # chain_id -> residue_ids
    output_files: tuple[Path, Path] | None = None  # (experimental_path, computational_aligned_path)


def align_dna_sequences(  # type: ignore[no-untyped-def]
    experimental_structure: gemmi.Structure, computational_structure: gemmi.Structure
):
    """
    Align DNA sequences between experimental and computational structures
    and create a mapping between corresponding nucleotides.

    Returns:
        mapping: Dictionary mapping experimental nucleotide full IDs to computational ones
        exp_sequence_dict: Dictionary of experimental DNA sequences by chain
        comp_sequence_dict: Dictionary of computational DNA sequences by chain
    """
    # Extract DNA sequences by chain
    exp_sequence_dict: dict[str, str] = {}  # Chain ID -> DNA sequence
    exp_residue_dict: dict[
        str, list[tuple[tuple, str]]
    ] = {}  # Chain ID -> list of (residue_id, full_id)
    comp_sequence_dict: dict[str, str] = {}  # Chain ID -> DNA sequence
    comp_residue_dict: dict[
        str, list[tuple[tuple, str]]
    ] = {}  # Chain ID -> list of (residue_id, full_id)

    # Process experimental structure
    for model in experimental_structure:
        for chain in model:
            chain_id = chain.name
            if chain_id not in exp_sequence_dict:
                exp_sequence_dict[chain_id] = ""
                exp_residue_dict[chain_id] = []

            # Sort nucleotides by ID for consistent processing
            residues = sorted(chain, key=lambda r: r.seqid.num)

            for residue in residues:
                residue_id = _get_residue_id(residue)
                residue_name = residue.name

                if residue_name in DNA_NUCLEOTIDE_MAP:
                    # Map to single letter nucleotide code
                    nuc = DNA_NUCLEOTIDE_MAP.get(residue_name)
                    if nuc:
                        exp_sequence_dict[chain_id] += nuc
                        exp_residue_dict[chain_id].append((residue_id, f"{chain_id}:{residue_id}"))

    # Process computational structure
    for model in computational_structure:
        for chain in model:
            chain_id = chain.name
            if chain_id not in comp_sequence_dict:
                comp_sequence_dict[chain_id] = ""
                comp_residue_dict[chain_id] = []

            # Sort nucleotides by ID for consistent processing
            residues = sorted(chain, key=lambda r: r.seqid.num)

            for residue in residues:
                residue_id = _get_residue_id(residue)
                residue_name = residue.name

                if residue_name in DNA_NUCLEOTIDE_MAP:
                    # Map to single letter nucleotide code
                    nuc = DNA_NUCLEOTIDE_MAP.get(residue_name)
                    if nuc:
                        comp_sequence_dict[chain_id] += nuc
                        comp_residue_dict[chain_id].append((residue_id, f"{chain_id}:{residue_id}"))

    # Create mapping between experimental and computational nucleotides
    mapping = {}

    # For each chain in experimental structure
    for chain_id in exp_sequence_dict:
        # Skip if chain doesn't exist in computational structure
        if chain_id not in comp_sequence_dict:
            continue

        exp_sequence = exp_sequence_dict[chain_id]
        comp_sequence = comp_sequence_dict[chain_id]

        if not exp_sequence or not comp_sequence:
            continue

        # Perform sequence alignment using Parasail (SIMD-accelerated)
        dna_matrix = parasail.matrix_create("ACGT", 2, -1)
        result = parasail.nw_trace_striped_32(exp_sequence, comp_sequence, 1, 1, dna_matrix)

        aligned_exp = result.traceback.query
        aligned_comp = result.traceback.ref

        # Process alignment to create nucleotide mapping
        exp_idx = 0
        comp_idx = 0

        for i in range(len(aligned_exp)):
            exp_char = aligned_exp[i]
            comp_char = aligned_comp[i]

            if exp_char != "-" and comp_char != "-":
                # Match or mismatch - create mapping between nucleotides
                if exp_idx < len(exp_residue_dict[chain_id]) and comp_idx < len(
                    comp_residue_dict[chain_id]
                ):
                    exp_full_id = exp_residue_dict[chain_id][exp_idx][1]
                    comp_full_id = comp_residue_dict[chain_id][comp_idx][1]
                    mapping[exp_full_id] = comp_full_id
                exp_idx += 1
                comp_idx += 1
            elif exp_char == "-":
                # Gap in experimental sequence
                comp_idx += 1
            elif comp_char == "-":
                # Gap in computational sequence
                exp_idx += 1

    return mapping, exp_sequence_dict, comp_sequence_dict


def align_protein_sequences(
    exp_structure: gemmi.Structure, comp_structure: gemmi.Structure
) -> dict[str, str]:
    """
    Align protein sequences between experimental and computational structures
    and create a mapping between corresponding residues.
    Returns: mapping (experimental_full_id -> computational_full_id)
    """
    exp_sequence_dict: dict[str, tuple[str, list[Any]]] = {}  # chain_id -> (sequence, residues)
    comp_sequence_dict: dict[str, tuple[str, list[Any]]] = {}

    # Extract sequences
    for model in exp_structure:
        for chain in model:
            chain_id = chain.name
            residues = [r for r in chain if r.name in AMINO_ACIDS]
            exp_seq = "".join(AMINO_ACID_MAP.get(r.name, "X") for r in residues)
            exp_sequence_dict[chain_id] = (exp_seq, residues)

    for model in comp_structure:
        for chain in model:
            chain_id = chain.name
            residues = [r for r in chain if r.name in AMINO_ACIDS]
            comp_seq = "".join(AMINO_ACID_MAP.get(r.name, "X") for r in residues)
            comp_sequence_dict[chain_id] = (comp_seq, residues)

    mapping = {}
    for chain_id in exp_sequence_dict:
        if chain_id not in comp_sequence_dict:
            continue
        exp_seq, exp_residues = exp_sequence_dict[chain_id]
        comp_seq, comp_residues = comp_sequence_dict[chain_id]

        if not exp_seq or not comp_seq:
            continue

        # Global alignment using Parasail (SIMD-accelerated)
        result = parasail.nw_trace_striped_32(exp_seq, comp_seq, 1, 1, parasail.blosum62)

        exp_aligned = result.traceback.query
        comp_aligned = result.traceback.ref

        # Map aligned positions
        exp_idx = comp_idx = 0
        for i in range(len(exp_aligned)):
            if exp_aligned[i] != "-" and comp_aligned[i] != "-":
                if exp_idx < len(exp_residues) and comp_idx < len(comp_residues):
                    exp_full_id = f"{chain_id}:{_get_residue_id(exp_residues[exp_idx])}"
                    comp_full_id = f"{chain_id}:{_get_residue_id(comp_residues[comp_idx])}"
                    mapping[exp_full_id] = comp_full_id
                exp_idx += 1
                comp_idx += 1
            elif exp_aligned[i] != "-":
                exp_idx += 1
            elif comp_aligned[i] != "-":
                comp_idx += 1

    return mapping


def align_protein_dna_complex(
    experimental_structure: gemmi.Structure,
    computational_structure: gemmi.Structure,
    interface_threshold: float = INTERFACE_DISTANCE_THRESHOLD,
    output_dir: Path | None = None,
    save_structures: bool = False,
) -> AlignmentResult:
    """
    Comprehensive alignment of protein-DNA binding complexes.

    Performs both sequence and structural alignment, calculating:
    - Overall structural RMSD
    - Per-residue RMSD for both protein and DNA components
    - Interface analysis and interface-specific RMSD
    - Orientation vs translational error decomposition

    Args:
        experimental_structure: Reference structure
        computational_structure: Structure to align
        interface_threshold: Distance threshold for interface detection (Angstroms)
        output_dir: Base directory for outputs (creates timestamped subdirectories)
        save_structures: Whether to save aligned structures to files

    Returns:
        AlignmentResult containing comprehensive alignment data

    TODO: Implement summary.json generation with run metadata
    """
    total_start = time.perf_counter()
    logger.debug("Starting protein-DNA complex alignment")

    # Classify chains and find best matches between structures
    classify_start = time.perf_counter()
    exp_prot_chains, exp_dna_chains = classify_chains(experimental_structure)
    comp_prot_chains, comp_dna_chains = classify_chains(computational_structure)
    logger.debug("Chain classification took %.2f ms", (time.perf_counter() - classify_start) * 1000)

    # Match chains based on sequence similarity
    chain_matches = match_chains_by_similarity(experimental_structure, computational_structure)

    # Create mapping based on matched chains
    sequence_mapping = {}

    # Process each chain match
    for match in chain_matches:
        if match.chain_type == "protein":
            chain_mapping = align_specific_protein_chains(
                experimental_structure,
                computational_structure,
                match.exp_chain_id,
                match.comp_chain_id,
            )
            sequence_mapping.update(chain_mapping)

        elif match.chain_type == "dna":
            chain_mapping = align_specific_dna_chains(
                experimental_structure,
                computational_structure,
                match.exp_chain_id,
                match.comp_chain_id,
            )
            sequence_mapping.update(chain_mapping)

    if not sequence_mapping:
        # Return empty result if no alignments found
        return AlignmentResult(
            sequence_mapping={},
            structural_rmsd=float("inf"),
            per_residue_rmsd={},
            protein_rmsd=float("inf"),
            dna_rmsd=float("inf"),
            interface_rmsd=float("inf"),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            orientation_error=0.0,
            translational_error=0.0,
            protein_chains=exp_prot_chains,
            dna_chains=exp_dna_chains,
            interface_residues={},
        )

    # Collect atoms for structural alignment
    exp_atoms_for_alignment = []  # For SVD superimposition
    comp_atoms_for_alignment = []
    exp_atoms_dict = {}  # For per-residue RMSD: residue_id -> atom_coords
    comp_atoms_dict = {}

    # Build residue dictionaries for both structures
    exp_residues = {}
    comp_residues = {}

    # Process experimental structure
    for model in experimental_structure:
        for chain in model:
            chain_id = chain.name
            for residue in chain:
                residue_id = f"{chain_id}:{_get_residue_id(residue)}"
                if residue_id in sequence_mapping:
                    exp_residues[residue_id] = residue

    # Process computational structure
    for model in computational_structure:
        for chain in model:
            chain_id = chain.name
            for residue in chain:
                residue_id = f"{chain_id}:{_get_residue_id(residue)}"
                comp_residues[residue_id] = residue

    # Align atoms only for residues that exist in both structures
    for exp_residue_id, comp_residue_id in sequence_mapping.items():
        if exp_residue_id in exp_residues and comp_residue_id in comp_residues:
            exp_residue = exp_residues[exp_residue_id]
            comp_residue = comp_residues[comp_residue_id]

            # Get atoms from both residues
            exp_atoms = {atom.name: atom for atom in exp_residue}
            comp_atoms = {atom.name: atom for atom in comp_residue}

            # Find common atoms
            common_atom_names = set(exp_atoms.keys()) & set(comp_atoms.keys())

            if common_atom_names:
                exp_residue_coords = []
                comp_residue_coords = []

                # Collect coordinates for common atoms in the same order
                for atom_name in sorted(common_atom_names):
                    exp_atom = exp_atoms[atom_name]
                    comp_atom = comp_atoms[atom_name]

                    exp_coord = [exp_atom.pos.x, exp_atom.pos.y, exp_atom.pos.z]
                    comp_coord = [comp_atom.pos.x, comp_atom.pos.y, comp_atom.pos.z]

                    exp_atoms_for_alignment.append(exp_coord)
                    comp_atoms_for_alignment.append(comp_coord)

                    exp_residue_coords.append(exp_coord)
                    comp_residue_coords.append(comp_coord)

                exp_atoms_dict[exp_residue_id] = exp_residue_coords
                comp_atoms_dict[comp_residue_id] = comp_residue_coords

    if not exp_atoms_for_alignment or not comp_atoms_for_alignment:
        return AlignmentResult(
            sequence_mapping=sequence_mapping,
            structural_rmsd=float("inf"),
            per_residue_rmsd={},
            protein_rmsd=float("inf"),
            dna_rmsd=float("inf"),
            interface_rmsd=float("inf"),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            orientation_error=0.0,
            translational_error=0.0,
            protein_chains=exp_prot_chains,
            dna_chains=exp_dna_chains,
            interface_residues={},
        )

    # Perform structural superimposition using SVD
    exp_coords = np.array(exp_atoms_for_alignment)
    comp_coords = np.array(comp_atoms_for_alignment)

    structural_rmsd, rotation_matrix, translation_vector = superimpose_structures(
        exp_coords, comp_coords
    )

    # Calculate per-residue RMSD
    per_residue_rmsd = calculate_per_residue_rmsd(
        exp_atoms_dict, comp_atoms_dict, sequence_mapping, rotation_matrix, translation_vector
    )

    # Calculate component-specific RMSDs
    protein_rmsds = [
        rmsd
        for res_id, rmsd in per_residue_rmsd.items()
        if any(res_id.startswith(f"{chain}:") for chain in exp_prot_chains)
    ]
    dna_rmsds = [
        rmsd
        for res_id, rmsd in per_residue_rmsd.items()
        if any(res_id.startswith(f"{chain}:") for chain in exp_dna_chains)
    ]

    protein_rmsd = np.mean(protein_rmsds) if protein_rmsds else float("inf")
    dna_rmsd = np.mean(dna_rmsds) if dna_rmsds else float("inf")

    # Find interface residues
    interface_residues = find_interface_residues(
        experimental_structure, exp_prot_chains, exp_dna_chains, interface_threshold
    )

    # Calculate interface RMSD
    interface_rmsds = []
    for chain_residues in interface_residues.values():
        for res_id in chain_residues:
            if res_id in per_residue_rmsd:
                interface_rmsds.append(per_residue_rmsd[res_id])

    interface_rmsd = np.mean(interface_rmsds) if interface_rmsds else float("inf")

    # Calculate orientation and translational errors
    orientation_error = calculate_orientation_error(rotation_matrix)
    translational_error = np.linalg.norm(translation_vector)

    # Save aligned structures if requested
    output_files = None
    if save_structures:
        # Create standardized output directory structure
        run_dir = create_output_directory_structure(output_dir)
        output_files = save_aligned_structures(
            experimental_structure,
            computational_structure,
            rotation_matrix,
            translation_vector,
            run_dir,
        )

    total_elapsed = (time.perf_counter() - total_start) * 1000
    logger.debug(
        "Alignment complete in %.2f ms: RMSD=%.3f Å, %d residues mapped, %d interface residues",
        total_elapsed, structural_rmsd, len(sequence_mapping),
        sum(len(v) for v in interface_residues.values())
    )

    return AlignmentResult(
        sequence_mapping=sequence_mapping,
        structural_rmsd=structural_rmsd,
        per_residue_rmsd=per_residue_rmsd,
        protein_rmsd=float(protein_rmsd),
        dna_rmsd=float(dna_rmsd),
        interface_rmsd=float(interface_rmsd),
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        orientation_error=orientation_error,
        translational_error=float(translational_error),
        protein_chains=exp_prot_chains,
        dna_chains=exp_dna_chains,
        interface_residues=interface_residues,
        output_files=output_files,
    )
