"""
Protein-DNA interface detection and analysis using GEMMI's spatial indexing.
"""

import logging
import time

import gemmi

from .sequences import AMINO_ACIDS, DNA_NUCLEOTIDE_MAP

logger = logging.getLogger(__name__)

# Distance threshold for protein-DNA interface detection (Angstroms)
INTERFACE_DISTANCE_THRESHOLD = 5.0


def _get_residue_id(residue: gemmi.Residue) -> tuple[str, int | None, str]:
    """Get residue ID tuple compatible with old BioPython format."""
    icode = residue.seqid.icode if residue.seqid.icode else " "
    return (" ", residue.seqid.num, icode)


def find_interface_residues(
    structure: gemmi.Structure,
    protein_chains: list[str],
    dna_chains: list[str],
    threshold: float = INTERFACE_DISTANCE_THRESHOLD,
) -> dict[str, list[str]]:
    """
    Find residues at the protein-DNA interface using spatial indexing.

    Uses GEMMI's NeighborSearch for O(n log n) performance instead of O(n²)
    pairwise distance calculations.

    Args:
        structure: GEMMI structure
        protein_chains: List of protein chain IDs
        dna_chains: List of DNA chain IDs
        threshold: Distance threshold in Angstroms

    Returns:
        Dict mapping chain_id to list of interface residue IDs
    """
    start_time = time.perf_counter()
    logger.debug(
        "Finding interface residues: protein chains %s, DNA chains %s, threshold %.1f Å",
        protein_chains,
        dna_chains,
        threshold,
    )

    interface_residues: dict[str, list[str]] = {}
    protein_chain_set = set(protein_chains)
    dna_chain_set = set(dna_chains)

    for model in structure:
        # Initialize result dict
        for chain_id in protein_chains + dna_chains:
            interface_residues[chain_id] = []

        # Build spatial index without periodic boundary conditions
        # Using an empty UnitCell disables PBC (we want real distances, not crystal contacts)
        index_start = time.perf_counter()
        ns = gemmi.NeighborSearch(model, gemmi.UnitCell(), threshold).populate()
        index_time = (time.perf_counter() - index_start) * 1000
        logger.debug("Built NeighborSearch spatial index in %.2f ms", index_time)

        # Track which residues we've already added (use set for O(1) lookup)
        seen_residues: set[str] = set()

        # For each DNA atom, find nearby protein atoms
        for chain in model:
            if chain.name not in dna_chain_set:
                continue

            for residue in chain:
                if residue.name not in DNA_NUCLEOTIDE_MAP:
                    continue

                dna_res_id = f"{chain.name}:{_get_residue_id(residue)}"

                for atom in residue:
                    # Find all atoms within threshold distance
                    for mark in ns.find_atoms(atom.pos, radius=threshold):
                        cra = mark.to_cra(model)
                        neighbor_chain = cra.chain.name

                        # Only care about protein neighbors
                        if neighbor_chain not in protein_chain_set:
                            continue

                        neighbor_residue = cra.residue
                        if neighbor_residue.name not in AMINO_ACIDS:
                            continue

                        # Add protein residue to interface
                        prot_res_id = f"{neighbor_chain}:{_get_residue_id(neighbor_residue)}"
                        if prot_res_id not in seen_residues:
                            seen_residues.add(prot_res_id)
                            interface_residues[neighbor_chain].append(prot_res_id)

                        # Add DNA residue to interface
                        if dna_res_id not in seen_residues:
                            seen_residues.add(dna_res_id)
                            interface_residues[chain.name].append(dna_res_id)

    total_interface = sum(len(v) for v in interface_residues.values())
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Interface detection complete: %d residues found in %.2f ms", total_interface, elapsed
    )
    return interface_residues
