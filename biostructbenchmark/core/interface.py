"""
Protein-DNA interface detection and analysis
"""

import math

import gemmi

from .sequences import AMINO_ACIDS, DNA_NUCLEOTIDE_MAP

# Distance threshold for protein-DNA interface detection (Angstroms)
INTERFACE_DISTANCE_THRESHOLD = 5.0


def _get_residue_id(residue: gemmi.Residue) -> tuple:
    """Get residue ID tuple compatible with old BioPython format."""
    icode = residue.seqid.icode if residue.seqid.icode else " "
    return (" ", residue.seqid.num, icode)


def _calculate_distance(atom1: gemmi.Atom, atom2: gemmi.Atom) -> float:
    """Calculate Euclidean distance between two atoms."""
    return math.sqrt(
        (atom1.pos.x - atom2.pos.x) ** 2
        + (atom1.pos.y - atom2.pos.y) ** 2
        + (atom1.pos.z - atom2.pos.z) ** 2
    )


def find_interface_residues(
    structure: gemmi.Structure,
    protein_chains: list[str],
    dna_chains: list[str],
    threshold: float = INTERFACE_DISTANCE_THRESHOLD,
) -> dict[str, list[str]]:
    """
    Find residues at the protein-DNA interface.

    Args:
        structure: GEMMI structure
        protein_chains: List of protein chain IDs
        dna_chains: List of DNA chain IDs
        threshold: Distance threshold in Angstroms

    Returns:
        Dict mapping chain_id to list of interface residue IDs
    """
    interface_residues: dict[str, list[str]] = {}

    for model in structure:
        # Get all protein and DNA atoms
        protein_atoms = []
        dna_atoms = []

        for chain in model:
            chain_id = chain.name
            if chain_id in protein_chains:
                for residue in chain:
                    if residue.name in AMINO_ACIDS:
                        protein_atoms.extend(
                            [(atom, chain_id, _get_residue_id(residue)) for atom in residue]
                        )
            elif chain_id in dna_chains:
                for residue in chain:
                    if residue.name in DNA_NUCLEOTIDE_MAP:
                        dna_atoms.extend(
                            [(atom, chain_id, _get_residue_id(residue)) for atom in residue]
                        )

        # Find interface residues
        for chain_id in protein_chains + dna_chains:
            interface_residues[chain_id] = []

        # Check distances between protein and DNA atoms
        for prot_atom, prot_chain, prot_res in protein_atoms:
            for dna_atom, dna_chain, dna_res in dna_atoms:
                distance = _calculate_distance(prot_atom, dna_atom)
                if distance <= threshold:
                    prot_res_id = f"{prot_chain}:{prot_res}"
                    dna_res_id = f"{dna_chain}:{dna_res}"
                    if prot_res_id not in interface_residues[prot_chain]:
                        interface_residues[prot_chain].append(prot_res_id)
                    if dna_res_id not in interface_residues[dna_chain]:
                        interface_residues[dna_chain].append(dna_res_id)

    return interface_residues
