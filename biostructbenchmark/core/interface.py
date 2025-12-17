"""
Protein-DNA interface detection and analysis
"""

import gemmi

from .sequences import DNA_NUCLEOTIDE_MAP, is_standard_aa

# Distance threshold for protein-DNA interface detection (Angstroms)
INTERFACE_DISTANCE_THRESHOLD = 5.0


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
                    if is_standard_aa(residue):
                        # Format residue ID to match BioPython format for compatibility
                        res_id_str = (
                            f"(' ', {residue.seqid.num}, "
                            f"'{residue.seqid.icode if residue.seqid.icode else ' '}')"
                        )
                        protein_atoms.extend([(atom, chain_id, res_id_str) for atom in residue])
            elif chain_id in dna_chains:
                for residue in chain:
                    if residue.name in DNA_NUCLEOTIDE_MAP:
                        # Format residue ID to match BioPython format for compatibility
                        res_id_str = (
                            f"(' ', {residue.seqid.num}, "
                            f"'{residue.seqid.icode if residue.seqid.icode else ' '}')"
                        )
                        dna_atoms.extend([(atom, chain_id, res_id_str) for atom in residue])

        # Find interface residues
        for chain_id in protein_chains + dna_chains:
            interface_residues[chain_id] = []

        # Check distances between protein and DNA atoms
        for prot_atom, prot_chain, prot_res in protein_atoms:
            for dna_atom, dna_chain, dna_res in dna_atoms:
                # Calculate distance using GEMMI Position objects
                distance = (prot_atom.pos - dna_atom.pos).length()
                if distance <= threshold:
                    prot_res_id = f"{prot_chain}:{prot_res}"
                    dna_res_id = f"{dna_chain}:{dna_res}"
                    if prot_res_id not in interface_residues[prot_chain]:
                        interface_residues[prot_chain].append(prot_res_id)
                    if dna_res_id not in interface_residues[dna_chain]:
                        interface_residues[dna_chain].append(dna_res_id)

    return interface_residues
