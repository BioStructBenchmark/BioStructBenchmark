"""
Protein-nucleic acid interface detection and analysis

Supports CAPRI-compliant interface detection for protein-DNA/RNA complexes.
"""

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import Structure
from .sequences import DNA_NUCLEOTIDE_MAP

# CAPRI standard: Interface residues with any heavy atoms within 10 Å of binding partner
CAPRI_INTERFACE_THRESHOLD = 10.0

# Tight interface definition: Higher specificity for core binding residues
TIGHT_INTERFACE_THRESHOLD = 5.0

# Default threshold (CAPRI standard for compatibility with published results)
INTERFACE_DISTANCE_THRESHOLD = CAPRI_INTERFACE_THRESHOLD


def find_interface_residues(
    structure: Structure,
    protein_chains: list[str],
    dna_chains: list[str],
    threshold: float = INTERFACE_DISTANCE_THRESHOLD,
) -> dict[str, list[str]]:
    """
    Find residues at the protein-nucleic acid interface.

    Uses CAPRI standard definition: residues with any heavy atoms within the
    specified threshold distance of the binding partner.

    Args:
        structure: BioPython structure
        protein_chains: List of protein chain IDs
        dna_chains: List of DNA/RNA chain IDs
        threshold: Distance threshold in Angstroms (default: 10.0 Å CAPRI standard)
                  Use TIGHT_INTERFACE_THRESHOLD (5.0 Å) for core binding residues

    Returns:
        Dict mapping chain_id to list of interface residue IDs

    Note:
        CAPRI standard uses 10 Å for interface definition. The previous default
        of 5 Å identifies only core binding residues and undercounts interfaces.
    """
    interface_residues: dict[str, list[str]] = {}
    
    for model in structure:
        # Get all protein and DNA atoms
        protein_atoms = []
        dna_atoms = []
        
        for chain in model:
            chain_id = chain.get_id()
            if chain_id in protein_chains:
                for residue in chain:
                    if is_aa(residue, standard=True):
                        protein_atoms.extend(
                            [
                                (atom, chain_id, residue.get_id())
                                for atom in residue.get_atoms()
                            ]
                        )
            elif chain_id in dna_chains:
                for residue in chain:
                    if residue.get_resname() in DNA_NUCLEOTIDE_MAP:
                        dna_atoms.extend(
                            [
                                (atom, chain_id, residue.get_id())
                                for atom in residue.get_atoms()
                            ]
                        )
        
        # Find interface residues
        for chain_id in protein_chains + dna_chains:
            interface_residues[chain_id] = []
        
        # Check distances between protein and DNA atoms
        for prot_atom, prot_chain, prot_res in protein_atoms:
            for dna_atom, dna_chain, dna_res in dna_atoms:
                distance = prot_atom - dna_atom  # BioPython calculates distance
                if distance <= threshold:
                    prot_res_id = f"{prot_chain}:{prot_res}"
                    dna_res_id = f"{dna_chain}:{dna_res}"
                    if prot_res_id not in interface_residues[prot_chain]:
                        interface_residues[prot_chain].append(prot_res_id)
                    if dna_res_id not in interface_residues[dna_chain]:
                        interface_residues[dna_chain].append(dna_res_id)
    
    return interface_residues