"""
Protein-nucleic acid interface detection and analysis

Supports CAPRI-compliant interface detection and contact analysis for
protein-DNA/RNA complexes, including fnat calculation.
"""

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import Structure
from .sequences import DNA_NUCLEOTIDE_MAP
from .atoms import is_protein_residue, is_nucleic_acid_residue

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


# CAPRI contact threshold for fnat calculation (Angstroms)
CONTACT_DISTANCE_THRESHOLD = 5.0


def get_interface_contacts(
    structure: Structure,
    protein_chains: list[str],
    dna_chains: list[str],
    contact_threshold: float = CONTACT_DISTANCE_THRESHOLD,
) -> set[tuple[str, str]]:
    """
    Get interface contacts for fnat calculation.

    A contact is defined as a pair of residues (one protein, one DNA/RNA)
    with any heavy atoms within the contact threshold distance.

    Args:
        structure: BioPython structure
        protein_chains: List of protein chain IDs
        dna_chains: List of DNA/RNA chain IDs
        contact_threshold: Distance threshold for contacts in Angstroms (default: 5.0 Å)

    Returns:
        Set of contact pairs as frozenset tuples: {frozenset({prot_res_id, dna_res_id}), ...}
        Using frozensets makes contacts order-independent for easy comparison

    Note:
        CAPRI standard uses 5 Å for contact definition (different from 10 Å interface threshold).
        Contacts are symmetric: if A contacts B, then B contacts A (same contact).
    """
    contacts = set()

    for model in structure:
        # Get all protein and DNA/RNA atoms with residue info
        protein_atoms = []
        dna_atoms = []

        for chain in model:
            chain_id = chain.get_id()
            if chain_id in protein_chains:
                for residue in chain:
                    if is_aa(residue, standard=True):
                        for atom in residue.get_atoms():
                            # Skip hydrogens
                            if atom.element != 'H' and not atom.get_name().startswith('H'):
                                res_id = f"{chain_id}:{residue.get_id()}"
                                protein_atoms.append((atom, res_id))

            elif chain_id in dna_chains:
                for residue in chain:
                    if residue.get_resname() in DNA_NUCLEOTIDE_MAP:
                        for atom in residue.get_atoms():
                            # Skip hydrogens
                            if atom.element != 'H' and not atom.get_name().startswith('H'):
                                res_id = f"{chain_id}:{residue.get_id()}"
                                dna_atoms.append((atom, res_id))

        # Find contacts between protein and DNA/RNA
        for prot_atom, prot_res_id in protein_atoms:
            for dna_atom, dna_res_id in dna_atoms:
                distance = prot_atom - dna_atom
                if distance <= contact_threshold:
                    # Use frozenset for order-independent contact representation
                    contact = frozenset({prot_res_id, dna_res_id})
                    contacts.add(contact)
                    break  # Found contact for this protein residue with this DNA residue

    return contacts


def calculate_fnat(
    experimental_structure: Structure,
    computational_structure: Structure,
    protein_chains: list[str],
    dna_chains: list[str],
    sequence_mapping: dict[str, str],
    contact_threshold: float = CONTACT_DISTANCE_THRESHOLD,
) -> float:
    """
    Calculate fnat (fraction of native contacts).

    fnat measures what fraction of interface contacts in the experimental structure
    are also present in the computational structure. This is a key CAPRI metric.

    Args:
        experimental_structure: Experimental (reference) structure
        computational_structure: Computational (predicted) structure
        protein_chains: List of protein chain IDs
        dna_chains: List of DNA/RNA chain IDs
        sequence_mapping: Mapping from experimental to computational residue IDs
        contact_threshold: Distance threshold for contacts (default: 5.0 Å)

    Returns:
        fnat: Fraction of native contacts preserved (0.0 to 1.0)

    Note:
        CAPRI quality thresholds for fnat:
        - Incorrect: fnat < 0.10
        - Acceptable: 0.10 ≤ fnat < 0.30
        - Medium: 0.30 ≤ fnat < 0.50
        - High: 0.50 ≤ fnat < 0.70
        - Very High: fnat ≥ 0.70

    Formula:
        fnat = (number of common contacts) / (number of native contacts)

    Example:
        If experimental has 100 interface contacts and computational has 60 of them,
        fnat = 60/100 = 0.60 (High quality according to CAPRI)
    """
    # Get contacts from experimental (native) structure
    native_contacts = get_interface_contacts(
        experimental_structure,
        protein_chains,
        dna_chains,
        contact_threshold
    )

    if len(native_contacts) == 0:
        # No native contacts means no interface - return 0.0
        return 0.0

    # Get contacts from computational structure
    computational_contacts = get_interface_contacts(
        computational_structure,
        protein_chains,
        dna_chains,
        contact_threshold
    )

    # Map computational contacts to experimental residue IDs for comparison
    # Need to map because residue numbering might differ
    reverse_mapping = {v: k for k, v in sequence_mapping.items()}

    mapped_comp_contacts = set()
    for contact in computational_contacts:
        # Contact is a frozenset of two residue IDs
        res_ids = list(contact)

        # Try to map both residues to experimental IDs
        mapped_res_ids = []
        for comp_res_id in res_ids:
            if comp_res_id in reverse_mapping:
                mapped_res_ids.append(reverse_mapping[comp_res_id])
            else:
                # Residue doesn't exist in experimental structure
                # Skip this contact
                break

        if len(mapped_res_ids) == 2:
            # Successfully mapped both residues
            mapped_contact = frozenset(mapped_res_ids)
            mapped_comp_contacts.add(mapped_contact)

    # Calculate fraction of native contacts that are preserved
    common_contacts = native_contacts & mapped_comp_contacts
    fnat = len(common_contacts) / len(native_contacts)

    return fnat