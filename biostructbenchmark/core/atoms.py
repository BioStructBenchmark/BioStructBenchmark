"""
Atom selection utilities for CAPRI-compliant structural analysis

Provides functions to extract backbone atoms according to CAPRI standards:
- Proteins: Cα atoms (or C, N, O, Cα for full backbone)
- DNA/RNA: P (phosphate) atoms or C3' atoms

These selections are critical for standardized RMSD calculations and
interface analysis that can be compared with published results.
"""

from Bio.PDB import Residue
from Bio.PDB.Polypeptide import is_aa
from .sequences import DNA_NUCLEOTIDE_MAP


# RNA nucleotides (add to support RNA in addition to DNA)
RNA_NUCLEOTIDE_MAP = {
    "A": "A",
    "U": "U",
    "G": "G",
    "C": "C",
    # 3-letter codes
    "ADE": "A",
    "URA": "U",
    "GUA": "G",
    "CYT": "C",
}

# Combined nucleic acid map
NUCLEIC_ACID_MAP = {**DNA_NUCLEOTIDE_MAP, **RNA_NUCLEOTIDE_MAP}


def is_protein_residue(residue: Residue) -> bool:
    """Check if residue is a standard amino acid."""
    return is_aa(residue, standard=True)


def is_nucleic_acid_residue(residue: Residue) -> bool:
    """Check if residue is a DNA or RNA nucleotide."""
    return residue.get_resname() in NUCLEIC_ACID_MAP


def get_backbone_atoms(residue: Residue, mode: str = "representative") -> list:
    """
    Extract backbone atoms from a residue according to CAPRI standards.

    Args:
        residue: BioPython Residue object
        mode: Backbone selection mode
              - "representative": Single representative atom (Cα or P) [DEFAULT]
              - "full": All backbone heavy atoms (N, Cα, C, O for protein; P, C3', C4', C5', O3', O5' for nucleic acids)
              - "ca_p": Synonym for "representative" (Cα for protein, P for nucleic acids)

    Returns:
        List of BioPython Atom objects representing the backbone

    Raises:
        ValueError: If mode is not recognized or residue type cannot be determined

    Note:
        CAPRI standard for i-RMSD uses representative backbone atoms:
        - Proteins: Cα (alpha carbon)
        - DNA/RNA: P (phosphate) or C3' if phosphate not available
    """
    residue_name = residue.get_resname()

    # Determine residue type
    if is_protein_residue(residue):
        if mode == "representative" or mode == "ca_p":
            # CAPRI standard: Cα atom
            ca_atom = residue.get_list()
            ca_atoms = [atom for atom in ca_atom if atom.get_name() == "CA"]
            return ca_atoms

        elif mode == "full":
            # Full protein backbone: N, CA, C, O
            backbone_names = {"N", "CA", "C", "O"}
            atoms = residue.get_list()
            backbone_atoms = [atom for atom in atoms if atom.get_name() in backbone_names]
            return backbone_atoms

        else:
            raise ValueError(
                f"Unknown backbone mode '{mode}'. "
                "Valid options: 'representative', 'ca_p', 'full'"
            )

    elif is_nucleic_acid_residue(residue):
        if mode == "representative" or mode == "ca_p":
            # CAPRI standard: P (phosphate) atom, fallback to C3'
            atoms = residue.get_list()

            # Try phosphate first (standard)
            p_atoms = [atom for atom in atoms if atom.get_name() == "P"]
            if p_atoms:
                return p_atoms

            # Fallback to C3' (sugar) if no phosphate (terminal residues)
            c3_atoms = [atom for atom in atoms if atom.get_name() == "C3'"]
            return c3_atoms

        elif mode == "full":
            # Full nucleic acid backbone: P, O5', C5', C4', C3', O3'
            backbone_names = {"P", "O5'", "C5'", "C4'", "C3'", "O3'"}
            atoms = residue.get_list()
            backbone_atoms = [atom for atom in atoms if atom.get_name() in backbone_names]
            return backbone_atoms

        else:
            raise ValueError(
                f"Unknown backbone mode '{mode}'. "
                "Valid options: 'representative', 'ca_p', 'full'"
            )

    else:
        # Unknown residue type
        raise ValueError(
            f"Cannot determine backbone atoms for residue {residue_name}. "
            "Residue is neither a standard amino acid nor a recognized nucleotide."
        )


def get_all_atoms(residue: Residue) -> list:
    """
    Get all atoms from a residue.

    Args:
        residue: BioPython Residue object

    Returns:
        List of all Atom objects in the residue
    """
    return list(residue.get_atoms())


def get_heavy_atoms(residue: Residue) -> list:
    """
    Get all heavy (non-hydrogen) atoms from a residue.

    Args:
        residue: BioPython Residue object

    Returns:
        List of heavy Atom objects (excludes hydrogens)
    """
    atoms = residue.get_list()
    # Filter out hydrogens (element 'H' or atom names starting with 'H')
    heavy_atoms = [
        atom for atom in atoms
        if atom.element != 'H' and not atom.get_name().startswith('H')
    ]
    return heavy_atoms
