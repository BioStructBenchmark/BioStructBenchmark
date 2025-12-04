"""
Sequence alignment and chain matching functionality
"""

from dataclasses import dataclass

import gemmi
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1

# Define DNA nucleotide mapping
DNA_NUCLEOTIDE_MAP = {
    "DA": "A",
    "A": "A",
    "DT": "T",
    "T": "T",
    "DG": "G",
    "G": "G",
    "DC": "C",
    "C": "C",
}

# Standard amino acids (3-letter codes)
AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def is_amino_acid(residue: gemmi.Residue) -> bool:
    """Check if a residue is a standard amino acid."""
    return residue.name in AMINO_ACIDS


@dataclass
class ChainMatch:
    """Represents a matched chain between structures"""

    exp_chain_id: str
    comp_chain_id: str
    chain_type: str  # 'protein' or 'dna'
    sequence_identity: float
    rmsd: float


def classify_chains(structure: gemmi.Structure) -> tuple[list[str], list[str]]:
    """
    Classify chains as protein or DNA based on residue content.

    Returns:
        tuple: (protein_chain_ids, dna_chain_ids)
    """
    protein_chains = []
    dna_chains = []

    for model in structure:
        for chain in model:
            chain_id = chain.name
            protein_residues = 0
            dna_residues = 0

            for residue in chain:
                if is_amino_acid(residue):
                    protein_residues += 1
                elif residue.name in DNA_NUCLEOTIDE_MAP:
                    dna_residues += 1

            # Classify based on predominant residue type
            if protein_residues > dna_residues:
                protein_chains.append(chain_id)
            elif dna_residues > 0:
                dna_chains.append(chain_id)

    return protein_chains, dna_chains


def get_protein_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    """Extract protein sequence from a specific chain."""
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                residues = [r for r in chain if is_amino_acid(r)]
                return "".join(seq1(r.name) for r in residues)
    return ""


def get_dna_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    """Extract DNA sequence from a specific chain."""
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                sequence = ""
                residues = sorted(chain, key=lambda r: r.seqid.num)
                for residue in residues:
                    if residue.name in DNA_NUCLEOTIDE_MAP:
                        sequence += DNA_NUCLEOTIDE_MAP[residue.name]
                return sequence
    return ""


def calculate_sequence_identity(sequence1: str, sequence2: str) -> float:
    """Calculate sequence identity between two sequences using pairwise alignment."""
    if not sequence1 or not sequence2:
        return 0.0

    aligner = PairwiseAligner()
    # Default scoring equivalent to globalxx
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    alignments = aligner.align(sequence1, sequence2)
    if not alignments:
        return 0.0

    best_alignment = alignments[0]
    alignment_str = str(best_alignment)
    lines = alignment_str.strip().split("\n")
    # Extract sequences from formatted alignment (3rd field after splitting by spaces)
    aligned_seq1 = lines[0].split()[2] if len(lines) >= 1 and len(lines[0].split()) >= 3 else ""
    aligned_seq2 = lines[2].split()[2] if len(lines) >= 3 and len(lines[2].split()) >= 3 else ""

    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != "-")
    total_aligned = len(aligned_seq1)

    return matches / total_aligned if total_aligned > 0 else 0.0


def match_chains_by_similarity(
    exp_structure: gemmi.Structure, comp_structure: gemmi.Structure
) -> list[ChainMatch]:
    """
    Match chains between experimental and computational structures based on sequence similarity.

    Args:
        exp_structure: Experimental structure
        comp_structure: Computational structure

    Returns:
        List of ChainMatch objects representing best chain pairings
    """
    # Extract chains and classify them
    exp_prot_chains, exp_dna_chains = classify_chains(exp_structure)
    comp_prot_chains, comp_dna_chains = classify_chains(comp_structure)

    chain_matches = []
    used_comp_chains = set()  # Track which computational chains have been matched

    # Match protein chains
    for exp_chain_id in exp_prot_chains:
        exp_seq = get_protein_sequence(exp_structure, exp_chain_id)
        best_match = None
        best_identity = 0.0

        for comp_chain_id in comp_prot_chains:
            if comp_chain_id in used_comp_chains:
                continue  # Skip already matched chains

            comp_seq = get_protein_sequence(comp_structure, comp_chain_id)
            if exp_seq and comp_seq:
                identity = calculate_sequence_identity(exp_seq, comp_seq)
                if identity > best_identity and identity > 0.3:  # Minimum 30% identity
                    best_match = comp_chain_id
                    best_identity = identity

        if best_match:
            used_comp_chains.add(best_match)  # Mark as used
            chain_matches.append(
                ChainMatch(
                    exp_chain_id=exp_chain_id,
                    comp_chain_id=best_match,
                    chain_type="protein",
                    sequence_identity=best_identity,
                    rmsd=0.0,  # Will be calculated later
                )
            )

    # Match DNA chains
    for exp_chain_id in exp_dna_chains:
        exp_seq = get_dna_sequence(exp_structure, exp_chain_id)
        best_match = None
        best_identity = 0.0

        for comp_chain_id in comp_dna_chains:
            if comp_chain_id in used_comp_chains:
                continue  # Skip already matched chains

            comp_seq = get_dna_sequence(comp_structure, comp_chain_id)
            if exp_seq and comp_seq:
                identity = calculate_sequence_identity(exp_seq, comp_seq)
                if identity > best_identity and identity > 0.5:  # Higher threshold for DNA
                    best_match = comp_chain_id
                    best_identity = identity

        if best_match:
            used_comp_chains.add(best_match)  # Mark as used
            chain_matches.append(
                ChainMatch(
                    exp_chain_id=exp_chain_id,
                    comp_chain_id=best_match,
                    chain_type="dna",
                    sequence_identity=best_identity,
                    rmsd=0.0,  # Will be calculated later
                )
            )

    return chain_matches


def _get_residue_id(residue: gemmi.Residue) -> tuple:
    """Get residue ID tuple compatible with old BioPython format."""
    # BioPython format: (' ', resnum, icode)
    icode = residue.seqid.icode if residue.seqid.icode else " "
    return (" ", residue.seqid.num, icode)


def align_specific_protein_chains(
    exp_structure: gemmi.Structure,
    comp_structure: gemmi.Structure,
    exp_chain_id: str,
    comp_chain_id: str,
) -> dict[str, str]:
    """
    Align protein sequences for specific chain pairs.

    Returns:
        Dict mapping experimental residue IDs to computational residue IDs
    """
    # Extract residues for specific chains
    exp_residues = []
    comp_residues = []

    # Get experimental chain residues
    for model in exp_structure:
        for chain in model:
            if chain.name == exp_chain_id:
                exp_residues = [r for r in chain if is_amino_acid(r)]
                break

    # Get computational chain residues
    for model in comp_structure:
        for chain in model:
            if chain.name == comp_chain_id:
                comp_residues = [r for r in chain if is_amino_acid(r)]
                break

    if not exp_residues or not comp_residues:
        return {}

    # Create sequences
    exp_seq = "".join(seq1(r.name) for r in exp_residues)
    comp_seq = "".join(seq1(r.name) for r in comp_residues)

    # Align sequences
    aligner = PairwiseAligner()
    # Default scoring equivalent to globalxx
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    alignments = aligner.align(exp_seq, comp_seq)
    if not alignments:
        return {}

    best = alignments[0]
    alignment_str = str(best)
    lines = alignment_str.strip().split("\n")
    # Extract sequences from formatted alignment (3rd field after splitting by spaces)
    exp_aligned = lines[0].split()[2] if len(lines) >= 1 and len(lines[0].split()) >= 3 else ""
    comp_aligned = lines[2].split()[2] if len(lines) >= 3 and len(lines[2].split()) >= 3 else ""

    # Create residue mapping
    mapping = {}
    exp_idx = comp_idx = 0

    for i in range(len(exp_aligned)):
        if exp_aligned[i] != "-" and comp_aligned[i] != "-":
            if exp_idx < len(exp_residues) and comp_idx < len(comp_residues):
                exp_full_id = f"{exp_chain_id}:{_get_residue_id(exp_residues[exp_idx])}"
                comp_full_id = f"{comp_chain_id}:{_get_residue_id(comp_residues[comp_idx])}"
                mapping[exp_full_id] = comp_full_id
            exp_idx += 1
            comp_idx += 1
        elif exp_aligned[i] != "-":
            exp_idx += 1
        elif comp_aligned[i] != "-":
            comp_idx += 1

    return mapping


def align_specific_dna_chains(
    exp_structure: gemmi.Structure,
    comp_structure: gemmi.Structure,
    exp_chain_id: str,
    comp_chain_id: str,
) -> dict[str, str]:
    """
    Align DNA sequences for specific chain pairs.

    Returns:
        Dict mapping experimental residue IDs to computational residue IDs
    """
    # Extract DNA residues for specific chains
    exp_residues = []
    comp_residues = []

    # Get experimental chain residues
    for model in exp_structure:
        for chain in model:
            if chain.name == exp_chain_id:
                residues = sorted(chain, key=lambda r: r.seqid.num)
                exp_residues = [
                    (_get_residue_id(r), f"{exp_chain_id}:{_get_residue_id(r)}")
                    for r in residues
                    if r.name in DNA_NUCLEOTIDE_MAP
                ]
                break

    # Get computational chain residues
    for model in comp_structure:
        for chain in model:
            if chain.name == comp_chain_id:
                residues = sorted(chain, key=lambda r: r.seqid.num)
                comp_residues = [
                    (_get_residue_id(r), f"{comp_chain_id}:{_get_residue_id(r)}")
                    for r in residues
                    if r.name in DNA_NUCLEOTIDE_MAP
                ]
                break

    if not exp_residues or not comp_residues:
        return {}

    # Create sequences
    exp_seq = get_dna_sequence(exp_structure, exp_chain_id)
    comp_seq = get_dna_sequence(comp_structure, comp_chain_id)

    if not exp_seq or not comp_seq:
        return {}

    # Align sequences using global alignment with match/mismatch scores
    aligner = PairwiseAligner()
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(exp_seq, comp_seq)
    if not alignments:
        return {}

    best_alignment = alignments[0]
    alignment_str = str(best_alignment)
    lines = alignment_str.strip().split("\n")
    # Extract sequences from formatted alignment (3rd field after splitting by spaces)
    aligned_exp = lines[0].split()[2] if len(lines) >= 1 and len(lines[0].split()) >= 3 else ""
    aligned_comp = lines[2].split()[2] if len(lines) >= 3 and len(lines[2].split()) >= 3 else ""

    # Create residue mapping
    mapping = {}
    exp_idx = comp_idx = 0

    for i in range(len(aligned_exp)):
        exp_char = aligned_exp[i]
        comp_char = aligned_comp[i]

        if exp_char != "-" and comp_char != "-":
            if exp_idx < len(exp_residues) and comp_idx < len(comp_residues):
                exp_full_id = exp_residues[exp_idx][1]
                comp_full_id = comp_residues[comp_idx][1]
                mapping[exp_full_id] = comp_full_id
            exp_idx += 1
            comp_idx += 1
        elif exp_char == "-":
            comp_idx += 1
        elif comp_char == "-":
            exp_idx += 1

    return mapping
