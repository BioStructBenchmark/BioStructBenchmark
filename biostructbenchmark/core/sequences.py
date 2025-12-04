"""
Sequence alignment and chain matching functionality using SIMD-accelerated Parasail.
"""

import logging
import time
from dataclasses import dataclass

import gemmi
import parasail

logger = logging.getLogger(__name__)

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

# Standard amino acids (3-letter to 1-letter mapping)
AMINO_ACID_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# Set for fast lookup
AMINO_ACIDS = set(AMINO_ACID_MAP.keys())


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
    start_time = time.perf_counter()
    protein_chains = []
    dna_chains = []

    for model in structure:
        for chain in model:
            chain_id = chain.name
            protein_residues = 0
            dna_residues = 0

            for residue in chain:
                # Inline check for performance (avoid function call overhead)
                res_name = residue.name
                if res_name in AMINO_ACIDS:
                    protein_residues += 1
                elif res_name in DNA_NUCLEOTIDE_MAP:
                    dna_residues += 1

            # Classify based on predominant residue type
            if protein_residues > dna_residues:
                protein_chains.append(chain_id)
                logger.debug(
                    "Chain %s classified as protein (%d protein, %d DNA residues)",
                    chain_id,
                    protein_residues,
                    dna_residues,
                )
            elif dna_residues > 0:
                dna_chains.append(chain_id)
                logger.debug(
                    "Chain %s classified as DNA (%d protein, %d DNA residues)",
                    chain_id,
                    protein_residues,
                    dna_residues,
                )

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Chain classification: %d protein chains %s, %d DNA chains %s in %.2f ms",
        len(protein_chains),
        protein_chains,
        len(dna_chains),
        dna_chains,
        elapsed,
    )
    return protein_chains, dna_chains


def get_protein_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    """Extract protein sequence from a specific chain."""
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                # Single pass: check membership and get mapping in one iteration
                chars = []
                for r in chain:
                    code = AMINO_ACID_MAP.get(r.name)
                    if code:
                        chars.append(code)
                return "".join(chars)
    return ""


def get_dna_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    """Extract DNA sequence from a specific chain."""
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                # Sort by sequence number and build list (faster than string concat)
                residues = sorted(chain, key=lambda r: r.seqid.num or 0)
                chars = []
                for residue in residues:
                    code = DNA_NUCLEOTIDE_MAP.get(residue.name)
                    if code:
                        chars.append(code)
                return "".join(chars)
    return ""


def calculate_sequence_identity(sequence1: str, sequence2: str) -> float:
    """Calculate sequence identity between two sequences using SIMD-accelerated alignment."""
    if not sequence1 or not sequence2:
        return 0.0

    # Use Parasail global alignment (Needleman-Wunsch)
    # nw_stats returns matches directly, avoiding traceback overhead
    result = parasail.nw_stats_striped_16(sequence1, sequence2, 0, 0, parasail.blosum62)

    # Use parasail's built-in match counting
    return result.matches / result.length if result.length > 0 else 0.0


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
    start_time = time.perf_counter()
    logger.debug("Matching chains between structures by sequence similarity")

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
                logger.debug(
                    "Protein chain similarity: exp %s vs comp %s = %.1f%%",
                    exp_chain_id,
                    comp_chain_id,
                    identity * 100,
                )
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
            logger.debug(
                "Matched protein chain %s -> %s (%.1f%% identity)",
                exp_chain_id,
                best_match,
                best_identity * 100,
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
                logger.debug(
                    "DNA chain similarity: exp %s vs comp %s = %.1f%%",
                    exp_chain_id,
                    comp_chain_id,
                    identity * 100,
                )
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
            logger.debug(
                "Matched DNA chain %s -> %s (%.1f%% identity)",
                exp_chain_id,
                best_match,
                best_identity * 100,
            )

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug("Chain matching complete: %d matches in %.2f ms", len(chain_matches), elapsed)
    return chain_matches


def _get_residue_id(residue: gemmi.Residue) -> tuple[str, int | None, str]:
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
    exp_seq = "".join(AMINO_ACID_MAP.get(r.name, "X") for r in exp_residues)
    comp_seq = "".join(AMINO_ACID_MAP.get(r.name, "X") for r in comp_residues)

    # Align sequences using Parasail (SIMD-accelerated)
    result = parasail.nw_trace_striped_32(exp_seq, comp_seq, 1, 1, parasail.blosum62)

    exp_aligned = result.traceback.query
    comp_aligned = result.traceback.ref

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
                residues = sorted(chain, key=lambda r: r.seqid.num or 0)
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
                residues = sorted(chain, key=lambda r: r.seqid.num or 0)
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

    # Align sequences using Parasail (SIMD-accelerated)
    dna_matrix = parasail.matrix_create("ACGT", 2, -1)
    result = parasail.nw_trace_striped_32(exp_seq, comp_seq, 1, 1, dna_matrix)

    aligned_exp = result.traceback.query
    aligned_comp = result.traceback.ref

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
