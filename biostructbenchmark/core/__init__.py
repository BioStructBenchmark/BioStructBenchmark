"""Core modules for structure alignment and analysis."""

from biostructbenchmark.core.alignment import AlignmentResult, align_protein_dna_complex
from biostructbenchmark.core.interface import find_interface_residues
from biostructbenchmark.core.io import get_structure, validate_file
from biostructbenchmark.core.sequences import classify_chains
from biostructbenchmark.core.structural import (
    calculate_orientation_error,
    calculate_per_residue_rmsd,
    superimpose_structures,
)

__all__ = [
    "AlignmentResult",
    "align_protein_dna_complex",
    "calculate_orientation_error",
    "calculate_per_residue_rmsd",
    "classify_chains",
    "find_interface_residues",
    "get_structure",
    "superimpose_structures",
    "validate_file",
]
