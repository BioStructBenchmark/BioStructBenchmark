"""
Tests for protein-DNA complex alignment functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from Bio.PDB import Structure, Model, Chain, Residue, Atom

from biostructbenchmark.core.alignment import align_protein_dna_complex, AlignmentResult
from biostructbenchmark.core.sequences import (
    classify_chains,
    match_chains_by_similarity,
    align_specific_protein_chains,
    align_specific_dna_chains,
    get_protein_sequence,
    get_dna_sequence,
    calculate_sequence_identity,
    ChainMatch,
    DNA_NUCLEOTIDE_MAP,
)
from biostructbenchmark.core.structural import (
    calculate_per_residue_rmsd,
    calculate_orientation_error,
)
from biostructbenchmark.core.interface import find_interface_residues


class TestChainClassification:
    """Test chain classification functionality."""

    def test_classify_protein_chains(self):
        """Test classification of protein chains."""
        # Mock structure with protein residues
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "A"

        # Mock protein residues
        protein_residues = []
        for i in range(3):
            residue = Mock()
            residue.get_resname.return_value = "ALA"
            protein_residues.append(residue)

        chain.__iter__ = Mock(return_value=iter(protein_residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        with patch("biostructbenchmark.core.sequences.is_aa") as mock_is_aa:
            mock_is_aa.return_value = True
            protein_chains, dna_chains = classify_chains(structure)

        assert protein_chains == ["A"]
        assert dna_chains == []

    def test_classify_dna_chains(self):
        """Test classification of DNA chains."""
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "B"

        # Mock DNA residues
        dna_residues = []
        for nucleotide in ["DA", "DT", "DG"]:
            residue = Mock()
            residue.get_resname.return_value = nucleotide
            dna_residues.append(residue)

        chain.__iter__ = Mock(return_value=iter(dna_residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        with patch("biostructbenchmark.core.sequences.is_aa") as mock_is_aa:
            mock_is_aa.return_value = False
            protein_chains, dna_chains = classify_chains(structure)

        assert protein_chains == []
        assert dna_chains == ["B"]


class TestSequenceExtraction:
    """Test sequence extraction functions."""

    def test_get_protein_sequence(self):
        """Test protein sequence extraction."""
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "A"

        # Mock protein residues
        residues = []
        for aa in ["ALA", "GLY", "VAL"]:
            residue = Mock()
            residue.get_resname.return_value = aa
            residues.append(residue)

        chain.__iter__ = Mock(return_value=iter(residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        with (
            patch("biostructbenchmark.core.sequences.is_aa") as mock_is_aa,
            patch("biostructbenchmark.core.sequences.seq1") as mock_seq1,
        ):
            mock_is_aa.return_value = True
            mock_seq1.side_effect = lambda x: {"ALA": "A", "GLY": "G", "VAL": "V"}[x]

            sequence = get_protein_sequence(structure, "A")

        assert sequence == "AGV"

    def test_get_dna_sequence(self):
        """Test DNA sequence extraction."""
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "B"

        # Mock DNA residues with proper ordering
        residues = []
        for i, nucleotide in enumerate(["DA", "DT", "DG", "DC"]):
            residue = Mock()
            residue.get_resname.return_value = nucleotide
            residue.get_id.return_value = (" ", i + 1, " ")  # Standard PDB residue ID
            residues.append(residue)

        chain.get_residues.return_value = residues
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        sequence = get_dna_sequence(structure, "B")
        assert sequence == "ATGC"


class TestSequenceIdentity:
    """Test sequence identity calculation."""

    def test_identical_sequences(self):
        """Test identity calculation for identical sequences."""
        seq1 = "ATGC"
        seq2 = "ATGC"
        identity = calculate_sequence_identity(seq1, seq2)
        assert identity == 1.0

    def test_partially_identical_sequences(self):
        """Test identity calculation for partially identical sequences."""
        seq1 = "ATGC"
        seq2 = "ATCC"
        identity = calculate_sequence_identity(seq1, seq2)
        assert (
            0.5 < identity <= 0.8
        )  # Expected around 3/4 matches but depends on alignment

    def test_empty_sequences(self):
        """Test identity calculation for empty sequences."""
        assert calculate_sequence_identity("", "ATGC") == 0.0
        assert calculate_sequence_identity("ATGC", "") == 0.0
        assert calculate_sequence_identity("", "") == 0.0


class TestOrientationError:
    """Test orientation error calculation."""

    def test_identity_matrix(self):
        """Test orientation error for identity matrix (no rotation)."""
        rotation_matrix = np.eye(3)
        error = calculate_orientation_error(rotation_matrix)
        assert abs(error) < 1e-10  # Should be essentially zero

    def test_90_degree_rotation(self):
        """Test orientation error for 90-degree rotation around z-axis."""
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        error = calculate_orientation_error(rotation_matrix)
        assert abs(error - 90.0) < 1e-10


class TestPerResidueRMSD:
    """Test per-residue RMSD calculation."""

    def test_identical_coordinates(self):
        """Test RMSD for identical coordinates."""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        exp_atoms = {"res1": coords}
        comp_atoms = {"res1": coords}
        mapping = {"res1": "res1"}

        rmsd_dict = calculate_per_residue_rmsd(exp_atoms, comp_atoms, mapping)
        assert "res1" in rmsd_dict
        assert abs(rmsd_dict["res1"]) < 1e-10

    def test_different_coordinates(self):
        """Test RMSD for different coordinates."""
        exp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        comp_coords = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

        exp_atoms = {"res1": exp_coords}
        comp_atoms = {"res1": comp_coords}
        mapping = {"res1": "res1"}

        rmsd_dict = calculate_per_residue_rmsd(exp_atoms, comp_atoms, mapping)
        assert "res1" in rmsd_dict
        assert rmsd_dict["res1"] == 1.0  # Each atom displaced by 1 unit


class TestInterfaceDetection:
    """Test protein-DNA interface detection."""

    def test_find_interface_residues_mock(self):
        """Test interface residue detection with mock structures."""
        structure = Mock()
        model = Mock()

        # Mock protein chain
        prot_chain = Mock()
        prot_chain.get_id.return_value = "A"

        # Mock DNA chain
        dna_chain = Mock()
        dna_chain.get_id.return_value = "B"

        # Mock protein residue with atoms
        prot_residue = Mock()
        prot_residue.get_id.return_value = (" ", 1, " ")
        prot_residue.get_resname.return_value = "ALA"  # Add residue name for is_aa check
        prot_atom = Mock()
        prot_atom.__sub__ = Mock(return_value=3.0)  # Distance of 3 Angstroms
        prot_residue.get_atoms.return_value = [prot_atom]

        # Mock DNA residue with atoms
        dna_residue = Mock()
        dna_residue.get_id.return_value = (" ", 1, " ")
        dna_residue.get_resname.return_value = "DA"
        dna_atom = Mock()
        dna_residue.get_atoms.return_value = [dna_atom]

        prot_chain.__iter__ = Mock(return_value=iter([prot_residue]))
        dna_chain.__iter__ = Mock(return_value=iter([dna_residue]))
        model.__iter__ = Mock(return_value=iter([prot_chain, dna_chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        with patch("biostructbenchmark.core.sequences.is_aa") as mock_is_aa:
            mock_is_aa.return_value = True
            interface_residues = find_interface_residues(structure, ["A"], ["B"], 5.0)

        assert "A" in interface_residues
        assert "B" in interface_residues


class TestChainMatching:
    """Test chain matching functionality."""

    def test_match_chains_by_similarity_mock(self):
        """Test chain matching with mock structures."""
        exp_structure = Mock()
        comp_structure = Mock()

        with (
            patch(
                "biostructbenchmark.core.sequences.classify_chains"
            ) as mock_classify,
            patch(
                "biostructbenchmark.core.sequences.get_protein_sequence"
            ) as mock_prot_seq,
            patch(
                "biostructbenchmark.core.sequences.get_dna_sequence"
            ) as mock_dna_seq,
            patch(
                "biostructbenchmark.core.sequences.calculate_sequence_identity"
            ) as mock_identity,
        ):

            # Mock chain classification
            mock_classify.side_effect = [(["A"], ["B"]), (["C"], ["D"])]

            # Mock protein sequences
            mock_prot_seq.side_effect = ["AGVL", "AGVL"]  # Identical sequences

            # Mock DNA sequences (empty for this protein-focused test)
            mock_dna_seq.return_value = ""

            # Mock high sequence identity
            mock_identity.return_value = 0.95

            matches = match_chains_by_similarity(exp_structure, comp_structure)

        assert len(matches) == 1
        assert matches[0].exp_chain_id == "A"
        assert matches[0].comp_chain_id == "C"
        assert matches[0].chain_type == "protein"
        assert matches[0].sequence_identity == 0.95


class TestAlignmentResult:
    """Test AlignmentResult data structure."""

    def test_alignment_result_creation(self):
        """Test creation of AlignmentResult."""
        result = AlignmentResult(
            sequence_mapping={"A:1": "B:1"},
            structural_rmsd=1.5,
            per_residue_rmsd={"A:1": 0.8},
            protein_rmsd=1.2,
            dna_rmsd=1.8,
            interface_rmsd=1.0,
            capri_i_rmsd=0.8,
            capri_l_rmsd=1.2,
            backbone_rmsd=1.3,
            backbone_protein_rmsd=1.1,
            backbone_dna_rmsd=1.5,
            fnat=0.65,
            dockq=0.70,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            orientation_error=15.0,
            translational_error=2.5,
            protein_chains=["A"],
            dna_chains=["B"],
            interface_residues={"A": ["A:1"], "B": ["B:1"]},
        )

        assert result.structural_rmsd == 1.5
        assert result.orientation_error == 15.0
        assert len(result.protein_chains) == 1
        assert len(result.dna_chains) == 1


class TestMainAlignmentFunction:
    """Test the main alignment function."""

    def test_align_protein_dna_complex_no_matches(self):
        """Test alignment function when no chain matches are found."""
        exp_structure = Mock()
        comp_structure = Mock()

        with (
            patch(
                "biostructbenchmark.core.alignment.classify_chains"
            ) as mock_classify,
            patch(
                "biostructbenchmark.core.alignment.match_chains_by_similarity"
            ) as mock_match,
        ):

            # Mock chain classification
            mock_classify.side_effect = [(["A"], ["B"]), (["C"], ["D"])]
            mock_match.return_value = []  # No matches

            result = align_protein_dna_complex(exp_structure, comp_structure)

        assert result.sequence_mapping == {}
        assert result.structural_rmsd == float("inf")
        assert result.protein_rmsd == float("inf")
        assert result.dna_rmsd == float("inf")
