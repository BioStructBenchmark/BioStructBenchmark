"""
Tests for protein-DNA complex alignment functionality
"""

from unittest.mock import Mock, patch

import numpy as np
from biostructbenchmark.core.alignment import AlignmentResult, align_protein_dna_complex
from biostructbenchmark.core.interface import find_interface_residues
from biostructbenchmark.core.sequences import (
    calculate_sequence_identity,
    classify_chains,
    get_dna_sequence,
    get_protein_sequence,
    match_chains_by_similarity,
)
from biostructbenchmark.core.structural import (
    calculate_orientation_error,
    calculate_per_residue_rmsd,
)


def create_mock_residue(name, seqid_num):
    """Create a mock residue with GEMMI-compatible interface."""
    residue = Mock()
    residue.name = name
    residue.seqid = Mock()
    residue.seqid.num = seqid_num
    residue.seqid.icode = ""
    return residue


def create_mock_chain(chain_name, residues):
    """Create a mock chain with GEMMI-compatible interface."""
    chain = Mock()
    chain.name = chain_name
    chain.__iter__ = Mock(return_value=iter(residues))
    return chain


def create_mock_structure(chains):
    """Create a mock structure with GEMMI-compatible interface."""
    structure = Mock()
    model = Mock()
    model.__iter__ = Mock(return_value=iter(chains))
    structure.__iter__ = Mock(return_value=iter([model]))
    return structure


class TestChainClassification:
    """Test chain classification functionality."""

    def test_classify_protein_chains(self):
        """Test classification of protein chains."""
        # Create protein residues
        protein_residues = [create_mock_residue("ALA", i) for i in range(1, 4)]
        chain = create_mock_chain("A", protein_residues)
        structure = create_mock_structure([chain])

        protein_chains, dna_chains = classify_chains(structure)

        assert protein_chains == ["A"]
        assert dna_chains == []

    def test_classify_dna_chains(self):
        """Test classification of DNA chains."""
        # Create DNA residues
        dna_residues = [
            create_mock_residue(nucleotide, i) for i, nucleotide in enumerate(["DA", "DT", "DG"], 1)
        ]
        chain = create_mock_chain("B", dna_residues)
        structure = create_mock_structure([chain])

        protein_chains, dna_chains = classify_chains(structure)

        assert protein_chains == []
        assert dna_chains == ["B"]


class TestSequenceExtraction:
    """Test sequence extraction functions."""

    def test_get_protein_sequence(self):
        """Test protein sequence extraction."""
        # Create protein residues with standard amino acids
        protein_residues = [
            create_mock_residue(aa, i) for i, aa in enumerate(["ALA", "GLY", "VAL"], 1)
        ]
        chain = create_mock_chain("A", protein_residues)
        structure = create_mock_structure([chain])

        sequence = get_protein_sequence(structure, "A")

        assert sequence == "AGV"

    def test_get_dna_sequence(self):
        """Test DNA sequence extraction."""
        # Create DNA residues
        dna_residues = [
            create_mock_residue(nucleotide, i)
            for i, nucleotide in enumerate(["DA", "DT", "DG", "DC"], 1)
        ]
        chain = create_mock_chain("B", dna_residues)
        structure = create_mock_structure([chain])

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
        assert 0.5 < identity <= 0.8  # Expected around 3/4 matches but depends on alignment

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
        # Create protein atom
        prot_atom = Mock()
        prot_atom.pos = Mock()
        prot_atom.pos.x = 0.0
        prot_atom.pos.y = 0.0
        prot_atom.pos.z = 0.0

        # Create DNA atom close to protein atom
        dna_atom = Mock()
        dna_atom.pos = Mock()
        dna_atom.pos.x = 3.0
        dna_atom.pos.y = 0.0
        dna_atom.pos.z = 0.0

        # Create protein residue
        prot_residue = create_mock_residue("ALA", 1)
        prot_residue.__iter__ = Mock(return_value=iter([prot_atom]))

        # Create DNA residue
        dna_residue = create_mock_residue("DA", 1)
        dna_residue.__iter__ = Mock(return_value=iter([dna_atom]))

        # Create chains
        prot_chain = create_mock_chain("A", [prot_residue])
        dna_chain = create_mock_chain("B", [dna_residue])

        # Create structure
        structure = create_mock_structure([prot_chain, dna_chain])

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
            patch("biostructbenchmark.core.sequences.classify_chains") as mock_classify,
            patch("biostructbenchmark.core.sequences.get_protein_sequence") as mock_prot_seq,
            patch("biostructbenchmark.core.sequences.get_dna_sequence") as mock_dna_seq,
            patch("biostructbenchmark.core.sequences.calculate_sequence_identity") as mock_identity,
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
            patch("biostructbenchmark.core.alignment.classify_chains") as mock_classify,
            patch("biostructbenchmark.core.alignment.match_chains_by_similarity") as mock_match,
        ):
            # Mock chain classification
            mock_classify.side_effect = [(["A"], ["B"]), (["C"], ["D"])]
            mock_match.return_value = []  # No matches

            result = align_protein_dna_complex(exp_structure, comp_structure)

        assert result.sequence_mapping == {}
        assert result.structural_rmsd == float("inf")
        assert result.protein_rmsd == float("inf")
        assert result.dna_rmsd == float("inf")
