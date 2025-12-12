"""
Tests for sequence analysis functionality
"""

from unittest.mock import Mock

import pytest

from conftest import create_mock_gemmi_chain, create_mock_gemmi_residue, create_mock_gemmi_structure

from biostructbenchmark.core.sequences import (
    DNA_NUCLEOTIDE_MAP,
    ChainMatch,
    calculate_sequence_identity,
    classify_chains,
    get_dna_sequence,
    get_protein_sequence,
    match_chains_by_similarity,
)


class TestChainMatching:
    """Test chain matching functionality - the most critical component"""

    def test_match_chains_prevents_duplicate_mapping(self):
        """Test that chain matching creates 1:1 mappings (the bug we fixed)"""
        # Create identical protein chains using GEMMI mocks
        protein_residues = [
            create_mock_gemmi_residue("ALA", 1),
            create_mock_gemmi_residue("GLY", 2),
            create_mock_gemmi_residue("VAL", 3),
        ]

        exp_chain_a = create_mock_gemmi_chain("A", protein_residues)
        exp_chain_b = create_mock_gemmi_chain("B", protein_residues)
        comp_chain_a = create_mock_gemmi_chain("A", protein_residues)
        comp_chain_b = create_mock_gemmi_chain("B", protein_residues)

        exp_structure = create_mock_gemmi_structure([exp_chain_a, exp_chain_b])
        comp_structure = create_mock_gemmi_structure([comp_chain_a, comp_chain_b])

        # Mock sequence functions to avoid BioPython dependency
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "biostructbenchmark.core.sequences.get_protein_sequence",
                lambda struct, chain_id: "AGV",
            )  # Same sequence for all
            mp.setattr(
                "biostructbenchmark.core.sequences.get_dna_sequence", lambda struct, chain_id: None
            )  # No DNA

            matches = match_chains_by_similarity(exp_structure, comp_structure)

        # Should have exactly 2 matches, no duplicates
        assert len(matches) == 2

        # Each computational chain should be matched at most once
        comp_chains_used = [match.comp_chain_id for match in matches]
        assert len(comp_chains_used) == len(set(comp_chains_used))  # No duplicates

    def test_match_chains_respects_sequence_threshold(self):
        """Test that chain matching respects minimum identity thresholds"""
        # Create chains with different sequences using GEMMI mocks
        good_residues = [create_mock_gemmi_residue("ALA", 1), create_mock_gemmi_residue("GLY", 2)]
        poor_residues = [create_mock_gemmi_residue("PHE", 1), create_mock_gemmi_residue("TRP", 2)]

        exp_chain = create_mock_gemmi_chain("A", good_residues)
        comp_chain_good = create_mock_gemmi_chain("A", good_residues)
        comp_chain_poor = create_mock_gemmi_chain("B", poor_residues)

        exp_structure = create_mock_gemmi_structure([exp_chain])
        comp_structure = create_mock_gemmi_structure([comp_chain_good, comp_chain_poor])

        with pytest.MonkeyPatch().context() as mp:

            def mock_protein_seq(struct, chain_id):
                if chain_id == "A":
                    return "AG"  # Good match
                return "FW"  # Poor match

            mp.setattr("biostructbenchmark.core.sequences.get_protein_sequence", mock_protein_seq)
            mp.setattr(
                "biostructbenchmark.core.sequences.get_dna_sequence", lambda struct, chain_id: None
            )

            matches = match_chains_by_similarity(exp_structure, comp_structure)

        # Should only match the good chain (identity = 100% > 30% threshold)
        assert len(matches) == 1
        assert matches[0].comp_chain_id == "A"
        assert matches[0].sequence_identity == 1.0


class TestSequenceIdentity:
    """Test sequence identity calculation"""

    def test_calculate_sequence_identity_identical(self):
        """Test sequence identity with identical sequences"""
        seq1 = "ACGT"
        seq2 = "ACGT"
        identity = calculate_sequence_identity(seq1, seq2)
        assert identity == 1.0

    def test_calculate_sequence_identity_different(self):
        """Test sequence identity with different sequences"""
        seq1 = "AAAA"
        seq2 = "TTTT"  # Completely different
        identity = calculate_sequence_identity(seq1, seq2)
        assert identity == 0.0

    def test_calculate_sequence_identity_partial(self):
        """Test sequence identity with partial matches"""
        seq1 = "AAAA"
        seq2 = "AAAT"  # Partial match
        identity = calculate_sequence_identity(seq1, seq2)
        # Should be between 0 and 1, and greater than completely different
        assert 0.0 < identity < 1.0

    def test_calculate_sequence_identity_different_lengths(self):
        """Test sequence identity with different length sequences"""
        seq1 = "ACG"
        seq2 = "ACGT"
        identity = calculate_sequence_identity(seq1, seq2)
        # Should handle length differences gracefully
        assert 0.0 <= identity <= 1.0


class TestChainClassification:
    """Test chain classification functionality"""

    def test_classify_chains_protein_and_dna(self):
        """Test classification of mixed protein and DNA chains"""
        # Create mixed structure using GEMMI mocks
        protein_residues = [create_mock_gemmi_residue(name, i+1)
                           for i, name in enumerate(["ALA", "GLY", "VAL"])]
        dna_residues = [create_mock_gemmi_residue(name, i+1)
                       for i, name in enumerate(["DA", "DT", "DG", "DC"])]

        protein_chain = create_mock_gemmi_chain("A", protein_residues)
        dna_chain = create_mock_gemmi_chain("B", dna_residues)

        structure = create_mock_gemmi_structure([protein_chain, dna_chain])

        protein_chains, dna_chains = classify_chains(structure)

        assert "A" in protein_chains
        assert "B" in dna_chains
        assert len(protein_chains) == 1
        assert len(dna_chains) == 1

    def test_classify_chains_empty_structure(self):
        """Test classification of empty structure"""
        structure = create_mock_gemmi_structure([])

        protein_chains, dna_chains = classify_chains(structure)

        assert len(protein_chains) == 0
        assert len(dna_chains) == 0


class TestSequenceExtraction:
    """Test sequence extraction functions with integration approach"""

    def test_dna_nucleotide_map_completeness(self):
        """Test that DNA nucleotide map contains expected nucleotides"""
        expected_nucleotides = ["DA", "DT", "DG", "DC"]
        for nucleotide in expected_nucleotides:
            assert nucleotide in DNA_NUCLEOTIDE_MAP
            assert len(DNA_NUCLEOTIDE_MAP[nucleotide]) == 1  # Single letter code


class TestChainMatchDataStructure:
    """Test ChainMatch dataclass"""

    def test_chain_match_creation(self):
        """Test ChainMatch object creation and attributes"""
        match = ChainMatch(
            exp_chain_id="A",
            comp_chain_id="B",
            chain_type="protein",
            sequence_identity=0.85,
            rmsd=1.5,
        )

        assert match.exp_chain_id == "A"
        assert match.comp_chain_id == "B"
        assert match.chain_type == "protein"
        assert match.sequence_identity == 0.85
        assert match.rmsd == 1.5


class TestSequenceFunctionsIntegration:
    """Integration tests using real structures to test sequence functions"""

    @pytest.mark.integration
    def test_sequence_functions_with_real_structure(self):
        """Test sequence extraction with real structure"""
        from pathlib import Path

        from biostructbenchmark.core.io import get_structure

        # Load a real structure
        structure = get_structure(Path("tests/data/complexes/experimental_9ny8.cif"))

        # Test protein sequence extraction
        protein_seq = get_protein_sequence(structure, "A")
        assert protein_seq is not None
        assert len(protein_seq) > 0
        assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in protein_seq)

        # Test DNA sequence extraction
        dna_seq = get_dna_sequence(structure, "C")
        assert dna_seq is not None
        assert len(dna_seq) > 0
        assert all(c in "ATGC" for c in dna_seq)

    @pytest.mark.integration
    def test_chain_classification_with_real_structure(self):
        """Test chain classification with real structure"""
        from pathlib import Path

        from biostructbenchmark.core.io import get_structure

        structure = get_structure(Path("tests/data/complexes/experimental_9ny8.cif"))
        protein_chains, dna_chains = classify_chains(structure)

        # 9ny8 should have protein chains A,B and DNA chains C,D
        assert len(protein_chains) == 2
        assert len(dna_chains) == 2
        assert "A" in protein_chains
        assert "B" in protein_chains
        assert "C" in dna_chains
        assert "D" in dna_chains


if __name__ == "__main__":
    pytest.main([__file__])
