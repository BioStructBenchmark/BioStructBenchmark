"""
Tests to verify sequence alignment functionality in alignment.py
"""

from unittest.mock import Mock

import pytest
from biostructbenchmark.core.alignment import (
    align_dna_sequences,
    align_protein_sequences,
)


def create_mock_residue(resname, resnum, chain_id="A"):
    """Create a mock residue with specified properties"""
    residue = Mock()
    residue.get_resname.return_value = resname
    residue.get_id.return_value = (" ", resnum, " ")  # (hetflag, resnum, icode)

    # Mock atoms for the residue
    atom = Mock()
    atom.get_name.return_value = "CA" if resname in ["ALA", "GLY", "VAL"] else "P"
    atom.get_coord.return_value = [0.0, 0.0, 0.0]
    residue.get_atoms.return_value = [atom]

    return residue


def create_mock_chain(chain_id, residues):
    """Create a mock chain with specified residues"""
    chain = Mock()
    chain.get_id.return_value = chain_id
    chain.get_residues.return_value = residues
    chain.__iter__ = lambda self: iter(residues)
    return chain


def create_mock_structure(chains):
    """Create a mock structure with specified chains"""
    structure = Mock()
    model = Mock()
    model.__iter__ = lambda self: iter(chains)
    structure.__iter__ = lambda self: iter([model])
    return structure


class TestProteinSequenceAlignment:
    """Test protein sequence alignment functionality"""

    def test_identical_sequences(self):
        """Test alignment of identical protein sequences"""
        # Create identical sequences
        residues_exp = [
            create_mock_residue("ALA", 1),
            create_mock_residue("GLY", 2),
            create_mock_residue("VAL", 3),
        ]
        residues_comp = [
            create_mock_residue("ALA", 1),
            create_mock_residue("GLY", 2),
            create_mock_residue("VAL", 3),
        ]

        chain_exp = create_mock_chain("A", residues_exp)
        chain_comp = create_mock_chain("A", residues_comp)

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        # Mock is_aa to return True for our test residues
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("biostructbenchmark.core.alignment.is_aa", lambda r, standard=True: True)
            mp.setattr(
                "biostructbenchmark.core.alignment.seq1",
                lambda name: {"ALA": "A", "GLY": "G", "VAL": "V"}[name],
            )

            mapping = align_protein_sequences(exp_structure, comp_structure)

        # Should have 1:1 mapping for all residues
        expected_mapping = {
            "A:(' ', 1, ' ')": "A:(' ', 1, ' ')",
            "A:(' ', 2, ' ')": "A:(' ', 2, ' ')",
            "A:(' ', 3, ' ')": "A:(' ', 3, ' ')",
        }
        assert mapping == expected_mapping

    def test_sequences_with_gaps(self):
        """Test alignment with insertion/deletion gaps"""
        # Experimental: ALA-GLY-VAL, Computational: ALA-VAL (missing GLY)
        residues_exp = [
            create_mock_residue("ALA", 1),
            create_mock_residue("GLY", 2),
            create_mock_residue("VAL", 3),
        ]
        residues_comp = [
            create_mock_residue("ALA", 1),
            create_mock_residue("VAL", 2),  # GLY deleted, VAL renumbered
        ]

        chain_exp = create_mock_chain("A", residues_exp)
        chain_comp = create_mock_chain("A", residues_comp)

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("biostructbenchmark.core.alignment.is_aa", lambda r, standard=True: True)
            mp.setattr(
                "biostructbenchmark.core.alignment.seq1",
                lambda name: {"ALA": "A", "GLY": "G", "VAL": "V"}[name],
            )

            mapping = align_protein_sequences(exp_structure, comp_structure)

        # Should map ALA->ALA and VAL->VAL, skip GLY
        expected_mapping = {
            "A:(' ', 1, ' ')": "A:(' ', 1, ' ')",
            "A:(' ', 3, ' ')": "A:(' ', 2, ' ')",
        }
        assert mapping == expected_mapping

    def test_empty_sequences(self):
        """Test alignment with empty sequences"""
        chain_exp = create_mock_chain("A", [])
        chain_comp = create_mock_chain("A", [])

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        mapping = align_protein_sequences(exp_structure, comp_structure)
        assert mapping == {}

    def test_mismatched_chains(self):
        """Test alignment when chains don't match"""
        residues_exp = [create_mock_residue("ALA", 1)]
        residues_comp = [create_mock_residue("GLY", 1)]

        chain_exp = create_mock_chain("A", residues_exp)
        chain_comp = create_mock_chain("B", residues_comp)  # Different chain ID

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        mapping = align_protein_sequences(exp_structure, comp_structure)
        assert mapping == {}


class TestDNASequenceAlignment:
    """Test DNA sequence alignment functionality"""

    def test_identical_dna_sequences(self):
        """Test alignment of identical DNA sequences"""
        residues_exp = [
            create_mock_residue("DA", 1),
            create_mock_residue("DT", 2),
            create_mock_residue("DG", 3),
            create_mock_residue("DC", 4),
        ]
        residues_comp = [
            create_mock_residue("DA", 1),
            create_mock_residue("DT", 2),
            create_mock_residue("DG", 3),
            create_mock_residue("DC", 4),
        ]

        chain_exp = create_mock_chain("A", residues_exp)
        chain_comp = create_mock_chain("A", residues_comp)

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        mapping, exp_seq_dict, comp_seq_dict = align_dna_sequences(exp_structure, comp_structure)

        # Check sequence extraction
        assert exp_seq_dict["A"] == "ATGC"
        assert comp_seq_dict["A"] == "ATGC"

        # Check 1:1 mapping
        expected_mapping = {
            "A:(' ', 1, ' ')": "A:(' ', 1, ' ')",
            "A:(' ', 2, ' ')": "A:(' ', 2, ' ')",
            "A:(' ', 3, ' ')": "A:(' ', 3, ' ')",
            "A:(' ', 4, ' ')": "A:(' ', 4, ' ')",
        }
        assert mapping == expected_mapping

    def test_dna_with_gaps(self):
        """Test DNA alignment with gaps"""
        residues_exp = [
            create_mock_residue("DA", 1),
            create_mock_residue("DT", 2),
            create_mock_residue("DG", 3),
        ]
        residues_comp = [
            create_mock_residue("DA", 1),
            create_mock_residue("DG", 2),  # DT deleted
        ]

        chain_exp = create_mock_chain("A", residues_exp)
        chain_comp = create_mock_chain("A", residues_comp)

        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([chain_comp])

        mapping, exp_seq_dict, comp_seq_dict = align_dna_sequences(exp_structure, comp_structure)

        assert exp_seq_dict["A"] == "ATG"
        assert comp_seq_dict["A"] == "AG"

        # Should map A->A and G->G, skip T
        expected_mapping = {
            "A:(' ', 1, ' ')": "A:(' ', 1, ' ')",
            "A:(' ', 3, ' ')": "A:(' ', 2, ' ')",
        }
        assert mapping == expected_mapping

    def test_non_dna_residues_filtered(self):
        """Test that non-DNA residues are filtered out"""
        residues_exp = [
            create_mock_residue("ALA", 1),  # Protein residue
            create_mock_residue("DA", 2),  # DNA residue
            create_mock_residue("HOH", 3),  # Water
        ]

        chain_exp = create_mock_chain("A", residues_exp)
        exp_structure = create_mock_structure([chain_exp])
        comp_structure = create_mock_structure([])

        _mapping, exp_seq_dict, _comp_seq_dict = align_dna_sequences(exp_structure, comp_structure)

        # Should only extract DNA residue
        assert exp_seq_dict["A"] == "A"


class TestAlignmentDebugFunctions:
    """Test debug and visualization functions for alignment verification"""

    def test_alignment_stats_calculation(self):
        """Test calculation of alignment statistics."""
        # TODO: Implement when we add debug functions


def test_alignment_quality_metrics():
    """Test alignment quality calculation"""
    # Mock structures with known alignment quality
    residues_exp = [create_mock_residue("ALA", i) for i in range(1, 6)]  # AAAAA
    residues_comp = [create_mock_residue("ALA", i) for i in range(1, 4)]  # AAA (shorter)

    chain_exp = create_mock_chain("A", residues_exp)
    chain_comp = create_mock_chain("A", residues_comp)

    exp_structure = create_mock_structure([chain_exp])
    comp_structure = create_mock_structure([chain_comp])

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("biostructbenchmark.core.alignment.is_aa", lambda r, standard=True: True)
        mp.setattr("biostructbenchmark.core.alignment.seq1", lambda name: "A")

        mapping = align_protein_sequences(exp_structure, comp_structure)

    # Should have partial alignment (3 out of 5 residues aligned)
    assert len(mapping) == 3
    alignment_coverage = len(mapping) / 5  # 5 residues in experimental
    assert alignment_coverage == 0.6


if __name__ == "__main__":
    pytest.main([__file__])
