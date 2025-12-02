"""Enhanced tests for sequence analysis with property-based testing."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume

from biostructbenchmark.core.sequences import (
    classify_chains,
    get_protein_sequence,
    get_dna_sequence,
    calculate_sequence_identity,
    match_chains_by_similarity,
    align_specific_protein_chains,
    align_specific_dna_chains,
    ChainMatch,
    DNA_NUCLEOTIDE_MAP,
)


class TestDNANucleotideMapping:
    """Test DNA nucleotide mapping constants."""
    
    @pytest.mark.parametrize("nucleotide,expected", [
        ("DA", "A"), ("A", "A"),
        ("DT", "T"), ("T", "T"), 
        ("DG", "G"), ("G", "G"),
        ("DC", "C"), ("C", "C"),
    ])
    def test_dna_nucleotide_mapping(self, nucleotide, expected):
        """Test DNA nucleotide mapping for all standard cases."""
        assert DNA_NUCLEOTIDE_MAP[nucleotide] == expected
    
    def test_dna_mapping_completeness(self):
        """Test that all standard DNA nucleotides are mapped."""
        expected_keys = {"DA", "A", "DT", "T", "DG", "G", "DC", "C"}
        assert set(DNA_NUCLEOTIDE_MAP.keys()) == expected_keys


class TestChainClassification:
    """Test chain classification with comprehensive mock scenarios."""
    
    def create_mock_residue(self, resname, is_standard_aa=False):
        """Helper to create mock residues."""
        residue = Mock()
        residue.get_resname.return_value = resname
        
        # Mock is_aa function behavior
        with patch('biostructbenchmark.core.sequences.is_aa') as mock_is_aa:
            mock_is_aa.return_value = is_standard_aa
            return residue, mock_is_aa
    
    def create_mock_structure(self, chain_configs):
        """Helper to create mock structure with specified chain configurations."""
        structure = Mock()
        model = Mock()
        chains = []
        
        for chain_id, residues_config in chain_configs.items():
            chain = Mock()
            chain.get_id.return_value = chain_id
            
            residues = []
            for resname, is_aa_flag in residues_config:
                residue = Mock()
                residue.get_resname.return_value = resname
                residues.append(residue)
            
            chain.__iter__ = Mock(return_value=iter(residues))
            chains.append(chain)
        
        model.__iter__ = Mock(return_value=iter(chains))
        structure.__iter__ = Mock(return_value=iter([model]))
        
        return structure
    
    @patch('biostructbenchmark.core.sequences.is_aa')
    def test_classify_pure_protein_chains(self, mock_is_aa):
        """Test classification of pure protein chains."""
        mock_is_aa.return_value = True
        
        chain_configs = {
            "A": [("ALA", True), ("GLY", True), ("VAL", True)],
            "B": [("LEU", True), ("SER", True)]
        }
        structure = self.create_mock_structure(chain_configs)
        
        protein_chains, dna_chains = classify_chains(structure)
        
        assert set(protein_chains) == {"A", "B"}
        assert dna_chains == []
    
    @patch('biostructbenchmark.core.sequences.is_aa')
    def test_classify_pure_dna_chains(self, mock_is_aa):
        """Test classification of pure DNA chains."""
        mock_is_aa.return_value = False
        
        chain_configs = {
            "C": [("DA", False), ("DT", False), ("DG", False)],
            "D": [("DC", False), ("DA", False)]
        }
        structure = self.create_mock_structure(chain_configs)
        
        protein_chains, dna_chains = classify_chains(structure)
        
        assert protein_chains == []
        assert set(dna_chains) == {"C", "D"}
    
    @patch('biostructbenchmark.core.sequences.is_aa')
    def test_classify_mixed_chains(self, mock_is_aa):
        """Test classification of mixed protein and DNA chains."""
        def is_aa_side_effect(residue, standard=True):
            return residue.get_resname() in ["ALA", "GLY", "VAL", "LEU"]
        
        mock_is_aa.side_effect = is_aa_side_effect
        
        chain_configs = {
            "A": [("ALA", True), ("GLY", True)],  # Protein
            "B": [("DA", False), ("DT", False)],  # DNA
            "C": [("VAL", True), ("LEU", True), ("DA", False)]  # Mixed - protein majority
        }
        structure = self.create_mock_structure(chain_configs)
        
        protein_chains, dna_chains = classify_chains(structure)
        
        assert set(protein_chains) == {"A", "C"}
        assert dna_chains == ["B"]


class TestSequenceExtraction:
    """Test sequence extraction functionality."""
    
    @pytest.mark.parametrize("amino_acids,expected_sequence", [
        (["ALA", "GLY", "VAL"], "AGV"),
        (["LEU", "SER", "THR"], "LST"),
        (["MET", "TRP", "PHE"], "MWF"),
        ([], ""),
    ])
    @patch('biostructbenchmark.core.sequences.is_aa')
    @patch('biostructbenchmark.core.sequences.seq1')
    def test_get_protein_sequence(self, mock_seq1, mock_is_aa, amino_acids, expected_sequence):
        """Test protein sequence extraction."""
        mock_is_aa.return_value = True
        # Map three-letter codes to single-letter codes like real seq1
        aa_map = {"ALA": "A", "GLY": "G", "VAL": "V", "LEU": "L",
                  "SER": "S", "THR": "T", "MET": "M", "TRP": "W", "PHE": "F"}
        mock_seq1.side_effect = lambda x: aa_map.get(x, x[0])
        
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "A"
        
        residues = []
        for aa in amino_acids:
            residue = Mock()
            residue.get_resname.return_value = aa
            residues.append(residue)
        
        chain.__iter__ = Mock(return_value=iter(residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))
        
        sequence = get_protein_sequence(structure, "A")
        assert sequence == expected_sequence
    
    @pytest.mark.parametrize("nucleotides,expected_sequence", [
        (["DA", "DT", "DG", "DC"], "ATGC"),
        (["A", "T", "G", "C"], "ATGC"),
        (["DG", "DG", "DA"], "GGA"),
        ([], ""),
    ])
    def test_get_dna_sequence(self, nucleotides, expected_sequence):
        """Test DNA sequence extraction."""
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "C"

        residues = []
        for idx, nt in enumerate(nucleotides):
            residue = Mock()
            residue.get_resname.return_value = nt
            residue.get_id.return_value = (None, idx, None)  # (hetatm, seqid, icode)
            residues.append(residue)

        chain.get_residues.return_value = residues
        chain.__iter__ = Mock(return_value=iter(residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        sequence = get_dna_sequence(structure, "C")
        assert sequence == expected_sequence
    
    def test_sequence_extraction_nonexistent_chain(self):
        """Test sequence extraction for nonexistent chain."""
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "A"
        
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))
        
        protein_seq = get_protein_sequence(structure, "NONEXISTENT")
        dna_seq = get_dna_sequence(structure, "NONEXISTENT")
        
        assert protein_seq == ""
        assert dna_seq == ""


class TestSequenceIdentity:
    """Test sequence identity calculation."""
    
    @pytest.mark.parametrize("seq1,seq2,expected_identity", [
        ("ATGC", "ATGC", 1.0),  # Perfect match
        ("ATGC", "ATGG", 0.6),  # Match based on PairwiseAligner alignment
        ("ATGC", "TTGC", 0.6),  # Match based on PairwiseAligner alignment
        ("ATGC", "GGGG", 0.14285714285714285),  # Match based on PairwiseAligner alignment
        ("ATGC", "TTTT", 0.14285714285714285),  # Match based on PairwiseAligner alignment
        ("", "", 0.0),           # Empty sequences
        ("ATGC", "", 0.0),       # One empty
    ])
    def test_calculate_sequence_identity(self, seq1, seq2, expected_identity):
        """Test sequence identity calculation for various cases."""
        identity = calculate_sequence_identity(seq1, seq2)
        assert abs(identity - expected_identity) < 1e-10
    
    def test_sequence_identity_different_lengths(self):
        """Test sequence identity with different length sequences."""
        identity = calculate_sequence_identity("ATGCAA", "ATGC")
        # Should align based on shorter sequence length
        assert identity > 0.0


class TestPropertyBasedSequences:
    """Property-based tests for sequence functions."""
    
    @given(st.lists(st.sampled_from(["A", "T", "G", "C"]), min_size=0, max_size=20))
    def test_dna_sequence_extraction_property(self, nucleotides):
        """Property-based test for DNA sequence extraction."""
        # Create mock structure
        structure = Mock()
        model = Mock()
        chain = Mock()
        chain.get_id.return_value = "DNA"

        residues = []
        for idx, nt in enumerate(nucleotides):
            residue = Mock()
            residue.get_resname.return_value = nt
            residue.get_id.return_value = (None, idx, None)  # (hetatm, seqid, icode)
            residues.append(residue)

        chain.get_residues.return_value = residues
        chain.__iter__ = Mock(return_value=iter(residues))
        model.__iter__ = Mock(return_value=iter([chain]))
        structure.__iter__ = Mock(return_value=iter([model]))

        sequence = get_dna_sequence(structure, "DNA")

        # Properties to test
        assert len(sequence) == len(nucleotides)
        assert all(base in "ATGC" for base in sequence)
        if nucleotides:
            assert sequence == "".join(nucleotides)
    
    @given(st.text(alphabet="ATGC", min_size=0, max_size=50))
    def test_sequence_identity_properties(self, sequence):
        """Property-based test for sequence identity properties."""
        # Identity with self should be 1.0 (unless empty)
        if sequence:
            assert calculate_sequence_identity(sequence, sequence) == 1.0
        
        # Identity with empty sequence should be 0.0
        assert calculate_sequence_identity(sequence, "") == 0.0
        assert calculate_sequence_identity("", sequence) == 0.0


class TestChainMatch:
    """Test ChainMatch dataclass."""
    
    def test_chain_match_creation(self):
        """Test ChainMatch dataclass creation."""
        match = ChainMatch(
            exp_chain_id="A",
            comp_chain_id="X", 
            chain_type="protein",
            sequence_identity=0.95,
            rmsd=1.2
        )
        
        assert match.exp_chain_id == "A"
        assert match.comp_chain_id == "X"
        assert match.chain_type == "protein"
        assert match.sequence_identity == 0.95
        assert match.rmsd == 1.2
    
    @pytest.mark.parametrize("chain_type", ["protein", "dna"])
    def test_chain_match_types(self, chain_type):
        """Test ChainMatch with different chain types."""
        match = ChainMatch("A", "B", chain_type, 0.8, 2.0)
        assert match.chain_type == chain_type


@pytest.mark.integration
class TestSequenceIntegration:
    """Integration tests with real structure data."""
    
    def test_chain_classification_real_data(self):
        """Test chain classification on real structure data."""
        from biostructbenchmark.core.io import get_structure
        from pathlib import Path

        structure = get_structure(Path("./tests/data/complexes/experimental_9ny8.cif"))
        protein_chains, dna_chains = classify_chains(structure)
        
        # Should have both protein and DNA chains
        assert len(protein_chains) > 0
        assert len(dna_chains) > 0
        
        # Chain IDs should be strings
        assert all(isinstance(chain_id, str) for chain_id in protein_chains)
        assert all(isinstance(chain_id, str) for chain_id in dna_chains)
    
    def test_sequence_extraction_real_data(self):
        """Test sequence extraction on real structure data."""
        from biostructbenchmark.core.io import get_structure
        from pathlib import Path

        structure = get_structure(Path("./tests/data/complexes/experimental_9ny8.cif"))
        protein_chains, dna_chains = classify_chains(structure)
        
        # Extract sequences
        if protein_chains:
            protein_seq = get_protein_sequence(structure, protein_chains[0])
            assert len(protein_seq) > 0
            assert all(aa.isalpha() for aa in protein_seq)
        
        if dna_chains:
            dna_seq = get_dna_sequence(structure, dna_chains[0])
            assert len(dna_seq) > 0
            assert all(base in "ATGC" for base in dna_seq)


@pytest.mark.benchmark
class TestSequencePerformance:
    """Performance tests for sequence operations."""
    
    def test_chain_classification_performance(self, benchmark):
        """Benchmark chain classification performance."""
        from biostructbenchmark.core.io import get_structure
        from pathlib import Path

        structure = get_structure(Path("./tests/data/complexes/experimental_9ny8.cif"))
        result = benchmark(classify_chains, structure)
        
        protein_chains, dna_chains = result
        assert len(protein_chains) > 0
        assert len(dna_chains) > 0