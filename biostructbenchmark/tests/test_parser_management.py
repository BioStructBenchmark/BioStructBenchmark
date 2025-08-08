"""Tests for parser management in io.py"""

import pytest
from pathlib import Path
from biostructbenchmark.core.io import get_structure
from Bio.PDB import MMCIFParser, PDBParser


class TestParserManagement:
    """Test suite for parser creation, caching, and file reading."""
    
    def setup_method(self):
        """Clear parser cache before each test."""
        _parsers.clear()
    
    def test_parser_creation_pdb(self):
        """Test PDB parser is created correctly."""
        pdb_path = Path("./tests/data/proteins_pdb/1bom.pdb")
        parser = get_parser(pdb_path)
        
        assert parser is not None
        assert isinstance(parser, PDBParser)
        assert parser.QUIET == True  # Should suppress warnings
    
    def test_parser_creation_cif(self):
        """Test CIF parser is created correctly."""
        cif_path = Path("./tests/data/proteins_cif/1bom.cif")
        parser = get_parser(cif_path)
        
        assert parser is not None
        assert isinstance(parser, MMCIFParser)
        assert parser.QUIET == True
    
    def test_parser_creation_mmcif(self):
        """Test mmCIF parser handles .mmcif extension."""
        mmcif_path = Path("test.mmcif")
        parser = get_parser(mmcif_path)
        
        assert parser is not None
        assert isinstance(parser, MMCIFParser)
    
    def test_parser_caching(self):
        """Test parsers are cached and reused."""
        pdb_path1 = Path("test1.pdb")
        pdb_path2 = Path("test2.pdb")
        
        parser1 = get_parser(pdb_path1)
        parser2 = get_parser(pdb_path2)
        
        # Should be the same parser instance
        assert parser1 is parser2
        assert len(_parsers) == 1
        
        # Different file type should create new parser
        cif_path = Path("test.cif")
        parser3 = get_parser(cif_path)
        
        assert parser3 is not parser1
        assert len(_parsers) == 2
    
    def test_unsupported_format(self):
        """Test unsupported file formats return None."""
        txt_path = Path("test.txt")
        parser = get_parser(txt_path)
        
        assert parser is None
    
    def test_structure_content_preservation_pdb(self):
        """Test PDB structure is read without distortion."""
        pdb_path = Path("./tests/data/proteins_pdb/1bom.pdb")
        structure = get_structure(pdb_path)
        
        assert structure is not None
        
        # Check structure hierarchy is preserved
        models = list(structure.get_models())
        assert len(models) > 0
        
        chains = list(models[0].get_chains())
        assert len(chains) > 0
        
        # Check atoms are preserved
        atoms = list(structure.get_atoms())
        assert len(atoms) > 0
        
        # Check specific atom properties are preserved
        first_atom = atoms[0]
        assert hasattr(first_atom, 'coord')
        assert hasattr(first_atom, 'name')
        assert hasattr(first_atom, 'element')
        
        # Check coordinates are numeric and reasonable
        coord = first_atom.coord
        assert len(coord) == 3
        assert all(isinstance(c, (int, float)) for c in coord)
        assert all(-1000 < c < 1000 for c in coord)  # Reasonable range for protein coords
    
    def test_structure_content_preservation_cif(self):
        """Test CIF structure is read without distortion."""
        cif_path = Path("./tests/data/proteins_cif/1bom.cif")
        structure = get_structure(cif_path)
        
        assert structure is not None
        
        # Get atom counts and properties
        atoms = list(structure.get_atoms())
        assert len(atoms) > 0
        
        # Check residues are preserved
        residues = list(structure.get_residues())
        assert len(residues) > 0
        
        # Check residue properties
        first_residue = residues[0]
        assert hasattr(first_residue, 'resname')
        assert hasattr(first_residue, 'id')
    
    def test_pdb_cif_consistency(self):
        """Test same structure from PDB and CIF has consistent content."""
        pdb_path = Path("./tests/data/proteins_pdb/1bom.pdb")
        cif_path = Path("./tests/data/proteins_cif/1bom.cif")
        
        pdb_structure = get_structure(pdb_path)
        cif_structure = get_structure(cif_path)
        
        assert pdb_structure is not None
        assert cif_structure is not None
        
        # Check both have same number of chains
        pdb_chains = list(pdb_structure.get_chains())
        cif_chains = list(cif_structure.get_chains())
        assert len(pdb_chains) == len(cif_chains)
        
        # Check CA atoms are preserved in both
        pdb_ca = [atom for atom in pdb_structure.get_atoms() if atom.name == "CA"]
        cif_ca = [atom for atom in cif_structure.get_atoms() if atom.name == "CA"]
        
        assert len(pdb_ca) > 0
        assert len(cif_ca) > 0
        # Note: Exact atom counts might differ slightly due to format differences
    
    def test_parser_handles_invalid_files(self):
        """Test parser gracefully handles invalid files."""
        invalid_path = Path("./tests/data/invalid.pdb")
        structure = get_structure(invalid_path)
        
        # Should return None for invalid files
        assert structure is None
    
    def test_parser_handles_empty_files(self):
        """Test parser gracefully handles empty files."""
        empty_path = Path("./tests/data/empty.cif")
        structure = get_structure(empty_path)
        
        assert structure is None


def test_coordinate_precision():
    """Test that atomic coordinates maintain precision."""
    pdb_path = Path("./tests/data/proteins_pdb/1bom.pdb")
    structure = get_structure(pdb_path)
    
    if structure:
        atoms = list(structure.get_atoms())[:10]  # Check first 10 atoms
        for atom in atoms:
            # Coordinates should be float arrays with 3 dimensions
            assert atom.coord.shape == (3,)
            # Check precision is maintained (not rounded to integers)
            assert any(c % 1 != 0 for c in atom.coord)  # At least some decimals
