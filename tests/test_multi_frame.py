"""
test_multi_frame_alignment.py
Test script for the enhanced multi-frame alignment functionality
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from biostructbenchmark.core.alignment import (
    perform_multi_frame_alignment,
    align_structures_by_reference_frame,
    get_atoms_by_molecule_type,
    calculate_center_of_mass,
    is_protein_residue,
    is_dna_residue,
    ResidueRMSD,
    AlignmentResult,
    MultiFrameAlignmentResult
)


class TestMultiFrameAlignment:
    """Test suite for multi-frame alignment functionality"""
    
    def test_multi_frame_alignment_result_dataclass(self):
        """Test MultiFrameAlignmentResult dataclass"""
        # Create mock alignment results
        mock_result = AlignmentResult(
            overall_rmsd=2.5,
            residue_rmsds=[],
            transformation_matrix=np.eye(4),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            aligned_atom_count=100,
            reference_frame='full'
        )
        
        multi_result = MultiFrameAlignmentResult(
            full_structure=mock_result,
            dna_to_protein=mock_result,
            dna_to_dna=mock_result
        )
        
        summary = multi_result.get_summary()
        assert summary['full_structure_rmsd'] == 2.5
        assert summary['dna_positioning_rmsd'] == 2.5
        assert summary['dna_standalone_rmsd'] == 2.5
        assert summary['full_atom_count'] == 100
    
    def test_molecule_selector(self):
        """Test MoleculeSelector for filtering residues"""
        from biostructbenchmark.core.alignment import MoleculeSelector
        
        # Test protein selector
        protein_selector = MoleculeSelector('protein')
        
        mock_protein_res = Mock()
        mock_protein_res.get_resname.return_value = 'ALA'
        assert protein_selector.accept_residue(mock_protein_res) == True
        
        mock_dna_res = Mock()
        mock_dna_res.get_resname.return_value = 'DA'
        assert protein_selector.accept_residue(mock_dna_res) == False
        
        # Test DNA selector
        dna_selector = MoleculeSelector('dna')
        assert dna_selector.accept_residue(mock_protein_res) == False
        assert dna_selector.accept_residue(mock_dna_res) == True
        
        # Test all selector
        all_selector = MoleculeSelector('all')
        assert all_selector.accept_residue(mock_protein_res) == True
        assert all_selector.accept_residue(mock_dna_res) == True
    
    def test_get_atoms_by_molecule_type(self):
        """Test atom extraction by molecule type"""
        # Create mock structure
        mock_structure = Mock()
        mock_model = Mock()
        mock_chain = Mock()
        
        # Create mixed residues
        residues = []
        
        # Add protein residues
        for i in range(3):
            res = Mock()
            res.get_resname.return_value = 'ALA'
            ca_atom = Mock()
            ca_atom.get_coord.return_value = np.array([i, 0, 0])
            res.__contains__ = lambda self, x: x == 'CA'
            res.__getitem__ = lambda self, x: ca_atom if x == 'CA' else None
            residues.append(res)
        
        # Add DNA residues
        for i in range(2):
            res = Mock()
            res.get_resname.return_value = 'DA'
            p_atom = Mock()
            p_atom.get_coord.return_value = np.array([i+3, 0, 0])
            res.__contains__ = lambda self, x: x == 'P'
            res.__getitem__ = lambda self, x: p_atom if x == 'P' else None
            residues.append(res)
        
        mock_chain.__iter__ = lambda self: iter(residues)
        mock_model.__iter__ = lambda self: iter([mock_chain])
        mock_model.__getitem__ = lambda self, x: mock_chain
        mock_structure.__getitem__ = lambda self, x: mock_model if x == 0 else None
        
        # Test full extraction
        atoms = get_atoms_by_molecule_type(mock_structure, 'full')
        assert len(atoms) == 5  # 3 CA + 2 P
        
        # Test protein only
        atoms = get_atoms_by_molecule_type(mock_structure, 'protein')
        assert len(atoms) == 3
        
        # Test DNA only
        atoms = get_atoms_by_molecule_type(mock_structure, 'dna')
        assert len(atoms) == 2
    
    def test_calculate_center_of_mass(self):
        """Test center of mass calculation"""
        # Create mock atoms with known coordinates
        atoms = []
        coords = [[0, 0, 0], [2, 0, 0], [1, 3, 0]]
        
        for coord in coords:
            atom = Mock()
            atom.get_coord.return_value = np.array(coord)
            atoms.append(atom)
        
        com = calculate_center_of_mass(atoms)
        expected = np.array([1, 1, 0])  # Mean of coordinates
        np.testing.assert_array_almost_equal(com, expected)
        
        # Test empty list
        com = calculate_center_of_mass([])
        np.testing.assert_array_equal(com, np.zeros(3))
    
    @patch('biostructbenchmark.core.alignment.get_structure')
    @patch('biostructbenchmark.core.alignment.save_aligned_structure')
    def test_perform_multi_frame_alignment(self, mock_save, mock_get_structure):
        """Test complete multi-frame alignment workflow"""
        # Create mock structures
        mock_obs = self._create_mock_structure()
        mock_pred = self._create_mock_structure()
        mock_get_structure.return_value = mock_obs
        
        # Mock the alignment functions to avoid complex setup
        with patch('biostructbenchmark.core.alignment.align_structures_by_reference_frame') as mock_align:
            mock_result = AlignmentResult(
                overall_rmsd=2.0,
                residue_rmsds=[],
                transformation_matrix=np.eye(4),
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros(3),
                aligned_atom_count=50,
                reference_frame='test'
            )
            mock_align.return_value = mock_result
            
            # Run multi-frame alignment
            result = perform_multi_frame_alignment(
                Path('test_obs.pdb'),
                Path('test_pred.pdb'),
                Path('output_dir')
            )
            
            # Check that all three alignments were performed
            assert mock_align.call_count == 3
            
            # Check result structure
            assert isinstance(result, MultiFrameAlignmentResult)
            assert result.full_structure.overall_rmsd == 2.0
            assert result.dna_to_protein.overall_rmsd == 2.0
            assert result.dna_to_dna.overall_rmsd == 2.0
            
            # Check that structures were saved
            assert mock_save.call_count == 4  # 3 aligned + 1 reference
    
    def test_align_structures_by_reference_frame(self):
        """Test alignment with different reference frames"""
        # This test would require more complex mocking of Bio.PDB structures
        # For now, just test that the function handles different parameters
        
        mock_obs = self._create_mock_structure()
        mock_pred = self._create_mock_structure()
        
        with patch('biostructbenchmark.core.alignment.get_atoms_by_molecule_type') as mock_get_atoms:
            with patch('biostructbenchmark.core.alignment.Superimposer') as mock_super:
                # Mock atom lists
                mock_atoms = [Mock() for _ in range(10)]
                for atom in mock_atoms:
                    atom.get_coord.return_value = np.random.rand(3)
                mock_get_atoms.return_value = mock_atoms
                
                # Mock superimposer
                mock_super_instance = Mock()
                mock_super_instance.rms = 2.5
                mock_super_instance.rotran = (np.eye(3), np.zeros(3))
                mock_super.return_value = mock_super_instance
                
                # Test different reference frames
                for ref_frame in ['full', 'protein', 'dna']:
                    for align_subset in ['full', 'protein', 'dna']:
                        result = align_structures_by_reference_frame(
                            mock_obs, mock_pred,
                            reference_frame=ref_frame,
                            align_subset=align_subset
                        )
                        
                        assert isinstance(result, AlignmentResult)
                        assert result.reference_frame == f"{ref_frame}_to_{align_subset}"
    
    def _create_mock_structure(self):
        """Helper to create a mock Bio.PDB structure"""
        mock_structure = Mock()
        mock_model = Mock()
        mock_chain = Mock()
        
        # Add some mock residues
        residues = []
        for i in range(5):
            res = Mock()
            res.get_resname.return_value = 'ALA' if i < 3 else 'DA'
            res.get_id.return_value = ('', i, '')
            residues.append(res)
        
        mock_chain.__iter__ = lambda self: iter(residues)
        mock_chain.get_id.return_value = 'A'
        
        mock_model.__iter__ = lambda self: iter([mock_chain])
        mock_model.get_atoms.return_value = []
        mock_model.__getitem__ = lambda self, x: mock_chain
        
        mock_structure.__getitem__ = lambda self, x: mock_model if x == 0 else None
        
        return mock_structure


class TestIntegrationWithCLI:
    """Test integration of multi-frame alignment with CLI"""
    
    def test_cli_reference_frame_option(self):
        """Test that CLI properly handles reference frame options"""
        from biostructbenchmark.cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Test default reference frame
        args = parser.parse_args(['-e', 'exp.pdb', '-p', 'pred.pdb'])
        assert args.reference_frame == 'full'
        
        # Test specific reference frames
        for frame in ['full', 'protein', 'dna']:
            args = parser.parse_args([
                '-e', 'exp.pdb', 
                '-p', 'pred.pdb',
                '--reference-frame', frame
            ])
            assert args.reference_frame == frame
    
    @patch('biostructbenchmark.core.alignment.perform_multi_frame_alignment')
    def test_cli_calls_multi_frame_alignment(self, mock_multi_align):
        """Test that CLI calls the multi-frame alignment when appropriate"""
        # This would test that the main CLI function properly calls
        # perform_multi_frame_alignment when --all-benchmarks is specified
        pass  # Implementation depends on CLI structure


def test_backwards_compatibility():
    """Test that legacy functions still work"""
    from biostructbenchmark.core.alignment import compare_structures, align_structures
    
    with patch('biostructbenchmark.core.alignment.get_structure') as mock_get:
        with patch('biostructbenchmark.core.alignment.align_structures_by_reference_frame') as mock_align:
            mock_result = AlignmentResult(
                overall_rmsd=2.5,
                residue_rmsds=[],
                transformation_matrix=np.eye(4),
                rotation_matrix=np.eye(3),
                translation_vector=np.array([1, 2, 3]),
                aligned_atom_count=100,
                reference_frame='full'
            )
            mock_align.return_value = mock_result
            
            # Test compare_structures
            result = compare_structures(Path('obs.pdb'), Path('pred.pdb'))
            assert result is not None
            assert result.overall_rmsd == 2.5
            
            # Test align_structures
            mock_obs = Mock()
            mock_pred = Mock()
            legacy_result = align_structures(mock_obs, mock_pred)
            
            assert legacy_result['rmsd'] == 2.5
            assert legacy_result['atom_count'] == 100
            np.testing.assert_array_equal(legacy_result['rotation'], np.eye(3))
            np.testing.assert_array_equal(legacy_result['translation'], np.array([1, 2, 3]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])