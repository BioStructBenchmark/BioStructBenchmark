"""
Tests for structural analysis functionality
"""

import numpy as np
import pytest
from biostructbenchmark.core.structural import (
    calculate_orientation_error,
    calculate_per_residue_rmsd,
    superimpose_structures,
)


class TestSuperimposeStructures:
    """Test structure superimposition functionality"""

    def test_superimpose_identical_coordinates(self):
        """Test superimposition of identical coordinate sets"""
        # Create identical coordinate arrays
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        rmsd, rotation, translation = superimpose_structures(coords, coords)

        # RMSD should be zero (or very close)
        assert rmsd < 1e-10

        # Rotation should be identity matrix (or very close)
        np.testing.assert_array_almost_equal(rotation, np.eye(3), decimal=10)

        # Translation should be zero vector (or very close)
        np.testing.assert_array_almost_equal(translation, np.zeros(3), decimal=10)

    def test_superimpose_translated_coordinates(self):
        """Test superimposition with pure translation"""
        # Create coordinates and translate one set
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        translation_vector = np.array([5.0, 3.0, 2.0])
        coords2 = coords1 + translation_vector

        rmsd, rotation, translation = superimpose_structures(coords1, coords2)

        # RMSD should be zero after superimposition
        assert rmsd < 1e-10

        # Rotation should be identity (no rotation needed)
        np.testing.assert_array_almost_equal(rotation, np.eye(3), decimal=6)

        # Translation should recover the original translation
        np.testing.assert_array_almost_equal(translation, -translation_vector, decimal=6)

    def test_superimpose_different_coordinates(self):
        """Test superimposition with genuinely different coordinates"""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        coords2 = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]])

        rmsd, rotation, translation = superimpose_structures(coords1, coords2)

        # RMSD should be positive
        assert rmsd > 0

        # Transformation matrices should be reasonable
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)

        # Check that rotation matrix is orthogonal
        should_be_identity = np.dot(rotation, rotation.T)
        np.testing.assert_array_almost_equal(should_be_identity, np.eye(3), decimal=6)

    def test_superimpose_insufficient_points(self):
        """Test superimposition with insufficient points"""
        # Need at least 3 points for meaningful superimposition
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        rmsd, rotation, translation = superimpose_structures(coords1, coords2)

        # Should still work but may not be as meaningful
        assert rmsd >= 0
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)


class TestPerResidueRMSD:
    """Test per-residue RMSD calculation"""

    def test_per_residue_rmsd_identical_atoms(self):
        """Test RMSD calculation with identical atom sets"""
        coords = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

        exp_atoms = {"RES1": coords, "RES2": coords}
        comp_atoms = {"RES1": coords, "RES2": coords}
        mapping = {"RES1": "RES1", "RES2": "RES2"}

        # Identity transformation
        rotation = np.eye(3)
        translation = np.zeros(3)

        rmsd_dict = calculate_per_residue_rmsd(
            exp_atoms, comp_atoms, mapping, rotation, translation
        )

        assert len(rmsd_dict) == 2
        assert rmsd_dict["RES1"] < 1e-10  # Should be essentially zero
        assert rmsd_dict["RES2"] < 1e-10

    def test_per_residue_rmsd_with_transformation(self):
        """Test RMSD calculation with coordinate transformation"""
        exp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        comp_coords = [[2.0, 3.0, 4.0], [3.0, 3.0, 4.0]]  # Different coordinates

        exp_atoms = {"RES1": exp_coords}
        comp_atoms = {"RES1": comp_coords}
        mapping = {"RES1": "RES1"}

        # Test with identity transformation (no change to coordinates)
        identity_rotation = np.eye(3)
        zero_translation = np.zeros(3)

        rmsd_dict = calculate_per_residue_rmsd(
            exp_atoms, comp_atoms, mapping, identity_rotation, zero_translation
        )

        # Should get some positive RMSD value
        assert rmsd_dict["RES1"] > 0

    def test_per_residue_rmsd_mismatched_atoms(self):
        """Test RMSD calculation with different number of atoms"""
        exp_atoms = {"RES1": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}  # 2 atoms
        comp_atoms = {"RES1": [[0.0, 0.0, 0.0]]}  # 1 atom
        mapping = {"RES1": "RES1"}

        rotation = np.eye(3)
        translation = np.zeros(3)

        rmsd_dict = calculate_per_residue_rmsd(
            exp_atoms, comp_atoms, mapping, rotation, translation
        )

        # Should skip residues with mismatched atom counts
        assert len(rmsd_dict) == 0

    def test_per_residue_rmsd_missing_residues(self):
        """Test RMSD calculation with missing residues in mapping"""
        exp_atoms = {"RES1": [[0.0, 0.0, 0.0]], "RES2": [[1.0, 1.0, 1.0]]}
        comp_atoms = {"RES1": [[0.0, 0.0, 0.0]]}  # Missing RES2
        mapping = {"RES1": "RES1", "RES2": "RES_MISSING"}

        rotation = np.eye(3)
        translation = np.zeros(3)

        rmsd_dict = calculate_per_residue_rmsd(
            exp_atoms, comp_atoms, mapping, rotation, translation
        )

        # Should only include residues present in both structures
        assert len(rmsd_dict) == 1
        assert "RES1" in rmsd_dict
        assert "RES2" not in rmsd_dict


class TestOrientationError:
    """Test orientation error calculation"""

    def test_orientation_error_identity_matrix(self):
        """Test orientation error with identity matrix (no rotation)"""
        identity_matrix = np.eye(3)
        error = calculate_orientation_error(identity_matrix)

        # No rotation should give zero error
        assert abs(error) < 1e-10

    def test_orientation_error_90_degree_rotation(self):
        """Test orientation error with known rotation"""
        # 90-degree rotation around Z-axis
        rotation_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        error = calculate_orientation_error(rotation_90z)

        # Should be close to 90 degrees
        assert abs(error - 90.0) < 1.0

    def test_orientation_error_180_degree_rotation(self):
        """Test orientation error with 180-degree rotation"""
        # 180-degree rotation around Z-axis
        rotation_180z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        error = calculate_orientation_error(rotation_180z)

        # Should be close to 180 degrees
        assert abs(error - 180.0) < 1.0

    def test_orientation_error_bounds(self):
        """Test that orientation error is within expected bounds"""
        # Test with random orthogonal matrix
        angles = [0, 30, 60, 90, 120, 150, 180]

        for angle in angles:
            rad = np.radians(angle)
            rotation = np.array(
                [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
            )

            error = calculate_orientation_error(rotation)

            # Error should be between 0 and 180 degrees
            assert 0 <= error <= 180
            # Should be close to expected angle
            assert abs(error - angle) < 1.0


class TestStructuralIntegration:
    """Integration tests using real structures"""

    @pytest.mark.integration
    def test_superimposition_with_real_structures(self):
        """Test superimposition with coordinates from real structures"""
        from pathlib import Path

        from biostructbenchmark.core.io import get_structure

        # Load real structures
        exp_structure = get_structure(Path("tests/data/complexes/experimental_9ny8.cif"))
        comp_structure = get_structure(Path("tests/data/complexes/predicted_9ny8.cif"))

        # Extract some coordinates (just first few atoms from chain A)
        exp_coords = []
        comp_coords = []

        exp_chain = list(exp_structure[0])[0]  # First chain
        comp_chain = list(comp_structure[0])[0]  # First chain

        for residue in list(exp_chain)[:5]:  # First 5 residues
            for atom in residue:
                if atom.name == "CA":  # Just CA atoms (GEMMI uses atom.name)
                    exp_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    break

        for residue in list(comp_chain)[:5]:  # First 5 residues
            for atom in residue:
                if atom.name == "CA":  # Just CA atoms (GEMMI uses atom.name)
                    comp_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    break

        if len(exp_coords) >= 3 and len(comp_coords) >= 3:
            exp_coords = np.array(exp_coords[:3])
            comp_coords = np.array(comp_coords[:3])

            rmsd, rotation, translation = superimpose_structures(exp_coords, comp_coords)

            # Should get reasonable results
            assert rmsd >= 0
            assert rotation.shape == (3, 3)
            assert translation.shape == (3,)

            # Rotation matrix should be orthogonal
            should_be_identity = np.dot(rotation, rotation.T)
            np.testing.assert_array_almost_equal(should_be_identity, np.eye(3), decimal=6)


if __name__ == "__main__":
    pytest.main([__file__])
