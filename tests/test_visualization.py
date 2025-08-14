"""Tests for visualization module"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from biostructbenchmark.visualization.structure import StructureVisualizer
from biostructbenchmark.core.alignment import ResidueRMSD


class TestStructureVisualizer:
    """Test StructureVisualizer class"""
    
    def test_visualizer_initialization(self):
        """Test StructureVisualizer initialization"""
        viz = StructureVisualizer()
        
        assert viz.width == 800
        assert viz.height == 600
        assert viz.backend in ['py3dmol', 'matplotlib']
        assert 'observed' in viz.colors
        assert 'predicted' in viz.colors
    
    def test_visualizer_custom_size(self):
        """Test custom size initialization"""
        viz = StructureVisualizer(width=1024, height=768)
        
        assert viz.width == 1024
        assert viz.height == 768
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', False)
    def test_backend_fallback_to_matplotlib(self):
        """Test fallback to matplotlib when py3dmol not available"""
        viz = StructureVisualizer()
        assert viz.backend == 'matplotlib'
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', True)
    def test_backend_prefers_py3dmol(self):
        """Test py3dmol is preferred when available"""
        viz = StructureVisualizer()
        assert viz.backend == 'py3dmol'


class TestVisualizationMethods:
    """Test visualization methods"""
    
    def test_save_summary(self, tmp_path):
        """Test summary JSON generation"""
        viz = StructureVisualizer()
        
        rmsd_data = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 3.0, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 4.0, 10, "dna")
        ]
        
        output_file = tmp_path / "summary.json"
        viz._save_summary(rmsd_data, output_file)
        
        assert output_file.exists()
        
        # Load and verify JSON
        with open(output_file) as f:
            summary = json.load(f)
        
        assert summary['overall']['count'] == 3
        assert abs(summary['overall']['mean'] - 3.0) < 0.01
        assert summary['protein']['count'] == 2
        assert summary['dna']['count'] == 1
        assert len(summary['worst_residues']) <= 5
    
    def test_save_summary_empty_data(self, tmp_path):
        """Test summary generation with empty data"""
        viz = StructureVisualizer()
        
        output_file = tmp_path / "empty_summary.json"
        viz._save_summary([], output_file)
        
        assert output_file.exists()
        
        with open(output_file) as f:
            summary = json.load(f)
        
        assert summary['overall']['count'] == 0


class TestMatplotlibVisualization:
    """Test matplotlib visualization backend"""
    
    @patch('biostructbenchmark.visualization.structure.get_structure')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_matplotlib(self, mock_savefig, mock_get_structure, tmp_path):
        """Test matplotlib visualization"""
        # Create mock structure
        mock_struct = MagicMock()
        mock_atom = MagicMock()
        mock_atom.name = 'CA'
        mock_atom.coord = np.array([1.0, 2.0, 3.0])
        
        # Setup structure iteration
        mock_struct.get_atoms.return_value = [mock_atom]
        mock_get_structure.return_value = mock_struct
        
        viz = StructureVisualizer()
        
        # Force matplotlib backend
        output_file = tmp_path / "test.png"
        fig = viz._visualize_matplotlib(
            Path("obs.pdb"), 
            Path("pred.pdb"),
            None,
            output_file
        )
        
        assert fig is not None
        # Check that figure was created with correct structure
        assert len(fig.axes) > 0
    
    @patch('biostructbenchmark.visualization.structure.get_structure')
    def test_visualize_matplotlib_with_rmsd(self, mock_get_structure, tmp_path):
        """Test matplotlib visualization with RMSD data"""
        mock_struct = MagicMock()
        mock_atom = MagicMock()
        mock_atom.name = 'CA'
        mock_atom.coord = np.array([1.0, 2.0, 3.0])
        mock_struct.get_atoms.return_value = [mock_atom]
        mock_get_structure.return_value = mock_struct
        
        viz = StructureVisualizer()
        
        rmsd_data = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("DG_5", "DG", "B", 5, 4.0, 10, "dna")
        ]
        
        output_file = tmp_path / "test_rmsd.png"
        fig = viz._visualize_matplotlib(
            Path("obs.pdb"),
            Path("pred.pdb"),
            rmsd_data,
            output_file
        )
        
        assert fig is not None
        # Should have 2 subplots (3D structure and RMSD plot)
        assert len(fig.axes) == 2
    
    @patch('biostructbenchmark.visualization.structure.get_structure')
    def test_visualize_matplotlib_invalid_structure(self, mock_get_structure):
        """Test handling of invalid structures"""
        mock_get_structure.return_value = None
        
        viz = StructureVisualizer()
        
        with pytest.raises(ValueError, match="Could not load structures"):
            viz._visualize_matplotlib(
                Path("invalid.pdb"),
                Path("invalid2.pdb"),
                None,
                None
            )


class TestPy3DmolVisualization:
    """Test py3Dmol visualization backend"""
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', True)
    @patch('biostructbenchmark.visualization.structure.py3Dmol')
    def test_visualize_py3dmol(self, mock_py3dmol, tmp_path):
        """Test py3Dmol visualization"""
        # Create mock view
        mock_view = MagicMock()
        mock_py3dmol.view.return_value = mock_view
        
        # Create test PDB files
        pdb_content = "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C\n"
        obs_file = tmp_path / "obs.pdb"
        pred_file = tmp_path / "pred.pdb"
        obs_file.write_text(pdb_content)
        pred_file.write_text(pdb_content)
        
        viz = StructureVisualizer()
        
        view = viz._visualize_py3dmol(obs_file, pred_file, None, None)
        
        assert view == mock_view
        # Check that structures were added
        assert mock_view.addModel.call_count == 2
        mock_view.zoomTo.assert_called_once()
        mock_view.setBackgroundColor.assert_called_once_with('white')
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', True)
    @patch('biostructbenchmark.visualization.structure.py3Dmol')
    def test_visualize_py3dmol_with_rmsd(self, mock_py3dmol, tmp_path):
        """Test py3Dmol visualization with RMSD coloring"""
        mock_view = MagicMock()
        mock_py3dmol.view.return_value = mock_view
        
        pdb_content = "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C\n"
        obs_file = tmp_path / "obs.pdb"
        pred_file = tmp_path / "pred.pdb"
        obs_file.write_text(pdb_content)
        pred_file.write_text(pdb_content)
        
        rmsd_data = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein"),
            ResidueRMSD("GLY_11", "GLY", "A", 11, 4.0, 5, "protein")
        ]
        
        viz = StructureVisualizer()
        view = viz._visualize_py3dmol(obs_file, pred_file, rmsd_data, None)
        
        assert view == mock_view
        # Check that setStyle was called for RMSD coloring
        assert mock_view.setStyle.called
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', True)
    @patch('biostructbenchmark.visualization.structure.py3Dmol')
    def test_save_html(self, mock_py3dmol, tmp_path):
        """Test HTML output generation"""
        mock_view = MagicMock()
        mock_view.js.return_value = "// JavaScript code"
        mock_py3dmol.view.return_value = mock_view
        
        pdb_content = "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C\n"
        obs_file = tmp_path / "obs.pdb"
        pred_file = tmp_path / "pred.pdb"
        obs_file.write_text(pdb_content)
        pred_file.write_text(pdb_content)
        
        output_html = tmp_path / "test.html"
        
        viz = StructureVisualizer()
        viz._visualize_py3dmol(obs_file, pred_file, None, output_html)
        
        assert output_html.exists()
        
        # Check HTML content
        html_content = output_html.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Structure Alignment" in html_content
        assert "3Dmol" in html_content
        assert "// JavaScript code" in html_content


class TestReportGeneration:
    """Test comprehensive report generation"""
    
    @patch('biostructbenchmark.visualization.structure.PY3DMOL_AVAILABLE', False)
    @patch('biostructbenchmark.visualization.structure.get_structure')
    def test_create_report(self, mock_get_structure, tmp_path):
        """Test complete report generation"""
        # Mock structure
        mock_struct = MagicMock()
        mock_atom = MagicMock()
        mock_atom.name = 'CA'
        mock_atom.coord = np.array([1.0, 2.0, 3.0])
        mock_struct.get_atoms.return_value = [mock_atom]
        mock_get_structure.return_value = mock_struct
        
        viz = StructureVisualizer()
        
        obs_file = tmp_path / "obs.pdb"
        pred_file = tmp_path / "pred.pdb"
        obs_file.write_text("HEADER TEST\n")
        pred_file.write_text("HEADER TEST\n")
        
        rmsd_data = [
            ResidueRMSD("ALA_10", "ALA", "A", 10, 2.0, 5, "protein")
        ]
        
        output_dir = tmp_path / "report"
        viz.create_report(obs_file, pred_file, rmsd_data, output_dir)
        
        assert output_dir.exists()
        assert (output_dir / "alignment.png").exists()
        assert (output_dir / "summary.json").exists()


class TestSaveAlignedStructures:
    """Test saving aligned structures"""
    
    @patch('Bio.PDB.PDBIO')
    def test_save_aligned_structures(self, mock_pdbio_class, tmp_path):
        """Test saving aligned structures to PDB files"""
        mock_io = MagicMock()
        mock_pdbio_class.return_value = mock_io
        
        # Create mock structures
        mock_observed = MagicMock()
        mock_predicted = MagicMock()
        
        viz = StructureVisualizer()
        
        output_dir = tmp_path / "aligned"
        viz.save_aligned_structures(mock_observed, mock_predicted, output_dir)
        
        assert output_dir.exists()
        
        # Check that structures were saved
        assert mock_io.set_structure.call_count == 2
        assert mock_io.save.call_count == 2
        
        # Check save paths
        save_calls = mock_io.save.call_args_list
        assert "observed_reference.pdb" in save_calls[0][0][0]
        assert "predicted_aligned.pdb" in save_calls[1][0][0]
