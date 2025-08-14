"""
biostructbenchmark/visualization/structure.py
Compact open source structure visualization using Py3Dmol and matplotlib
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
from mpl_toolkits.mplot3d import Axes3D

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

from biostructbenchmark.core.io import get_structure
from biostructbenchmark.core.alignment import ResidueRMSD


class StructureVisualizer:
    """Open source structure visualization without PyMOL"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.backend = 'py3dmol' if PY3DMOL_AVAILABLE else 'matplotlib'
        self.colors = {
            'observed': 'green', 'predicted': 'cyan',
            'protein': 'lightblue', 'dna': 'orange'
        }
    
    def visualize_alignment(self, observed_path: Path, predicted_path: Path,
                          rmsd_data: Optional[List[ResidueRMSD]] = None,
                          output_path: Optional[Path] = None) -> Union['py3Dmol.view', plt.Figure]:
        """Visualize structure alignment with optional RMSD coloring"""
        if self.backend == 'py3dmol' and PY3DMOL_AVAILABLE:
            return self._visualize_py3dmol(observed_path, predicted_path, rmsd_data, output_path)
        return self._visualize_matplotlib(observed_path, predicted_path, rmsd_data, output_path)
    
    def _visualize_py3dmol(self, obs_path: Path, pred_path: Path,
                          rmsd_data: Optional[List[ResidueRMSD]], output_path: Optional[Path]):
        """Create interactive 3D visualization using py3Dmol"""
        view = py3Dmol.view(width=self.width, height=self.height)
        
        # Load structures
        with open(obs_path) as f:
            view.addModel(f.read(), 'pdb')  # Model 0
        with open(pred_path) as f:
            view.addModel(f.read(), 'pdb')  # Model 1
        
        # Apply coloring
        if rmsd_data:
            self._apply_rmsd_coloring(view, rmsd_data)
        else:
            self._apply_default_coloring(view)
        
        view.zoomTo()
        view.setBackgroundColor('white')
        
        if output_path:
            self._save_html(view, output_path)
        
        return view
    
    def _apply_default_coloring(self, view):
        """Apply default structure coloring"""
        # Protein residues
        protein_res = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
                      'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                      'THR', 'TRP', 'TYR', 'VAL']
        # DNA residues  
        dna_res = ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']
        
        # Color observed (model 0)
        view.setStyle({'model': 0, 'resn': protein_res},
                     {'cartoon': {'color': self.colors['observed']}})
        view.setStyle({'model': 0, 'resn': dna_res},
                     {'stick': {'color': self.colors['dna']}})
        
        # Color predicted (model 1)
        view.setStyle({'model': 1, 'resn': protein_res},
                     {'cartoon': {'color': self.colors['predicted']}})
        view.setStyle({'model': 1, 'resn': dna_res},
                     {'stick': {'color': '#FFD700'}})
    
    def _apply_rmsd_coloring(self, view, rmsd_data: List[ResidueRMSD]):
        """Color by per-residue RMSD"""
        rmsd_values = [r.rmsd for r in rmsd_data]
        rmsd_min, rmsd_max = min(rmsd_values), max(rmsd_values)
        
        for res in rmsd_data:
            # Map RMSD to color (blue->yellow->red)
            norm_rmsd = (res.rmsd - rmsd_min) / (rmsd_max - rmsd_min) if rmsd_max > rmsd_min else 0
            if norm_rmsd < 0.5:
                color = f"#{int(norm_rmsd*2*255):02x}{int(norm_rmsd*2*255):02x}{int(255*(1-norm_rmsd*2)):02x}"
            else:
                color = f"#ff{int(255*(2-norm_rmsd*2)):02x}00"
            
            selection = {'chain': res.chain_id, 'resi': res.position}
            style = {'cartoon' if res.molecule_type == 'protein' else 'stick': {'color': color}}
            view.setStyle(selection, style)
    
    def _save_html(self, view, output_path: Path):
        """Save as standalone HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Structure Alignment</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f0f0f0; }}
        #viewer {{ width: {self.width}px; height: {self.height}px; margin: auto; border: 2px solid #ccc; }}
        .controls {{ text-align: center; margin-top: 10px; }}
        button {{ margin: 5px; padding: 10px 20px; cursor: pointer; }}
    </style>
</head>
<body>
    <h2 style="text-align:center">Structure Alignment Visualization</h2>
    <div id="viewer"></div>
    <div class="controls">
        <button onclick="viewer.spin('y')">Spin</button>
        <button onclick="viewer.spin(false)">Stop</button>
        <button onclick="viewer.zoomTo()">Reset</button>
    </div>
    <script>
        let viewer = $3Dmol.createViewer("viewer");
        {view.js()}
    </script>
</body>
</html>"""
        output_path.write_text(html)
        print(f"Saved interactive visualization to: {output_path}")
    
    def _visualize_matplotlib(self, obs_path: Path, pred_path: Path,
                            rmsd_data: Optional[List[ResidueRMSD]], output_path: Optional[Path]):
        """Create static visualization using matplotlib"""
        obs_struct = get_structure(obs_path)
        pred_struct = get_structure(pred_path)
        
        if not obs_struct or not pred_struct:
            raise ValueError("Could not load structures")
        
        fig = plt.figure(figsize=(14, 6))
        
        # 3D structure plot
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_backbone(ax1, obs_struct, pred_struct)
        
        # RMSD plot
        if rmsd_data:
            ax2 = fig.add_subplot(122)
            self._plot_rmsd(ax2, rmsd_data)
        
        plt.suptitle('Structure Alignment Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved static visualization to: {output_path}")
        
        return fig
    
    def _plot_backbone(self, ax, obs_struct, pred_struct):
        """Plot backbone traces"""
        for struct, color, label in [(obs_struct, self.colors['observed'], 'Observed'),
                                     (pred_struct, self.colors['predicted'], 'Predicted')]:
            coords = []
            for atom in struct.get_atoms():
                if atom.name in ['CA', 'P']:  # Backbone atoms
                    coords.append(atom.coord)
            
            if coords:
                coords = np.array(coords)
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                       color=color, label=label, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('3D Structure Alignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rmsd(self, ax, rmsd_data: List[ResidueRMSD]):
        """Plot per-residue RMSD"""
        protein = [r for r in rmsd_data if r.molecule_type == 'protein']
        dna = [r for r in rmsd_data if r.molecule_type == 'dna']
        
        for data, color, label in [(protein, 'blue', 'Protein'), (dna, 'orange', 'DNA')]:
            if data:
                positions = [r.position for r in data]
                rmsds = [r.rmsd for r in data]
                ax.scatter(positions, rmsds, alpha=0.6, color=color, label=label, s=30)
        
        ax.axhline(y=2.0, color='yellow', linestyle='--', alpha=0.5, label='2Å')
        ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='4Å')
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title('Per-Residue RMSD')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_aligned_structures(self, observed, predicted, output_dir: Path):
        """Save aligned structures to PDB files"""
        from Bio.PDB import PDBIO
        io = PDBIO()
        
        output_dir.mkdir(exist_ok=True)
        
        io.set_structure(observed)
        io.save(str(output_dir / "observed_reference.pdb"))
        
        io.set_structure(predicted)
        io.save(str(output_dir / "predicted_aligned.pdb"))
        
        print(f"Saved aligned structures to: {output_dir}")
    
    def create_report(self, obs_path: Path, pred_path: Path,
                     rmsd_data: List[ResidueRMSD], output_dir: Path):
        """Generate comprehensive comparison report"""
        output_dir.mkdir(exist_ok=True)
        
        # Interactive visualization
        if PY3DMOL_AVAILABLE:
            self.visualize_alignment(obs_path, pred_path, rmsd_data,
                                    output_dir / "alignment.html")
        
        # Static visualization  
        self.visualize_alignment(obs_path, pred_path, rmsd_data,
                               output_dir / "alignment.png")
        
        # Summary statistics
        if rmsd_data:
            self._save_summary(rmsd_data, output_dir / "summary.json")
    
    def _save_summary(self, rmsd_data: list, output_path: Path):
        """Save summary statistics as JSON"""
        import json
        import numpy as np
        
        # Handle empty data case
        if not rmsd_data:
            summary = {
                "overall": {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0
                },
                "protein": {"mean": None, "count": 0},
                "dna": {"mean": None, "count": 0},
                "worst_residues": []
            }
        else:
            # Extract RMSD values
            rmsds = [r.rmsd for r in rmsd_data]
            protein = [r.rmsd for r in rmsd_data if r.molecule_type == 'protein']
            dna = [r.rmsd for r in rmsd_data if r.molecule_type == 'dna']
            
            summary = {
                "overall": {
                    "mean": float(np.mean(rmsds)) if rmsds else None,
                    "std": float(np.std(rmsds)) if rmsds else None,
                    "min": float(np.min(rmsds)) if rmsds else None,
                    "max": float(np.max(rmsds)) if rmsds else None,
                    "count": len(rmsds)
                },
                "protein": {
                    "mean": float(np.mean(protein)) if protein else None,
                    "count": len(protein)
                },
                "dna": {
                    "mean": float(np.mean(dna)) if dna else None,
                    "count": len(dna)
                },
                "worst_residues": [
                    {"id": r.residue_id, "rmsd": r.rmsd}
                    for r in sorted(rmsd_data, key=lambda x: x.rmsd, reverse=True)[:5]
                ]
            }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

#Fix for failed import bug in __init__.py in visualization.

def create_structure_visualization(width: int = 800, height: int = 600):
    """
    Factory function to create a StructureVisualizer instance
    
    Args:
        width: Width of visualization window
        height: Height of visualization window
        
    Returns:
        StructureVisualizer instance
    """
    return StructureVisualizer(width=width, height=height)
