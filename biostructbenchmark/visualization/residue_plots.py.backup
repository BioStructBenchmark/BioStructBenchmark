"""
biostructbenchmark/visualization/residue_plots.py
Minimal, efficient residue-level visualization for RMSD and analysis data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

from ..core.alignment import ResidueRMSD


class ResidueVisualizer:
    """Efficient residue-level data visualization with minimal resource usage"""
    
    def __init__(self, style: str = 'seaborn-v0_8' if SNS_AVAILABLE else 'default', 
                 figsize: Tuple[int, int] = (12, 8)):
        """Initialize with style and figure size preferences"""
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {'protein': '#2E86AB', 'dna': '#A23B72', 'high_rmsd': '#F18F01', 'low_rmsd': '#C73E1D'}
    
    def plot_rmsd_heatmap(self, residue_data: List[ResidueRMSD], 
                         output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create per-residue RMSD heatmap using minimal memory footprint
        
        Args:
            residue_data: List of ResidueRMSD objects
            output_path: Optional save path
            
        Returns:
            matplotlib Figure object
        """
        if not residue_data:
            raise ValueError("Empty residue data provided")
        
        # Convert to minimal DataFrame view (avoid data duplication)
        df = pd.DataFrame([{
            'chain': r.chain_id, 'pos': r.position, 'rmsd': r.rmsd, 
            'type': r.molecule_type, 'res': r.residue_type
        } for r in residue_data])
        
        # Pivot for heatmap (memory efficient)
        heatmap_data = df.pivot_table(index='chain', columns='pos', values='rmsd', fill_value=0)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Use matplotlib's efficient imshow for large datasets
        im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Minimal labeling for efficiency
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('Chain ID')
        ax.set_title('Per-Residue RMSD Heatmap', fontweight='bold')
        
        # Colorbar with RMSD values
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSD (Å)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_rmsd_distribution(self, residue_data: List[ResidueRMSD], 
                              split_by_type: bool = True) -> plt.Figure:
        """
        Plot RMSD distribution with optional protein/DNA splitting
        
        Args:
            residue_data: Residue RMSD data
            split_by_type: Whether to separate protein and DNA
            
        Returns:
            matplotlib Figure
        """
        rmsds = np.array([r.rmsd for r in residue_data])  # Efficient array view
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if split_by_type and any(r.molecule_type == 'dna' for r in residue_data):
            # Split data efficiently using boolean indexing
            protein_rmsds = [r.rmsd for r in residue_data if r.molecule_type == 'protein']
            dna_rmsds = [r.rmsd for r in residue_data if r.molecule_type == 'dna']
            
            ax.hist([protein_rmsds, dna_rmsds], bins=30, alpha=0.7, 
                   color=[self.colors['protein'], self.colors['dna']], 
                   label=['Protein', 'DNA'])
            ax.legend()
        else:
            ax.hist(rmsds, bins=30, alpha=0.7, color=self.colors['protein'])
        
        ax.set_xlabel('RMSD (Å)')
        ax.set_ylabel('Frequency')
        ax.set_title('RMSD Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add summary stats efficiently
        ax.axvline(np.mean(rmsds), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(rmsds):.2f}Å')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_chain_comparison(self, residue_data: List[ResidueRMSD]) -> plt.Figure:
        """Plot per-chain RMSD comparison with minimal processing"""
        # Group by chain efficiently using dictionary comprehension
        chain_rmsds = {}
        for r in residue_data:
            chain_rmsds.setdefault(r.chain_id, []).append(r.rmsd)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Box plot for chain comparison
        chains, rmsds = zip(*chain_rmsds.items())
        ax.boxplot(rmsds, labels=chains)
        
        ax.set_xlabel('Chain ID')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title('Per-Chain RMSD Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def create_residue_report(residue_data: List[ResidueRMSD], output_dir: Path) -> Dict[str, Path]:
    """
    Generate comprehensive residue analysis report with minimal resource usage
    
    Args:
        residue_data: List of ResidueRMSD objects
        output_dir: Directory for output files
        
    Returns:
        Dictionary mapping plot types to output paths
    """
    output_dir.mkdir(exist_ok=True)
    viz = ResidueVisualizer()
    output_paths = {}
    
    # Generate core plots efficiently (one at a time to minimize memory)
    plots = [
        ('heatmap', lambda: viz.plot_rmsd_heatmap(residue_data)),
        ('distribution', lambda: viz.plot_rmsd_distribution(residue_data)),
        ('chains', lambda: viz.plot_chain_comparison(residue_data))
    ]
    
    for plot_name, plot_func in plots:
        fig = plot_func()
        path = output_dir / f'residue_{plot_name}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Free memory immediately
        output_paths[plot_name] = path
    
    # Generate summary CSV efficiently
    summary_path = output_dir / 'residue_summary.csv'
    pd.DataFrame([{
        'residue_id': r.residue_id, 'chain': r.chain_id, 'position': r.position,
        'rmsd': f'{r.rmsd:.3f}', 'type': r.molecule_type, 'residue': r.residue_type
    } for r in residue_data]).to_csv(summary_path, index=False)
    
    output_paths['summary'] = summary_path
    return output_paths


def quick_residue_plot(residue_data: List[ResidueRMSD], plot_type: str = 'heatmap',
                      output_path: Optional[Path] = None) -> plt.Figure:
    """
    Single-function plotting for quick analysis
    
    Args:
        residue_data: Residue RMSD data
        plot_type: 'heatmap', 'distribution', or 'chains'
        output_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    viz = ResidueVisualizer()
    
    plot_funcs = {
        'heatmap': viz.plot_rmsd_heatmap,
        'distribution': viz.plot_rmsd_distribution, 
        'chains': viz.plot_chain_comparison
    }
    
    if plot_type not in plot_funcs:
        raise ValueError(f"Plot type '{plot_type}' not supported. Use: {list(plot_funcs.keys())}")
    
    fig = plot_funcs[plot_type](residue_data)
    if output_path: fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig
