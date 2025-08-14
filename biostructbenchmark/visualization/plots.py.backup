"""
biostructbenchmark/visualization/plots.py
Lightweight general plotting utilities for cross-module visualization and publication outputs
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Any

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

# Import data structures from analysis modules
from ..core.alignment import ResidueRMSD


class PublicationPlotter:
    """Memory-efficient plotting for publication-quality figures across all analysis modules"""
    
    def __init__(self, style: str = 'seaborn-v0_8' if SNS_AVAILABLE else 'default'):
        """Initialize with publication-ready styling"""
        plt.style.use(style)
        self.fig_params = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
        self.colors = {
            'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01',
            'protein': '#4CAF50', 'dna': '#FF9800', 'error': '#F44336'
        }
    
    def summary_dashboard(self, data_dict: Dict[str, Any], output_path: Path) -> plt.Figure:
        """
        Create comprehensive analysis dashboard from multiple data sources
        
        Args:
            data_dict: Dictionary with keys 'rmsd', 'bfactor', 'consensus', 'mutations'
            output_path: Save path for dashboard
            
        Returns:
            matplotlib Figure with 2x2 subplot layout
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BioStructBenchmark Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # RMSD overview (top-left)
        if 'rmsd' in data_dict and data_dict['rmsd']:
            self._plot_rmsd_summary(axes[0,0], data_dict['rmsd'])
        else:
            axes[0,0].text(0.5, 0.5, 'No RMSD Data', ha='center', va='center', transform=axes[0,0].transAxes)
        
        # B-factor correlation (top-right) 
        if 'bfactor' in data_dict and data_dict['bfactor']:
            self._plot_bfactor_correlation(axes[0,1], data_dict['bfactor'])
        else:
            axes[0,1].text(0.5, 0.5, 'No B-factor Data', ha='center', va='center', transform=axes[0,1].transAxes)
        
        # Consensus errors (bottom-left)
        if 'consensus' in data_dict and data_dict['consensus']:
            self._plot_consensus_summary(axes[1,0], data_dict['consensus'])
        else:
            axes[1,0].text(0.5, 0.5, 'No Consensus Data', ha='center', va='center', transform=axes[1,0].transAxes)
        
        # Mutation impact (bottom-right)
        if 'mutations' in data_dict and data_dict['mutations']:
            self._plot_mutation_impact(axes[1,1], data_dict['mutations'])
        else:
            axes[1,1].text(0.5, 0.5, 'No Mutation Data', ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        fig.savefig(output_path, **self.fig_params)
        return fig
    
    def _plot_rmsd_summary(self, ax: plt.Axes, rmsd_data: List[ResidueRMSD]) -> None:
        """Efficient RMSD summary plot"""
        # Group data efficiently
        protein_rmsds = [r.rmsd for r in rmsd_data if r.molecule_type == 'protein']
        dna_rmsds = [r.rmsd for r in rmsd_data if r.molecule_type == 'dna']
        
        # Violin plot for compact visualization
        data_to_plot = [protein_rmsds] if protein_rmsds else []
        labels = ['Protein'] if protein_rmsds else []
        
        if dna_rmsds:
            data_to_plot.append(dna_rmsds)
            labels.append('DNA')
        
        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(1, len(data_to_plot)+1))
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels)
            
            # Color by molecule type
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(self.colors['protein'] if i == 0 else self.colors['dna'])
                pc.set_alpha(0.7)
        
        ax.set_ylabel('RMSD (Å)')
        ax.set_title('RMSD Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_bfactor_correlation(self, ax: plt.Axes, bfactor_data: List) -> None:
        """B-factor vs confidence correlation plot"""
        if not bfactor_data:
            return
            
        # Extract data efficiently (assuming BFactorComparison objects)
        exp_vals = [b.experimental_bfactor for b in bfactor_data]
        pred_vals = [b.predicted_confidence for b in bfactor_data]
        
        # Scatter plot with correlation
        ax.scatter(exp_vals, pred_vals, alpha=0.6, s=20, c=self.colors['secondary'])
        
        # Add correlation line
        if len(exp_vals) > 1:
            correlation = np.corrcoef(exp_vals, pred_vals)[0,1]
            z = np.polyfit(exp_vals, pred_vals, 1)
            p = np.poly1d(z)
            ax.plot(exp_vals, p(exp_vals), 'r--', alpha=0.8)
            ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Experimental B-factor')
        ax.set_ylabel('Predicted Confidence')
        ax.set_title('B-factor vs Confidence', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_consensus_summary(self, ax: plt.Axes, consensus_data: List) -> None:
        """Consensus error visualization"""
        if not consensus_data:
            return
            
        # Extract RMSD values efficiently
        rmsds = [c.mean_rmsd for c in consensus_data]
        confidences = [c.confidence for c in consensus_data]
        
        # Scatter plot: RMSD vs confidence
        scatter = ax.scatter(confidences, rmsds, alpha=0.7, s=30, c=rmsds, 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Consensus Confidence')
        ax.set_ylabel('Mean RMSD (Å)')
        ax.set_title('Consensus Error Map', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar if space allows
        plt.colorbar(scatter, ax=ax, label='RMSD (Å)', shrink=0.8)
    
    def _plot_mutation_impact(self, ax: plt.Axes, mutation_data: List) -> None:
        """Mutation impact summary"""
        if not mutation_data:
            return
            
        # Group by mutation type efficiently
        impact_by_type = {}
        for mut in mutation_data:
            mut_type = getattr(mut, 'mutation_type', 'unknown')
            impact_by_type.setdefault(mut_type, []).append(getattr(mut, 'impact_score', 0))
        
        if impact_by_type:
            # Box plot of impact scores by type
            types, impacts = zip(*impact_by_type.items())
            ax.boxplot(impacts, labels=types)
            ax.set_ylabel('Impact Score')
            ax.set_title('Mutation Impact by Type', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def comparison_plot(self, data1: List[float], data2: List[float], 
                       labels: Tuple[str, str], title: str = "Comparison",
                       output_path: Optional[Path] = None) -> plt.Figure:
        """
        Generic comparison plot for any two datasets
        
        Args:
            data1, data2: Datasets to compare
            labels: Labels for datasets
            title: Plot title
            output_path: Optional save path
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Side-by-side histograms
        ax1.hist([data1, data2], bins=20, alpha=0.7, label=labels, 
                color=[self.colors['primary'], self.colors['secondary']])
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot comparison
        min_len = min(len(data1), len(data2))
        if min_len > 0:
            ax2.scatter(data1[:min_len], data2[:min_len], alpha=0.6, s=20)
            
            # Perfect correlation line
            max_val = max(max(data1[:min_len]), max(data2[:min_len]))
            ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Perfect correlation')
            
            ax2.set_xlabel(labels[0])
            ax2.set_ylabel(labels[1])
            ax2.set_title(f'{title} - Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_path: fig.savefig(output_path, **self.fig_params)
        return fig


def create_publication_report(analysis_results: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """
    Generate publication-ready figures from all analysis modules
    
    Args:
        analysis_results: Dictionary containing results from different analyses
        output_dir: Output directory for plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(exist_ok=True)
    plotter = PublicationPlotter()
    output_paths = {}
    
    # Main dashboard
    dashboard_path = output_dir / 'analysis_dashboard.png'
    fig = plotter.summary_dashboard(analysis_results, dashboard_path)
    plt.close(fig)
    output_paths['dashboard'] = dashboard_path
    
    # Individual comparison plots if data exists
    if 'rmsd' in analysis_results and 'bfactor' in analysis_results:
        # RMSD vs B-factor comparison
        rmsd_vals = [r.rmsd for r in analysis_results['rmsd']]
        bfactor_vals = [b.experimental_bfactor for b in analysis_results['bfactor']]
        
        comp_path = output_dir / 'rmsd_vs_bfactor.png'
        fig = plotter.comparison_plot(rmsd_vals, bfactor_vals, 
                                    ('RMSD (Å)', 'B-factor'), 
                                    'RMSD vs B-factor Analysis', comp_path)
        plt.close(fig)
        output_paths['rmsd_bfactor'] = comp_path
    
    return output_paths


def quick_comparison(data1: List[float], data2: List[float], 
                    labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2'),
                    output_path: Optional[Path] = None) -> plt.Figure:
    """
    Quick two-dataset comparison plot - single function interface
    
    Args:
        data1, data2: Datasets to compare  
        labels: Dataset labels
        output_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    plotter = PublicationPlotter()
    return plotter.comparison_plot(data1, data2, labels, output_path=output_path)


def save_all_plots(figures: List[plt.Figure], output_dir: Path, 
                  prefix: str = 'plot') -> List[Path]:
    """
    Efficiently save multiple figures with automatic naming
    
    Args:
        figures: List of matplotlib figures
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir.mkdir(exist_ok=True)
    paths = []
    
    for i, fig in enumerate(figures):
        path = output_dir / f'{prefix}_{i+1:02d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        paths.append(path)
        plt.close(fig)  # Free memory immediately
    
    return paths
