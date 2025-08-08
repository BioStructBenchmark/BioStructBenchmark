"""
biostructbenchmark/visualization/curves_plots.py
Compact visualization functions for CURVES+ analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap

from ..analysis.curves import CurvesParameters, HydrogenBond


class CurvesVisualizer:
    """Compact visualization tools for CURVES+ analysis results"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.base_colors = {'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#FFA07A',
                           'DA': '#FF6B6B', 'DT': '#4ECDC4', 'DG': '#45B7D1', 'DC': '#FFA07A'}
        self.param_cmaps = {'twist': 'RdYlBu_r', 'roll': 'RdBu_r', 'tilt': 'PRGn',
                           'rise': 'viridis', 'slide': 'plasma', 'shift': 'coolwarm'}
    
    def _setup_subplot(self, ax, title: str, xlabel: str, ylabel: str, grid: bool = True):
        """Quick subplot formatting"""
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid: ax.grid(True, alpha=0.3)
    
    def _add_bar_labels(self, ax, bars, values, threshold: float = 0.1):
        """Add value labels to bars"""
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if abs(height) > threshold:
                ax.text(bar.get_x() + bar.get_width()/2., height + np.sign(height)*0.1,
                       f'{value:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    def plot_parameter_comparison(self, comparison_df: pd.DataFrame, parameter: str, 
                                output_path: Optional[Path] = None) -> plt.Figure:
        """Plot comparison of geometric parameter with distribution"""
        if f'{parameter}_diff' not in comparison_df.columns:
            raise ValueError(f"Parameter '{parameter}_diff' not found in data")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        steps = comparison_df['step'].values
        differences = comparison_df[f'{parameter}_diff'].values
        colors = ['red' if x > 0 else 'blue' for x in differences]
        
        bars = ax1.bar(range(len(steps)), differences, color=colors, alpha=0.7)
        self._setup_subplot(ax1, f'{parameter.replace("_", " ").title()} Parameter Comparison',
                           'Base Pair Step', f'{parameter.replace("_", " ").title()} Difference (Predicted - Experimental)')
        ax1.set_xticks(range(len(steps)))
        ax1.set_xticklabels(steps, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        self._add_bar_labels(ax1, bars, differences)
        
        # Distribution
        ax2.hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        mean_diff = np.mean(differences)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No difference')
        ax2.axvline(x=mean_diff, color='orange', linestyle='-', label=f'Mean: {mean_diff:.2f}')
        self._setup_subplot(ax2, 'Distribution of Differences', 
                           f'{parameter.replace("_", " ").title()} Difference', 'Frequency')
        ax2.legend()
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_parameter_heatmap(self, comparison_df: pd.DataFrame, 
                             output_path: Optional[Path] = None) -> plt.Figure:
        """Create heatmap of all parameter differences"""
        diff_cols = [col for col in comparison_df.columns if col.endswith('_diff')]
        if not diff_cols:
            raise ValueError("No difference columns found in data")
        
        heatmap_data = comparison_df.set_index('step')[diff_cols]
        heatmap_data.columns = [col.replace('_diff', '').replace('_', ' ').title() for col in diff_cols]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
        
        sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                   annot=True, fmt='.1f', 
                   cbar_kws={'label': 'Parameter Difference (Predicted - Experimental)'}, ax=ax)
        
        ax.set_title('DNA Geometric Parameter Differences Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Base Pair Step')
        ax.set_ylabel('Geometric Parameters')
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_groove_geometry(self, experimental_params: Dict[str, CurvesParameters],
                           predicted_params: Dict[str, CurvesParameters],
                           output_path: Optional[Path] = None) -> plt.Figure:
        """Plot major and minor groove geometry comparison"""
        common_steps = set(experimental_params.keys()) & set(predicted_params.keys())
        if not common_steps:
            raise ValueError("No common steps found between experimental and predicted data")
        
        # Extract data
        data = {'steps': [], 'exp_major': [], 'pred_major': [], 'exp_minor': [], 'pred_minor': []}
        for step in sorted(common_steps):
            exp, pred = experimental_params[step], predicted_params[step]
            if exp.major_groove_width is not None and pred.major_groove_width is not None:
                data['steps'].append(step)
                data['exp_major'].append(exp.major_groove_width)
                data['pred_major'].append(pred.major_groove_width)
                data['exp_minor'].append(exp.minor_groove_width or 0)
                data['pred_minor'].append(pred.minor_groove_width or 0)
        
        if not data['steps']:
            raise ValueError("No valid groove width data found")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        x_pos, width = np.arange(len(data['steps'])), 0.35
        
        # Major/Minor groove widths
        for ax, exp_data, pred_data, title in [
            (ax1, data['exp_major'], data['pred_major'], 'Major Groove Width'),
            (ax2, data['exp_minor'], data['pred_minor'], 'Minor Groove Width')
        ]:
            ax.bar(x_pos - width/2, exp_data, width, label='Experimental', color='skyblue', alpha=0.8)
            ax.bar(x_pos + width/2, pred_data, width, label='Predicted', color='lightcoral', alpha=0.8)
            self._setup_subplot(ax, f'{title} Comparison', 'Base Pair Step', f'{title} (Å)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data['steps'], rotation=45, ha='right')
            ax.legend()
        
        # Correlation plots
        for ax, exp_data, pred_data, title in [
            (ax3, data['exp_major'], data['pred_major'], 'Major Groove Width'),
            (ax4, data['exp_minor'], data['pred_minor'], 'Minor Groove Width')
        ]:
            ax.scatter(exp_data, pred_data, alpha=0.7, color='blue' if 'Major' in title else 'orange')
            min_val, max_val = min(min(exp_data), min(pred_data)), max(max(exp_data), max(pred_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect correlation')
            
            corr = np.corrcoef(exp_data, pred_data)[0, 1]
            ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            self._setup_subplot(ax, f'{title} Correlation', f'Experimental {title} (Å)', f'Predicted {title} (Å)')
            ax.legend()
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_hydrogen_bond_network(self, hbonds: List[HydrogenBond], structure_name: str = "Structure",
                                 output_path: Optional[Path] = None) -> plt.Figure:
        """Create network visualization of hydrogen bonds"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if not hbonds:
            ax.text(0.5, 0.5, 'No hydrogen bonds detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{structure_name} - Hydrogen Bond Network')
            return fig
        
        # Build network
        G = nx.Graph()
        residues = set()
        for hb in hbonds:
            residues.add(hb.donor_residue)
            residues.add(hb.acceptor_residue)
        
        # Add nodes with type annotation
        aa_names = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
        for residue in residues:
            res_type = 'protein' if any(aa in residue for aa in aa_names) else 'DNA'
            G.add_node(residue, type=res_type)
        
        # Add edges
        for hb in hbonds:
            G.add_edge(hb.donor_residue, hb.acceptor_residue, weight=1.0/hb.distance,
                      distance=hb.distance, strength=hb.strength,
                      atoms=f"{hb.donor_atom}-{hb.acceptor_atom}")
        
        # Layout and draw
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes by type
        protein_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'protein']
        dna_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'DNA']
        
        nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_color='lightblue', 
                              node_size=500, alpha=0.8, label='Protein', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=dna_nodes, node_color='lightcoral', 
                              node_size=500, alpha=0.8, label='DNA', ax=ax)
        
        # Draw edges by strength
        edge_styles = [
            ([(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'strong'], 3, 'red', 'Strong H-bonds'),
            ([(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'medium'], 2, 'orange', 'Medium H-bonds'),
            ([(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'weak'], 1, 'gray', 'Weak H-bonds')
        ]
        
        for edges, width, color, label in edge_styles:
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=width, alpha=0.8-width*0.1,
                                     edge_color=color, label=label, ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'{structure_name} - Hydrogen Bond Network\n'
                    f'({len(hbonds)} H-bonds, {len(protein_nodes)} protein, {len(dna_nodes)} DNA)', 
                    fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_hbond_statistics(self, exp_hbonds: List[HydrogenBond], pred_hbonds: List[HydrogenBond],
                            output_path: Optional[Path] = None) -> plt.Figure:
        """Compare hydrogen bond statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distance distributions
        exp_distances = [hb.distance for hb in exp_hbonds]
        pred_distances = [hb.distance for hb in pred_hbonds]
        
        ax1.hist(exp_distances, bins=20, alpha=0.7, label='Experimental', color='blue')
        ax1.hist(pred_distances, bins=20, alpha=0.7, label='Predicted', color='red')
        self._setup_subplot(ax1, 'H-bond Distance Distribution', 'H-bond Distance (Å)', 'Frequency')
        ax1.legend()
        
        # Strength distribution
        strength_categories = ['strong', 'medium', 'weak']
        exp_counts = [sum(1 for hb in exp_hbonds if hb.strength == s) for s in strength_categories]
        pred_counts = [sum(1 for hb in pred_hbonds if hb.strength == s) for s in strength_categories]
        
        x, width = np.arange(len(strength_categories)), 0.35
        ax2.bar(x - width/2, exp_counts, width, label='Experimental', alpha=0.8, color='blue')
        ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='red')
        self._setup_subplot(ax2, 'H-bond Strength Distribution', 'H-bond Strength', 'Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strength_categories)
        ax2.legend()
        
        # Summary counts
        aa_names = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS',
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
        
        def count_protein_hbonds(hbonds):
            return sum(1 for hb in hbonds if any(aa in hb.donor_residue for aa in aa_names) or
                      any(aa in hb.acceptor_residue for aa in aa_names))
        
        categories = ['Total H-bonds', 'Protein-DNA H-bonds']
        exp_values = [len(exp_hbonds), count_protein_hbonds(exp_hbonds)]
        pred_values = [len(pred_hbonds), count_protein_hbonds(pred_hbonds)]
        
        x = np.arange(len(categories))
        ax3.bar(x - width/2, exp_values, width, label='Experimental', alpha=0.8, color='blue')
        ax3.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.8, color='red')
        self._setup_subplot(ax3, 'H-bond Count Comparison', 'H-bond Type', 'Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # Summary table
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = [
            ['Total H-bonds', len(exp_hbonds), len(pred_hbonds), len(pred_hbonds) - len(exp_hbonds)],
            ['Mean distance (Å)', f'{np.mean(exp_distances):.2f}' if exp_distances else 'N/A',
             f'{np.mean(pred_distances):.2f}' if pred_distances else 'N/A',
             f'{np.mean(pred_distances) - np.mean(exp_distances):.2f}' if exp_distances and pred_distances else 'N/A'],
            ['Strong bonds', exp_counts[0], pred_counts[0], pred_counts[0] - exp_counts[0]],
            ['Medium bonds', exp_counts[1], pred_counts[1], pred_counts[1] - exp_counts[1]],
            ['Weak bonds', exp_counts[2], pred_counts[2], pred_counts[2] - exp_counts[2]]
        ]
        
        table = ax4.table(cellText=summary_data, colLabels=['Metric', 'Experimental', 'Predicted', 'Difference'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_pymol_script(self, hbonds: List[HydrogenBond], structure_path: Path, 
                          output_path: Path) -> None:
        """Generate PyMOL script for hydrogen bond visualization"""
        script_lines = [
            f"# PyMOL script for {structure_path.name}",
            f"load {structure_path.absolute()}",
            "hide everything", "show cartoon, polymer", "show sticks, nucleic",
            "color skyblue, polymer and chain A", "color lightpink, nucleic",
            "color red, resn DA+A", "color blue, resn DT+T", 
            "color green, resn DG+G", "color orange, resn DC+C",
            "distance hbonds, none, none", ""
        ]
        
        for i, hb in enumerate(hbonds):
            donor_parts, acceptor_parts = hb.donor_residue.split('_'), hb.acceptor_residue.split('_')
            if len(donor_parts) >= 2 and len(acceptor_parts) >= 2:
                script_lines.append(f"distance hbond_{i}, resi {donor_parts[1]} and name {hb.donor_atom}, "
                                   f"resi {acceptor_parts[1]} and name {hb.acceptor_atom}")
        
        script_lines.extend([
            "hide labels, hbonds", "color yellow, hbonds", "orient", "zoom",
            f"png {output_path.parent / 'hbond_visualization.png'}, dpi=300"
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(script_lines))


def create_curves_report(analysis_dir: Path, output_dir: Path) -> None:
    """Generate comprehensive CURVES+ analysis report"""
    visualizer = CurvesVisualizer()
    output_dir.mkdir(exist_ok=True)
    
    comparison_file = analysis_dir / 'all_geometry_comparisons.csv'
    if not comparison_file.exists():
        print(f"No comparison data found at {comparison_file}")
        return
    
    comparison_df = pd.read_csv(comparison_file)
    
    # Parameter plots
    parameters = ['twist', 'roll', 'tilt', 'rise', 'slide', 'shift', 
                 'shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']
    
    for param in parameters:
        if f'{param}_diff' in comparison_df.columns:
            try:
                fig = visualizer.plot_parameter_comparison(comparison_df, param, 
                                                          output_dir / f'{param}_comparison.png')
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not create {param} plot: {e}")
    
    # Heatmap
    try:
        heatmap_fig = visualizer.plot_parameter_heatmap(comparison_df, output_dir / 'parameter_heatmap.png')
        plt.close(heatmap_fig)
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
    
    # Process individual structure data
    for struct_dir in [d for d in analysis_dir.iterdir() if d.is_dir()]:
        exp_file, pred_file = struct_dir / 'experimental_hbonds.csv', struct_dir / 'predicted_hbonds.csv'
        
        if exp_file.exists() and pred_file.exists():
            try:
                exp_df, pred_df = pd.read_csv(exp_file), pd.read_csv(pred_file)
                exp_hbonds = [HydrogenBond(**row) for _, row in exp_df.iterrows()]
                pred_hbonds = [HydrogenBond(**row) for _, row in pred_df.iterrows()]
                
                struct_output = output_dir / struct_dir.name
                struct_output.mkdir(exist_ok=True)
                
                # Create visualizations
                for hbonds, name in [(exp_hbonds, 'experimental'), (pred_hbonds, 'predicted')]:
                    fig = visualizer.plot_hydrogen_bond_network(hbonds, f"{struct_dir.name} ({name.title()})",
                                                               struct_output / f'{name}_hbond_network.png')
                    plt.close(fig)
                
                stats_fig = visualizer.plot_hbond_statistics(exp_hbonds, pred_hbonds,
                                                            struct_output / 'hbond_statistics.png')
                plt.close(stats_fig)
                
            except Exception as e:
                print(f"Warning: Could not process {struct_dir.name}: {e}")
    
    print(f"CURVES+ analysis report generated in {output_dir}")