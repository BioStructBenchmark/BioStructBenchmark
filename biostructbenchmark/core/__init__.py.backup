"""
biostructbenchmark.core
Core functionality for structure analysis and comparison
"""

# Main analysis functions
from .alignment import compare_structures, export_residue_rmsd_csv
from .metrics import generate_comprehensive_metrics

# Data structures
from .alignment import AlignmentResult, ResidueRMSD
from .metrics import StructureMetrics, ErrorComponents

# I/O utilities
from .io import get_structure, validate_file

__all__ = [
    # Main functions
    'compare_structures',
    'generate_comprehensive_metrics', 
    'export_residue_rmsd_csv',
    'get_structure',
    'validate_file',
    
    # Data structures
    'AlignmentResult',
    'ResidueRMSD', 
    'StructureMetrics',
    'ErrorComponents'
]