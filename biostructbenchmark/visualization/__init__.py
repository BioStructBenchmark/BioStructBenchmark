"""
biostructbenchmark.visualization
Visualization tools for structure analysis results
"""

# Core residue visualization
from .residue_plots import ResidueVisualizer, create_residue_report, quick_residue_plot

# CURVES+ visualization (conditional import)
try:
    from .curves_plots import CurvesVisualizer, create_curves_report
    _curves_viz_available = True
except ImportError:
    _curves_viz_available = False
    CurvesVisualizer = None
    create_curves_report = None

__all__ = [
    # Core visualization
    'ResidueVisualizer',
    'create_residue_report',
    'quick_residue_plot'
]

# Add CURVES+ visualization if available
if _curves_viz_available:
    __all__.extend([
        'CurvesVisualizer',
        'create_curves_report'
    ])