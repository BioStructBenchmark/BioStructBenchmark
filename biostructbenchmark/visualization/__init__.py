"""
biostructbenchmark.visualization
Visualization tools with graceful matplotlib handling
"""

import os
import warnings

# Set matplotlib backend before importing
try:
    import matplotlib
    # Use non-interactive backend if no display
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available - visualization features disabled")

# Now import visualization modules
if MATPLOTLIB_AVAILABLE:
    try:
        from .residue_plots import ResidueVisualizer, create_residue_report, quick_residue_plot
        from .plots import PublicationPlotter, create_publication_report
        from .structure import StructureVisualizer, create_structure_visualization
        _viz_available = True
    except ImportError as e:
        warnings.warn(f"Some visualization modules failed to import: {e}")
        _viz_available = False
        ResidueVisualizer = None
        create_residue_report = None
        quick_residue_plot = None
else:
    _viz_available = False
    ResidueVisualizer = None
    create_residue_report = None
    quick_residue_plot = None

# Conditional CURVES+ visualization
try:
    from .curves_plots import CurvesVisualizer, create_curves_report
    _curves_viz_available = True
except (ImportError, RuntimeError):
    _curves_viz_available = False
    CurvesVisualizer = None
    create_curves_report = None

__all__ = []

# Add available visualizations to __all__
if _viz_available:
    __all__.extend([
        'ResidueVisualizer',
        'create_residue_report', 
        'quick_residue_plot'
    ])

if _curves_viz_available:
    __all__.extend([
        'CurvesVisualizer',
        'create_curves_report'
    ])
