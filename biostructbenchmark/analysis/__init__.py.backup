"""
biostructbenchmark.analysis
Analysis modules for structure comparison
"""

# Import analysis classes
from .bfactor import *
from .secondary import *
from .consensus import *
from .mutations import *
from .curves import *

__all__ = ['consensus', 'bfactor', 'secondary', 'mutations', 'curves']

# Conditional import for CURVES+ (may not have executable)
try:
    from .curves import CurvesAnalyzer, CurvesParameters, HydrogenBond
    _curves_available = True
except (ImportError, RuntimeError):
    _curves_available = False
    CurvesAnalyzer = None
    CurvesParameters = None
    HydrogenBond = None

__all__ = [
    # B-factor analysis
    'BFactorAnalyzer',
    'BFactorComparison', 
    'BFactorStatistics',
    
    # Secondary structure
    'SecondaryStructureAnalyzer',
    'SecondaryStructure',
    
    # Consensus errors
    'ConsensusAnalyzer',
    'ConsensusError',
    
    # Mutations
    'MutationAnalyzer',
    'Mutation',
]

# Add CURVES+ if available
if _curves_available:
    __all__.extend([
        'CurvesAnalyzer',
        'CurvesParameters',
        'HydrogenBond'
    ])
