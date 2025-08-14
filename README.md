# BioStructBenchmark 🧬

A comprehensive toolkit for benchmarking computational structure predictions against experimental reality, with a focus on DNA-protein complexes.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Usage](#command-line-usage)
- [Module Architecture](#module-architecture)
- [Analysis Capabilities](#analysis-capabilities)
- [Output Files](#output-files)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

BioStructBenchmark compares experimental DNA-protein complex structures with their computationally predicted counterparts (AlphaFold, RoseTTAFold, etc.) to identify systematic prediction errors. The toolkit provides comprehensive structural analysis through multiple reference frame alignments, detailed error decomposition, and extensive DNA geometry analysis.

### Why BioStructBenchmark?

- **Multi-frame alignment**: Three different reference frames to distinguish between positioning and structural errors
- **Comprehensive metrics**: Beyond simple RMSD - error decomposition, per-residue analysis, DNA geometry
- **DNA-focused**: Specialized tools for DNA structure analysis including CURVES+ integration
- **Production-ready**: Robust error handling, batch processing, and publication-quality outputs

## Key Features

### 🎯 Multi-Frame Alignment Analysis
- **Full structure alignment**: Overall structural accuracy assessment
- **DNA-to-protein positioning**: Evaluates DNA placement relative to protein
- **DNA standalone**: Assesses DNA structure independent of protein context

### 📊 Comprehensive Metrics
- Per-residue/nucleotide RMSD calculations
- Error decomposition (translational vs rotational components)
- B-factor vs pLDDT confidence metric comparison
- Consensus error mapping across multiple structures
- Mutation impact analysis

### 🧬 DNA Geometry Analysis (CURVES+ Integration)
- Base pair parameters (shear, stretch, stagger, buckle, propeller, opening)
- Base pair step parameters (shift, slide, rise, tilt, roll, twist)
- Groove geometry (major/minor groove width and depth)
- Hydrogen bond network analysis
- DNA conformation classification (A/B/Z-form)

### 📈 Visualization
- Publication-quality plots and dashboards
- Per-residue RMSD heatmaps
- DNA geometry comparison plots
- Hydrogen bond network visualization
- 3D structure alignment visualization

## Installation

### Requirements

- Python 3.8+
- BioPython ≥ 1.79
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- Matplotlib ≥ 3.4
- SciPy ≥ 1.7

### Optional Dependencies

- CURVES+ (for DNA geometry analysis)
- py3Dmol (for interactive 3D visualization)
- Seaborn (for enhanced plotting)

### Install from PyPI

```bash
pip install biostructbenchmark
```

### Install from Source

```bash
git clone https://github.com/yourusername/biostructbenchmark
cd biostructbenchmark
pip install -e .
```

### Verify Installation

```bash
biostructbenchmark --version
# BioStructBenchmark v0.1.0

# Run tests
pytest tests/
```

## Quick Start

### Basic Single Structure Comparison

```bash
# Simple RMSD calculation
biostructbenchmark observed.pdb predicted.pdb

# With output directory
biostructbenchmark -e experimental.pdb -p predicted.pdb -o results/
```

### Multi-Frame Alignment Analysis

```bash
# Perform all three reference frame alignments
biostructbenchmark -e exp.pdb -p pred.pdb --multi-frame --save-aligned

# This generates:
# 1. Full structure alignment
# 2. DNA positioning relative to protein
# 3. Standalone DNA structure comparison
```

### Batch Processing

```bash
# Process entire directories
biostructbenchmark -e experimental_dir/ -p predicted_dir/ -o results/ --all-benchmarks

# With parallel processing
biostructbenchmark -e exp/ -p pred/ -o out/ --parallel 4
```

## Command Line Usage

### Basic Syntax

```bash
biostructbenchmark [OPTIONS] [LEGACY_FILES]
```

### Primary Options

| Option | Description |
|--------|-------------|
| `-e, --experimental PATH` | Path to experimental structure(s) |
| `-p, --predicted PATH` | Path to predicted structure(s) |
| `-o, --output PATH` | Output directory (default: ./biostructbenchmark_results) |
| `-v, --version` | Show version and exit |

### Analysis Options

| Option | Description |
|--------|-------------|
| `--all-benchmarks` | Run all available analyses |
| `--multi-frame` | Perform multi-frame alignment (3 reference frames) |
| `--rmsd-only` | Basic RMSD analysis only (fastest) |
| `--curves` | CURVES+ DNA geometry analysis |
| `--bfactor` | B-factor vs confidence metric analysis |
| `--consensus` | Consensus error mapping (requires multiple pairs) |
| `--mutations` | Detect and analyze mutations |
| `--visualize` | Generate publication-quality plots |

### Advanced Options

| Option | Description |
|--------|-------------|
| `--reference-frame {full,protein,dna,multi}` | Alignment reference frame |
| `--output-format {csv,json,both}` | Output data format |
| `--rmsd-threshold FLOAT` | RMSD threshold for consensus (default: 3.0 Å) |
| `--save-aligned` | Save aligned PDB structures |
| `--export-all` | Export all intermediate files |
| `--parallel N` | Number of parallel processes |
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress non-essential output |

### Example Commands

```bash
# Full analysis with all features
biostructbenchmark -e exp.pdb -p pred.pdb --all-benchmarks --visualize

# DNA-focused analysis
biostructbenchmark -e exp.pdb -p pred.pdb --curves --multi-frame

# Batch with specific analyses
biostructbenchmark -e exp_dir/ -p pred_dir/ --bfactor --mutations --consensus

# High-throughput processing
biostructbenchmark -e structures/ -p predictions/ --parallel 8 --quiet --export-all
```

## Module Architecture

### Core Modules (`biostructbenchmark/core/`)

#### `alignment.py`
- **Purpose**: Structure superposition and RMSD calculations
- **Key Functions**:
  - `perform_multi_frame_alignment()`: Execute all three reference frame alignments
  - `align_structures_by_reference_frame()`: Flexible alignment with different references
  - `calculate_rmsd()`: RMSD calculation for atoms or coordinates
  - `calculate_per_residue_rmsd_for_subset()`: Per-residue RMSD for specific molecule types
- **Classes**:
  - `AlignmentResult`: Container for alignment results
  - `MultiFrameAlignmentResult`: Results from all three alignments
  - `ResidueRMSD`: Per-residue RMSD data

#### `metrics.py`
- **Purpose**: Error decomposition and comprehensive metrics
- **Key Functions**:
  - `decompose_structural_error()`: Separate translation/rotation components
  - `calculate_per_chain_metrics()`: Chain-specific analysis
  - `generate_comprehensive_metrics()`: Full metrics report
- **Key Metrics**:
  - Translation error magnitude
  - Rotation error (Frobenius norm)
  - Per-molecule type statistics
  - Backbone vs side-chain RMSD

#### `io.py`
- **Purpose**: Robust structure file handling
- **Features**:
  - Automatic PDB/CIF format detection
  - Missing residue/atom handling
  - Chain mismatch resolution
  - Structure validation

### Analysis Modules (`biostructbenchmark/analysis/`)

#### `curves.py`
- **Purpose**: CURVES+ integration for DNA geometry
- **Analyzes**:
  - Base pair parameters (6 parameters)
  - Base pair step parameters (6 parameters)
  - Groove geometry (width, depth)
  - DNA conformation classification
- **Classes**:
  - `CurvesAnalyzer`: Main analysis class
  - `CurvesParameters`: Parameter container
  - `HydrogenBond`: H-bond information

#### `bfactor.py`
- **Purpose**: B-factor and confidence metric analysis
- **Features**:
  - B-factor extraction from experimental structures
  - pLDDT extraction from AlphaFold predictions
  - Correlation analysis
  - Disorder region identification

#### `consensus.py`
- **Purpose**: Identify consistent prediction errors
- **Requirements**: Minimum 3 structure pairs
- **Outputs**:
  - Consensus error positions
  - Statistical significance
  - Error frequency maps

#### `mutations.py`
- **Purpose**: Mutation detection and impact analysis
- **Features**:
  - Automatic mutation detection
  - Local RMSD impact calculation
  - Mutation clustering analysis

### Visualization Modules (`biostructbenchmark/visualization/`)

#### `plots.py`
- **Purpose**: Publication-quality matplotlib visualizations
- **Generates**:
  - RMSD distribution plots
  - B-factor correlation plots
  - Consensus error heatmaps
  - Multi-panel dashboards

#### `structure.py`
- **Purpose**: 3D structure visualization
- **Features**:
  - Matplotlib 3D backbone traces
  - Per-residue RMSD coloring
  - Alignment visualization
  - Summary statistics export

#### `curves_plots.py`
- **Purpose**: DNA geometry visualization
- **Plots**:
  - Base pair parameter comparisons
  - Groove geometry analysis
  - Hydrogen bond networks
  - Parameter correlation matrices

## Analysis Capabilities

### 1. Multi-Frame Alignment Analysis

The cornerstone feature providing three complementary perspectives:

```python
# Alignment 1: Full structure
# Answers: How accurate is the overall prediction?
full_rmsd = result.full_structure.overall_rmsd

# Alignment 2: DNA aligned using protein reference
# Answers: Is the DNA correctly positioned relative to the protein?
positioning_rmsd = result.dna_to_protein.overall_rmsd

# Alignment 3: DNA-only alignment
# Answers: Is the DNA structure itself accurate?
dna_rmsd = result.dna_to_dna.overall_rmsd
```

**Interpretation Guide**:
- High positioning RMSD + Low DNA RMSD = Interface prediction problem
- Low positioning RMSD + High DNA RMSD = DNA geometry problem
- Both high = Systematic prediction issues

### 2. DNA Geometry Analysis

Comprehensive DNA structure characterization:

- **Base Pair Parameters**: Quantify intra-base pair distortions
- **Step Parameters**: Inter-base pair relationships
- **Groove Analysis**: Major/minor groove dimensions
- **Global Metrics**: Bending, writhe, handedness

### 3. Hydrogen Bond Network Analysis

Critical for understanding DNA-protein interfaces:

- Distance and angle-based detection
- Strength classification (strong/medium/weak)
- Network topology analysis
- Interface stability assessment

## Output Files

### Directory Structure

```
biostructbenchmark_results/
├── structure1_vs_structure2/
│   ├── alignments/                    # Aligned PDB files
│   │   ├── aligned_1_full_to_experimental.pdb
│   │   ├── aligned_2_dna_to_protein_reference.pdb
│   │   ├── aligned_3_dna_to_dna.pdb
│   │   └── experimental_reference.pdb
│   ├── rmsd_full_structure.csv       # Per-residue RMSD
│   ├── rmsd_dna_to_protein.csv      
│   ├── rmsd_dna_standalone.csv      
│   ├── multi_frame_analysis.json     # Comprehensive summary
│   ├── geometry_comparison.csv       # CURVES+ results
│   ├── hydrogen_bonds.csv           
│   ├── bfactor_comparison.csv       
│   └── multi_frame_dashboard.png     # Visualization
└── batch_analysis_report.json        # Batch summary
```

### Key Output Files

#### `multi_frame_analysis.json`
Complete analysis summary including:
- Overall RMSD values for all three alignments
- Per-molecule type statistics
- Worst-performing residues
- Automated interpretation

#### `geometry_comparison.csv`
DNA geometry parameters with differences:
- All 12 base pair/step parameters
- Groove dimensions
- Statistical summaries

#### `hydrogen_bonds.csv`
Complete H-bond listing:
- Donor/acceptor atoms and residues
- Bond distances and angles
- Strength classification

## Best Practices

### 1. Structure Preparation

```bash
# Clean PDB files before analysis
# Remove water molecules, heteroatoms if needed
grep -v "HOH\|HETATM" input.pdb > cleaned.pdb

# Ensure consistent chain naming
# Experimental: Chain A (protein), Chain B (DNA)
# Predicted: Same chain naming scheme
```

### 2. Choosing Analysis Modes

- **Quick assessment**: Use `--rmsd-only` for rapid screening
- **Publication figures**: Always use `--visualize --export-all`
- **DNA focus**: Combine `--multi-frame --curves`
- **Comprehensive**: Use `--all-benchmarks` for complete analysis

### 3. Interpreting Results

#### RMSD Thresholds
- < 2.0 Å: Excellent (near-experimental accuracy)
- 2.0-3.5 Å: Good (reliable for most analyses)
- 3.5-5.0 Å: Moderate (useful with caution)
- \> 5.0 Å: Poor (significant deviations)

#### DNA-Specific Metrics
- Twist deviation > 5°: Significant helical distortion
- Rise deviation > 0.5 Å: Inter-base pair spacing issues
- Groove width deviation > 2 Å: Major conformational differences

### 4. Batch Processing

```bash
# For large datasets, use parallel processing
biostructbenchmark -e exp_dir/ -p pred_dir/ --parallel 8

# Monitor progress with verbose mode
biostructbenchmark -e exp/ -p pred/ --verbose --parallel 4

# For HPC environments
export OMP_NUM_THREADS=1  # Prevent nested parallelism
biostructbenchmark -e exp/ -p pred/ --parallel $SLURM_CPUS_PER_TASK
```

### 5. Memory Management

- Process large structures individually rather than in batch
- Use `--rmsd-only` for initial screening of large datasets
- Consider splitting very large complexes into domains

## Troubleshooting

### Common Issues and Solutions

#### "CURVES+ executable not found"
```bash
# Install CURVES+
# Add to PATH or specify explicitly
export CURVES_EXEC=/path/to/curves+
```

#### "No common residues found"
- Check chain naming consistency
- Verify sequence alignment
- Use `--verbose` to see detailed matching

#### Memory errors with large structures
```bash
# Reduce parallel processes
biostructbenchmark -e exp/ -p pred/ --parallel 2

# Process individually
for file in exp/*.pdb; do
    biostructbenchmark -e $file -p pred/$(basename $file) -o results/
done
```

#### Visualization errors
```bash
# Check matplotlib backend
export MPLBACKEND=Agg  # For headless systems

# Install optional dependencies
pip install seaborn py3Dmol
```

## API Usage

### Python Integration

```python
from biostructbenchmark.core.alignment import perform_multi_frame_alignment
from biostructbenchmark.analysis.curves import CurvesAnalyzer
from biostructbenchmark.visualization.plots import PublicationPlotter

# Multi-frame alignment
result = perform_multi_frame_alignment(
    experimental_path="exp.pdb",
    predicted_path="pred.pdb",
    output_dir="results/"
)

# Access results
print(f"Full RMSD: {result.full_structure.overall_rmsd:.2f} Å")
print(f"DNA positioning: {result.dna_to_protein.overall_rmsd:.2f} Å")
print(f"DNA structure: {result.dna_to_dna.overall_rmsd:.2f} Å")

# DNA geometry analysis
analyzer = CurvesAnalyzer()
params = analyzer.analyze_structure("dna_structure.pdb")
hbonds = analyzer.detect_hydrogen_bonds("complex.pdb")

# Visualization
plotter = PublicationPlotter()
plotter.summary_dashboard(
    {"rmsd": result.full_structure.residue_rmsds},
    "dashboard.png"
)
```

## Contributing

We welcome contributions! Please follow these guidelines:

### Code Style
- Follow PEP 8
- Use type hints for function arguments
- Add comprehensive docstrings
- Include unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_alignment.py -v

# Check coverage
pytest --cov=biostructbenchmark tests/
```

## Citation

If you use BioStructBenchmark in your research, please cite:

```bibtex
@software{biostructbenchmark,
  title = {BioStructBenchmark: A toolkit for benchmarking DNA-protein structure predictions},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/biostructbenchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- BioPython team for excellent structure handling tools
- CURVES+ developers for DNA geometry analysis
- AlphaFold team for advancing structure prediction
- All contributors and users of BioStructBenchmark

---

*Built with coffee, BioPython, and an unhealthy obsession with structural accuracy* 🐱

For questions, issues, or suggestions, please open an issue on [GitHub](https://github.com/yourusername/biostructbenchmark).