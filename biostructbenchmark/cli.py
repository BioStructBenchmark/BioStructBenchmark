"""
biostructbenchmark/cli.py
Command-line interface for BioStructBenchmark with multi-frame alignment support
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import csv
from datetime import datetime


def validate_path(path_str: str, must_exist: bool = False, 
                 allow_directory: bool = False) -> Path:
    """
    Validate and convert path string to Path object
    
    Args:
        path_str: Path string to validate
        must_exist: Whether path must exist
        allow_directory: Whether directories are allowed
        
    Returns:
        Validated Path object
        
    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    path = Path(path_str)
    
    if must_exist and not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    
    if path.exists() and not allow_directory and path.is_dir():
        raise argparse.ArgumentTypeError(f"Expected file, got directory: {path}")
    
    return path


def validate_file_path(path_str: str) -> Path:
    """Legacy validator for backward compatibility"""
    return validate_path(path_str, must_exist=True, allow_directory=False)


def get_version() -> str:
    """Get package version from __init__.py or pyproject.toml"""
    try:
        # Try to import from package
        import biostructbenchmark
        if hasattr(biostructbenchmark, '__version__'):
            return biostructbenchmark.__version__
    except ImportError:
        pass
    
    # Fallback to reading from file
    try:
        init_file = Path(__file__).parent / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r') as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"\'')
    except Exception:
        pass
    
    return "0.1.0"  # Default version


def find_structure_files(directory: Path) -> List[Path]:
    """Find all structure files (PDB/CIF) in directory"""
    structure_extensions = {'.pdb', '.cif', '.mmcif', '.ent'}
    files = []
    
    for ext in structure_extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser with multi-frame alignment options"""
    parser = argparse.ArgumentParser(
        prog='biostructbenchmark',
        description='Compare experimental and predicted DNA-protein complex structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file comparison with basic RMSD
  biostructbenchmark observed.pdb predicted.pdb
  
  # Multi-frame alignment analysis
  biostructbenchmark -e experimental.pdb -p predicted.pdb --multi-frame
  
  # Directory comparison with full analysis
  biostructbenchmark -e experimental/ -p predicted/ -o results/ --all-benchmarks
  
  # Specific analyses with visualization
  biostructbenchmark -e exp.pdb -p pred.pdb -o out/ --curves --bfactor --visualize
  
  # Multi-frame with detailed export
  biostructbenchmark -e exp.pdb -p pred.pdb --multi-frame --export-all
        """
    )
    
    # Version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'biostructbenchmark {get_version()}',
        help='Show version and exit'
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # New style - explicit experimental/predicted
    input_group.add_argument(
        '-e', '--experimental',
        type=lambda x: validate_path(x, must_exist=True, allow_directory=True),
        help='Path to experimental structure file or directory'
    )
    
    # Legacy positional arguments
    input_group.add_argument(
        'legacy_files',
        nargs='*',
        help='Legacy format: observed_file predicted_file'
    )
    
    # Predicted structures
    parser.add_argument(
        '-p', '--predicted',
        type=lambda x: validate_path(x, must_exist=True, allow_directory=True),
        help='Path to predicted structure file or directory (required with -e)'
    )
    
    # Output directory
    parser.add_argument(
        '-o', '--output',
        type=lambda x: validate_path(x, must_exist=False, allow_directory=True),
        default=Path('./biostructbenchmark_results'),
        help='Output directory for results (default: ./biostructbenchmark_results)'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    
    analysis_group.add_argument(
        '--all-benchmarks',
        action='store_true',
        help='Run all analyses including multi-frame alignment'
    )
    
    analysis_group.add_argument(
        '--multi-frame',
        action='store_true',
        help='Perform multi-frame alignment analysis (3 reference frames)'
    )
    
    analysis_group.add_argument(
        '--rmsd-only',
        action='store_true', 
        help='Perform only basic RMSD analysis (fastest)'
    )
    
    analysis_group.add_argument(
        '--curves',
        action='store_true',
        help='Perform CURVES+ DNA geometry analysis'
    )
    
    analysis_group.add_argument(
        '--bfactor',
        action='store_true',
        help='Analyze B-factors vs confidence metrics'
    )
    
    analysis_group.add_argument(
        '--consensus',
        action='store_true', 
        help='Generate consensus error mapping (requires multiple structure pairs)'
    )
    
    analysis_group.add_argument(
        '--mutations',
        action='store_true',
        help='Detect and analyze mutations between structures'
    )
    
    analysis_group.add_argument(
        '--visualize',
        action='store_true',
        help='Generate publication-quality plots and visualizations'
    )
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    
    processing_group.add_argument(
        '--parallel',
        type=int,
        metavar='N',
        help='Number of parallel processes for batch analysis (default: auto-detect)'
    )
    
    processing_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    processing_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    
    advanced_group.add_argument(
        '--rmsd-threshold',
        type=float,
        default=3.0,
        help='RMSD threshold for consensus error analysis (default: 3.0 Å)'
    )
    
    advanced_group.add_argument(
        '--reference-frame',
        choices=['full', 'protein', 'dna', 'multi'],
        default='full',
        help='Reference frame for alignment (default: full, use "multi" for all three frames)'
    )
    
    advanced_group.add_argument(
        '--output-format',
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Output format for data files (default: csv)'
    )
    
    advanced_group.add_argument(
        '--export-all',
        action='store_true',
        help='Export all intermediate files and detailed analyses'
    )
    
    advanced_group.add_argument(
        '--save-aligned',
        action='store_true',
        help='Save aligned structure PDB files'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and post-process parsed arguments"""
    
    # Handle legacy format
    if args.legacy_files:
        if len(args.legacy_files) != 2:
            print("Error: Legacy format requires exactly 2 files (observed predicted)", file=sys.stderr)
            sys.exit(1)
        
        args.experimental = validate_path(args.legacy_files[0], must_exist=True)
        args.predicted = validate_path(args.legacy_files[1], must_exist=True)
        
        # Set default for legacy mode
        if not any([args.all_benchmarks, args.multi_frame, args.rmsd_only, args.curves, 
                   args.bfactor, args.consensus, args.mutations, args.visualize]):
            args.rmsd_only = True
    
    # Validate experimental/predicted pair
    if args.experimental and not args.predicted:
        print("Error: --predicted/-p is required when using --experimental/-e", file=sys.stderr)
        sys.exit(1)
    
    # Handle multi-frame and reference-frame interaction
    if args.multi_frame or args.reference_frame == 'multi':
        args.multi_frame = True
        args.reference_frame = 'multi'
    
    # All benchmarks includes multi-frame
    if args.all_benchmarks:
        args.multi_frame = True
    
    # Set default analysis if none specified
    if not any([args.all_benchmarks, args.multi_frame, args.rmsd_only, args.curves, 
               args.bfactor, args.consensus, args.mutations, args.visualize]):
        args.rmsd_only = True
    
    # Handle conflicting options
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Detect available CPU cores
    if args.parallel is None:
        args.parallel = min(4, os.cpu_count() or 1)
    
    # Set export flags based on options
    if args.export_all:
        args.save_aligned = True
        args.output_format = 'both'
    
    return args


def arg_parser() -> argparse.Namespace:
    """Main argument parsing function with validation"""
    parser = create_argument_parser()
    args = parser.parse_args()
    return validate_arguments(args)


def get_analysis_flags(args: argparse.Namespace) -> Dict[str, bool]:
    """Extract analysis flags as dictionary"""
    return {
        'rmsd': True,  # Always perform basic RMSD
        'multi_frame': args.multi_frame or args.all_benchmarks,
        'rmsd_only': args.rmsd_only,
        'curves': args.curves or args.all_benchmarks,
        'bfactor': args.bfactor or args.all_benchmarks,
        'consensus': args.consensus or args.all_benchmarks,
        'mutations': args.mutations or args.all_benchmarks,
        'visualize': args.visualize or args.all_benchmarks,
        'all_benchmarks': args.all_benchmarks
    }


def get_structure_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    """Get list of (experimental, predicted) structure file pairs"""
    pairs = []
    
    if args.experimental.is_file() and args.predicted.is_file():
        # Single file pair
        pairs.append((args.experimental, args.predicted))
    elif args.experimental.is_dir() and args.predicted.is_dir():
        # Directory matching
        exp_files = find_structure_files(args.experimental)
        pred_files = find_structure_files(args.predicted)
        
        # Match by filename
        exp_stems = {f.stem: f for f in exp_files}
        pred_stems = {f.stem: f for f in pred_files}
        
        common_stems = set(exp_stems.keys()) & set(pred_stems.keys())
        pairs = [(exp_stems[stem], pred_stems[stem]) for stem in sorted(common_stems)]
        
        if not pairs:
            print(f"Warning: No matching structure pairs found between directories")
    else:
        print("Error: Both experimental and predicted must be files or both must be directories", file=sys.stderr)
        sys.exit(1)
    
    return pairs


def export_multi_frame_results(result, output_dir: Path, format: str = 'both'):
    """Export multi-frame alignment results to CSV and/or JSON"""
    from biostructbenchmark.core.alignment import export_residue_rmsd_csv
    
    if format in ['csv', 'both']:
        # Export CSV for each frame
        export_residue_rmsd_csv(
            result.full_structure.residue_rmsds,
            output_dir / "rmsd_full_structure.csv",
            "full_to_experimental"
        )
        
        export_residue_rmsd_csv(
            result.dna_to_protein.residue_rmsds,
            output_dir / "rmsd_dna_to_protein.csv",
            "dna_to_protein_reference"
        )
        
        export_residue_rmsd_csv(
            result.dna_to_dna.residue_rmsds,
            output_dir / "rmsd_dna_standalone.csv",
            "dna_to_dna"
        )
    
    if format in ['json', 'both']:
        # Create comprehensive JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'version': get_version(),
            'summary': result.get_summary(),
            'detailed': {
                'full_structure': {
                    'overall_rmsd': result.full_structure.overall_rmsd,
                    'atom_count': result.full_structure.aligned_atom_count,
                    'residue_count': len(result.full_structure.residue_rmsds),
                    'protein_residues': len([r for r in result.full_structure.residue_rmsds 
                                            if r.molecule_type == 'protein']),
                    'dna_residues': len([r for r in result.full_structure.residue_rmsds 
                                       if r.molecule_type == 'dna']),
                    'worst_residues': [
                        {'id': r.residue_id, 'rmsd': r.rmsd}
                        for r in sorted(result.full_structure.residue_rmsds, 
                                      key=lambda x: x.rmsd, reverse=True)[:5]
                    ]
                },
                'dna_positioning': {
                    'overall_rmsd': result.dna_to_protein.overall_rmsd,
                    'atom_count': result.dna_to_protein.aligned_atom_count,
                    'interpretation': interpret_dna_positioning(result.dna_to_protein.overall_rmsd)
                },
                'dna_standalone': {
                    'overall_rmsd': result.dna_to_dna.overall_rmsd,
                    'atom_count': result.dna_to_dna.aligned_atom_count,
                    'interpretation': interpret_dna_accuracy(result.dna_to_dna.overall_rmsd)
                },
                'comparative_analysis': generate_comparative_analysis(result)
            }
        }
        
        with open(output_dir / "multi_frame_analysis.json", 'w') as f:
            json.dump(summary, f, indent=2)


def interpret_dna_positioning(rmsd: float) -> str:
    """Interpret DNA positioning RMSD"""
    if rmsd < 3.0:
        return "Excellent - Accurate DNA-protein interface prediction"
    elif rmsd < 5.0:
        return "Good - Minor interface positioning errors"
    elif rmsd < 8.0:
        return "Moderate - Significant interface errors"
    else:
        return "Poor - DNA positioning is unreliable"


def interpret_dna_accuracy(rmsd: float) -> str:
    """Interpret standalone DNA RMSD"""
    if rmsd < 2.0:
        return "Excellent - Near-crystallographic accuracy"
    elif rmsd < 3.0:
        return "Good - Minor geometric distortions"
    elif rmsd < 4.5:
        return "Moderate - Noticeable structural deviations"
    else:
        return "Poor - Major geometric errors"


def generate_comparative_analysis(result) -> str:
    """Generate comparative analysis text"""
    positioning_error = result.dna_to_protein.overall_rmsd
    structural_error = result.dna_to_dna.overall_rmsd
    
    if positioning_error > structural_error * 1.5:
        return "DNA structure is accurate but positioning relative to protein is poor - interface prediction issues"
    elif structural_error > positioning_error * 1.5:
        return "DNA positioning is reasonable but internal structure is distorted - geometry prediction issues"
    else:
        return "DNA structural and positioning errors are comparable - uniform prediction quality"


def print_multi_frame_summary(result):
    """Print formatted summary of multi-frame alignment results"""
    print("\n" + "=" * 70)
    print("MULTI-FRAME ALIGNMENT ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\n1. FULL STRUCTURE ALIGNMENT")
    print(f"   Overall RMSD: {result.full_structure.overall_rmsd:.2f} Å")
    print(f"   Aligned atoms: {result.full_structure.aligned_atom_count}")
    print(f"   Residues analyzed: {len(result.full_structure.residue_rmsds)}")
    
    print(f"\n2. DNA POSITIONING (relative to protein)")
    print(f"   DNA RMSD: {result.dna_to_protein.overall_rmsd:.2f} Å")
    print(f"   Assessment: {interpret_dna_positioning(result.dna_to_protein.overall_rmsd)}")
    
    print(f"\n3. DNA STRUCTURE (standalone)")
    print(f"   DNA RMSD: {result.dna_to_dna.overall_rmsd:.2f} Å")
    print(f"   Assessment: {interpret_dna_accuracy(result.dna_to_dna.overall_rmsd)}")
    
    print(f"\n4. COMPARATIVE ANALYSIS")
    print(f"   {generate_comparative_analysis(result)}")
    
    print("\n" + "=" * 70)


# Main execution function
def main():
    """Main execution function with multi-frame support"""
    args = arg_parser()
    analysis_flags = get_analysis_flags(args)
    
    # Print header if not quiet
    if not args.quiet:
        print(f"BioStructBenchmark v{get_version()}")
        print("=" * 70)
    
    # Get structure pairs
    structure_pairs = get_structure_pairs(args)
    
    if not structure_pairs:
        print("Error: No structure pairs to analyze", file=sys.stderr)
        return 1
    
    # Process each pair
    for exp_path, pred_path in structure_pairs:
        if args.verbose:
            print(f"\nProcessing: {exp_path.name} vs {pred_path.name}")
        
        # Create output subdirectory for this pair
        pair_output = args.output / f"{exp_path.stem}_vs_{pred_path.stem}"
        pair_output.mkdir(parents=True, exist_ok=True)
        
        # Multi-frame alignment
        if analysis_flags['multi_frame']:
            from biostructbenchmark.core.alignment import perform_multi_frame_alignment
            
            if not args.quiet:
                print("\nPerforming multi-frame alignment analysis...")
            
            # Determine if we should save aligned structures
            alignment_output = pair_output / "alignments" if args.save_aligned else None
            
            result = perform_multi_frame_alignment(
                exp_path, pred_path, alignment_output
            )
            
            if result:
                # Print summary
                if not args.quiet:
                    print_multi_frame_summary(result)
                
                # Export results
                export_multi_frame_results(result, pair_output, args.output_format)
                
                # Visualization if requested
                if analysis_flags['visualize']:
                    try:
                        from biostructbenchmark.visualization.plots import PublicationPlotter
                        plotter = PublicationPlotter()
                        
                        # Create multi-frame dashboard
                        data_dict = {
                            'rmsd': result.full_structure.residue_rmsds,
                            'dna_positioning': result.dna_to_protein.residue_rmsds,
                            'dna_standalone': result.dna_to_dna.residue_rmsds
                        }
                        
                        plotter.summary_dashboard(
                            data_dict,
                            pair_output / "multi_frame_dashboard.png"
                        )
                        
                        if not args.quiet:
                            print(f"Saved visualization to {pair_output / 'multi_frame_dashboard.png'}")
                    except ImportError:
                        print("Warning: Visualization module not available")
        
        # Single-frame alignment (backward compatibility)
        elif analysis_flags['rmsd_only'] or args.reference_frame != 'multi':
            from biostructbenchmark.core.alignment import compare_structures
            
            if not args.quiet:
                print(f"\nPerforming {args.reference_frame} frame alignment...")
            
            result = compare_structures(exp_path, pred_path)
            
            if result:
                if not args.quiet:
                    print(f"Overall RMSD: {result.overall_rmsd:.2f} Å")
                    print(f"Aligned atoms: {result.aligned_atom_count}")
                
                # Export basic results
                if args.output_format in ['csv', 'both']:
                    from biostructbenchmark.core.alignment import export_residue_rmsd_csv
                    export_residue_rmsd_csv(
                        result.residue_rmsds,
                        pair_output / "rmsd_analysis.csv",
                        args.reference_frame
                    )
        
        # Additional analyses
        if analysis_flags['curves']:
            if not args.quiet:
                print("\nPerforming CURVES+ analysis...")
            # Import and run CURVES+ analysis
            try:
                from biostructbenchmark.analysis.curves import run_curves_analysis
                curves_result = run_curves_analysis(exp_path, pred_path, pair_output)
                if not args.quiet and curves_result:
                    print("CURVES+ analysis complete")
            except ImportError:
                print("Warning: CURVES+ module not available")
        
        if analysis_flags['bfactor']:
            if not args.quiet:
                print("\nAnalyzing B-factors...")
            try:
                from biostructbenchmark.analysis.bfactor import analyze_bfactors
                bfactor_result = analyze_bfactors(exp_path, pred_path, pair_output)
                if not args.quiet and bfactor_result:
                    print("B-factor analysis complete")
            except ImportError:
                print("Warning: B-factor module not available")
    
    if not args.quiet:
        print("\n" + "=" * 70)
        print(f"Analysis complete! Results saved to: {args.output}")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())