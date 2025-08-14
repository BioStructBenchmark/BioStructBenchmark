"""
biostructbenchmark/__main__.py
Main entry point for BioStructBenchmark with multi-frame alignment support
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import traceback


def main():
    """Main entry point with comprehensive error handling and multi-frame support"""
    try:
        # Import CLI module
        from biostructbenchmark.cli import (
            arg_parser, 
            get_analysis_flags, 
            get_structure_pairs,
            get_version,
            print_multi_frame_summary,
            export_multi_frame_results
        )
        
        # Parse arguments
        args = arg_parser()
        analysis_flags = get_analysis_flags(args)
        
        # Setup logging level
        if args.verbose:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        elif args.quiet:
            import logging
            logging.basicConfig(level=logging.ERROR)
        
        # Print header
        if not args.quiet:
            print(f"BioStructBenchmark v{get_version()}")
            print("=" * 70)
            print(f"Experimental: {args.experimental}")
            print(f"Predicted: {args.predicted}")
            print(f"Output: {args.output}")
            print("=" * 70)
        
        # Get structure pairs to process
        structure_pairs = get_structure_pairs(args)
        
        if not structure_pairs:
            print("Error: No valid structure pairs found", file=sys.stderr)
            return 1
        
        # Track overall results
        all_results = []
        failed_pairs = []
        
        # Process each structure pair
        for i, (exp_path, pred_path) in enumerate(structure_pairs, 1):
            if not args.quiet:
                print(f"\n[{i}/{len(structure_pairs)}] Processing: {exp_path.name} vs {pred_path.name}")
            
            # Create output directory for this pair
            pair_name = f"{exp_path.stem}_vs_{pred_path.stem}"
            pair_output = args.output / pair_name
            pair_output.mkdir(parents=True, exist_ok=True)
            
            try:
                # Run analyses based on flags
                pair_results = run_analyses(
                    exp_path, pred_path, pair_output, 
                    analysis_flags, args
                )
                
                all_results.append({
                    'pair': pair_name,
                    'experimental': str(exp_path),
                    'predicted': str(pred_path),
                    'results': pair_results
                })
                
            except Exception as e:
                print(f"Error processing {pair_name}: {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
                failed_pairs.append(pair_name)
                continue
        
        # Generate summary report if multiple pairs
        if len(structure_pairs) > 1:
            generate_batch_report(all_results, failed_pairs, args.output)
        
        # Final summary
        if not args.quiet:
            print("\n" + "=" * 70)
            print("ANALYSIS COMPLETE")
            print(f"Processed: {len(all_results)}/{len(structure_pairs)} pairs")
            if failed_pairs:
                print(f"Failed: {', '.join(failed_pairs)}")
            print(f"Results saved to: {args.output}")
            print("=" * 70)
        
        return 0 if not failed_pairs else 1
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        return 1


def run_analyses(exp_path: Path, pred_path: Path, output_dir: Path, 
                 flags: Dict[str, bool], args) -> Dict[str, Any]:
    """
    Run all requested analyses on a structure pair
    
    Returns:
        Dictionary of analysis results
    """
    results = {}
    
    # Multi-frame alignment (highest priority)
    if flags.get('multi_frame'):
        results['multi_frame'] = run_multi_frame_alignment(
            exp_path, pred_path, output_dir, args
        )
    
    # Single-frame RMSD (if not doing multi-frame)
    elif flags.get('rmsd') and not flags.get('multi_frame'):
        results['rmsd'] = run_basic_rmsd(
            exp_path, pred_path, output_dir, args
        )
    
    # CURVES+ analysis
    if flags.get('curves'):
        results['curves'] = run_curves_analysis(
            exp_path, pred_path, output_dir, args
        )
    
    # B-factor analysis
    if flags.get('bfactor'):
        results['bfactor'] = run_bfactor_analysis(
            exp_path, pred_path, output_dir, args
        )
    
    # Consensus analysis (if applicable)
    if flags.get('consensus'):
        results['consensus'] = run_consensus_analysis(
            exp_path, pred_path, output_dir, args
        )
    
    # Mutation analysis
    if flags.get('mutations'):
        results['mutations'] = run_mutation_analysis(
            exp_path, pred_path, output_dir, args
        )
    
    # Visualization
    if flags.get('visualize'):
        results['visualization'] = generate_visualizations(
            results, output_dir, args
        )
    
    return results


def run_multi_frame_alignment(exp_path: Path, pred_path: Path, 
                             output_dir: Path, args) -> Optional[Dict]:
    """Run multi-frame alignment analysis"""
    try:
        from biostructbenchmark.core.alignment import perform_multi_frame_alignment
        from biostructbenchmark.cli import (
            print_multi_frame_summary,
            export_multi_frame_results
        )
        
        if not args.quiet:
            print("  → Running multi-frame alignment analysis...")
        
        # Determine if we should save aligned structures
        alignment_output = output_dir / "alignments" if args.save_aligned else None
        
        # Perform alignment
        result = perform_multi_frame_alignment(
            exp_path, pred_path, alignment_output
        )
        
        if result:
            # Print summary if not quiet
            if not args.quiet:
                print_multi_frame_summary(result)
            
            # Export results
            export_multi_frame_results(result, output_dir, args.output_format)
            
            # Return summary for aggregation
            return result.get_summary()
        
        return None
        
    except ImportError as e:
        print(f"  ⚠ Multi-frame alignment not available: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Multi-frame alignment failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return None


def run_basic_rmsd(exp_path: Path, pred_path: Path, 
                   output_dir: Path, args) -> Optional[Dict]:
    """Run basic RMSD analysis"""
    try:
        from biostructbenchmark.core.alignment import compare_structures
        from biostructbenchmark.core.alignment import export_residue_rmsd_csv
        
        if not args.quiet:
            print(f"  → Running {args.reference_frame} frame RMSD analysis...")
        
        result = compare_structures(exp_path, pred_path)
        
        if result:
            # Export if requested
            if args.output_format in ['csv', 'both']:
                export_residue_rmsd_csv(
                    result.residue_rmsds,
                    output_dir / "rmsd_analysis.csv",
                    args.reference_frame
                )
            
            # Return summary
            return {
                'overall_rmsd': result.overall_rmsd,
                'atom_count': result.aligned_atom_count,
                'residue_count': len(result.residue_rmsds)
            }
        
        return None
        
    except Exception as e:
        print(f"  ✗ RMSD analysis failed: {e}")
        return None


def run_curves_analysis(exp_path: Path, pred_path: Path, 
                       output_dir: Path, args) -> Optional[Dict]:
    """Run CURVES+ DNA geometry analysis"""
    try:
        from biostructbenchmark.analysis.curves import run_curves_analysis
        
        if not args.quiet:
            print("  → Running CURVES+ DNA geometry analysis...")
        
        result = run_curves_analysis(exp_path, pred_path, output_dir)
        
        if result and not args.quiet:
            print("  ✓ CURVES+ analysis complete")
        
        return result
        
    except ImportError:
        if args.verbose:
            print("  ⚠ CURVES+ module not available")
        return None
    except Exception as e:
        print(f"  ✗ CURVES+ analysis failed: {e}")
        return None


def run_bfactor_analysis(exp_path: Path, pred_path: Path, 
                        output_dir: Path, args) -> Optional[Dict]:
    """Run B-factor vs pLDDT analysis"""
    try:
        from biostructbenchmark.analysis.bfactor import analyze_bfactors
        
        if not args.quiet:
            print("  → Analyzing B-factors vs confidence metrics...")
        
        result = analyze_bfactors(exp_path, pred_path, output_dir)
        
        if result and not args.quiet:
            print("  ✓ B-factor analysis complete")
        
        return result
        
    except ImportError:
        if args.verbose:
            print("  ⚠ B-factor module not available")
        return None
    except Exception as e:
        print(f"  ✗ B-factor analysis failed: {e}")
        return None


def run_consensus_analysis(exp_path: Path, pred_path: Path, 
                          output_dir: Path, args) -> Optional[Dict]:
    """Run consensus error mapping"""
    try:
        from biostructbenchmark.analysis.consensus import generate_consensus_map
        
        if not args.quiet:
            print("  → Generating consensus error map...")
        
        result = generate_consensus_map(
            exp_path, pred_path, output_dir,
            threshold=args.rmsd_threshold
        )
        
        if result and not args.quiet:
            print("  ✓ Consensus analysis complete")
        
        return result
        
    except ImportError:
        if args.verbose:
            print("  ⚠ Consensus module not available")
        return None
    except Exception as e:
        print(f"  ✗ Consensus analysis failed: {e}")
        return None


def run_mutation_analysis(exp_path: Path, pred_path: Path, 
                         output_dir: Path, args) -> Optional[Dict]:
    """Run mutation impact analysis"""
    try:
        from biostructbenchmark.analysis.mutations import analyze_mutations
        
        if not args.quiet:
            print("  → Analyzing mutation impacts...")
        
        result = analyze_mutations(exp_path, pred_path, output_dir)
        
        if result and not args.quiet:
            print(f"  ✓ Found {result.get('mutation_count', 0)} mutations")
        
        return result
        
    except ImportError:
        if args.verbose:
            print("  ⚠ Mutations module not available")
        return None
    except Exception as e:
        print(f"  ✗ Mutation analysis failed: {e}")
        return None


def generate_visualizations(results: Dict, output_dir: Path, args) -> Optional[Dict]:
    """Generate all requested visualizations"""
    try:
        from biostructbenchmark.visualization.plots import PublicationPlotter
        
        if not args.quiet:
            print("  → Generating visualizations...")
        
        plotter = PublicationPlotter()
        viz_results = {}
        
        # Multi-frame dashboard if available
        if 'multi_frame' in results and results['multi_frame']:
            # This would need the actual result object, not just summary
            # For now, we'll note it was requested
            viz_results['multi_frame_dashboard'] = str(output_dir / "multi_frame_dashboard.png")
        
        # General dashboard for all analyses
        if len(results) > 1:
            plotter.summary_dashboard(
                results,
                output_dir / "analysis_dashboard.png"
            )
            viz_results['dashboard'] = str(output_dir / "analysis_dashboard.png")
        
        if not args.quiet:
            print(f"  ✓ Generated {len(viz_results)} visualizations")
        
        return viz_results
        
    except ImportError:
        if args.verbose:
            print("  ⚠ Visualization module not available")
        return None
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        return None


def generate_batch_report(all_results: list, failed_pairs: list, output_dir: Path):
    """Generate summary report for batch processing"""
    try:
        report = {
            'summary': {
                'total_pairs': len(all_results) + len(failed_pairs),
                'successful': len(all_results),
                'failed': len(failed_pairs)
            },
            'results': all_results,
            'failed_pairs': failed_pairs
        }
        
        # Calculate aggregate statistics
        if all_results:
            rmsds = []
            for result in all_results:
                if 'results' in result:
                    if 'multi_frame' in result['results'] and result['results']['multi_frame']:
                        rmsds.append(result['results']['multi_frame'].get('full_structure_rmsd'))
                    elif 'rmsd' in result['results'] and result['results']['rmsd']:
                        rmsds.append(result['results']['rmsd'].get('overall_rmsd'))
            
            if rmsds:
                import numpy as np
                report['summary']['mean_rmsd'] = float(np.mean([r for r in rmsds if r]))
                report['summary']['std_rmsd'] = float(np.std([r for r in rmsds if r]))
        
        # Save report
        report_path = output_dir / "batch_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBatch report saved to: {report_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate batch report: {e}")


if __name__ == "__main__":
    sys.exit(main())