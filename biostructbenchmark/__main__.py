#!/usr/bin/env python3

"""Entry point for biostructbenchmark"""

from biostructbenchmark.core.alignment import align_protein_dna_complex
from biostructbenchmark.cli import arg_parser
from biostructbenchmark.core.io import get_structure


def main() -> None:
    """Main entry point for structure comparison."""
    args = arg_parser()

    print("Loading structures...")
    experimental_structure = get_structure(args.file_path_observed)
    computational_structure = get_structure(args.file_path_predicted)

    print("Performing protein-DNA complex alignment...")
    alignment_result = align_protein_dna_complex(
        experimental_structure, 
        computational_structure,
        output_dir=args.output_dir,
        save_structures=args.save_structures
    )

    print("\n=== CAPRI-COMPLIANT METRICS ===")
    print(f"DockQ Score:                 {alignment_result.dockq:.3f}")
    print(f"i-RMSD (Interface Backbone): {alignment_result.capri_i_rmsd:.3f} Å")
    print(f"l-RMSD (Ligand Backbone):    {alignment_result.capri_l_rmsd:.3f} Å")
    print(f"fnat (Native Contacts):      {alignment_result.fnat:.3f} ({alignment_result.fnat*100:.1f}%)")
    print(f"Backbone RMSD (Overall):     {alignment_result.backbone_rmsd:.3f} Å")

    # DockQ quality classification (primary)
    if alignment_result.dockq >= 0.80:
        quality = "High/Very High Quality"
    elif alignment_result.dockq >= 0.49:
        quality = "Medium Quality"
    elif alignment_result.dockq >= 0.23:
        quality = "Acceptable"
    else:
        quality = "Incorrect"

    print(f"\nDockQ Classification: {quality}")

    print("\n=== ALL-ATOM METRICS (Detailed) ===")
    print(f"Overall RMSD:    {alignment_result.structural_rmsd:.3f} Å")
    print(f"Protein RMSD:    {alignment_result.protein_rmsd:.3f} Å")
    print(f"DNA RMSD:        {alignment_result.dna_rmsd:.3f} Å")
    print(f"Interface RMSD:  {alignment_result.interface_rmsd:.3f} Å")

    print("\n=== ERROR DECOMPOSITION ===")
    print(f"Orientation Error:     {alignment_result.orientation_error:.2f}°")
    print(f"Translational Error:   {alignment_result.translational_error:.3f} Å")

    print(f"\n=== STRUCTURE INFORMATION ===")
    print(f"Protein Chains: {', '.join(alignment_result.protein_chains)}")
    print(f"DNA Chains: {', '.join(alignment_result.dna_chains)}")
    print(f"Aligned Residues: {len(alignment_result.sequence_mapping)}")

    if alignment_result.interface_residues:
        total_interface = sum(
            len(residues) for residues in alignment_result.interface_residues.values()
        )
        print(f"Interface Residues: {total_interface}")

    print(f"\nPer-residue RMSD statistics:")
    if alignment_result.per_residue_rmsd:
        rmsds = list(alignment_result.per_residue_rmsd.values())
        print(f"  Min: {min(rmsds):.3f} Å")
        print(f"  Max: {max(rmsds):.3f} Å")
        print(f"  Mean: {sum(rmsds)/len(rmsds):.3f} Å")

    # Display output file information if structures were saved
    if alignment_result.output_files:
        exp_path, comp_path = alignment_result.output_files
        print(f"\n=== OUTPUT FILES ===")
        print(f"Experimental structure: {exp_path}")
        print(f"Aligned computational structure: {comp_path}")

    # Perform B-factor analysis if requested
    if args.analyze_bfactor:
        from biostructbenchmark.analysis.bfactor import BFactorAnalyzer
        from pathlib import Path

        print("\nPerforming B-factor/pLDDT analysis...")
        analyzer = BFactorAnalyzer()

        try:
            comparisons, stats = analyzer.analyze_structures(
                experimental_structure,
                computational_structure
            )

            print("\n=== B-FACTOR ANALYSIS ===")
            print(f"Mean Experimental B-factor: {stats.mean_experimental:.2f}")
            print(f"Mean Predicted Confidence (pLDDT): {stats.mean_predicted:.2f}")
            print(f"Correlation: {stats.correlation:.3f}")
            print(f"RMSD: {stats.rmsd:.2f}")
            print(f"High Confidence Accuracy (pLDDT>70): {stats.high_confidence_accuracy:.2f}")
            print(f"Low Confidence Accuracy (pLDDT≤70): {stats.low_confidence_accuracy:.2f}")
            print(f"Total Residues Analyzed: {len(comparisons)}")

            # Determine output path
            if args.bfactor_output:
                csv_path = Path(args.bfactor_output)
            elif args.output_dir:
                csv_path = Path(args.output_dir) / "analysis" / "bfactor_comparison.csv"
            else:
                csv_path = Path("./analysis/bfactor_comparison.csv")

            # Save CSV
            analyzer.save_to_csv(comparisons, csv_path)
            print(f"\nB-factor comparison saved to: {csv_path}")

        except ValueError as e:
            print(f"\nB-factor analysis failed: {e}")
        except Exception as e:
            print(f"\nUnexpected error during B-factor analysis: {e}")


if __name__ == "__main__":
    main()
