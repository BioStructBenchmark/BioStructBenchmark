#!/usr/bin/env python3

"""Entry point for biostructbenchmark"""

from pathlib import Path
from biostructbenchmark.core.alignment import compare_structures, export_residue_rmsd_csv
from biostructbenchmark.cli import arg_parser


def main() -> None:
    """Main entry point for structure comparison."""
    args = arg_parser()

    # Perform detailed structure comparison
    result = compare_structures(args.file_path_observed, args.file_path_predicted)

    if result is not None:
        print(f"RMSD: {result.overall_rmsd:.3f} Å")
        print(f"Aligned atoms: {result.aligned_atom_count}")
        print(f"Per-residue analysis: {len(result.residue_rmsds)} residues")
        
        # Show statistics
        protein_residues = [r for r in result.residue_rmsds if r.molecule_type == 'protein']
        dna_residues = [r for r in result.residue_rmsds if r.molecule_type == 'dna']
        
        if protein_residues:
            protein_rmsd_avg = sum(r.rmsd for r in protein_residues) / len(protein_residues)
            print(f"Protein average per-residue RMSD: {protein_rmsd_avg:.3f} Å ({len(protein_residues)} residues)")
        
        if dna_residues:
            dna_rmsd_avg = sum(r.rmsd for r in dna_residues) / len(dna_residues)
            print(f"DNA average per-residue RMSD: {dna_rmsd_avg:.3f} Å ({len(dna_residues)} residues)")
        
        # Export detailed results
        output_path = Path(f"{args.file_path_observed.stem}_vs_{args.file_path_predicted.stem}_residue_rmsd.csv")
        export_residue_rmsd_csv(result.residue_rmsds, output_path)
        print(f"Detailed results exported to: {output_path}")
        
    else:
        print("Error: Unable to compare structures")


if __name__ == "__main__":
    main()