#!/usr/bin/env python3
"""
Profiling script for BioStructBenchmark with flamegraph generation.

Usage:
    # Run with py-spy (sampling profiler, low overhead):
    sudo py-spy record -o profile.svg --native -- python scripts/profile_benchmark.py

    # Run with scalene (line-level profiler):
    scalene --html --outfile profile.html scripts/profile_benchmark.py

    # Run standalone to verify it works:
    python scripts/profile_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biostructbenchmark.core.io import get_structure
from biostructbenchmark.core.sequences import classify_chains
from biostructbenchmark.core.alignment import align_protein_dna_complex


def find_structure_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find experimental/predicted structure pairs for benchmarking."""
    pairs = []

    # Look for paired structures in complexes directory
    complexes_dir = data_dir / "complexes"
    if complexes_dir.exists():
        exp_files = list(complexes_dir.glob("experimental_*.cif"))
        for exp_file in exp_files:
            pdb_id = exp_file.stem.replace("experimental_", "")
            pred_file = complexes_dir / f"predicted_{pdb_id}.cif"
            if pred_file.exists():
                pairs.append((exp_file, pred_file))

    return pairs


def benchmark_parsing(files: list[Path]) -> dict:
    """Benchmark structure parsing performance."""
    results = {"success": 0, "failed": 0, "total_time": 0.0}

    for f in files:
        start = time.perf_counter()
        try:
            structure = get_structure(f)
            if structure is not None:
                # Force full traversal to get accurate timing
                atom_count = sum(
                    1 for model in structure for chain in model
                    for res in chain for atom in res
                )
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception:
            results["failed"] += 1
        results["total_time"] += time.perf_counter() - start

    return results


def benchmark_alignment(pairs: list[tuple[Path, Path]]) -> dict:
    """Benchmark full alignment pipeline."""
    results = {"success": 0, "failed": 0, "total_time": 0.0, "alignments": []}

    for exp_path, pred_path in pairs:
        start = time.perf_counter()
        try:
            exp_struct = get_structure(exp_path)
            pred_struct = get_structure(pred_path)

            if exp_struct and pred_struct:
                result = align_protein_dna_complex(exp_struct, pred_struct)
                results["success"] += 1
                results["alignments"].append({
                    "name": exp_path.stem,
                    "rmsd": result.structural_rmsd,
                    "residues": len(result.sequence_mapping),
                })
            else:
                results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            print(f"  Failed {exp_path.name}: {e}")

        results["total_time"] += time.perf_counter() - start

    return results


def main():
    """Main profiling entry point."""
    data_dir = Path("tests/data")

    print("=" * 60)
    print("BioStructBenchmark Profiling Run")
    print("=" * 60)

    # Collect all structure files
    all_files = list(data_dir.rglob("*.cif")) + list(data_dir.rglob("*.pdb"))
    print(f"\nFound {len(all_files)} structure files")

    # Find pairs for alignment
    pairs = find_structure_pairs(data_dir)
    print(f"Found {len(pairs)} experimental/predicted pairs")

    # Phase 1: Parse all structures
    print("\n--- Phase 1: Parsing all structures ---")
    start = time.perf_counter()
    parse_results = benchmark_parsing(all_files)
    phase1_time = time.perf_counter() - start

    print(f"Parsed {parse_results['success']}/{len(all_files)} structures")
    print(f"Total time: {phase1_time:.2f}s ({phase1_time/len(all_files)*1000:.2f} ms/file)")

    # Phase 2: Run alignments on pairs
    if pairs:
        print("\n--- Phase 2: Running alignments ---")
        start = time.perf_counter()
        align_results = benchmark_alignment(pairs)
        phase2_time = time.perf_counter() - start

        print(f"Aligned {align_results['success']}/{len(pairs)} pairs")
        print(f"Total time: {phase2_time:.2f}s ({phase2_time/len(pairs)*1000:.2f} ms/pair)")

        for align in align_results["alignments"]:
            print(f"  {align['name']}: RMSD={align['rmsd']:.3f}Å, {align['residues']} residues")

    # Phase 3: Intensive workload (for profiling hot paths)
    print("\n--- Phase 3: Intensive parsing (5 iterations) ---")
    sample_files = all_files[:100]  # Use subset for intensive profiling
    start = time.perf_counter()
    for i in range(5):
        benchmark_parsing(sample_files)
        print(f"  Iteration {i+1}/5 complete")
    phase3_time = time.perf_counter() - start
    print(f"Total time: {phase3_time:.2f}s")

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
