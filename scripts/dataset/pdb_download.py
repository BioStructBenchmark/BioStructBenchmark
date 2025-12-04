#!/usr/bin/env python3
"""
Download mmCIF structures from RCSB PDB and extract sequences.

Usage:
    uv run python scripts/dataset/pdb_download.py --input search_results.json --output-dir data/experimental
    uv run python scripts/dataset/pdb_download.py --pdb-ids 8ABC,8DEF --output-dir data/experimental
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gemmi
import httpx

# Add parent to path for biostructbenchmark imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biostructbenchmark.core.sequences import (
    AMINO_ACID_MAP,
    DNA_NUCLEOTIDE_MAP,
    classify_chains,
    get_dna_sequence,
    get_protein_sequence,
)

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


def download_structure(pdb_id: str, output_dir: Path, format: str = "cif") -> Path | None:
    """Download structure file from RCSB."""
    pdb_id = pdb_id.upper()
    ext = "cif" if format == "cif" else "pdb"
    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.{ext}"
    output_path = output_dir / f"{pdb_id.lower()}.{ext}"

    if output_path.exists():
        print(f"  {pdb_id}: Already exists, skipping")
        return output_path

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        try:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return output_path
        except httpx.HTTPError as e:
            print(f"  {pdb_id}: Download failed - {e}")
            return None


def extract_sequences(structure_path: Path) -> dict:
    """Extract protein and DNA sequences from a structure file."""
    try:
        structure = gemmi.read_structure(str(structure_path))
    except Exception as e:
        return {"error": str(e)}

    protein_chains, dna_chains = classify_chains(structure)

    sequences = {
        "protein": {},
        "dna": {},
        "protein_chains": protein_chains,
        "dna_chains": dna_chains,
    }

    for chain_id in protein_chains:
        seq = get_protein_sequence(structure, chain_id)
        if seq:
            sequences["protein"][chain_id] = seq

    for chain_id in dna_chains:
        seq = get_dna_sequence(structure, chain_id)
        if seq:
            sequences["dna"][chain_id] = seq

    return sequences


def validate_structure(structure_path: Path) -> dict:
    """Validate that structure has expected protein-DNA content."""
    try:
        structure = gemmi.read_structure(str(structure_path))
    except Exception as e:
        return {"valid": False, "error": str(e)}

    protein_chains, dna_chains = classify_chains(structure)

    if not protein_chains:
        return {"valid": False, "error": "No protein chains found"}
    if not dna_chains:
        return {"valid": False, "error": "No DNA chains found"}

    # Count residues
    protein_residues = 0
    dna_residues = 0
    for model in structure:
        for chain in model:
            if chain.name in protein_chains:
                protein_residues += sum(
                    1 for r in chain if r.name in AMINO_ACID_MAP
                )
            elif chain.name in dna_chains:
                dna_residues += sum(
                    1 for r in chain if r.name in DNA_NUCLEOTIDE_MAP
                )

    return {
        "valid": True,
        "protein_chains": len(protein_chains),
        "dna_chains": len(dna_chains),
        "protein_residues": protein_residues,
        "dna_residues": dna_residues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PDB structures and extract sequences"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input JSON file from pdb_search.py",
    )
    parser.add_argument(
        "--pdb-ids",
        type=str,
        help="Comma-separated list of PDB IDs",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for structures",
    )
    parser.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format",
    )
    parser.add_argument(
        "--extract-sequences",
        action="store_true",
        help="Extract and save sequences",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate downloaded structures",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of downloads",
    )
    args = parser.parse_args()

    # Get PDB IDs
    pdb_ids = []
    if args.input:
        data = json.loads(args.input.read_text())
        pdb_ids = [e["pdb_id"] for e in data.get("entries", [])]
    elif args.pdb_ids:
        pdb_ids = [p.strip().upper() for p in args.pdb_ids.split(",")]
    else:
        print("Error: Must provide --input or --pdb-ids")
        return 1

    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    print(f"Downloading {len(pdb_ids)} structures to {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = args.output_dir / "sequences"
    if args.extract_sequences:
        sequences_dir.mkdir(exist_ok=True)

    results = {"downloaded": [], "failed": [], "validated": []}

    for i, pdb_id in enumerate(pdb_ids):
        print(f"[{i + 1}/{len(pdb_ids)}] {pdb_id}")

        # Download
        path = download_structure(pdb_id, args.output_dir, args.format)
        if not path:
            results["failed"].append(pdb_id)
            continue

        results["downloaded"].append(pdb_id)

        # Validate
        if args.validate:
            validation = validate_structure(path)
            if validation["valid"]:
                results["validated"].append(
                    {
                        "pdb_id": pdb_id,
                        **validation,
                    }
                )
            else:
                print(f"  Validation failed: {validation.get('error')}")

        # Extract sequences
        if args.extract_sequences:
            sequences = extract_sequences(path)
            if "error" not in sequences:
                seq_file = sequences_dir / f"{pdb_id.lower()}_sequences.json"
                seq_file.write_text(json.dumps(sequences, indent=2))

    # Summary
    print(f"\nSummary:")
    print(f"  Downloaded: {len(results['downloaded'])}")
    print(f"  Failed: {len(results['failed'])}")
    if args.validate:
        print(f"  Validated: {len(results['validated'])}")

    # Save results
    results_file = args.output_dir / "download_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
