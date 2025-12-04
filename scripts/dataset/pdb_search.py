#!/usr/bin/env python3
"""
Query RCSB PDB for protein-DNA complexes matching benchmark criteria.

Usage:
    uv run python scripts/dataset/pdb_search.py --output results.json
    uv run python scripts/dataset/pdb_search.py --min-date 2020-01-01 --max-resolution 2.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"


def build_search_query(
    min_date: str = "2020-01-01",
    max_resolution: float = 2.5,
    exclude_rna: bool = True,
) -> dict:
    """Build RCSB Search API query for protein-DNA complexes."""
    nodes = [
        # Resolution filter
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less",
                "value": max_resolution,
            },
        },
        # Must have DNA
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_DNA",
                "operator": "greater",
                "value": 0,
            },
        },
        # Must have protein
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "greater",
                "value": 0,
            },
        },
        # Release date filter
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater",
                "value": min_date,
            },
        },
    ]

    # Optionally exclude RNA-containing structures
    if exclude_rna:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                    "operator": "equals",
                    "value": 0,
                },
            }
        )

    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"]},
    }


def search_pdb(query: dict) -> list[str]:
    """Execute search query and return list of PDB IDs."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(RCSB_SEARCH_URL, json=query)
        response.raise_for_status()
        data = response.json()

    return [hit["identifier"] for hit in data.get("result_set", [])]


def get_entry_metadata(pdb_id: str) -> dict | None:
    """Fetch metadata for a single PDB entry."""
    url = f"{RCSB_DATA_URL}/{pdb_id}"
    with httpx.Client(timeout=10.0) as client:
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None


def extract_entry_info(metadata: dict) -> dict:
    """Extract relevant fields from PDB entry metadata."""
    entry = metadata.get("rcsb_entry_info", {})
    accession = metadata.get("rcsb_accession_info", {})

    return {
        "pdb_id": metadata.get("rcsb_id", ""),
        "title": metadata.get("struct", {}).get("title", ""),
        "resolution": entry.get("resolution_combined", [None])[0],
        "release_date": accession.get("initial_release_date", ""),
        "protein_chains": entry.get("polymer_entity_count_protein", 0),
        "dna_chains": entry.get("polymer_entity_count_DNA", 0),
        "rna_chains": entry.get("polymer_entity_count_RNA", 0),
        "experimental_method": entry.get("experimental_method", ""),
        "molecular_weight": entry.get("molecular_weight", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search RCSB PDB for protein-DNA complexes"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file", default=None
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2020-01-01",
        help="Minimum release date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-resolution",
        type=float,
        default=2.5,
        help="Maximum resolution in Angstroms",
    )
    parser.add_argument(
        "--include-rna",
        action="store_true",
        help="Include structures with RNA",
    )
    parser.add_argument(
        "--fetch-metadata",
        action="store_true",
        help="Fetch detailed metadata for each entry",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of results"
    )
    args = parser.parse_args()

    print(f"Searching for protein-DNA complexes...")
    print(f"  Min date: {args.min_date}")
    print(f"  Max resolution: {args.max_resolution}A")
    print(f"  Exclude RNA: {not args.include_rna}")

    query = build_search_query(
        min_date=args.min_date,
        max_resolution=args.max_resolution,
        exclude_rna=not args.include_rna,
    )

    pdb_ids = search_pdb(query)
    print(f"\nFound {len(pdb_ids)} structures")

    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
        print(f"Limited to {len(pdb_ids)} structures")

    if args.fetch_metadata:
        print("\nFetching metadata...")
        results = []
        for i, pdb_id in enumerate(pdb_ids):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(pdb_ids)}")
            metadata = get_entry_metadata(pdb_id)
            if metadata:
                results.append(extract_entry_info(metadata))
    else:
        results = [{"pdb_id": pdb_id} for pdb_id in pdb_ids]

    output = {
        "query_date": datetime.now().isoformat(),
        "query_params": {
            "min_date": args.min_date,
            "max_resolution": args.max_resolution,
            "exclude_rna": not args.include_rna,
        },
        "count": len(results),
        "entries": results,
    }

    if args.output:
        args.output.write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
