# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioStructBenchmark is a Python tool for analyzing protein-DNA complexes. It compares experimentally-determined structures and computationally-predicted structures to analyze the differences between them. Analyses include alignment and per-residue rmsd to determine orientation vs translational errors. It processes structural biology files (PDB and CIF formats) using BioPython for validation and parsing.

## Before Coding
- Ask the user clarifying questions.
- If the task requires significant domain-specific knowledge, research online in bioinformatics-related sources.

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv (macOS), pipx install uv, or pip install uv

# Install project with all dev dependencies
uv sync

# Activate the virtual environment (optional - uv run handles this automatically)
source .venv/bin/activate
```

### Testing
- Always create Pytest unit tests for new features (functions, classes, etc).
- Tests live in `tests/`
```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_io.py   # Run specific test file
```

### Linting and Formatting
```bash
uv run black .                   # Format code
uv run pylint biostructbenchmark # Lint code
```

### Building and Distribution
```bash
uv build                 # Build package
uv run twine upload dist/*      # Upload to PyPI
```

### Core Structure
- `biostructbenchmark/__main__.py`: Entry point that orchestrates CLI parsing and core functionality
- `biostructbenchmark/cli.py`: Argument parsing with file validation
- `biostructbenchmark/core/io.py`: File format detection, validation, and BioPython parsing

### Key Design Patterns
- **File Validation Pipeline**: Files go through multiple validation stages in `cli.py` (existence, permissions, size) then format validation in `core/io.py`
- **Parser Selection**: Auto-detection of file format (.pdb/.cif) with corresponding BioPython parser selection
- **Graceful Error Handling**: Invalid files are skipped with informative error messages rather than crashing

### Dependencies
- **BioPython**: Core dependency for structural file parsing (MMCIFParser, PDBParser)
- **Python 3.13+**: Required minimum version
- **Test Framework**: pytest with test data in `tests/data/`

### Current Limitations
- Single file processing only (batch processing TODO)
- Missing atom/residue handling not implemented
- Only supports PDB and CIF formats