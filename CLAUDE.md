# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioStructBenchmark is a Python tool for analyzing protein-DNA complexes. It compares experimentally-determined structures and computationally-predicted structures to analyze the differences between them. Analyses include alignment and per-residue rmsd to determine orientation vs translational errors. It processes structural biology files (PDB and CIF formats) using BioPython for validation and parsing.

## Before Coding
- Ask the user clarifying questions.
- If the task requires significant domain-specific knowledge, research online in bioinformatics-related sources.

### Installation
```bash
pip install -e .[dev]
```

### Testing
- Always create Pytest unit tests for new features (functions, classes, etc).
- Tests live in `tests/`
```bash
pytest                    # Run all tests
pytest tests/test_io.py   # Run specific test file
```

### Linting and Formatting
```bash
black .                   # Format code
pylint biostructbenchmark # Lint code
```

### Building and Distribution
```bash
python -m build          # Build package
twine upload dist/*      # Upload to PyPI
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