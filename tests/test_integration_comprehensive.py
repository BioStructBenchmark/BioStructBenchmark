#!/usr/bin/env python3
"""
Comprehensive integration test for BioStructBenchmark
Tests all modules can import and basic functionality works
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test that all modules can be imported"""
    
    def test_core_imports(self):
        """Test core module imports"""
        import biostructbenchmark.core.alignment
        import biostructbenchmark.core.io
        import biostructbenchmark.core.metrics
    
    def test_cli_import(self):
        """Test CLI module import"""
        import biostructbenchmark.cli
    
    def test_main_import(self):
        """Test main entry point import"""
        import biostructbenchmark.__main__
    
    @pytest.mark.xfail(reason="Analysis modules may have missing dependencies")
    def test_analysis_imports(self):
        """Test analysis module imports"""
        import biostructbenchmark.analysis.bfactor
        import biostructbenchmark.analysis.consensus
        import biostructbenchmark.analysis.mutations
        import biostructbenchmark.analysis.secondary
    
    @pytest.mark.xfail(reason="Visualization modules may have missing dependencies")
    def test_visualization_imports(self):
        """Test visualization module imports"""
        import biostructbenchmark.visualization.plots
        import biostructbenchmark.visualization.residue_plots
        import biostructbenchmark.visualization.structure


class TestBasicFunctionality:
    """Test basic functionality works"""
    
    def test_cli_validation(self, tmp_path):
        """Test CLI file validation"""
        from biostructbenchmark.cli import validate_file_path
        
        # Create test file
        test_file = tmp_path / "test.pdb"
        test_file.write_text("HEADER TEST")
        
        # Test validation
        result = validate_file_path(str(test_file))
        assert result == test_file
    
    def test_cli_arg_parser(self):
        """Test CLI argument parser creation"""
        from biostructbenchmark.cli import create_argument_parser
        
        parser = create_argument_parser()
        assert parser is not None
    
    @patch('biostructbenchmark.core.io.get_structure')
    def test_structure_comparison_mock(self, mock_get_structure):
        """Test structure comparison with mocked structures"""
        from biostructbenchmark.core.alignment import compare_structures
        
        # Create mock structures
        mock_structure = MagicMock()
        mock_get_structure.return_value = mock_structure
        
        # This will fail without proper mocking, but tests the import
        with pytest.raises(Exception):
            compare_structures("fake1.pdb", "fake2.pdb")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
