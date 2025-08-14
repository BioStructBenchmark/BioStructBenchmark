"""
test_cli_multiframe.py
Test script to verify CLI integration with multi-frame alignment
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import argparse


class TestCLIMultiFrameIntegration:
    """Test suite for CLI multi-frame alignment integration"""
    
    def test_cli_multi_frame_flag(self):
        """Test that --multi-frame flag is recognized"""
        from biostructbenchmark.cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Test multi-frame flag
        args = parser.parse_args([
            '-e', 'exp.pdb',
            '-p', 'pred.pdb',
            '--multi-frame'
        ])
        
        assert hasattr(args, 'multi_frame')
        assert args.multi_frame == True
    
    def test_cli_reference_frame_multi(self):
        """Test that reference-frame=multi triggers multi-frame"""
        from biostructbenchmark.cli import create_argument_parser, validate_arguments
        
        parser = create_argument_parser()
        
        # Create mock paths
        with patch('biostructbenchmark.cli.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.is_file.return_value = True
            
            args = parser.parse_args([
                '-e', 'exp.pdb',
                '-p', 'pred.pdb',
                '--reference-frame', 'multi'
            ])
            
            # Mock the paths
            args.experimental = Path('exp.pdb')
            args.predicted = Path('pred.pdb')
            args.output = Path('output')
            
            # Validate should set multi_frame=True
            with patch.object(Path, 'mkdir'):
                validated = validate_arguments(args)
                assert validated.multi_frame == True
                assert validated.reference_frame == 'multi'
    
    def test_all_benchmarks_includes_multi_frame(self):
        """Test that --all-benchmarks includes multi-frame analysis"""
        from biostructbenchmark.cli import get_analysis_flags
        
        # Create mock args
        args = Mock()
        args.all_benchmarks = True
        args.multi_frame = False
        args.rmsd_only = False
        args.curves = False
        args.bfactor = False
        args.consensus = False
        args.mutations = False
        args.visualize = False
        
        flags = get_analysis_flags(args)
        
        assert flags['multi_frame'] == True
        assert flags['curves'] == True
        assert flags['bfactor'] == True
    
    def test_export_multi_frame_results(self):
        """Test export of multi-frame results"""
        from biostructbenchmark.cli import export_multi_frame_results
        from biostructbenchmark.core.alignment import (
            AlignmentResult, MultiFrameAlignmentResult, ResidueRMSD
        )
        
        # Create mock results
        mock_residues = [
            ResidueRMSD(
                residue_id="ALA_1",
                residue_type="ALA",
                chain_id="A",
                position=1,
                rmsd=2.5,
                atom_count=5,
                molecule_type="protein"
            )
        ]
        
        mock_alignment = AlignmentResult(
            overall_rmsd=3.0,
            residue_rmsds=mock_residues,
            transformation_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_vector=[0, 0, 0],
            aligned_atom_count=100,
            reference_frame="full"
        )
        
        mock_result = MultiFrameAlignmentResult(
            full_structure=mock_alignment,
            dna_to_protein=mock_alignment,
            dna_to_dna=mock_alignment
        )
        
        # Test export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Export both formats
            export_multi_frame_results(mock_result, output_dir, 'both')
            
            # Check CSV files exist
            assert (output_dir / "rmsd_full_structure.csv").exists()
            assert (output_dir / "rmsd_dna_to_protein.csv").exists()
            assert (output_dir / "rmsd_dna_standalone.csv").exists()
            
            # Check JSON file exists
            assert (output_dir / "multi_frame_analysis.json").exists()
            
            # Verify JSON content
            import json
            with open(output_dir / "multi_frame_analysis.json") as f:
                data = json.load(f)
                assert 'summary' in data
                assert 'detailed' in data
                assert data['summary']['full_structure_rmsd'] == 3.0
    
    @patch('biostructbenchmark.core.alignment.perform_multi_frame_alignment')
    def test_main_calls_multi_frame(self, mock_multi_align):
        """Test that main() calls multi-frame alignment when flag is set"""
        from biostructbenchmark.__main__ import run_multi_frame_alignment
        
        # Create mock result
        mock_result = Mock()
        mock_result.get_summary.return_value = {
            'full_structure_rmsd': 2.5,
            'dna_positioning_rmsd': 3.0,
            'dna_standalone_rmsd': 2.0
        }
        mock_multi_align.return_value = mock_result
        
        # Create mock args
        args = Mock()
        args.quiet = False
        args.verbose = False
        args.save_aligned = True
        args.output_format = 'both'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Run multi-frame alignment
            result = run_multi_frame_alignment(
                Path('exp.pdb'),
                Path('pred.pdb'),
                output_dir,
                args
            )
            
            # Verify it was called
            assert mock_multi_align.called
            assert result == mock_result.get_summary()
    
    def test_cli_help_shows_multi_frame(self):
        """Test that help text includes multi-frame options"""
        from biostructbenchmark.cli import create_argument_parser
        
        parser = create_argument_parser()
        help_text = parser.format_help()
        
        assert '--multi-frame' in help_text
        assert 'multi-frame alignment' in help_text.lower()
        assert 'reference frames' in help_text.lower()
    
    def test_legacy_compatibility(self):
        """Test that legacy CLI format still works"""
        from biostructbenchmark.cli import create_argument_parser, validate_arguments
        
        parser = create_argument_parser()
        
        # Test legacy positional arguments
        args = parser.parse_args(['obs.pdb', 'pred.pdb'])
        
        assert args.legacy_files == ['obs.pdb', 'pred.pdb']
        
        # Mock file existence
        with patch('biostructbenchmark.cli.validate_path') as mock_validate:
            mock_validate.return_value = Path('test.pdb')
            
            with patch.object(Path, 'mkdir'):
                args.output = Path('output')
                validated = validate_arguments(args)
                
                # Should set experimental and predicted from legacy
                assert validated.experimental is not None
                assert validated.predicted is not None
                assert validated.rmsd_only == True  # Default for legacy


class TestCLIErrorHandling:
    """Test error handling in CLI"""
    
    def test_missing_predicted_error(self):
        """Test error when predicted is missing with experimental"""
        from biostructbenchmark.cli import create_argument_parser, validate_arguments
        
        parser = create_argument_parser()
        args = parser.parse_args(['-e', 'exp.pdb'])
        
        args.predicted = None
        args.legacy_files = []
        args.output = Path('output')
        
        with pytest.raises(SystemExit):
            validate_arguments(args)
    
    def test_verbose_quiet_conflict(self):
        """Test that verbose and quiet are mutually exclusive"""
        from biostructbenchmark.cli import validate_arguments
        
        args = Mock()
        args.verbose = True
        args.quiet = True
        args.experimental = Path('exp.pdb')
        args.predicted = Path('pred.pdb')
        args.legacy_files = []
        args.output = Path('output')
        
        with pytest.raises(SystemExit):
            validate_arguments(args)


def test_integration_workflow():
    """Test complete workflow from CLI to multi-frame alignment"""
    
    # This is a comprehensive integration test
    print("\n" + "=" * 70)
    print("TESTING COMPLETE CLI → MULTI-FRAME WORKFLOW")
    print("=" * 70)
    
    # 1. Test CLI parsing
    print("\n1. Testing CLI argument parsing...")
    from biostructbenchmark.cli import create_argument_parser
    
    parser = create_argument_parser()
    test_args = [
        '-e', 'test_exp.pdb',
        '-p', 'test_pred.pdb',
        '-o', 'test_output',
        '--multi-frame',
        '--export-all',
        '--verbose'
    ]
    
    args = parser.parse_args(test_args)
    print(f"   ✓ Parsed arguments: multi_frame={args.multi_frame}, export_all={args.export_all}")
    
    # 2. Test analysis flags
    print("\n2. Testing analysis flag extraction...")
    from biostructbenchmark.cli import get_analysis_flags
    
    # Mock the args
    args.all_benchmarks = False
    args.rmsd_only = False
    args.curves = False
    args.bfactor = False
    args.consensus = False
    args.mutations = False
    args.visualize = True
    
    flags = get_analysis_flags(args)
    print(f"   ✓ Analysis flags: multi_frame={flags['multi_frame']}, visualize={flags['visualize']}")
    
    # 3. Test result interpretation
    print("\n3. Testing result interpretation...")
    from biostructbenchmark.cli import (
        interpret_dna_positioning,
        interpret_dna_accuracy,
        generate_comparative_analysis
    )
    
    print(f"   DNA positioning (3.5 Å): {interpret_dna_positioning(3.5)}")
    print(f"   DNA accuracy (2.5 Å): {interpret_dna_accuracy(2.5)}")
    
    # Mock result for comparative analysis
    mock_result = Mock()
    mock_result.dna_to_protein.overall_rmsd = 5.0
    mock_result.dna_to_dna.overall_rmsd = 2.0
    
    analysis = generate_comparative_analysis(mock_result)
    print(f"   Comparative: {analysis}")
    
    print("\n" + "=" * 70)
    print("✓ All integration tests passed!")
    print("=" * 70)


if __name__ == '__main__':
    # Run specific integration test
    if '--integration' in sys.argv:
        test_integration_workflow()
    else:
        # Run pytest
        pytest.main([__file__, '-v'])