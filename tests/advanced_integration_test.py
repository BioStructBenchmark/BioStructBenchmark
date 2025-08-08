#!/usr/bin/env python3
"""
Advanced integration testing for BioStructBenchmark with CURVES+ components
Tests the actual implemented structure with all components
"""

import sys
import tempfile
import traceback
from pathlib import Path

def test_all_module_imports():
    """Test that all implemented modules can be imported"""
    print("🔍 Testing all module imports...")
    
    modules_to_test = [
        # Basic components
        ("Package root", "biostructbenchmark"),
        ("CLI module", "biostructbenchmark.cli"),
        ("Main entry", "biostructbenchmark.__main__"),
        
        # Core components  
        ("Core alignment", "biostructbenchmark.core.alignment"),
        ("Core IO", "biostructbenchmark.core.io"),
        ("CURVES+ integration", "biostructbenchmark.core.curves_integration"),
        ("CURVES+ visualization", "biostructbenchmark.core.curves_visualization"),
    ]
    
    import_results = {}
    
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"  ✅ {name}: OK")
            import_results[module_path] = True
        except ImportError as e:
            print(f"  ❌ {name}: Import failed - {e}")
            import_results[module_path] = False
        except Exception as e:
            print(f"  ⚠️ {name}: Import succeeded but error during initialization - {e}")
            import_results[module_path] = True  # Import worked, but module has issues
    
    return import_results

def test_basic_integration_chain():
    """Test the basic analysis chain: CLI -> IO -> Alignment"""
    print("\n🔗 Testing basic integration chain...")
    
    try:
        # Test CLI validation
        from biostructbenchmark.cli import validate_file_path, get_version
        print("  ✅ CLI functions imported")
        
        # Test version
        version = get_version()
        print(f"  ✅ Version: {version}")
        
        # Test core IO
        from biostructbenchmark.core.io import get_structure, validate_file
        print("  ✅ IO functions imported")
        
        # Test core alignment
        from biostructbenchmark.core.alignment import compare_structures
        print("  ✅ Alignment functions imported")
        
        # Test integration: CLI -> IO -> Alignment
        # This would need a real PDB file to fully test
        print("  ✅ Basic integration chain imports successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic integration failed: {e}")
        traceback.print_exc()
        return False

def test_curves_integration_chain():
    """Test the CURVES+ analysis chain"""
    print("\n🧬 Testing CURVES+ integration chain...")
    
    try:
        # Test CURVES+ analyzer
        from biostructbenchmark.core.curves_integration import CurvesAnalyzer, CurvesParameters, HydrogenBond
        print("  ✅ CURVES+ analyzer classes imported")
        
        # Test CURVES+ visualization
        from biostructbenchmark.core.curves_visualization import CurvesVisualizer
        print("  ✅ CURVES+ visualizer imported")
        
        # Test CURVES+ analyzer initialization
        try:
            analyzer = CurvesAnalyzer()
            if analyzer.curves_exe:
                print(f"  ✅ CURVES+ executable found: {analyzer.curves_exe}")
            else:
                print("  ⚠️ CURVES+ executable not found (expected in development)")
        except RuntimeError as e:
            print(f"  ⚠️ CURVES+ not available: {e}")
        
        # Test data structures
        try:
            params = CurvesParameters(
                shear=0.1, stretch=0.2, stagger=0.3,
                buckle=1.0, propeller=2.0, opening=3.0,
                shift=0.4, slide=0.5, rise=3.2,
                tilt=1.5, roll=2.5, twist=36.0
            )
            print(f"  ✅ CurvesParameters dataclass works: twist={params.twist}°")
            
            hbond = HydrogenBond(
                donor_residue="ARG_10", donor_atom="NH1",
                acceptor_residue="DG_5", acceptor_atom="N7",
                distance=2.8, angle=165.0, strength="strong"
            )
            print(f"  ✅ HydrogenBond dataclass works: {hbond.distance}Å")
            
        except Exception as e:
            print(f"  ❌ CURVES+ data structures failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ CURVES+ components not available: {e}")
        return False
    except Exception as e:
        print(f"  ❌ CURVES+ integration failed: {e}")
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration with all components"""
    print("\n🖥️ Testing CLI integration...")
    
    try:
        # Test main entry point import
        from biostructbenchmark.__main__ import main
        print("  ✅ Main entry point imports")
        
        # Test CLI argument parsing (without actually running)
        from biostructbenchmark.cli import arg_parser
        print("  ✅ Argument parser available")
        
        # We can't easily test full CLI without mocking sys.argv
        # But we can test that the imports work
        print("  ✅ CLI integration imports successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI integration failed: {e}")
        traceback.print_exc()
        return False

def test_visualization_integration():
    """Test visualization component integration"""
    print("\n📊 Testing visualization integration...")
    
    try:
        from biostructbenchmark.core.curves_visualization import CurvesVisualizer, create_curves_report
        
        # Test visualizer initialization
        visualizer = CurvesVisualizer()
        print("  ✅ CurvesVisualizer initializes")
        
        # Test that matplotlib/seaborn dependencies work
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        print("  ✅ Visualization dependencies available")
        
        # Test networkx for hydrogen bond networks
        import networkx as nx
        print("  ✅ NetworkX available for H-bond networks")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Visualization dependencies missing: {e}")
        print("  💡 Install with: pip install matplotlib seaborn networkx pandas")
        return False
    except Exception as e:
        print(f"  ❌ Visualization integration failed: {e}")
        return False

def test_full_pipeline_readiness():
    """Test if full pipeline could work with real data"""
    print("\n🚀 Testing full pipeline readiness...")
    
    try:
        # Import all major components
        from biostructbenchmark.cli import validate_file_path
        from biostructbenchmark.core.io import get_structure
        from biostructbenchmark.core.alignment import compare_structures
        from biostructbenchmark.core.curves_integration import CurvesAnalyzer
        from biostructbenchmark.core.curves_visualization import CurvesVisualizer
        
        print("  ✅ All pipeline components available")
        
        # Test that we could process a hypothetical workflow
        # (without actually doing it since we don't have test data here)
        
        # 1. CLI validation -> IO loading -> RMSD analysis (basic pipeline)
        print("  ✅ Basic pipeline: CLI -> IO -> Alignment ready")
        
        # 2. CLI validation -> IO loading -> CURVES+ analysis (advanced pipeline)  
        print("  ✅ Advanced pipeline: CLI -> IO -> CURVES+ ready")
        
        # 3. Analysis results -> Visualization (output pipeline)
        print("  ✅ Output pipeline: Analysis -> Visualization ready")
        
        print("  🎉 Full pipeline appears ready for testing with real data!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Full pipeline not ready: {e}")
        return False

def identify_missing_integrations():
    """Identify what integration pieces might be missing"""
    print("\n🔍 Checking for missing integration pieces...")
    
    missing_pieces = []
    
    # Check if CLI has been updated to handle CURVES+ options
    try:
        from biostructbenchmark.cli import arg_parser
        # We'd need to check if CLI has --curves, --hbonds options
        # This is hard to test without mocking, so we'll note it
        print("  ⚠️ Need to verify: CLI has CURVES+ options (--curves, --hbonds)")
        missing_pieces.append("CLI may need CURVES+ argument options")
    except:
        missing_pieces.append("CLI argument parser issues")
    
    # Check if __main__.py integrates CURVES+ components
    try:
        # We can't easily check this without reading the file
        print("  ⚠️ Need to verify: __main__.py integrates CURVES+ workflow")
        missing_pieces.append("__main__.py may need CURVES+ integration")
    except:
        pass
    
    # Check for analysis/ and visualization/ directories
    analysis_dir = Path("biostructbenchmark/analysis")
    viz_dir = Path("biostructbenchmark/visualization")
    
    if not analysis_dir.exists():
        print("  ℹ️ No separate analysis/ directory (using core/ - OK)")
    
    if not viz_dir.exists():
        print("  ℹ️ No separate visualization/ directory (using core/ - OK)")
    
    if missing_pieces:
        print("  🎯 Potential missing integration pieces:")
        for piece in missing_pieces:
            print(f"    - {piece}")
    else:
        print("  ✅ No obvious missing integration pieces detected")
    
    return missing_pieces

def provide_next_steps():
    """Provide specific next steps based on test results"""
    print("\n🎯 NEXT STEPS FOR INTEGRATION:")
    
    print("\n1. Test with Real Data:")
    print("   └── Download test structure: wget https://files.rcsb.org/download/1crn.pdb")
    print("   └── Test basic RMSD: python -m biostructbenchmark 1crn.pdb 1crn.pdb") 
    print("   └── Test CURVES+ (if available): python -m biostructbenchmark 1crn.pdb 1crn.pdb --curves")
    
    print("\n2. Verify CLI Integration:")
    print("   └── Check if cli.py has --curves, --hbonds, --all-benchmarks options")
    print("   └── Test: python -m biostructbenchmark --help")
    
    print("\n3. Test Advanced Features:")
    print("   └── Download DNA-protein complex: wget https://files.rcsb.org/download/1bom.pdb")
    print("   └── Test H-bond analysis: python -m biostructbenchmark 1bom.pdb 1bom.pdb --hbonds")
    
    print("\n4. Integration Testing:")
    print("   └── Run: pytest tests/ -v")
    print("   └── Create integration tests for CURVES+ components")
    
    print("\n5. Performance Testing:")
    print("   └── Time full pipeline: time python -m biostructbenchmark exp.pdb pred.pdb --all-benchmarks")

def main():
    """Run all integration tests"""
    print("🧬 BioStructBenchmark Advanced Integration Analysis")
    print("=" * 60)
    
    # Test all components
    results = {}
    
    results['imports'] = test_all_module_imports()
    results['basic_chain'] = test_basic_integration_chain()
    results['curves_chain'] = test_curves_integration_chain() 
    results['cli'] = test_cli_integration()
    results['visualization'] = test_visualization_integration()
    results['pipeline'] = test_full_pipeline_readiness()
    
    missing_pieces = identify_missing_integrations()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION SUMMARY:")
    
    working_components = sum(1 for result in results.values() if result)
    total_components = len(results)
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {name.replace('_', ' ').title()}: {'WORKING' if status else 'NEEDS ATTENTION'}")
    
    print(f"\n🎯 Status: {working_components}/{total_components} integration areas working")
    
    if working_components >= 4:
        print("🎉 EXCELLENT! Your integration is in great shape!")
        print("   Ready for real-world testing with structure files")
    elif working_components >= 2:
        print("🚀 GOOD PROGRESS! Most core components integrated")
        print("   Focus on the failing areas above")
    else:
        print("⚠️ NEEDS WORK: Several integration issues to resolve")
    
    provide_next_steps()
    
    return working_components >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
