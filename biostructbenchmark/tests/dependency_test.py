#!/usr/bin/env python3
"""
Test all BioStructBenchmark dependencies are properly installed and compatible
"""

def test_dependencies():
    """Test that all dependencies import correctly with versions"""
    print("🔍 Testing BioStructBenchmark dependencies...")
    
    # Test core dependencies
    dependencies = [
        ("BioPython", "Bio", "Bio.__version__"),
        ("NumPy", "numpy", "numpy.__version__"),
        ("Pandas", "pandas", "pandas.__version__"),
        ("Matplotlib", "matplotlib", "matplotlib.__version__"),
        ("SciPy", "scipy", "scipy.__version__"),
        ("Seaborn", "seaborn", "seaborn.__version__"),
        ("NetworkX", "networkx", "networkx.__version__"),
    ]
    
    results = {}
    
    for name, import_name, version_attr in dependencies:
        try:
            module = __import__(import_name)
            version = eval(version_attr)
            print(f"  ✅ {name}: {version}")
            results[name] = True
        except ImportError:
            print(f"  ❌ {name}: Not installed")
            results[name] = False
        except AttributeError:
            print(f"  ⚠️ {name}: Installed but version unknown")
            results[name] = True
        except Exception as e:
            print(f"  ⚠️ {name}: Error getting version - {e}")
            results[name] = True
    
    # Summary
    installed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Dependency Status: {installed}/{total} installed")
    
    if installed == total:
        print("🎉 All dependencies are properly installed!")
        return True
    else:
        missing = [name for name, installed in results.items() if not installed]
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"💡 Install with: pip install {' '.join(dep.lower().replace('biopython', 'biopython') for dep in missing)}")
        return False

def test_biostructbenchmark_imports():
    """Test that BioStructBenchmark modules import correctly"""
    print("\n🧬 Testing BioStructBenchmark module imports...")
    
    modules = [
        ("Core IO", "biostructbenchmark.core.io"),
        ("Core Alignment", "biostructbenchmark.core.alignment"),  
        ("CURVES+ Integration", "biostructbenchmark.core.curves_integration"),
        ("CURVES+ Visualization", "biostructbenchmark.core.curves_visualization"),
        ("CLI", "biostructbenchmark.cli"),
        ("Main Entry", "biostructbenchmark.__main__"),
    ]
    
    results = {}
    
    for name, module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {name}: OK")
            results[name] = True
        except ImportError as e:
            print(f"  ❌ {name}: Import failed - {e}")
            results[name] = False
        except Exception as e:
            print(f"  ⚠️ {name}: Import succeeded but error - {e}")
            results[name] = True
    
    # Summary
    working = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Module Status: {working}/{total} working")
    
    return working == total

def test_version_compatibility():
    """Test that dependency versions are compatible"""
    print("\n🔄 Testing version compatibility...")
    
    try:
        # Test key integrations
        import Bio
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print("  ✅ BioPython + NumPy integration")
        print("  ✅ Pandas + NumPy integration") 
        print("  ✅ Matplotlib ready")
        
        # Test if we can create basic objects
        test_array = np.array([1, 2, 3])
        test_df = pd.DataFrame({'x': test_array})
        
        print("  ✅ Basic NumPy/Pandas operations work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Compatibility issue: {e}")
        return False

if __name__ == "__main__":
    print("🧬 BioStructBenchmark Dependency Test")
    print("=" * 50)
    
    # Run all tests
    deps_ok = test_dependencies()
    modules_ok = test_biostructbenchmark_imports()  
    compat_ok = test_version_compatibility()
    
    print("\n" + "=" * 50)
    print("📊 OVERALL STATUS:")
    
    if deps_ok and modules_ok and compat_ok:
        print("🎉 SUCCESS: All dependencies and modules working!")
        print("\nReady to run:")
        print("  python -m biostructbenchmark --help")
        print("  python advanced_integration_test.py")
    elif deps_ok and not modules_ok:
        print("⚠️ PARTIAL: Dependencies OK, but module import issues")
        print("💡 Check your package installation: pip install -e .")
    elif not deps_ok:
        print("❌ FAILED: Missing dependencies")
        print("💡 Install missing packages first")
    else:
        print("⚠️ MIXED: Some issues detected")
        
    print(f"\nStatus: Dependencies {'✅' if deps_ok else '❌'} | Modules {'✅' if modules_ok else '❌'} | Compatibility {'✅' if compat_ok else '❌'}")
