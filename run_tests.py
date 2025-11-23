"""
Test runner for Petri Net assignment
"""
import unittest
import sys
import os
import importlib

def clear_module_cache():
    """Clear cache for our custom modules to avoid import issues"""
    modules_to_clear = ['Task1_pnml_parser', 'Task2_explicit', 'Task3_bdd_reach', 
                       'Task4_deadlock', 'Task5_optimize', 'model', 'utils']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

def run_tests():
    """Discover and run all tests"""
    # Clear module cache first
    clear_module_cache()
    
    # Add src to Python path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Change to project root directory
    os.chdir(os.path.dirname(__file__))
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Test dd availability at top level
    try:
        from dd.autoref import BDD
        print("✓ dd available at test runner level")
    except ImportError as e:
        print(f"✗ dd NOT available at test runner level: {e}")
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)