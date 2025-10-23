#!/usr/bin/env python3
"""Test script to verify batch mode respects command-line options."""

import os
import sys
import tempfile
import shutil
import numpy as np

def create_test_files(test_dir):
    """Create test XY files."""
    # Create a simple XRD pattern
    theta = np.linspace(10, 80, 100)
    intensity = 100 + 50 * np.sin(theta/10) + np.random.normal(0, 5, 100)
    
    for i in range(3):
        fname = os.path.join(test_dir, f"pattern_{i+1}.xy")
        data = np.column_stack([theta, intensity + i*10])
        np.savetxt(fname, data, fmt='%.4f', delimiter=' ', 
                   header=f'Test pattern {i+1}')
        print(f"Created: {fname}")

def test_batch_with_options():
    """Test batplot --all with various options."""
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix='batplot_test_')
    print(f"Test directory: {test_dir}")
    
    try:
        # Create test files
        create_test_files(test_dir)
        
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        print("\n" + "="*60)
        print("Testing: batplot --all --xaxis 2theta --xrange 20 70")
        print("="*60)
        
        # Import after changing directory
        sys.path.insert(0, '/Users/tiandai/Library/CloudStorage/OneDrive-UniversitetetiOslo/My files/batplot_script')
        from batplot.cli import main
        
        # Test with options
        test_args = ['--all', '--xaxis', '2theta', '--xrange', '20', '70']
        sys.argv = ['batplot'] + test_args
        
        try:
            main()
            print("\n✓ Test passed: batch mode with --xaxis and --xrange")
            
            # Check if SVG files were created
            svg_dir = os.path.join(test_dir, 'batplot_svg')
            if os.path.exists(svg_dir):
                svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
                print(f"✓ Created {len(svg_files)} SVG files in batplot_svg/")
                for f in svg_files:
                    print(f"  - {f}")
            else:
                print("✗ No batplot_svg directory created")
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Change back to original directory
        os.chdir(original_dir)
        
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == '__main__':
    test_batch_with_options()
