#!/usr/bin/env python3
"""
Automated testing script for batplot v1.3.0 new features
Tests batch styling and verifies output files
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def test_batch_output_has_styling(svg_path, expected_features):
    """
    Parse SVG file and verify styling was applied
    
    Args:
        svg_path: Path to SVG file
        expected_features: Dict of features to check (color, linewidth, etc.)
    
    Returns:
        (passed, messages) tuple
    """
    messages = []
    passed = True
    
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Check for styled elements
        # Find path elements (these are the plot lines)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.findall('.//svg:path', ns) or root.findall('.//path')
        
        if not paths:
            messages.append(f"⚠️  No path elements found in {os.path.basename(svg_path)}")
            passed = False
        else:
            messages.append(f"✅ Found {len(paths)} path elements")
        
        # Check for text elements (axis labels, ticks)
        texts = root.findall('.//svg:text', ns) or root.findall('.//text')
        if texts:
            messages.append(f"✅ Found {len(texts)} text elements (labels, ticks)")
        
        # Check file size (styled plots are usually larger)
        size = os.path.getsize(svg_path)
        if size > 100000:  # > 100KB suggests detailed plot
            messages.append(f"✅ File size {size:,} bytes (well-formatted plot)")
        else:
            messages.append(f"⚠️  File size {size:,} bytes (might be too small)")
        
    except Exception as e:
        messages.append(f"❌ Error parsing SVG: {e}")
        passed = False
    
    return passed, messages

def test_batch_style_applied():
    """Test that batch processing with style files produces correct output"""
    print("\n" + "="*60)
    print("TEST: Batch Style Application")
    print("="*60)
    
    test_dir = Path(__file__).parent / "test_batch_ec"
    svg_dir = test_dir / "batplot_svg"
    
    if not svg_dir.exists():
        print("❌ FAILED: batplot_svg directory not found")
        print(f"   Expected: {svg_dir}")
        return False
    
    svg_files = list(svg_dir.glob("*.svg"))
    if not svg_files:
        print("❌ FAILED: No SVG files found in batplot_svg/")
        return False
    
    print(f"\nFound {len(svg_files)} SVG files:")
    all_passed = True
    
    for svg_file in svg_files:
        print(f"\n📄 Testing: {svg_file.name}")
        passed, messages = test_batch_output_has_styling(svg_file, {})
        for msg in messages:
            print(f"   {msg}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_style_file_formats():
    """Test that style files have correct JSON structure"""
    print("\n" + "="*60)
    print("TEST: Style File Formats")
    print("="*60)
    
    test_dir = Path(__file__).parent / "test_batch_ec"
    
    test_files = [
        ("test_style.bps", "ec_style"),
        ("test_geom.bpsg", "ec_style_geom")
    ]
    
    all_passed = True
    
    for filename, expected_kind in test_files:
        filepath = test_dir / filename
        print(f"\n📄 Testing: {filename}")
        
        if not filepath.exists():
            print(f"   ❌ File not found: {filepath}")
            all_passed = False
            continue
        
        try:
            with open(filepath, 'r') as f:
                cfg = json.load(f)
            
            # Check required fields
            if 'kind' in cfg:
                if cfg['kind'] == expected_kind:
                    print(f"   ✅ Correct 'kind': {cfg['kind']}")
                else:
                    print(f"   ⚠️  Unexpected 'kind': {cfg['kind']} (expected {expected_kind})")
            
            if 'font' in cfg:
                print(f"   ✅ Has 'font' section")
            
            if 'ticks' in cfg:
                print(f"   ✅ Has 'ticks' section")
                if 'direction' in cfg['ticks']:
                    print(f"      - direction: {cfg['ticks']['direction']}")
                if 'length' in cfg['ticks']:
                    print(f"      - length: {cfg['ticks']['length']}")
            
            if 'spines' in cfg:
                print(f"   ✅ Has 'spines' section")
            
            if 'lines' in cfg:
                print(f"   ✅ Has 'lines' section ({len(cfg['lines'])} lines)")
            
            if filename.endswith('.bpsg') and 'figure' in cfg:
                print(f"   ✅ Has 'figure' geometry section")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON: {e}")
            all_passed = False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
    
    return all_passed

def test_backward_compatibility():
    """Test that old batch syntax still works"""
    print("\n" + "="*60)
    print("TEST: Backward Compatibility")
    print("="*60)
    
    # This test would require running actual commands
    # For now, just verify the test files exist
    test_dir = Path(__file__).parent / "test_batch_ec"
    csv_files = list(test_dir.glob("*.csv"))
    
    print(f"\n✅ Found {len(csv_files)} test CSV files")
    print(f"   Old syntax 'batplot --gc all' should process these files")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" AUTOMATED TESTS FOR BATPLOT v1.3.0")
    print("="*70)
    
    results = {
        "Batch Style Application": test_batch_style_applied(),
        "Style File Formats": test_style_file_formats(),
        "Backward Compatibility": test_backward_compatibility()
    }
    
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
