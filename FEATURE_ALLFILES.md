# New Feature: `allfiles` Command

## Overview

The `allfiles` command plots all XY files in the current directory on the **same figure**, with full support for all command-line flags and interactive mode.

## Comparison of Batch Commands

| Command | What it does | Interactive? | Output | Use case |
|---------|--------------|--------------|--------|----------|
| `batplot file1.xy file2.xy` | Plot specific files together | ✅ Yes | Single figure | Compare selected files |
| `batplot allfiles` | Plot all XY files together | ✅ Yes | Single figure | Compare all patterns at once |
| `batplot --all` | Export each file separately | ❌ No | Multiple SVG files | Prepare publication figures |

## Usage Examples

### Basic Usage
```bash
# Plot all XY files in current directory on same figure
batplot allfiles

# With interactive menu
batplot allfiles --interactive

# Stack all files vertically
batplot allfiles --stack

# Stack with interactive menu
batplot allfiles --stack --interactive
```

### With Options
```bash
# Custom X-axis type
batplot allfiles --xaxis 2theta

# Custom X-axis range
batplot allfiles --xrange 10 80

# With wavelength conversion
batplot allfiles --wl 1.5406

# With vertical offset
batplot allfiles --delta 0.2

# Combined options
batplot allfiles --xaxis 2theta --xrange 10 80 --stack --interactive
```

### Save Output
```bash
# Save to file
batplot allfiles --out comparison.svg

# Save with custom styling
batplot allfiles --stack --out comparison.png --interactive
```

## Supported File Types

All standard XY file formats:
- `.xye`, `.xy`, `.qye`, `.dat`, `.csv` (XRD, general XY data)
- `.gr` (PDF)
- `.nor`, `.chik`, `.chir` (XAS)
- `.txt` (2-column data)

## Features

✅ **All flags supported**: `--stack`, `--interactive`, `--xaxis`, `--xrange`, `--wl`, `--delta`, `--raw`, `--out`

✅ **Interactive menu**: Full access to styling, colors, ranges, fonts, export, session save

✅ **Automatic discovery**: Finds all supported files in current directory

✅ **Sorted by filename**: Files plotted in alphabetical order

✅ **CIF files excluded**: Only XY data files are plotted (CIF overlays must be specified manually)

## Implementation Details

### Code Location
- Main logic: `batplot/batplot.py` (lines ~1120-1132)
- Detects `allfiles` as sole argument
- Finds all XY files in current directory
- Replaces `args.files` with list of found files
- Continues to normal multi-file plotting mode

### File Discovery
```python
supported_ext = {'.xye', '.xy', '.qye', '.dat', '.csv', '.gr', '.nor', '.chik', '.chir', '.txt'}
all_xy_files = [f for f in sorted(os.listdir(os.getcwd()))
               if os.path.splitext(f)[1].lower() in supported_ext 
               and os.path.isfile(os.path.join(os.getcwd(), f))
               and not f.lower().endswith('.cif')]
```

## Documentation Updates

### Files Updated
1. `batplot/args.py` - General help and XY help sections
2. `USER_MANUAL.md` - Added plotting modes comparison table and examples
3. `README.md` - Added quick start examples

### Help Messages
```bash
# View general help
batplot -h

# View XY-specific help
batplot -h xy
```

## Testing

Test script available: `test_allfiles.py`

```bash
# Run test (creates temp directory with sample files)
python test_allfiles.py
```

## Typical Workflows

### Compare Multiple XRD Patterns
```bash
cd /path/to/xrd/patterns
batplot allfiles --xaxis 2theta --stack --interactive
# Adjust colors, offsets, export in interactive menu
```

### Compare PDF Data
```bash
cd /path/to/pdf/data
batplot allfiles --xrange 1 8 --interactive
# Fine-tune ranges and styling
```

### Quick Visual Check
```bash
cd /path/to/data
batplot allfiles
# Quick overlay of all files
```

### Export Comparison Figure
```bash
batplot allfiles --stack --out comparison_figure.svg
# Direct export without interactive menu
```

## Notes

- If no files found, displays error message
- Files are sorted alphabetically before plotting
- All files use same axis type (set via --xaxis or auto-detected)
- Wavelength conversion (--wl) applied to all files
- Interactive menu has full functionality (colors, styles, export, sessions)
