# batplot

**Interactive plotting tool for battery and materials characterization data**

`batplot` is a Python CLI tool for visualizing and analyzing electrochemical and structural characterization data with interactive styling and session management. The electrochemistry and operando plots were inspired from Amalie Skurtveit's python scripts (https://github.com/piieceofcake?tab=repositories).

## Features

- **Electrochemistry Modes**: Galvanostatic cycling (GC), cyclic voltammetry (CV), differential capacity (dQdV), capacity per cycle (CPC) with multi-file support
- **Structural Characterization**: XRD, PDF, XAS (XANES/EXAFS)
- **Operando Analysis**: Correlate in-situ structural changes with electrochemical data
- **Interactive Menu**: Real-time styling, cycle visibility control, individual color customization for multi-file plots
- **Session Persistence**: Save and reload complete plot states with `.pkl` files
- **Style Management**: Import/export plot styles as `.bpcfg` files
- **Batch Processing**: Plot multiple files simultaneously with automatic SVG export

## Installation

```bash
pip install batplot
```

## Quick Start

### XRD / PDF / XAS

```bash
# Single diffraction pattern
batplot pattern.xye

# Multiple patterns with interactive styling
batplot *.xye --interactive

# Batch mode: export all XY files to SVG
batplot all
```

### Electrochemistry

```bash
# Galvanostatic cycling with interactive menu
batplot battery.csv --gc --interactive

# Cyclic voltammetry
batplot cyclic.mpt --cv --interactive

# Differential capacity
batplot battery.csv --dqdv

# Capacity per cycle - single file
batplot stability.mpt --cpc --mass 5.4 --interactive

# Capacity per cycle - multiple files with individual color control
batplot file1.csv file2.csv file3.mpt --cpc --mass 5.4 --interactive

# Batch processing: export all EC files to SVG
batplot --gc all         # All .mpt/.csv files (.mpt needs --mass, .csv doesn't)
batplot --cv all                    # All .mpt files (CV mode)
batplot --dqdv all                  # All .csv files (dQdV mode)
batplot --cpc all --mass 6.2        # All .mpt/.csv files (.mpt needs --mass, .csv doesn't)
```

### Operando Analysis

```bash
# Correlate in-situ XRD with electrochemistry
# (Place both .xye and .mpt files in same directory)
batplot --operando --interactive
```

## Supported File Formats

| Type | Formats |
|------|---------|
| **Electrochemistry** | `.csv` (Neware), `.mpt` (Biologic) |
| **XRD / PDF** | `.xye`, `.xy`, `.qye`, `.dat` |
| **XAS** | `.nor`, `.chik`, `.chir` |

## Interactive Features

When launched with `--interactive`:
- **Cycle/Scan Control**: Toggle visibility, change colors
- **Styling**: Line widths, markers, transparency
- **Axes**: Labels, limits, ticks, spine styles
- **Export**: Save sessions (`.pkl`), styles (`.bpcfg`), or high-res images
- **Live Preview**: All changes update in real-time

## Documentation

For detailed usage, see [USER_MANUAL.md](USER_MANUAL.md)

## Requirements

- Python ≥ 3.7
- numpy
- matplotlib

## License

See [LICENSE](LICENSE)

## Author

Tian Dai (tianda@uio.no)  
University of Oslo
