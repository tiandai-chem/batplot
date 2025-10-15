# batplot User Manual
**v1.0.18, 2025-01-15**

Batplot is a lightweight CLI tool for plotting XRD, PDF, XAS, electrochemistry, and operando data, featuring interactive and batch modes.
The electrochemistry and operando plotting functions are inspired by the script written by Amalie, Erlend and Casper.

**Supported Python versions:** 3.7–3.13

---

## Installation

```bash
pip install batplot
```

## Table of Contents

1. [Overview](#overview)
2. [XY Mode](#normal-xy-mode)
3. [Electrochemistry Mode](#electrochemistry-mode)
4. [Operando Mode](#operando-mode)

---

## 1. Overview

Batplot supports three main figure types:
- **Normal XY**: For XRD, PDF, XAS, and general 2D data.
- **Electrochemistry (EC)**: For battery cycling and related data.
- **Operando**: For synchronized plotting of structural and electrochemical data.

**Key features:**
- Interactive menu for live editing.
- Save and reuse styles via `.bpcfg` files.
- Save and reload full sessions as `.pkl` files for future editing.

---

## 2. Normal XY Mode

### Supported Inputs

- XRD: `.xye`, `.xy`, `.qye`, `.dat`, `.csv`, `.txt`
- PDF: `.gr`
- XAS: `.nor` (energy), `.chik` (k), `.chir` (FT-EXAFS R)
- Crystallography: `.cif` (reflection ticks/labels only)
- Generic/undefined: `.xy`, `.dat` or other types (will read the first two columns and plot as x and y)

### Example Usage

```bash
batplot file1.xye:1.54 file2.qye
# Plot two files; .xye is converted to Q with wavelength 1.54 Å

batplot file1.xye file2.dat --wl 1.54
# Plot two 2theta files in Q space with wavelength 1.54 Å

batplot file1.xye file2.xye --xaxis 2theta --interactive
# Plot with 2theta as X axis and open interactive menu

batplot file1.xye:0.25995 file2.qye --stack --interactive
# Stack two files and open the interactive menu

batplot file1.xye:0.25995 file2.qye --stack --interactive
# Stack two files and open the interactive menu

batplot file1.xye:0.25995 file2.qye structure1.cif structure2.cif --stack --interactive
# Stack two files with reference cif ticks and open the interactive menu

batplot all
# Plot all supported files in the current folder
```

---

## 3. Electrochemistry Mode

### Supported Inputs

- Neware `.csv` (GC, dQdV, CPC)
- Biologic `.mpt` (GC, CV, CPC)

### Plotting Modes

**GC (Galvanostatic Cycling)**: Voltage vs. capacity plots showing charge/discharge cycles.

**CV (Cyclic Voltammetry)**: Voltage vs. current plots for electrochemical characterization. Supports full interactive menu with cycle-by-cycle styling, colors, visibility control, and session save/load.

**dQdV**: Differential capacity analysis (dQ/dV vs. voltage).

**CPC (Chronopotentiometry/Chronoamperometry)**: Time-domain electrochemical measurements.

### Example Usage

```bash
batplot file.csv --gc --interactive
# Plot GC data with interactive menu

batplot file.csv --dqdv
# Plot dQdV curve

batplot file.mpt --cpc --mass 6.2 --interactive
# Plot CPC from .mpt; capacity calculated using provided mass (mg)
# For .csv, mass is not required

batplot file.mpt --cv --interactive
# Plot CV data with full interactive menu support
```

### Batch Mode

Export all EC files in a directory to SVG format:

```bash
batplot --gc all --mass 7.0
# Process all .mpt and .csv files in current directory (GC mode)
# Note: --mass only required for .mpt files; .csv files already contain capacity data
# Outputs saved to batplot_svg/ subdirectory

batplot --cv all
# Process all .mpt files (CV mode)

batplot --dqdv all
# Process all .csv files (dQdV mode)

batplot --cpc all --mass 5.4
# Process all .mpt and .csv files (CPC mode)
# Note: --mass only required for .mpt files

batplot --gc /path/to/folder --mass 6.0
# Process files in specific directory
```

**Note**: 
- Batch mode automatically exports SVG plots to `batplot_svg/` subdirectory
- For GC and CPC modes: `.csv` files don't need `--mass` (capacity already in file)
- For GC and CPC modes: `.mpt` files require `--mass` parameter
- Interactive mode (`--interactive`) is only available for single-file plotting

---

## 4. Operando Mode

### Requirements

- Place both operando files (`.xye`, `.qye`, `.xy`, `.dat`) and EC files (`.mpt`) in the same directory.
- Navigate to the folder before running Batplot.

### Example Usage

```bash
batplot --operando --interactive
# Launch operando mode with interactive editing

batplot --operando --wl 0.25995 --interactive
# Launch operando mode with interactive editing, converting x axis from 2theta to Q space
```