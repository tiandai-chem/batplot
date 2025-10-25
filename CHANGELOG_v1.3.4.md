# Changelog for v1.3.4 (Draft)

## Major Changes

### 1. Normalization Behavior Overhaul

#### Removed `--raw` Flag
- The `--raw` flag has been completely removed from the codebase.

#### Added `--norm` Flag
- New `--norm` flag for explicit normalization (0-1 range).
- Use `batplot file.xy --norm` to normalize intensity.

#### Changed Default Behavior

**XY Plots (Regular Mode)**
- **OLD**: Auto-normalized by default (use `--raw` to disable)
- **NEW**: NO auto-normalization (use `--norm` to normalize)
- This allows plotting raw intensity values without modification.

**Stack Mode (`--stack`)**
- **Unchanged**: Auto-normalizes data (required for stacking)
- Normalization happens automatically for `--stack` mode.

**Operando Mode (`--operando`)**
- **OLD**: Auto-normalized (0-1) unless `--raw` specified
- **NEW**: NO normalization applied to raw data
- Z-scale (colorbar) automatically spans from minimum to maximum intensity
- In interactive mode, commands like `oz` adjust Z-scale relative to current X/Y range while preserving actual intensity values

**Batch Mode (`--all`)**
- **NEW**: Respects `--norm` flag
- Example: `batplot --all --norm` exports normalized plots
- Default: exports raw intensity plots

#### Updated Files
- `args.py`: Updated help messages and added `--norm` argument
- `batplot.py`: Changed normalization logic for XY plots
- `batch.py`: Updated to use `--norm` instead of `--raw`
- `operando.py`: Removed normalization, kept Z-scale auto-adjustment
- `session.py`: Updated session handling to save/restore `norm` flag
- `interactive.py`: Updated X-range zoom to respect new normalization logic

### 2. Operando Interactive Mode Without EC Data

#### New Behavior
- **Interactive mode now works without `.mpt` file**: The operando interactive menu is available even when there's no electrochemistry data
- **Dynamic menu**: Shows different options based on whether EC panel exists
- **EC commands disabled gracefully**: Commands like `el`, `ew`, `et`, `ey`, `er` show helpful message when no EC data is present

#### Operando-Only Menu Features
When no `.mpt` file is found, the interactive menu now includes:
- **All operando commands**: `oc`, `ow`, `ox`, `oy`, `oz`, `or` work as before
- **Tick & spine controls**: `t` command for toggle axes works for operando panel
- **Additional decoration commands**: `d1`-`d5` for advanced tick styling:
  - `d1`: tick & spine visibility
  - `d2`: tick width
  - `d3`: tick length
  - `d4`: minor ticks
  - `d5`: nice ticks (auto)
- **Style/geometry export**: `p`, `i`, `e`, `s` commands all functional

#### Technical Changes
- `batplot.py`: Always calls `operando_ec_interactive_menu`, even when `ec_ax` is None
- `operando_ec_interactive.py`:
  - Updated `_get_geometry_snapshot` to handle missing EC panel
  - Updated `_ensure_fixed_params` to work with `ec_ax=None`
  - Updated `_apply_group_layout_inches` to skip EC positioning when not present
  - Added guards to EC-specific commands (`el`, `ew`, `et`, `ey`, `er`)
  - Updated `print_menu` to show context-appropriate menu
  - Updated `t` command to show only available panes

### 3. Bug Fixes

#### Fixed `oz` Command Error
- **Issue**: `oz` command in operando interactive failed with "cannot access local variable 'np'"
- **Fix**: Initialize `auto_lo`, `auto_hi`, and `auto_available` before try block, use local numpy import
- **File**: `operando_ec_interactive.py`

## Examples

### Normalization Examples
```bash
# XY plots - raw intensity (default)
batplot file1.xy file2.xy

# XY plots - normalized intensity
batplot file1.xy file2.xy --norm

# Stack mode - auto-normalized (unchanged)
batplot file1.xy file2.xy --stack

# Operando - raw intensity with auto-scaled colorbar
batplot --operando --interactive

# Batch mode - raw intensity
batplot --all

# Batch mode - normalized
batplot --all --norm
```

### Operando Interactive Examples
```bash
# Operando with EC data (full menu)
batplot --operando --interactive  # in folder with .mpt file

# Operando without EC data (operando-only menu with d1-d5)
batplot --operando --interactive  # in folder without .mpt file

# Both cases: use 't' for tick toggles, 'ox'/'oy'/'oz' for ranges
```

### Migration Guide
If you have existing scripts using `--raw`:
- **Remove `--raw` flag** (it's now the default behavior)
- If you want normalization, **add `--norm` flag** instead

## Git Backups
- `backup-20251025-1857`: Before any changes
- `backup-norm-changes-20251025-1915`: After normalization changes
- `backup-operando-no-ec-20251025-1954`: After operando interactive improvements (current)

To restore any backup:
```bash
git checkout backup-operando-no-ec-20251025-1954
git checkout main  # return to latest
```
