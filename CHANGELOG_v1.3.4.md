# Changelog for v1.3.4 (Draft)

## Major Changes: Normalization Behavior

### 1. Removed `--raw` Flag
- The `--raw` flag has been completely removed from the codebase.

### 2. Added `--norm` Flag
- New `--norm` flag for explicit normalization (0-1 range).
- Use `batplot file.xy --norm` to normalize intensity.

### 3. Changed Default Behavior

#### XY Plots (Regular Mode)
- **OLD**: Auto-normalized by default (use `--raw` to disable)
- **NEW**: NO auto-normalization (use `--norm` to normalize)
- This allows plotting raw intensity values without modification.

#### Stack Mode (`--stack`)
- **Unchanged**: Auto-normalizes data (required for stacking)
- Normalization happens automatically for `--stack` mode.

#### Operando Mode (`--operando`)
- **OLD**: Auto-normalized (0-1) unless `--raw` specified
- **NEW**: NO normalization applied to raw data
- Z-scale (colorbar) automatically spans from minimum to maximum intensity
- In interactive mode, commands like `oz` adjust Z-scale relative to current X/Y range while preserving actual intensity values

#### Batch Mode (`--all`)
- **NEW**: Respects `--norm` flag
- Example: `batplot --all --norm` exports normalized plots
- Default: exports raw intensity plots

### 4. Updated Files
- `args.py`: Updated help messages and added `--norm` argument
- `batplot.py`: Changed normalization logic for XY plots
- `batch.py`: Updated to use `--norm` instead of `--raw`
- `operando.py`: Removed normalization, kept Z-scale auto-adjustment
- `session.py`: Updated session handling to save/restore `norm` flag
- `interactive.py`: Updated X-range zoom to respect new normalization logic

### Examples

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

### Migration Guide
If you have existing scripts using `--raw`:
- **Remove `--raw` flag** (it's now the default behavior)
- If you want normalization, **add `--norm` flag** instead

## Git Backups
- Backup tag created: `backup-norm-changes-20251025-1915`
- Previous backup: `backup-20251025-1857`
- To restore: `git checkout backup-norm-changes-20251025-1915`
