# Batplot User Manual (v1.0.13)

Batplot is a lightweight CLI tool for plotting XRD, PDF, and XAS data with an interactive editing mode.

Supported inputs:
- XRD-like data: `.xye`, `.xy`, `.qye`, `.dat`, `.csv`, `.txt`
- PDF data: `.gr`
- XAS data: `.nor` (energy), `.chik` (k), `.chir` (FT-EXAFS R)
- Crystallography: `.cif` (renders reflection ticks/labels; not intensity curves)

Python 3.7–3.13 are supported.

---

## Installation

From PyPI (recommended):

```bash
python -m pip install -U batplot
```

If a venv is used on macOS with spaces in the path (e.g., OneDrive), prefer module invocations (`python -m ...`).

Editable (local dev):

```bash
cd "/path/to/batplot_script"
python -m pip install -U pip
python -m pip install -e .
```

---

## Quick start

- Plot two datasets and export an SVG:

```bash
batplot a.qye b.qye --delta 0.2 --out figure.svg
```

- Enter interactive mode to edit the figure:

```bash
batplot a.qye b.qye --delta 0.2 --interactive
```

- Load a saved interactive session (`.pkl`) exactly as saved:

```bash
batplot mysession.pkl            # static
batplot mysession.pkl --interactive  # interactive
```

Note: On some systems, the `batplot` shim can be stale; use the module runner:

```bash
python -m batplot.cli --help
```

---

## File types and axis selection

Batplot chooses the X-axis domain automatically from file types:
- `.qye` → Q
- `.gr` → r (Å)
- `.nor` → energy (eV)
- `.chik` → k (Å⁻¹)
- `.chir` → r (FT-EXAFS R)
- `.txt` → generic 2-column; requires `--xaxis` (Q, 2theta, r, k, energy, rft)
- `.cif` → simulated reflection ticks in Q by default; convert to 2θ when plotting in 2θ

You may not mix fundamentally different axis domains in one run (e.g., `.gr` with `.nor`). If needed, split into separate runs.

### Wavelength and 2θ
- Provide a wavelength with each filename (`file.xye:1.5406`) or globally via `--wl 1.5406`.
- For `.cif` used in 2θ mode, wavelength is required to convert Q ↔ 2θ (uses saved wl or `--wl`, default 1.5406 Å).

---

## Command-line options (common)

```text
--interactive          Keep the window open and enter interactive menu
--stack                Stack curves top→bottom (waterfall)
--delta, -d FLOAT      Vertical offset spacing (default 0.0 unless --stack)
--autoscale            Scale delta by curve’s Y-span when not stacked
--xrange, -r MIN MAX   Initial X range (also available in interactive ‘x’)
--raw                  Plot raw Y instead of normalized [0..1]
--xaxis TYPE           Override axis type (Q, 2theta, r, k, energy, rft)
--wl FLOAT             Wavelength (Å) for Q/2θ conversions
--fullprof ...         FullProf matrix: xstart xend xstep [wavelength]
--convert, -c FILES    Convert to .qye (requires --wl)
--out, -o PATH         Write image (SVG by default if no extension)
--savefig PATH         Pickle the Matplotlib Figure object (advanced)
```

Examples:

```bash
# Two curves in Q (global wavelength for .xye)
batplot a.xye b.qye --wl 1.5406 --delta 0.2 --interactive

# 2θ mode with explicit axis
a.xye b.xye --xaxis 2theta --wl 1.5406 --out plot.svg

# Generic .txt requires an axis choice
batplot data.txt --xaxis Q

# Convert to .qye
batplot --convert file1.xye file2.xy --wl 1.5406
```

Paths with spaces (macOS/OneDrive):

```bash
batplot "/Users/.../My files/a.qye" "/Users/.../My files/b.qye"
```

---

## Batch mode

- `batplot DIR` → export all supported files in `DIR` to `DIR/batplot_svg/*.svg`
- `batplot all` → export all supported files in the current directory

---

## Interactive mode (keyboard)

Press keys in the terminal after launching with `--interactive`.

Main commands:
- `c` Colors: manual per-curve colors; or apply a colormap to a range; change CIF tick colors (if present)
- `f` Font: change size or family (Arial/Helvetica/Times/STIX/DejaVu)
- `l` Lines: set curve line widths; frame/tick widths; dots-only or line+dots style
- `t` Toggles: ticks/labels on each side, axis titles (bottom/top X, left/right Y), and frame spines
- `a` Rearrange curves: reorder by indices (preserves axis title visibility)
- `x` Set X-range; `y` Set Y-range (disabled in `--stack` for Y)
- `d` Delta/offset spacing (disabled in `--stack`)
- `r` Rename: curve labels, axis labels, CIF tick set names
- `g` Resize: `p` plot frame inside fixed canvas; `c` canvas size
- `v` Find peaks: lists peaks within a given X window and threshold
- `n` Crosshair: show movable crosshair with live readout (Q/2θ only)
- `p` Print style info; export style
- `i` Import style from `.bpcfg` (applies fonts, spines, ticks, lines, etc.)
- `e` Export image (temporarily removes numbering from labels)
- `s` Save interactive session to `.pkl`
- `z` Toggle hkl labels on CIF ticks (safety limits when too many peaks)
- `b` Undo last change
- `q` Quit interactive

Tips:
- Minor ticks can be toggled independently (mbx/mtx/mly/mry in `t`).
- Duplicate axis titles (top X, right Y) are supported and respect tick-side spacing.

---

## Sessions (`.pkl`) and fidelity

- `s` saves an interactive session with data, labels, styles, tick states, spines, fonts, figure/canvas size, and CIF metadata.
- Load with:

```bash
batplot session.pkl
batplot session.pkl --interactive
```

Loaded sessions reproduce the saved appearance (including minor ticks, duplicated axis titles, spines, figure size/dpi). In 2θ, CIF ticks are drawn by converting stored Q positions using the saved or provided wavelength.

---

## Styles (`.bpcfg`)

- In `p` (style) menu, press `e` to export a style file.
- Apply a style in another session with `i` and selecting the `.bpcfg` file.
- Styles include: figure/canvas, margins, fonts, ticks/minor ticks, spines, line styles, and label layout.

---

## CIF ticks and hkl labels

- Add `.cif` files alongside data to draw reflection ticks (Q or 2θ depending on current axis).
- In mixed mode (data + CIF), only ticks are drawn for CIF; intensities are not plotted.
- `z` toggles hkl labels on/off; label density is automatically limited if peaks are too many.
- Hover tooltips (when enabled) show the nearest CIF tick and hkl label.

For 2θ, provide a wavelength (saved in sessions and used on reload).

---

## Troubleshooting

- “Mixed axis types” error: split into separate runs (e.g., `.gr` with `.nor` is not allowed together).
- `.txt` without `--xaxis`: specify the domain (`--xaxis Q` or `2theta`, etc.).
- CLI shim issues (no response or wrong Python): use `python -m batplot.cli ...`.
- Headless export: omit `--interactive` and use `--out file.svg`.
- OneDrive/paths with spaces: quote paths or the interpreter in VS Code tasks.
- Pip script failures (Exit 126): always prefer `python -m pip ...`.

---

## Examples

- Stacked plot with CIF ticks in Q:

```bash
batplot sample1.qye sample2.qye structure.cif --stack --interactive
```

- 2θ with wavelength and style round-trip:

```bash
batplot a.xye b.xye --xaxis 2theta --wl 1.5406 --interactive
# p -> e (export mystyle.bpcfg), s (save session.pkl), q
batplot session.pkl --interactive
# i -> mystyle.bpcfg
```

- Batch export current directory:

```bash
batplot all
```

- Convert to `.qye`:

```bash
batplot --convert file1.xy file2.xye --wl 1.5406
```

---

## Notes and limits

- Do not mix `.gr`, `.nor`, `.chik`, `.chir`, and Q/2θ/CIF together.
- `.cif` intensity curves are not drawn; only reflection ticks/labels are supported.
- Very large CIF peak lists may suppress hkl labels for responsiveness.

---

## Version

This manual applies to batplot v1.0.13.
