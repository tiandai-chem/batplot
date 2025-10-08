# batplot

Interactive plotting for XRD, PDF, and XAS data (.xye, .xy, .qye, .dat, .csv, .gr, .nor, .chik, .chir)

## Install

```bash
python -m pip install -U batplot
```

## Usage

CLI entry point: `batplot` (or `python -m batplot.cli`)

- Launch interactive plot with default settings:
  ```bash
  batplot file1.xye file2.xy --interactive
  ```
- Convert 2-theta data to QYE with wavelength:
  ```bash
  batplot --convert pattern.xy --wl 1.5406

- Load a saved session exactly as saved:
  ```bash
  batplot mysession.pkl
  batplot mysession.pkl --interactive
  ```
  ```

## Features
- Interactive styling (colors, fonts, lines, ticks)
- Stacked or overlaid curves
- Crosshair readouts (Q, d, 2θ)
- CIF reflection positions and simulated patterns (basic)
- Session/style export and import

## Documentation
- User Manual: see `USER_MANUAL.md` for installation, CLI options, interactive controls, sessions/styles, CIF behavior, batch/conversion, troubleshooting, and examples.
- Publishing guide: see `README_PUBLISH.md` for TestPyPI/PyPI steps.

## Python
```python
from batplot.batplot import main
# or run via CLI: batplot
```

