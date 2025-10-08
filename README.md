# batplot

Interactive plotting for XRD, PDF, and XAS data (.xye, .xy, .qye, .dat, .csv, .gr, .nor, .chik, .chir)

## Install

```bash
pip install batplot
```

## Usage

CLI entry point: `batplot`

- Launch interactive plot with default settings:
  ```bash
  batplot file1.xye file2.xy
  ```
- Convert 2-theta data to QYE with wavelength:
  ```bash
  batplot --convert pattern.xy --wl 1.5406
  ```

## Features
- Interactive styling (colors, fonts, lines, ticks)
- Stacked or overlaid curves
- Crosshair readouts (Q, d, 2θ)
- CIF reflection positions and simulated patterns (basic)
- Session/style export and import

## Python
```python
from batplot.batplot import main
# or run via CLI: batplot
```

