"""Data conversion utilities for batplot."""

from __future__ import annotations

import os
import numpy as np


def convert_to_qye(filenames, wavelength: float):
    """Convert 2θ-based files (deg, intensity[, error]) to Q-based .qye files.

    Saves alongside input as <base>.qye with header noting the wavelength.
    """
    for fname in filenames:
        if not os.path.isfile(fname):
            print(f"File not found: {fname}")
            continue
        try:
            data = np.loadtxt(fname, comments="#")
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            print(f"Invalid data format in {fname}")
            continue
        x, y = data[:, 0], data[:, 1]
        e = data[:, 2] if data.shape[1] >= 3 else None
        theta_rad = np.radians(x / 2)
        q = 4 * np.pi * np.sin(theta_rad) / wavelength
        out_data = np.column_stack((q, y)) if e is None else np.column_stack((q, y, e))
        base, _ = os.path.splitext(fname)
        out_fname = f"{base}.qye"
        np.savetxt(out_fname, out_data, fmt="% .6f",
                   header=f"# Converted from {fname} using λ={wavelength} Å")
        print(f"Saved {out_fname}")


__all__ = ["convert_to_qye"]
