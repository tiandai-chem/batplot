"""Readers for various data formats used by batplot."""

from __future__ import annotations

import numpy as np


def read_csv_file(fname: str):
    for delim in [",", ";", "\t"]:
        try:
            data = np.genfromtxt(fname, delimiter=delim, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 2:
                return data
        except Exception:
            continue
    raise ValueError(f"Invalid CSV format in {fname}, need at least 2 columns (x,y).")


def read_gr_file(fname: str):
    """Read a PDF .gr file (r, G(r))."""
    r_vals = []
    g_vals = []
    with open(fname, "r") as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            parts = ls.replace(",", " ").split()
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    break
            if len(floats) >= 2:
                r_vals.append(floats[0])
                g_vals.append(floats[1])
    if not r_vals:
        raise ValueError(f"No numeric data found in {fname}")
    return np.array(r_vals, dtype=float), np.array(g_vals, dtype=float)


def read_fullprof_rowwise(fname: str):
    with open(fname, "r") as f:
        lines = f.readlines()[1:]
    y_rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        y_rows.extend([float(val) for val in line.split()])
    y = np.array(y_rows)
    return y, len(lines)


def robust_loadtxt_skipheader(fname: str):
    """Skip comments/non-numeric lines and load at least 2-column numeric data."""
    data_lines = []
    with open(fname, "r") as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            floats = []
            for p in ls.replace(",", " ").split():
                try:
                    floats.append(float(p))
                except ValueError:
                    break
            if len(floats) >= 2:
                data_lines.append(ls)
    if not data_lines:
        raise ValueError(f"No numeric data found in {fname}")
    from io import StringIO
    return np.loadtxt(StringIO("\n".join(data_lines)))
