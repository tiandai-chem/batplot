"""Operando (time/sequence) contour plotting utilities.

This module provides a single helper `plot_operando_folder` that scans a folder
for normalized diffraction data files (.xy, .xye, .qye, .dat) and renders them as
an intensity contour (imshow / pcolormesh) stack vs scan index.

Rules:
- X axis: 2θ by default; if --xaxis Q provided, or files are .qye, or a global
  wavelength is specified and conversion is desired, Q will be used.
- Input files are assumed already normalized (0-1) unless --raw passed in the
  parent CLI; for simplicity we re-normalize each curve individually unless
  raw mode chosen.
- Sort files alphabetically for deterministic order.

Returned figure/axes so caller can further tweak or save.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .converters import convert_to_qye
from .readers import robust_loadtxt_skipheader, read_mpt_file

SUPPORTED_EXT = {".xy", ".xye", ".qye", ".dat"}

_two_theta_re = re.compile(r"2[tT]heta|2th", re.IGNORECASE)
_q_re = re.compile(r"^q$", re.IGNORECASE)

def _infer_axis_mode(args, any_qye: bool):
    # Priority: explicit --xaxis, else .qye presence (Q), else wavelength (Q), else default 2theta with warning
    if args.xaxis:
        if _q_re.match(args.xaxis.strip()):
            return "Q"
        if _two_theta_re.search(args.xaxis):
            return "2theta"
        print(f"[operando] Unrecognized --xaxis '{args.xaxis}', assuming 2theta.")
        return "2theta"
    if any_qye:
        return "Q"
    if getattr(args, 'wl', None) is not None:
        return "Q"
    print("[operando] No --xaxis or --wl supplied and no .qye files; assuming 2theta (degrees). Use --xaxis 2theta to silence this message.")
    return "2theta"

def _load_curve(path: Path):
    data = robust_loadtxt_skipheader(str(path))
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"File {path} has insufficient numeric data")
        x = data[0::2]
        y = data[1::2]
    else:
        x = data[:,0]
        y = data[:,1]
    return np.asarray(x, float), np.asarray(y, float)

def _maybe_convert_to_Q(x, wl):
    # Accept degrees (2theta) -> Q
    # Q = 4π sin(theta)/λ ; theta = (2θ)/2
    theta = np.radians(x/2.0)
    return 4.0 * np.pi * np.sin(theta) / wl

def plot_operando_folder(folder: str, args) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """Plot operando contour from a folder of diffraction files.
    
    Args:
        folder: Path to directory containing diffraction data files
        args: Argument namespace with attributes: xaxis, wl, raw, interactive, savefig, out
        
    Returns:
        Tuple of (figure, axes, metadata_dict)
        metadata_dict contains: files, axis_mode, x_grid, imshow, colorbar, has_ec, ec_ax
    """
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
    if not files:
        raise FileNotFoundError("No supported diffraction data files (.xy/.xye/.qye/.dat) found in folder")
    any_qye = any(f.suffix.lower()==".qye" for f in files)
    axis_mode = _infer_axis_mode(args, any_qye)
    wl = getattr(args, 'wl', None)

    x_arrays = []
    y_arrays = []
    for f in files:
        try:
            x, y = _load_curve(f)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
            continue
        # Convert to Q if needed
        if axis_mode == "Q":
            if f.suffix.lower() == ".qye":
                pass  # already Q
            else:
                if wl is None:
                    # If user wants Q without wavelength we cannot proceed for this file
                    print(f"Skip {f.name}: need wavelength (--wl) for Q conversion")
                    continue
                x = _maybe_convert_to_Q(x, wl)
        # Normalize unless raw
        if not args.raw and y.size:
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            span = ymax - ymin
            if span > 0:
                y = (y - ymin)/span
            else:
                y = np.zeros_like(y)
        x_arrays.append(x)
        y_arrays.append(y)

    if not x_arrays:
        raise RuntimeError("No curves loaded after filtering/conversion.")

    # Create common X grid (union, simple linear interpolation) for contour
    # Determine global min/max and pick reasonable number of points (~max original length)
    xmin = min(arr.min() for arr in x_arrays if arr.size)
    xmax = max(arr.max() for arr in x_arrays if arr.size)
    base_len = int(max(arr.size for arr in x_arrays))
    grid_x = np.linspace(xmin, xmax, base_len)
    stack = []
    for x, y in zip(x_arrays, y_arrays):
        if x.size < 2:
            interp = np.full_like(grid_x, np.nan)
        else:
            interp = np.interp(grid_x, x, y, left=np.nan, right=np.nan)
        stack.append(interp)
    Z = np.vstack(stack)  # shape (n_scans, n_x)

    # Detect an electrochemistry .mpt file in the same folder (if any)
    mpt_files = sorted([f for f in p.iterdir() if f.suffix.lower() == ".mpt"])  # pick first if present
    has_ec = len(mpt_files) > 0
    ec_ax = None

    if has_ec:
        # Wider canvas to accommodate side-by-side plots
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.5, 1.2], wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig, ax = plt.subplots(figsize=(8,6))
    # Use imshow for speed; mask nans
    Zm = np.ma.masked_invalid(Z)
    extent = (grid_x.min(), grid_x.max(), 0, Zm.shape[0]-1)
    # Top-to-down visual order (scan 0 at top) -> origin='upper'
    im = ax.imshow(Zm, aspect='auto', origin='upper', extent=extent, cmap='viridis', interpolation='nearest')
    # Place colorbar on the left
    cbar = fig.colorbar(im, ax=ax, location='left', pad=0.15)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label('Intensity (norm)' if not args.raw else 'Intensity')
    ax.set_ylabel('Scan index')
    if axis_mode == 'Q':
        # Use mathtext for reliable superscript minus; plain unicode '⁻' can fail with some fonts
        ax.set_xlabel(r'Q (Å$^{-1}$)')  # renders as Å^{-1}
    else:
        ax.set_xlabel('2θ (deg)')
    # No title for operando plot (requested)

    # If an EC .mpt exists, attach it to the right with the same height (Voltage vs Time in hours)
    if has_ec:
        try:
            ec_path = mpt_files[0]
            # Read time series from .mpt
            time_s, voltage_v, current_mA = read_mpt_file(str(ec_path), mode='time')
            time_h = np.asarray(time_s, float) / 3600.0
            voltage_v = np.asarray(voltage_v, float)
            # Add the EC axes on the right
            ec_ax = fig.add_subplot(gs[0, 1])
            ln_ec, = ec_ax.plot(voltage_v, time_h, lw=1.0, color='tab:blue')
            ec_ax.set_xlabel('Voltage (V)')
            ec_ax.set_ylabel('Time (h)')
            # Match interactive defaults: put EC Y axis on the right
            try:
                ec_ax.yaxis.tick_right()
                ec_ax.yaxis.set_label_position('right')
                _title = ec_ax.get_title()
                if isinstance(_title, str) and _title.strip():
                    ec_ax.set_title(_title, loc='right')
            except Exception:
                pass
            # Keep a clean look, no grid
            # Align visually: ensure similar vertical span display
            try:
                # Remove vertical margins and clamp to exact data bounds
                ec_ax.margins(y=0)
                tmin = float(np.nanmin(time_h)) if getattr(np, 'nanmin', None) else float(np.min(time_h))
                tmax = float(np.nanmax(time_h)) if getattr(np, 'nanmax', None) else float(np.max(time_h))
                ec_ax.set_ylim(tmin, tmax)
            except Exception:
                pass
            # Add a small right margin on EC X to give space for right-side ticks/labels
            try:
                x0, x1 = ec_ax.get_xlim()
                xr = (x1 - x0) if x1 > x0 else 0.0
                if xr > 0:
                    ec_ax.set_xlim(x0, x1 + 0.02 * xr)
                    setattr(ec_ax, '_xlim_expanded_default', True)
            except Exception:
                pass
            # Stash EC data and line for interactive transforms
            try:
                ec_ax._ec_time_h = time_h
                ec_ax._ec_voltage_v = voltage_v
                ec_ax._ec_current_mA = current_mA
                ec_ax._ec_line = ln_ec
                ec_ax._ec_y_mode = 'time'  # or 'ions'
                ec_ax._ion_annots = []
                ec_ax._ion_params = {"mass_mg": None, "cap_per_ion_mAh_g": None}
            except Exception:
                pass
        except Exception as e:
            print(f"[operando] Failed to attach electrochem plot: {e}")

    # --- Default layout: set operando plot width to 5 inches (centered) ---
    try:
        fig_w_in, fig_h_in = fig.get_size_inches()
        # Current geometry in fractions
        ax_x0, ax_y0, ax_wf, ax_hf = ax.get_position().bounds
        cb_x0, cb_y0, cb_wf, cb_hf = cbar.ax.get_position().bounds
        # Convert to inches
        desired_ax_w_in = 5.0
        ax_h_in = ax_hf * fig_h_in
        cb_w_in = cb_wf * fig_w_in
        cb_gap_in = max(0.0, (ax_x0 - (cb_x0 + cb_wf)) * fig_w_in)
        ec_gap_in = 0.0
        ec_w_in = 0.0
        if ec_ax is not None:
            ec_x0, ec_y0, ec_wf, ec_hf = ec_ax.get_position().bounds
            ec_gap_in = max(0.0, (ec_x0 - (ax_x0 + ax_wf)) * fig_w_in)
            ec_w_in = ec_wf * fig_w_in
            # Match interactive default: shrink EC gap and rebalance widths
            try:
                # Decrease gap more aggressively with a sensible minimum
                ec_gap_in = max(0.02, ec_gap_in * 0.2)
                # Transfer a fraction of width from EC to operando while keeping total similar
                combined = (desired_ax_w_in if desired_ax_w_in > 0 else ax_wf * fig_w_in) + ec_w_in
                ax_w_in_current = desired_ax_w_in if desired_ax_w_in > 0 else (ax_wf * fig_w_in)
                if combined > 0 and ec_w_in > 0.5:
                    transfer = min(ec_w_in * 0.18, combined * 0.12)
                    min_ec = 0.8
                    if ec_w_in - transfer < min_ec:
                        transfer = max(0.0, ec_w_in - min_ec)
                    desired_ax_w_in = ax_w_in_current + transfer
                    ec_w_in = max(min_ec, ec_w_in - transfer)
            except Exception:
                pass
        # Clamp desired width if it would overflow the canvas
        reserved = cb_w_in + cb_gap_in + ec_gap_in + ec_w_in
        max_ax_w = max(0.25, fig_w_in - reserved - 0.02)
        ax_w_in = min(desired_ax_w_in, max_ax_w)
        # Convert inches to fractions
        ax_wf_new = max(0.0, ax_w_in / fig_w_in)
        ax_hf_new = max(0.0, ax_h_in / fig_h_in)
        cb_wf_new = max(0.0, cb_w_in / fig_w_in)
        cb_gap_f = max(0.0, cb_gap_in / fig_w_in)
        ec_gap_f = max(0.0, ec_gap_in / fig_w_in)
        ec_wf_new = max(0.0, ec_w_in / fig_w_in)
        # Center group horizontally
        total_wf = cb_wf_new + cb_gap_f + ax_wf_new + ec_gap_f + ec_wf_new
        group_left = 0.5 - total_wf / 2.0
        y0 = 0.5 - ax_hf_new / 2.0
        # Positions
        cb_x0_new = group_left
        ax_x0_new = cb_x0_new + cb_wf_new + cb_gap_f
        ec_x0_new = ax_x0_new + ax_wf_new + ec_gap_f if ec_ax is not None else None
        # Apply
        ax.set_position([ax_x0_new, y0, ax_wf_new, ax_hf_new])
        cbar.ax.set_position([cb_x0_new, y0, cb_wf_new, ax_hf_new])
        if ec_ax is not None and ec_x0_new is not None:
            ec_ax.set_position([ec_x0_new, y0, ec_wf_new, ax_hf_new])
        # Persist inches so interactive menu can pick them up
        try:
            setattr(cbar.ax, '_fixed_cb_w_in', cb_w_in)
            # Store both names for compatibility across interactive menus
            setattr(cbar.ax, '_fixed_cb_gap_in', cb_gap_in)
            setattr(cbar.ax, '_fixed_gap_in', cb_gap_in)
            if ec_ax is not None:
                setattr(ec_ax, '_fixed_ec_gap_in', ec_gap_in)
                setattr(ec_ax, '_fixed_ec_w_in', ec_w_in)
                # Mark as adjusted so interactive menu won't adjust twice
                setattr(ec_ax, '_ec_gap_adjusted', True)
                setattr(ec_ax, '_ec_op_width_adjusted', True)
            setattr(ax, '_fixed_ax_w_in', ax_w_in)
            setattr(ax, '_fixed_ax_h_in', ax_h_in)
        except Exception:
            pass
        try:
            fig.canvas.draw()
        except Exception:
            fig.canvas.draw_idle()
    except Exception:
        # Non-fatal: keep Matplotlib's default layout
        pass

    meta = {
        'files': [f.name for f in files],
        'axis_mode': axis_mode,
        'x_grid': grid_x,
        'imshow': im,
        'colorbar': cbar,
        'has_ec': bool(has_ec),
    }
    if ec_ax is not None:
        meta['ec_ax'] = ec_ax
    return fig, ax, meta

__all__ = ["plot_operando_folder"]
