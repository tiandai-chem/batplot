"""Batch processing for exporting plots to SVG."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .readers import read_gr_file, robust_loadtxt_skipheader
from .utils import _confirm_overwrite


def batch_process(directory: str, args):
    print(f"Batch mode: scanning {directory}")
    supported_ext = {'.xye', '.xy', '.qye', '.dat', '.csv', '.gr', '.nor', '.chik', '.chir', '.txt'}
    out_dir = os.path.join(directory, "batplot_svg")
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(directory))
             if os.path.splitext(f)[1].lower() in supported_ext and os.path.isfile(os.path.join(directory, f))]
    if not files:
        print("No supported data files found.")
        return
    print(f"Found {len(files)} supported files. Exporting SVG plots to {out_dir}")
    for fname in files:
        fpath = os.path.join(directory, fname)
        ext = os.path.splitext(fname)[1].lower()
        try:
            # ---- Read data ----
            if ext == '.gr':
                x, y = read_gr_file(fpath); e = None
                axis_mode = 'r'
            elif ext == '.nor':
                data = np.loadtxt(fpath, comments="#")
                if data.ndim == 1: data = data.reshape(1, -1)
                if data.shape[1] < 2: raise ValueError("Invalid .nor format")
                x, y = data[:,0], data[:,1]
                e = data[:,2] if data.shape[1] >= 3 else None
                axis_mode = 'energy'
            elif 'chik' in ext:
                data = np.loadtxt(fpath, comments="#")
                if data.ndim == 1: data = data.reshape(1, -1)
                if data.shape[1] < 2: raise ValueError("Invalid .chik data")
                x, y = data[:,0], data[:,1]; e = data[:,2] if data.shape[1] >= 3 else None
                axis_mode = 'k'
            elif 'chir' in ext:
                data = np.loadtxt(fpath, comments="#")
                if data.ndim == 1: data = data.reshape(1, -1)
                if data.shape[1] < 2: raise ValueError("Invalid .chir data")
                x, y = data[:,0], data[:,1]; e = data[:,2] if data.shape[1] >= 3 else None
                axis_mode = 'rft'
            else:
                data = robust_loadtxt_skipheader(fpath)
                if data.ndim == 1: data = data.reshape(1, -1)
                if data.shape[1] < 2: raise ValueError("Invalid 2-column data")
                x, y = data[:,0], data[:,1]
                e = data[:,2] if data.shape[1] >= 3 else None
                if ext == '.qye':
                    axis_mode = 'Q'
                elif ext == '.gr':
                    axis_mode = 'r'
                elif ext == '.nor':
                    axis_mode = 'energy'
                elif 'chik' in ext:
                    axis_mode = 'k'
                elif 'chir' in ext:
                    axis_mode = 'rft'
                elif args.xaxis:
                    axis_mode = args.xaxis
                else:
                    raise ValueError(f"Cannot determine X-axis type for file {fname} (need .qye/.gr/.nor/.chik/.chir or --xaxis).")

            # Convert to Q if needed
            if axis_mode == 'Q' and ext not in ('.qye', '.gr', '.nor'):
                if args.wl is None:
                    axis_mode = '2theta'
                    x_plot = x
                else:
                    theta_rad = np.radians(x/2)
                    x_plot = 4*np.pi*np.sin(theta_rad)/args.wl
            else:
                x_plot = x

            # Normalize or raw
            if args.raw:
                y_plot = y.copy()
            else:
                if y.size:
                    ymin = float(y.min()); ymax = float(y.max())
                    span = ymax - ymin
                    y_plot = (y - ymin)/span if span > 0 else np.zeros_like(y)
                else:
                    y_plot = y

            # Plot and save
            fig_b, ax_b = plt.subplots(figsize=(6,4))
            ax_b.plot(x_plot, y_plot, lw=1)
            if axis_mode == 'Q':
                ax_b.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
            elif axis_mode == 'r':
                ax_b.set_xlabel("r (Å)")
            elif axis_mode == 'energy':
                ax_b.set_xlabel("Energy (eV)")
            elif axis_mode == 'k':
                ax_b.set_xlabel(r"k ($\mathrm{\AA}^{-1}$)")
            elif axis_mode == 'rft':
                ax_b.set_xlabel("Radial distance (Å)")
            else:
                ax_b.set_xlabel(r"$2\theta\ (\mathrm{deg})$")
            ax_b.set_ylabel("Intensity" if args.raw else "Normalized intensity (a.u.)")
            ax_b.set_title(fname)
            fig_b.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.90)
            out_name = os.path.splitext(fname)[0] + ".svg"
            out_path = os.path.join(out_dir, out_name)
            target = _confirm_overwrite(out_path)
            if not target:
                plt.close(fig_b)
                print(f"  Skipped {out_name} (user canceled)")
            else:
                fig_b.savefig(target, dpi=300)
                plt.close(fig_b)
                print(f"  Saved {os.path.basename(target)}")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    print("Batch processing complete.")


__all__ = ["batch_process"]
