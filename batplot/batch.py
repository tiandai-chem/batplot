"""Batch processing for exporting plots to SVG."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .readers import (
    read_gr_file, 
    robust_loadtxt_skipheader,
    read_mpt_file,
    read_ec_csv_file,
    read_ec_csv_dqdv_file,
)
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
                # Transparent background for SVG exports
                _, _ext = os.path.splitext(target)
                if _ext.lower() == '.svg':
                    try:
                        _fig_fc = fig_b.get_facecolor()
                    except Exception:
                        _fig_fc = None
                    try:
                        _ax_fc = ax_b.get_facecolor()
                    except Exception:
                        _ax_fc = None
                    try:
                        if getattr(fig_b, 'patch', None) is not None:
                            fig_b.patch.set_alpha(0.0); fig_b.patch.set_facecolor('none')
                        if getattr(ax_b, 'patch', None) is not None:
                            ax_b.patch.set_alpha(0.0); ax_b.patch.set_facecolor('none')
                    except Exception:
                        pass
                    try:
                        fig_b.savefig(target, dpi=300, transparent=True, facecolor='none', edgecolor='none')
                    finally:
                        try:
                            if _fig_fc is not None and getattr(fig_b, 'patch', None) is not None:
                                fig_b.patch.set_alpha(1.0); fig_b.patch.set_facecolor(_fig_fc)
                        except Exception:
                            pass
                        try:
                            if _ax_fc is not None and getattr(ax_b, 'patch', None) is not None:
                                ax_b.patch.set_alpha(1.0); ax_b.patch.set_facecolor(_ax_fc)
                        except Exception:
                            pass
                else:
                    fig_b.savefig(target, dpi=300)
                plt.close(fig_b)
                print(f"  Saved {os.path.basename(target)}")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    print("Batch processing complete.")


def batch_process_ec(directory: str, args):
    """Batch process electrochemistry files in a directory.
    
    Supports GC (.mpt/.csv), CV (.mpt), dQdV (.csv), and CPC (.mpt/.csv) modes.
    Exports SVG plots to batplot_svg subdirectory.
    
    Note: For GC and CPC modes with .csv files, --mass is not required as the
    capacity data is already in the file. For .mpt files, --mass is required.
    
    Args:
        directory: Directory containing EC files
        args: Argument namespace with mode flags (gc, cv, dqdv, cpc) and mass
    """
    print(f"EC Batch mode: scanning {directory}")
    
    # Determine which EC mode is active
    mode = None
    if getattr(args, 'gc', False):
        mode = 'gc'
        supported_ext = {'.mpt', '.csv'}
    elif getattr(args, 'cv', False):
        mode = 'cv'
        supported_ext = {'.mpt', '.txt'}
    elif getattr(args, 'dqdv', False):
        mode = 'dqdv'
        supported_ext = {'.csv'}
    elif getattr(args, 'cpc', False):
        mode = 'cpc'
        supported_ext = {'.mpt', '.csv'}
    else:
        print("EC batch mode requires one of: --gc, --cv, --dqdv, or --cpc")
        return
    
    out_dir = os.path.join(directory, "batplot_svg")
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in sorted(os.listdir(directory))
             if os.path.splitext(f)[1].lower() in supported_ext 
             and os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print(f"No supported {mode.upper()} files found ({', '.join(supported_ext)}).")
        return
    
    print(f"Found {len(files)} {mode.upper()} files. Exporting SVG plots to {out_dir}")
    
    # Color palette
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for fname in files:
        fpath = os.path.join(directory, fname)
        ext = os.path.splitext(fname)[1].lower()
        
        try:
            fig_b, ax_b = plt.subplots(figsize=(6, 4))
            
            # ---- GC Mode ----
            if mode == 'gc':
                if ext == '.mpt':
                    mass_mg = getattr(args, 'mass', None)
                    if mass_mg is None:
                        print(f"  Skipped {fname}: GC mode (.mpt) requires --mass parameter")
                        plt.close(fig_b)
                        continue
                    specific_capacity, voltage, cycle_numbers, charge_mask, discharge_mask = \
                        read_mpt_file(fpath, mode='gc', mass_mg=mass_mg)
                    cap_x = specific_capacity
                    x_label = r'Specific Capacity (mAh g$^{-1}$)'
                elif ext == '.csv':
                    cap_x, voltage, cycle_numbers, charge_mask, discharge_mask = \
                        read_ec_csv_file(fpath, prefer_specific=True)
                    x_label = r'Specific Capacity (mAh g$^{-1}$)'
                else:
                    raise ValueError(f"Unsupported file type for GC: {ext}")
                
                # Plot cycles
                if cycle_numbers is not None:
                    cyc_int_raw = np.array(np.rint(cycle_numbers), dtype=int)
                    if cyc_int_raw.size:
                        min_c = int(np.min(cyc_int_raw))
                        shift = 1 - min_c if min_c <= 0 else 0
                        cyc_int = cyc_int_raw + shift
                        cycles_present = sorted(int(c) for c in np.unique(cyc_int))
                    else:
                        cycles_present = [1]
                else:
                    cycles_present = [1]
                
                for cyc in cycles_present[:10]:  # Limit to first 10 cycles for clarity
                    if cycle_numbers is not None:
                        mask_c = (cyc_int == cyc) & charge_mask
                        mask_d = (cyc_int == cyc) & discharge_mask
                    else:
                        mask_c = charge_mask
                        mask_d = discharge_mask
                    
                    color = base_colors[(cyc-1) % len(base_colors)]
                    
                    if np.any(mask_c):
                        ax_b.plot(cap_x[mask_c], voltage[mask_c], '-', 
                                 color=color, linewidth=1.5, alpha=0.8)
                    if np.any(mask_d):
                        ax_b.plot(cap_x[mask_d], voltage[mask_d], '-', 
                                 color=color, linewidth=1.5, alpha=0.8)
                
                ax_b.set_xlabel(x_label)
                ax_b.set_ylabel('Voltage (V)')
                ax_b.set_title(f"{fname}")
            
            # ---- CV Mode ----
            elif mode == 'cv':
                if ext == '.txt':
                    from .readers import read_biologic_txt_file
                    voltage, current, cycles = read_biologic_txt_file(fpath, mode='cv')
                elif ext == '.mpt':
                    voltage, current, cycles = read_mpt_file(fpath, mode='cv')
                else:
                    raise ValueError("CV mode requires .mpt or .txt file")
                
                cyc_int_raw = np.array(np.rint(cycles), dtype=int)
                if cyc_int_raw.size:
                    min_c = int(np.min(cyc_int_raw))
                    shift = 1 - min_c if min_c <= 0 else 0
                    cyc_int = cyc_int_raw + shift
                    cycles_present = sorted(int(c) for c in np.unique(cyc_int))
                else:
                    cycles_present = [1]
                
                for cyc in cycles_present[:10]:  # Limit to first 10 cycles
                    mask = (cyc_int == cyc)
                    idx = np.where(mask)[0]
                    if idx.size >= 2:
                        color = base_colors[(cyc-1) % len(base_colors)]
                        ax_b.plot(voltage[mask], current[mask], '-', 
                                 color=color, linewidth=1.5, alpha=0.8)
                
                ax_b.set_xlabel('Voltage (V)')
                ax_b.set_ylabel('Current (mA)')
                ax_b.set_title(f"{fname}")
            
            # ---- dQdV Mode ----
            elif mode == 'dqdv':
                if ext != '.csv':
                    raise ValueError("dQdV mode requires .csv file")
                
                voltage, dqdv = read_ec_csv_dqdv_file(fpath)
                ax_b.plot(voltage, dqdv, '-', color='#1f77b4', linewidth=1.5)
                ax_b.set_xlabel('Voltage (V)')
                ax_b.set_ylabel(r'dQ/dV (mAh g$^{-1}$ V$^{-1}$)')
                ax_b.set_title(f"{fname}")
            
            # ---- CPC Mode ----
            elif mode == 'cpc':
                if ext == '.mpt':
                    mass_mg = getattr(args, 'mass', None)
                    if mass_mg is None:
                        print(f"  Skipped {fname}: CPC mode (.mpt) requires --mass parameter")
                        plt.close(fig_b)
                        continue
                    cyc_nums, cap_charge, cap_discharge, eff = \
                        read_mpt_file(fpath, mode='cpc', mass_mg=mass_mg)
                    x_label = r'Specific Capacity (mAh g$^{-1}$)'
                elif ext == '.csv':
                    # For CSV CPC, read as GC-like data
                    cap_x, voltage, cycle_numbers, charge_mask, discharge_mask = \
                        read_ec_csv_file(fpath, prefer_specific=True)
                    # Plot capacity vs cycle number
                    if cycle_numbers is not None:
                        cyc_int_raw = np.array(np.rint(cycle_numbers), dtype=int)
                        if cyc_int_raw.size:
                            cycles_present = sorted(int(c) for c in np.unique(cyc_int_raw))
                            # Calculate capacity per cycle
                            cap_charge = []
                            cap_discharge = []
                            for cyc in cycles_present:
                                mask_c = (cyc_int_raw == cyc) & charge_mask
                                mask_d = (cyc_int_raw == cyc) & discharge_mask
                                cap_charge.append(np.max(cap_x[mask_c]) if np.any(mask_c) else 0)
                                cap_discharge.append(np.max(cap_x[mask_d]) if np.any(mask_d) else 0)
                            cyc_nums = np.array(cycles_present)
                            cap_charge = np.array(cap_charge)
                            cap_discharge = np.array(cap_discharge)
                        else:
                            cyc_nums = np.array([1])
                            cap_charge = np.array([0])
                            cap_discharge = np.array([0])
                    else:
                        cyc_nums = np.array([1])
                        cap_charge = np.array([0])
                        cap_discharge = np.array([0])
                    x_label = r'Specific Capacity (mAh g$^{-1}$)'
                else:
                    raise ValueError(f"Unsupported file type for CPC: {ext}")
                
                # Plot CPC data
                ax_b.plot(cyc_nums, cap_charge, 'o-', color='#1f77b4', 
                         linewidth=1.5, markersize=4, label='Charge', alpha=0.8)
                ax_b.plot(cyc_nums, cap_discharge, 's-', color='#ff7f0e', 
                         linewidth=1.5, markersize=4, label='Discharge', alpha=0.8)
                ax_b.set_xlabel('Cycle Number')
                ax_b.set_ylabel(x_label)
                ax_b.legend()
                ax_b.set_title(f"{fname}")
            
            # Adjust layout and save
            fig_b.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.90)
            out_name = os.path.splitext(fname)[0] + f"_{mode}.svg"
            out_path = os.path.join(out_dir, out_name)
            
            target = _confirm_overwrite(out_path)
            if not target:
                plt.close(fig_b)
                print(f"  Skipped {out_name} (user canceled)")
            else:
                # Transparent background for SVG
                _, _ext = os.path.splitext(target)
                if _ext.lower() == '.svg':
                    try:
                        _fig_fc = fig_b.get_facecolor()
                    except Exception:
                        _fig_fc = None
                    try:
                        _ax_fc = ax_b.get_facecolor()
                    except Exception:
                        _ax_fc = None
                    try:
                        if getattr(fig_b, 'patch', None) is not None:
                            fig_b.patch.set_alpha(0.0)
                            fig_b.patch.set_facecolor('none')
                        if getattr(ax_b, 'patch', None) is not None:
                            ax_b.patch.set_alpha(0.0)
                            ax_b.patch.set_facecolor('none')
                    except Exception:
                        pass
                    try:
                        fig_b.savefig(target, dpi=300, transparent=True, 
                                     facecolor='none', edgecolor='none')
                    finally:
                        try:
                            if _fig_fc is not None and getattr(fig_b, 'patch', None) is not None:
                                fig_b.patch.set_alpha(1.0)
                                fig_b.patch.set_facecolor(_fig_fc)
                        except Exception:
                            pass
                        try:
                            if _ax_fc is not None and getattr(ax_b, 'patch', None) is not None:
                                ax_b.patch.set_alpha(1.0)
                                ax_b.patch.set_facecolor(_ax_fc)
                        except Exception:
                            pass
                else:
                    fig_b.savefig(target, dpi=300)
                plt.close(fig_b)
                print(f"  Saved {os.path.basename(target)}")
                
        except Exception as e:
            plt.close(fig_b)
            print(f"  Skipped {fname}: {e}")
    
    print(f"EC batch processing complete ({mode.upper()} mode).")


__all__ = ["batch_process", "batch_process_ec"]
