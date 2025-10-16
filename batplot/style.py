"""Style helpers for batplot: print diagnostics, export/import style configs.

These utilities keep batplot.py slimmer by centralizing style logic.
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
import json
import numpy as np
import matplotlib.pyplot as plt

from .utils import _confirm_overwrite
from .ui import (
    ensure_text_visibility as _ui_ensure_text_visibility,
    update_tick_visibility as _ui_update_tick_visibility,
    position_top_xlabel as _ui_position_top_xlabel,
    position_right_ylabel as _ui_position_right_ylabel,
    position_bottom_xlabel as _ui_position_bottom_xlabel,
    position_left_ylabel as _ui_position_left_ylabel,
)


def print_style_info(
    fig,
    ax,
    y_data_list: List[np.ndarray],
    labels: List[str],
    offsets_list: List[float],
    x_full_list: List[np.ndarray],
    raw_y_full_list: List[np.ndarray],
    args,
    delta: float,
    label_text_objects: List,
    tick_state: Dict[str, bool],
    cif_tick_series: Optional[List[tuple]] = None,
    show_cif_hkl: Optional[bool] = None,
) -> None:
    print("\n--- Style / Diagnostics ---")
    fw, fh = fig.get_size_inches()
    print(f"Figure size (inches): {fw:.3f} x {fh:.3f}")
    # DPI omitted from compact style print
    bbox = ax.get_position()
    print(
        f"Axes position (figure fraction): x0={bbox.x0:.3f}, y0={bbox.y0:.3f}, w={bbox.width:.3f}, h={bbox.height:.3f}"
    )
    frame_w_in = bbox.width * fw
    frame_h_in = bbox.height * fh
    print(f"Plot frame size (inches):  {frame_w_in:.3f} x {frame_h_in:.3f}")
    sp = fig.subplotpars
    print(
        f"Margins (subplot fractions): left={sp.left:.3f}, right={sp.right:.3f}, bottom={sp.bottom:.3f}, top={sp.top:.3f}"
    )
    # Omit ranges and axis labels from style print
    # Font info
    if label_text_objects:
        fs_any = label_text_objects[0].get_fontsize()
        ff_any = label_text_objects[0].get_fontfamily()
    else:
        fs_any = plt.rcParams.get("font.size")
        ff_any = plt.rcParams.get("font.family")
    print(f"Effective font size (labels/ticks): {fs_any}")
    print(f"Font family chain (rcParams['font.sans-serif']): {plt.rcParams.get('font.sans-serif')}")
    print(f"Mathtext fontset: {plt.rcParams.get('mathtext.fontset')}")

    # Per-side matrix summary (spine, major, minor, labels, title)
    def _onoff(v):
        return 'ON ' if bool(v) else 'off'
    sides = (
        ('bottom',
         ax.spines.get('bottom').get_visible() if ax.spines.get('bottom') else False,
         tick_state.get('b_ticks', tick_state.get('bx', True)),
         tick_state.get('mbx', False),
         tick_state.get('b_labels', tick_state.get('bx', True)),
         bool(ax.get_xlabel())),
        ('top',
         ax.spines.get('top').get_visible() if ax.spines.get('top') else False,
         tick_state.get('t_ticks', tick_state.get('tx', False)),
         tick_state.get('mtx', False),
         tick_state.get('t_labels', tick_state.get('tx', False)),
         bool(getattr(ax, '_top_xlabel_on', False))),
        ('left',
         ax.spines.get('left').get_visible() if ax.spines.get('left') else False,
         tick_state.get('l_ticks', tick_state.get('ly', True)),
         tick_state.get('mly', False),
         tick_state.get('l_labels', tick_state.get('ly', True)),
         bool(ax.get_ylabel())),
        ('right',
         ax.spines.get('right').get_visible() if ax.spines.get('right') else False,
         tick_state.get('r_ticks', tick_state.get('ry', False)),
         tick_state.get('mry', False),
         tick_state.get('r_labels', tick_state.get('ry', False)),
         bool(getattr(ax, '_right_ylabel_on', False))),
    )
    print("Per-side: spine, major, minor, labels, title")
    for name, spine, mj, mn, lbl, title in sides:
        print(f"  {name:<6}: spine={_onoff(spine)} major={_onoff(mj)} minor={_onoff(mn)} labels={_onoff(lbl)} title={_onoff(title)}")

    # Tick widths helper
    def axis_tick_width(axis, which):
        ticks = axis.get_major_ticks() if which == "major" else axis.get_minor_ticks()
        for t in ticks:
            line = t.tick1line
            if line.get_visible():
                return line.get_linewidth()
        return None

    x_major_w = axis_tick_width(ax.xaxis, "major")
    x_minor_w = axis_tick_width(ax.xaxis, "minor")
    y_major_w = axis_tick_width(ax.yaxis, "major")
    y_minor_w = axis_tick_width(ax.yaxis, "minor")
    print(
        f"Tick widths (major/minor): X=({x_major_w}, {x_minor_w})  Y=({y_major_w}, {y_minor_w})"
    )

    # Spines
    print("Spines:")
    for name, spn in ax.spines.items():
        print(
            f"  {name:<5} lw={spn.get_linewidth()} color={spn.get_edgecolor()} visible={spn.get_visible()}"
        )

    # Omit CIF/HKL details from compact style print

    # Omit non-style global flags (mode/raw/autoscale/delta)

    # Curves
    print("Lines (style):")
    for i, ln in enumerate(ax.lines):
        col = ln.get_color(); lw = ln.get_linewidth(); ls = ln.get_linestyle()
        mk = ln.get_marker(); ms = ln.get_markersize(); a = ln.get_alpha()
        base_label = labels[i] if i < len(labels) else ""
        print(f"  {i+1:02d}: label='{base_label}' color={col} lw={lw} ls={ls} marker={mk} ms={ms} alpha={a}")
    print("--- End diagnostics ---\n")


def export_style_config(
    filename: str,
    fig,
    ax,
    y_data_list: List[np.ndarray],
    labels: List[str],
    delta: float,
    args,
    tick_state: Dict[str, bool],
    cif_tick_series: Optional[List[tuple]] = None,
) -> None:
    """Export style configuration after displaying a summary and prompting the user.
    
    This function now matches the EC menu workflow: display summary, then prompt for export.
    """
    try:
        fw, fh = fig.get_size_inches()
        sp = fig.subplotpars

        def axis_tick_width(axis, which):
            ticks = axis.get_major_ticks() if which == "major" else axis.get_minor_ticks()
            for t in ticks:
                line = t.tick1line
                if line.get_visible():
                    return line.get_linewidth()
            return None

        spine_vis = {name: spn.get_visible() for name, spn in ax.spines.items()}

        bbox = ax.get_position()
        frame_w_in = bbox.width * fw
        frame_h_in = bbox.height * fh
        cfg = {
            "figure": {
                "size": [fw, fh],
                "dpi": fig.dpi,
                "frame_size": [frame_w_in, frame_h_in],
                "axes_fraction": [bbox.x0, bbox.y0, bbox.width, bbox.height],
            },
            "margins": {
                "left": sp.left,
                "right": sp.right,
                "bottom": sp.bottom,
                "top": sp.top,
            },
            "font": {
                "size": plt.rcParams.get("font.size"),
                "family_chain": plt.rcParams.get("font.sans-serif"),
            },
            "ticks": {
                "visibility": tick_state.copy(),
                "x_major_width": axis_tick_width(ax.xaxis, "major"),
                "x_minor_width": axis_tick_width(ax.xaxis, "minor"),
                "y_major_width": axis_tick_width(ax.yaxis, "major"),
                "y_minor_width": axis_tick_width(ax.yaxis, "minor"),
            },
            "spines": {
                name: {
                    "linewidth": spn.get_linewidth(),
                    "color": spn.get_edgecolor(),
                    "visible": spine_vis.get(name, True),
                }
                for name, spn in ax.spines.items()
            },
            "lines": [
                {
                    "index": i,
                    # label text is not a style item (handled by 'r'), don't export it
                    "color": ln.get_color(),
                    "linewidth": ln.get_linewidth(),
                    "linestyle": ln.get_linestyle(),
                    "marker": ln.get_marker(),
                    "markersize": ln.get_markersize(),
                    "markerfacecolor": ln.get_markerfacecolor(),
                    "markeredgecolor": ln.get_markeredgecolor(),
                    "alpha": ln.get_alpha(),
                }
                for i, ln in enumerate(ax.lines)
            ],
            }
        cfg["axis_titles"] = {
            "top_x": bool(getattr(ax, "_top_xlabel_on", False)),
            "right_y": bool(getattr(ax, "_right_ylabel_on", False)),
            "has_bottom_x": bool(ax.get_xlabel()),
            "has_left_y": bool(ax.get_ylabel()),
        }
        if cif_tick_series:
            cfg["cif_ticks"] = [
                {"index": i, "color": color}
                for i, (lab, fname, peaksQ, wl, qmax_sim, color) in enumerate(cif_tick_series)
            ]
        
        # List existing .bpcfg files for user convenience
        import os
        try:
            bpcfg_files = sorted([f for f in os.listdir(os.getcwd()) if f.lower().endswith('.bpcfg')])
        except Exception:
            bpcfg_files = []

        if bpcfg_files:
            print("Existing .bpcfg files:")
            for i, f in enumerate(bpcfg_files, 1):
                print(f"  {i}: {f}")

        choice = input("Export style to file? Enter filename or number to overwrite (q=cancel): ").strip()
        if not choice or choice.lower() == 'q':
            print("Style export canceled.")
            return

        # Determine the target path
        if choice.isdigit() and bpcfg_files and 1 <= int(choice) <= len(bpcfg_files):
            target_path = bpcfg_files[int(choice) - 1]
        else:
            target_path = choice if choice.lower().endswith('.bpcfg') else f"{choice}.bpcfg"

        # Only prompt ONCE for overwrite if the file exists
        if os.path.exists(target_path):
            yn = input(f"Overwrite '{target_path}'? (y/n): ").strip().lower()
            if yn != 'y':
                print("Style export canceled.")
                return

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Exported style to {target_path}")
    except Exception as e:
        print(f"Error exporting style: {e}")


def apply_style_config(
    filename: str,
    fig,
    ax,
    x_data_list: List[np.ndarray] | None,
    y_data_list: List[np.ndarray],
    orig_y: List[np.ndarray] | None,
    offsets_list: List[float] | None,
    label_text_objects: List,
    args,
    tick_state: Dict[str, bool],
    labels: List[str],
    update_labels_func: Callable[[Any, List[np.ndarray], List, bool], None],
    cif_tick_series: Optional[List[tuple]] = None,
    cif_hkl_label_map: Optional[Dict[str, Dict[float, str]]] = None,
    adjust_margins_cb: Optional[Callable[[], None]] = None,
    keep_canvas_fixed: bool = False,
) -> None:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"Could not read config: {e}")
        return
    try:
        figure_cfg = cfg.get("figure", {})
        sz = figure_cfg.get("size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            try:
                fw = float(sz[0])
                fh = float(sz[1])
                if not keep_canvas_fixed:
                    fig.set_size_inches(fw, fh, forward=True)
                else:
                    print("(Canvas fixed) Ignoring style figure size request.")
            except Exception as e:
                print(f"Warning: could not parse figure size: {e}")
        try:
            frame_size = figure_cfg.get("frame_size")
            axes_frac = figure_cfg.get("axes_fraction")
            if axes_frac and isinstance(axes_frac, (list, tuple)) and len(axes_frac) == 4:
                x0, y0, w, h = axes_frac
                left = float(x0)
                bottom = float(y0)
                right = left + float(w)
                top = bottom + float(h)
                if 0 < left < right <= 1 and 0 < bottom < top <= 1:
                    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            elif frame_size and isinstance(frame_size, (list, tuple)) and len(frame_size) == 2:
                cur_fw, cur_fh = fig.get_size_inches()
                des_w, des_h = float(frame_size[0]), float(frame_size[1])
                min_margin = 0.05
                w_frac = min(des_w / cur_fw, 1 - 2 * min_margin)
                h_frac = min(des_h / cur_fh, 1 - 2 * min_margin)
                left = (1 - w_frac) / 2
                bottom = (1 - h_frac) / 2
                fig.subplots_adjust(left=left, right=left + w_frac, bottom=bottom, top=bottom + h_frac)
        except Exception as e:
            print(f"[DEBUG] Exception in frame/axes fraction adjustment: {e}")
        # Don't restore DPI from style - use system default to avoid display-dependent issues
        # (Retina displays, Windows scaling, etc. can cause saved DPI to differ)

        # Font
        font_cfg = cfg.get("font", {})
        fam_chain = font_cfg.get("family_chain")
        if not fam_chain:
            # Accept legacy/simple form: { "family": "Arial" }
            fam = font_cfg.get("family")
            if isinstance(fam, str) and fam.strip():
                fam_chain = [fam.strip(), 'DejaVu Sans', 'Arial', 'Helvetica']
        size_val = font_cfg.get("size")
        if fam_chain:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = fam_chain
        numeric_size = None
        if size_val is not None:
            try:
                numeric_size = float(size_val)
                plt.rcParams["font.size"] = numeric_size
            except Exception as e:
                print(f"[DEBUG] Exception parsing font size: {e}")
                numeric_size = None

    # Do not change axis labels or limits in Styles import

        # Apply font changes to existing text objects
        if fam_chain or numeric_size is not None:
            for txt in label_text_objects:
                if numeric_size is not None:
                    txt.set_fontsize(numeric_size)
                if fam_chain:
                    txt.set_fontfamily(fam_chain[0])
            for axis_label in (ax.xaxis.label, ax.yaxis.label):
                if numeric_size is not None:
                    axis_label.set_fontsize(numeric_size)
                if fam_chain:
                    axis_label.set_fontfamily(fam_chain[0])
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                if numeric_size is not None:
                    lbl.set_fontsize(numeric_size)
                if fam_chain:
                    lbl.set_fontfamily(fam_chain[0])
            # Also update top/right tick labels (label2)
            try:
                for t in ax.xaxis.get_major_ticks():
                    if hasattr(t, 'label2'):
                        if numeric_size is not None:
                            t.label2.set_fontsize(numeric_size)
                        if fam_chain:
                            t.label2.set_fontfamily(fam_chain[0])
                for t in ax.yaxis.get_major_ticks():
                    if hasattr(t, 'label2'):
                        if numeric_size is not None:
                            t.label2.set_fontsize(numeric_size)
                        if fam_chain:
                            t.label2.set_fontfamily(fam_chain[0])
            except Exception:
                pass
            # Also update duplicate top/right artists if they exist
            try:
                art = getattr(ax, '_top_xlabel_artist', None)
                if art is not None:
                    if numeric_size is not None:
                        art.set_fontsize(numeric_size)
                    if fam_chain:
                        art.set_fontfamily(fam_chain[0])
            except Exception:
                pass
            try:
                art = getattr(ax, '_right_ylabel_artist', None)
                if art is not None:
                    if numeric_size is not None:
                        art.set_fontsize(numeric_size)
                    if fam_chain:
                        art.set_fontfamily(fam_chain[0])
            except Exception:
                pass

        # Tick visibility + widths
        ticks_cfg = cfg.get("ticks", {})
        vis_cfg = ticks_cfg.get("visibility", {})
        changed_visibility = False
        for k, v in vis_cfg.items():
            if k in tick_state and isinstance(v, bool):
                tick_state[k] = v
                changed_visibility = True
        if changed_visibility:
            try:
                _ui_update_tick_visibility(ax, tick_state)
            except Exception as e:
                print(f"[DEBUG] Exception updating tick visibility: {e}")

        xmaj = ticks_cfg.get("x_major_width")
        xminr = ticks_cfg.get("x_minor_width")
        ymaj = ticks_cfg.get("y_major_width")
        yminr = ticks_cfg.get("y_minor_width")
        if any(v is not None for v in (xmaj, xminr, ymaj, yminr)):
            try:
                if xmaj is not None:
                    ax.tick_params(axis="x", which="major", width=xmaj)
                if xminr is not None:
                    ax.tick_params(axis="x", which="minor", width=xminr)
                if ymaj is not None:
                    ax.tick_params(axis="y", which="major", width=ymaj)
                if yminr is not None:
                    ax.tick_params(axis="y", which="minor", width=yminr)
            except Exception as e:
                print(f"[DEBUG] Exception setting tick widths: {e}")

    # Spines
        for name, sp_dict in cfg.get("spines", {}).items():
            if name in ax.spines:
                if "linewidth" in sp_dict:
                    ax.spines[name].set_linewidth(sp_dict["linewidth"])
                if "color" in sp_dict:
                    try:
                        ax.spines[name].set_edgecolor(sp_dict["color"])
                    except Exception:
                        pass
                if "visible" in sp_dict:
                    ax.spines[name].set_visible(sp_dict["visible"])

    # Lines
        for entry in cfg.get("lines", []):
            idx = entry.get("index")
            if idx is None or not (0 <= idx < len(ax.lines)):
                continue
            ln = ax.lines[idx]
            if "color" in entry:
                ln.set_color(entry["color"])
            if "linewidth" in entry:
                ln.set_linewidth(entry["linewidth"])
            if "linestyle" in entry:
                try:
                    ln.set_linestyle(entry["linestyle"])
                except Exception:
                    pass
            if "marker" in entry:
                try:
                    ln.set_marker(entry["marker"])
                except Exception:
                    pass
            if "markersize" in entry:
                try:
                    ln.set_markersize(entry["markersize"])
                except Exception:
                    pass
            if "markerfacecolor" in entry:
                try:
                    ln.set_markerfacecolor(entry["markerfacecolor"])
                except Exception:
                    pass
            if "markeredgecolor" in entry:
                try:
                    ln.set_markeredgecolor(entry["markeredgecolor"])
                except Exception:
                    pass
            if "alpha" in entry and entry["alpha"] is not None:
                try:
                    ln.set_alpha(entry["alpha"])
                except Exception:
                    pass
        # CIF tick sets (labels & colors)
        cif_cfg = cfg.get("cif_ticks", [])
        if cif_cfg and cif_tick_series is not None:
            for entry in cif_cfg:
                idx = entry.get("index")
                if idx is None:
                    continue
                if 0 <= idx < len(cif_tick_series):
                    lab, fname, peaksQ, wl, qmax_sim, color_old = cif_tick_series[idx]
                    lab_new = entry.get("label", lab)
                    color_new = entry.get("color", color_old)
                    cif_tick_series[idx] = (lab_new, fname, peaksQ, wl, qmax_sim, color_new)
            if hasattr(ax, "_cif_draw_func"):
                try:
                    ax._cif_draw_func()
                except Exception:
                    pass

        # Re-run label placement with current mode (no mode changes via Styles)
        update_labels_func(ax, y_data_list, label_text_objects, args.stack)

        # Margin / overflow handling
        try:
            overflow = _ui_ensure_text_visibility(fig, ax, label_text_objects, check_only=True)
        except Exception:
            overflow = False
        if overflow and adjust_margins_cb is not None:
            try:
                adjust_margins_cb()
            except Exception as e:
                print(f"[DEBUG] Exception in adjust_margins callback: {e}")
            try:
                _ui_ensure_text_visibility(fig, ax, label_text_objects)
            except Exception as e:
                print(f"[DEBUG] Exception in ensure_text_visibility: {e}")

        try:
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"[DEBUG] Exception in fig.canvas.draw_idle: {e}")
        print(f"Applied style from {filename}")

    # Axis title toggle state
        try:
            # Preserve current pads to avoid drift when toggling presence via styles
            try:
                ax._pending_xlabelpad = getattr(ax.xaxis, 'labelpad', None)
            except Exception:
                pass
            try:
                ax._pending_ylabelpad = getattr(ax.yaxis, 'labelpad', None)
            except Exception:
                pass
            at_cfg = cfg.get("axis_titles", {})
            # Top X duplicate via artist
            ax._top_xlabel_on = bool(at_cfg.get("top_x", False))
            try:
                _ui_position_top_xlabel(ax, fig, tick_state)
            except Exception:
                pass
            # Bottom X presence
            if not at_cfg.get("has_bottom_x", True):
                ax.set_xlabel("")
            elif at_cfg.get("has_bottom_x", True) and not ax.get_xlabel():
                if hasattr(ax, "_stored_xlabel"):
                    ax.set_xlabel(ax._stored_xlabel)
            # Always re-position bottom xlabel to consume pending pad or set deterministic pad
            try:
                _ui_position_bottom_xlabel(ax, fig, tick_state)
            except Exception:
                pass
            # Right Y duplicate via artist
            ax._right_ylabel_on = bool(at_cfg.get("right_y", False))
            try:
                _ui_position_right_ylabel(ax, fig, tick_state)
            except Exception:
                pass
            # Left Y presence
            if not at_cfg.get("has_left_y", True):
                ax.set_ylabel("")
            elif at_cfg.get("has_left_y", True) and not ax.get_ylabel():
                if hasattr(ax, "_stored_ylabel"):
                    ax.set_ylabel(ax._stored_ylabel)
            # Always re-position left ylabel to consume pending pad or set deterministic pad
            try:
                _ui_position_left_ylabel(ax, fig, tick_state)
            except Exception:
                pass
            # After positioning, ensure duplicate top/right title artists adopt imported font
            try:
                if numeric_size is not None:
                    art = getattr(ax, '_top_xlabel_artist', None)
                    if art is not None:
                        art.set_fontsize(numeric_size)
                    art = getattr(ax, '_right_ylabel_artist', None)
                    if art is not None:
                        art.set_fontsize(numeric_size)
                if fam_chain:
                    fam0 = fam_chain[0]
                    art = getattr(ax, '_top_xlabel_artist', None)
                    if art is not None:
                        art.set_fontfamily(fam0)
                    art = getattr(ax, '_right_ylabel_artist', None)
                    if art is not None:
                        art.set_fontfamily(fam0)
            except Exception:
                pass
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"[DEBUG] Exception in axis title toggle: {e}")
    except Exception as e:
        print(f"Error applying config: {e}")


__all__ = [
    "print_style_info",
    "export_style_config",
    "apply_style_config",
]
