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
) -> None:
    print("\n--- Style / Diagnostics ---")
    fw, fh = fig.get_size_inches()
    print(f"Figure size (inches): {fw:.3f} x {fh:.3f}")
    print(f"Figure DPI: {fig.dpi}")
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
    # Axes ranges
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(f"X range: {xlim[0]:.6g} .. {xlim[1]:.6g}")
    print(f"Y range: {ylim[0]:.6g} .. {ylim[1]:.6g}")
    # Axis labels
    print(f"X label: {ax.get_xlabel()}")
    print(f"Y label: {ax.get_ylabel()}")
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

    # Tick state
    print("Tick visibility state:")
    for k in sorted(tick_state.keys()):
        print(f"  {k:<3} : {'ON ' if tick_state[k] else 'off'}")

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

    # Global flags
    print(
        f"Mode: stack={'yes' if args.stack else 'no'}, autoscale={'yes' if args.autoscale else 'no'}, raw={'yes' if args.raw else 'no'}"
    )
    print(f"Current delta (offset spacing): {delta} (initial args.delta={args.delta})")

    # Curves
    print("Curves:")
    for i, ln in enumerate(ax.lines):
        col = ln.get_color()
        lw = ln.get_linewidth()
        ls = ln.get_linestyle()
        mk = ln.get_marker()
        a = ln.get_alpha()
        xd, yd = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
        npts = len(xd)
        xmn = np.min(xd) if npts else None
        xmx = np.max(xd) if npts else None
        ymn = np.min(yd) if npts else None
        ymx = np.max(yd) if npts else None
        off = offsets_list[i] if i < len(offsets_list) else None
        base_label = labels[i] if i < len(labels) else ""
        print(
            f"  {i+1:02d}: label='{base_label}' n={npts} color={col} lw={lw} ls={ls} marker={mk} alpha={a} "
            f"x=[{xmn},{xmx}] y=[{ymn},{ymx}] offset={off}"
        )
    print(f"Number of curves: {len(ax.lines)}")
    print(
        f"Stored full-length arrays: {len(x_full_list)} (x_full_list), {len(raw_y_full_list)} (raw_y_full_list)"
    )
    print(
        f"Normalization: {'raw intensities' if args.raw else 'per-curve max scaled to 1 (current window)'}"
    )
    # Axis title placement state
    try:
        print(
            f"Axis titles: bottom_x={'ON' if ax.get_xlabel() else 'off'} top_x={'ON' if getattr(ax,'_top_xlabel_on', False) else 'off'} "
            f"left_y={'ON' if ax.get_ylabel() else 'off'} right_y={'ON' if getattr(ax,'_right_ylabel_on', False) else 'off'}"
        )
    except Exception:
        pass
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
    try:
        fw, fh = fig.get_size_inches()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
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
            "axes": {
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": list(xlim),
                "ylim": list(ylim),
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
                    "label": (labels[i] if i < len(labels) else ""),
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
            "delta": delta,
            "mode": {"stack": bool(args.stack), "autoscale": bool(args.autoscale), "raw": bool(args.raw)},
            "layout": {
                "label_layout": "stack" if args.stack else "block_top_right",
                "xaxis_type": (
                    "Q"
                    if getattr(args, "xaxis", "") == "Q"
                    else getattr(args, "xaxis", "") or ""
                ),
            },
            "normalization": "raw" if args.raw else "normalized",
        }
        cfg["axis_titles"] = {
            "top_x": bool(getattr(ax, "_top_xlabel_on", False)),
            "right_y": bool(getattr(ax, "_right_ylabel_on", False)),
            "has_bottom_x": bool(ax.get_xlabel()),
            "has_left_y": bool(ax.get_ylabel()),
        }
        if cif_tick_series:
            cfg["cif_ticks"] = [
                {"index": i, "label": lab, "color": color}
                for i, (lab, fname, peaksQ, wl, qmax_sim, color) in enumerate(cif_tick_series)
            ]
        if not filename.endswith(".bpcfg"):
            filename += ".bpcfg"
        target = _confirm_overwrite(filename)
        if not target:
            print("Style export canceled.")
            return
        with open(target, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Exported style to {target}")
    except Exception as e:
        print(f"Error exporting style: {e}")


def apply_style_config(
    filename: str,
    fig,
    ax,
    y_data_list: List[np.ndarray],
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
        if "dpi" in figure_cfg:
            try:
                fig.set_dpi(int(figure_cfg["dpi"]))
            except Exception as e:
                print(f"[DEBUG] Exception setting dpi: {e}")

        # Font
        font_cfg = cfg.get("font", {})
        fam_chain = font_cfg.get("family_chain")
        size_val = font_cfg.get("size")
        if fam_chain:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = fam_chain
        numeric_size = None
        if size_val is not None:
            try:
                numeric_size = float(size_val)
            except Exception as e:
                print(f"[DEBUG] Exception parsing font size: {e}")
                numeric_size = None

        # Axes labels
        axes_cfg = cfg.get("axes", {})
        if "xlabel" in axes_cfg:
            ax.set_xlabel(axes_cfg["xlabel"])
        if "ylabel" in axes_cfg:
            ax.set_ylabel(axes_cfg["ylabel"])

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

        # Label layout
        layout_cfg = cfg.get("layout", {})
        cfg_layout = layout_cfg.get("label_layout")
        if cfg_layout == "block_top_right" and not args.stack:
            update_labels_func(ax, y_data_list, label_text_objects, False)
        elif cfg_layout == "stack" and not args.stack:
            print("Warning: Style file was created in stacked mode; current plot not stacked. Labels kept in block layout.")
        else:
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
            at_cfg = cfg.get("axis_titles", {})
            # Top X title
            if at_cfg.get("top_x") and not getattr(ax, "_top_xlabel_on", False) and ax.get_xlabel():
                txt = ax.xaxis.get_label()
                txt.set_position((0.5, 1.02))
                txt.set_verticalalignment("bottom")
                ax._top_xlabel_on = True
            if not at_cfg.get("top_x") and getattr(ax, "_top_xlabel_on", False):
                txt = ax.xaxis.get_label()
                txt.set_position((0.5, -0.12))
                txt.set_verticalalignment("top")
                ax._top_xlabel_on = False
            # Bottom X presence
            if not at_cfg.get("has_bottom_x", True):
                ax.set_xlabel("")
            elif at_cfg.get("has_bottom_x", True) and not ax.get_xlabel():
                if hasattr(ax, "_stored_xlabel"):
                    ax.set_xlabel(ax._stored_xlabel)
            # Right Y duplicate
            if at_cfg.get("right_y") and not getattr(ax, "_right_ylabel_on", False):
                if not hasattr(ax, "_right_label_axis") or ax._right_label_axis is None:
                    ax._right_label_axis = ax.twinx()
                    ax._right_label_axis.set_frame_on(False)
                    ax._right_label_axis.tick_params(
                        which="both", length=0, labelleft=False, labelright=False
                    )
                ax._right_label_axis.set_ylabel(ax.get_ylabel())
                ax._right_ylabel_on = True
            if not at_cfg.get("right_y") and getattr(ax, "_right_ylabel_on", False):
                if hasattr(ax, "_right_label_axis") and ax._right_label_axis is not None:
                    ax._right_label_axis.set_ylabel("")
                ax._right_ylabel_on = False
            # Left Y presence
            if not at_cfg.get("has_left_y", True):
                ax.set_ylabel("")
            elif at_cfg.get("has_left_y", True) and not ax.get_ylabel():
                if hasattr(ax, "_stored_ylabel"):
                    ax.set_ylabel(ax._stored_ylabel)
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
