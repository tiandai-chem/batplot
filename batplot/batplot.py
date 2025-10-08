#!/usr/bin/env python3
"""
batplot_v1.0.10: Interactively plot: 
    XRD data .xye, .xy, .qye, .dat, .csv
    PDF data .gr
    XAS data .nor, .chik, .chir
 More features to be added.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import random
import sys
from matplotlib.ticker import AutoMinorLocator, NullFormatter
import re
from .plotting import update_labels
from .utils import _confirm_overwrite, normalize_label_text
from .readers import read_csv_file, read_fullprof_rowwise, robust_loadtxt_skipheader
from .cif import (
    simulate_cif_pattern_Q,
    cif_reflection_positions,
    list_reflections_with_hkl,
    build_hkl_label_map_from_list,
)
from .args import parse_args as _bp_parse_args
from .interactive import interactive_menu
from .batch import batch_process
from .converters import convert_to_qye
from .session import dump_session as _bp_dump_session
from .ui import (
    apply_font_changes as _ui_apply_font_changes,
    sync_fonts as _ui_sync_fonts,
    position_top_xlabel as _ui_position_top_xlabel,
    position_right_ylabel as _ui_position_right_ylabel,
    update_tick_visibility as _ui_update_tick_visibility,
    ensure_text_visibility as _ui_ensure_text_visibility,
    resize_plot_frame as _ui_resize_plot_frame,
    resize_canvas as _ui_resize_canvas,
)
from .style import (
    print_style_info as _bp_print_style_info,
    export_style_config as _bp_export_style_config,
    apply_style_config as _bp_apply_style_config,
)

# --- Global flag to allow canvas resizing by style import ---
keep_canvas_fixed = False

# Set global default font
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'STIXGeneral', 'Liberation Sans', 'Arial Unicode MS'],
    'mathtext.fontset': 'dejavusans',   # keeps math consistent with Arial-like sans
    'font.size': 16
})


## normalize_label_text now imported from utils

## _confirm_overwrite now imported from utils

"""
Note: CIF parsing and simulation helpers now come from batplot.cif.
This file defers to simulate_cif_pattern_Q and cif_reflection_positions
imported above to avoid duplicating heavy logic here.
"""

# ---------------- Conversion Function ----------------
# Implemented in batplot.converters as convert_to_qye

# Readers now live in batplot.readers; avoid duplicating implementations here.

# ---------------- .gr (Pair Distribution Function) Reading ----------------
from .readers import read_gr_file

# Label layout handled by plotting.update_labels imported at top.

# ---------------- Interactive menu ----------------
def interactive_menu(fig, ax, y_data_list, x_data_list, labels, orig_y,
                     label_text_objects, delta, x_label, args,
                     x_full_list, raw_y_full_list, offsets_list,
                     use_Q, use_r, use_E, use_k, use_rft):
    # Ensure we operate on a single window: close any other figures accidentally opened
    try:
        import matplotlib.pyplot as _plt
    except Exception:
        _plt = None
    if _plt is not None:
        try:
            for num in list(_plt.get_fignums()):
                f = _plt.figure(num)
                if f is not fig:
                    try:
                        _plt.close(f)
                    except Exception:
                        pass
        except Exception:
            pass
    # Ensure globals declared before any assignment in nested handlers
    global show_cif_hkl, cif_extend_suspended
    # REPLACED print_main_menu with column layout (now hides 'd' and 'y' in --stack)
    is_diffraction = use_Q or (not use_r and not use_E and not use_k and not use_rft)  # 2θ or Q
    def print_main_menu():
        has_cif = False
        try:
            has_cif = any(f.lower().endswith('.cif') for f in args.files)
        except Exception:
            pass
        col1 = ["c: colors", "f: font", "l: line", "t: ticks"]
        if has_cif:
            col1.append("z: hkl")
        col2 = ["a: rearrange", "d: offset", "r: rename", "g: size","x: change X", "y: change Y"]
        col3 = ["v: find peaks", "p: print style", "i: import style", "n: crosshair", "e: export", "s: save", "b: undo", "q: quit"]
        if args.stack:
            col2 = [item for item in col2 if not item.startswith("d:") and not item.startswith("y:")]
        if not is_diffraction:
            col3 = [item for item in col3 if not item.startswith("n:")]
        rows = max(len(col1), len(col2), len(col3))
        print("\nInteractive menu:")
        print("  (Styles)         (Geometries)     (Options)")
        for i in range(rows):
            p1 = col1[i] if i < len(col1) else ""
            p2 = col2[i] if i < len(col2) else ""
            p3 = col3[i] if i < len(col3) else ""
            print(f"  {p1:<16} {p2:<16} {p3:<16}")

    # --- Helper for spine visibility ---
    def set_spine_visible(which, visible):
        if which in ax.spines:
            ax.spines[which].set_visible(visible)
            fig.canvas.draw_idle()

    def get_spine_visible(which):
        if which in ax.spines:
            return ax.spines[which].get_visible()
        return False
    # Initial menu display REMOVED to avoid double print
    # print_main_menu()
    ax.set_aspect('auto', adjustable='datalim')

    def on_xlim_change(event_ax):
        update_labels(event_ax, y_data_list, label_text_objects, args.stack)
        # Extend CIF ticks if needed when user pans/zooms horizontally
        try:
            if (not globals().get('cif_extend_suspended', False) and
                hasattr(ax, '_cif_extend_func') and hasattr(ax, '_cif_draw_func') and callable(ax._cif_extend_func)):
                current_xlim = ax.get_xlim()
                xmax = current_xlim[1]
                ax._cif_extend_func(xmax)
        except Exception:
            pass
        fig.canvas.draw()
    ax.callbacks.connect('xlim_changed', on_xlim_change)

    # --------- UPDATED unified font update helper ----------
    def apply_font_changes(new_size=None, new_family=None):
        return _ui_apply_font_changes(ax, fig, label_text_objects, normalize_label_text, new_size, new_family)

    # Generic font sync (even when size/family unchanged) so newly created labels/twin axes inherit the rcParams size
    def sync_fonts():
        return _ui_sync_fonts(ax, fig, label_text_objects)

    # Adjust vertical position of duplicate top X label depending on top tick visibility
    def position_top_xlabel():
        return _ui_position_top_xlabel(ax, fig, tick_state)

    def position_right_ylabel():
        return _ui_position_right_ylabel(ax, fig, tick_state)
    def play_jump_game():
        """
        Simple terminal 'jumping bird' (Flappy-style) game.
        Controls: j = jump, Enter = let bird fall, q = quit game.
        Avoid hitting '#' pillars. Gap moves left each tick.
        Score = pillars passed.
        """
        WIDTH = 32
        HEIGHT =  nine = 9  # make height obvious
        HEIGHT = 9
        BIRD_X = 5
        GRAVITY = 1
        JUMP_VEL = -2
        GAP_SIZE = 3
        MIN_OBS_SPACING = 6

        class Obstacle:
            __slots__ = ("x", "gap_start", "scored")
            def __init__(self, x):
                self.x = x
                self.gap_start = random.randint(1, HEIGHT - GAP_SIZE - 1)
                self.scored = False

        bird_y = HEIGHT // 2
        vel = 0
        tick = 0
        score = 0
        obstacles = [Obstacle(WIDTH - 1)]

        def need_new():
            if not obstacles:
                return True
            rightmost = max(o.x for o in obstacles)
            return rightmost < WIDTH - MIN_OBS_SPACING

        def new_obstacle():
            obstacles.append(Obstacle(WIDTH - 1))
            update_tick_visibility(); update_labels(ax, y_data_list, label_text_objects, args.stack); sync_fonts(); position_top_xlabel(); position_right_ylabel()
            # Ensure interactive changes propagate to module-level state used by other commands
            try:
                globals()['tick_state'] = tick_state
            except Exception:
                pass
        def collision():
            """Return True if the bird collides with a pillar or goes out of bounds."""
            # Out of bounds
            if bird_y < 0 or bird_y >= HEIGHT:
                return True
            # Pillar collisions: any obstacle at bird column without a gap covering bird_y
            for o in obstacles:
                if o.x == BIRD_X:
                    if not (o.gap_start <= bird_y < o.gap_start + GAP_SIZE):
                        return True
                # Also check when passing between columns (approx): treat x==BIRD_X-1 as hit if not in gap
                if o.x == BIRD_X - 1:
                    if not (o.gap_start <= bird_y < o.gap_start + GAP_SIZE):
                        return True
            return False
        def move_obstacles():
            for o in obstacles:
                o.x -= 1

        def purge_obstacles():
            while obstacles and obstacles[0].x < -1:
                obstacles.pop(0)

        def render():
            top_border = "+" + "-" * WIDTH + "+"
            print("\n" + top_border)
            for y in range(HEIGHT):
                row_chars = []
                for x in range(WIDTH):
                    ch = " "
                    # bird
                    if x == BIRD_X and y == bird_y:
                        ch = "@"
                    # obstacle
        print("  q         -> quit game")
        print("Bird = @  | Score increments when you pass a pillar.\n")

        while True:
            render()
            cmd = input("> ").strip().lower()
            if cmd == 'q':
                print("Exited game. Returning to interactive menu.\n")
                break
            if cmd == 'j':
                vel = JUMP_VEL  # jump impulse
            else:
                vel += GRAVITY  # falling acceleration

            bird_y += vel
            # Soft clamp: if hits boundary, treat as collision next loop render
            move_obstacles()
            if need_new():
                new_obstacle()
            purge_obstacles()

            # Scoring: pillar passed when it moves left of bird
            for o in obstacles:
                if not o.scored and o.x < BIRD_X:
                    o.scored = True
                    score += 1

            tick += 1
            if collision():
                render()
                print(f"Game Over! Final score: {score}\n")
                break

    # -------------------------------------------------------

    # --------- NEW: Resize only the plotting frame (axes), keep canvas (figure) size fixed ----------
    def resize_plot_frame():
        return _ui_resize_plot_frame(fig, ax, y_data_list, label_text_objects, args, update_labels)

    def resize_canvas():
        return _ui_resize_canvas(fig, ax)
    # -------------------------------------------------

    # ---- Tick / label visibility state ----
    # Prefer any state saved on the axes (set by session loader) to preserve fidelity
    tick_state = getattr(ax, '_saved_tick_state', {
        'bx': True,
        'tx': False,
        'ly': True,
        'ry': False,
        'mbx': False,
        'mtx': False,
        'mly': False,
        'mry': False
    })
    if hasattr(ax, '_saved_tick_state'):
        try:
            delattr(ax, '_saved_tick_state')
        except Exception:
            pass

    # NEW: dynamic margin adjustment for top/right ticks
    # Flag to preserve a manual/initial interactive top margin override
    if not hasattr(fig, '_interactive_top_locked'):
        fig._interactive_top_locked = False

    def adjust_margins():
        """Lightweight margin tweak based on tick visibility.

        Unlike the old version this DOES NOT try to aggressively reallocate
        space or change apparent plot size; it only adds a small padding on
        sides that show ticks so labels have breathing room. Intended to be
        idempotent and minimally invasive. Called during initial setup & some
        style operations, but not on every tick toggle anymore.
        """
        sp = fig.subplotpars
        # Start from current to avoid jumping
        left, right, bottom, top = sp.left, sp.right, sp.bottom, sp.top
        pad = 0.01  # modest expansion per active side
        max_pad = 0.10
        # Expand outward (shrinks axes) only if room
        if tick_state['ly'] and left < 0.25:
            left = min(left + pad, 0.40)
        if tick_state['ry'] and (1 - right) < 0.25:
            right = max(right - pad, 0.60)
        if tick_state['bx'] and bottom < 0.25:
            bottom = min(bottom + pad, 0.40)
        if tick_state['tx'] and (1 - top) < 0.25:
            top = max(top - pad, 0.60)

        # Keep minimum plot span
        if right - left < 0.25:
            # Undo horizontal change proportionally
            mid = (left + right) / 2
            left = mid - 0.125
            right = mid + 0.125
        if top - bottom < 0.25:
            mid = (bottom + top) / 2
            bottom = mid - 0.125
            top = mid + 0.125

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    def ensure_text_visibility(max_iterations=4, check_only=False):
        return _ui_ensure_text_visibility(fig, ax, label_text_objects, max_iterations, check_only)

    def update_tick_visibility():
        ax.tick_params(axis='x',
                       bottom=tick_state['bx'], labelbottom=tick_state['bx'],
                       top=tick_state['tx'],    labeltop=tick_state['tx'])
        ax.tick_params(axis='y',
                       left=tick_state['ly'],  labelleft=tick_state['ly'],
                       right=tick_state['ry'], labelright=tick_state['ry'])

        if tick_state['mbx'] or tick_state['mtx']:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', which='minor',
                           bottom=tick_state['mbx'],
                           top=tick_state['mtx'],
                           labelbottom=False, labeltop=False)
        else:
            ax.tick_params(axis='x', which='minor',
                           bottom=False, top=False,
                           labelbottom=False, labeltop=False)

        if tick_state['mly'] or tick_state['mry']:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='y', which='minor',
                           left=tick_state['mly'],
                           right=tick_state['mry'],
                           labelleft=False, labelright=False)
        else:
            ax.tick_params(axis='y', which='minor',
                           left=False, right=False,
                           labelleft=False, labelright=False)

    # NOTE: Previously we auto-adjusted subplot margins here based on which
    # sides had ticks visible (calling adjust_margins()). This caused the
    # plotted data area to resize every time the user toggled bx/tx/ly/ry
    # (or minor variants) in the 'h' menu. Per user request, we disable
    # that behavior so changing tick visibility does NOT change the plot
    # size or axes extent. If manual margin adjustments are desired, they
    # can still be triggered via figure resize (menu 'g') or style import.
    # If needed in future, re-enable by uncommenting the next line.
    # adjust_margins()
    ensure_text_visibility()
    fig.canvas.draw_idle()

    # NEW helper (was referenced in 'h' menu but not defined previously)
    def print_tick_state():
        print("Tick visibility state:")
        for k in sorted(tick_state.keys()):
            print(f"  {k:<3} : {'ON ' if tick_state[k] else 'off'}")

    # NEW: style / diagnostics printer (clean version)
    def print_style_info():
        return _bp_print_style_info(fig, ax, y_data_list, labels, offsets_list, x_full_list, raw_y_full_list, args, delta, label_text_objects, tick_state)

    # NEW: export current style to .bpcfg
    def export_style_config(filename):
        return _bp_export_style_config(filename, fig, ax, y_data_list, labels, delta, args, tick_state, cif_tick_series if 'cif_tick_series' in globals() else None)

    # NEW: apply imported style config (restricted application)
    def apply_style_config(filename):
        return _bp_apply_style_config(filename, fig, ax, y_data_list, label_text_objects, args, tick_state, labels, update_labels, cif_tick_series if 'cif_tick_series' in globals() else None, cif_hkl_label_map if 'cif_hkl_label_map' in globals() else None, adjust_margins)

    # Initialize with current defaults
    update_tick_visibility()

    # --- Crosshair state & toggle function (UPDATED) ---
    crosshair = {
        'active': False,
        'hline': None,
        'vline': None,
        'text': None,
        'cid_motion': None,
        'wavelength': None  # only used when axis is 2theta
    }

    def toggle_crosshair():
        if not crosshair['active']:
            if not use_Q:
                try:
                    wl_in = input("Enter wavelength in Å for Q,d display (blank=skip, q=cancel): ").strip()
                    if wl_in.lower() == 'q':
                        print("Canceled.")
                        return
                    if wl_in:
                        crosshair['wavelength'] = float(wl_in)
                    else:
                        crosshair['wavelength'] = None
                except ValueError:
                    print("Invalid wavelength. Skipping Q,d calculation.")
                    crosshair['wavelength'] = None
            vline = ax.axvline(x=ax.get_xlim()[0], color='0.35', ls='--', lw=0.8, alpha=0.85, zorder=9999)
            hline = ax.axhline(y=ax.get_ylim()[0], color='0.35', ls='--', lw=0.8, alpha=0.85, zorder=9999)
            txt = ax.text(1.0, 1.0, "",
                          ha='right', va='bottom',
                          transform=ax.transAxes,
                          fontsize=max(9, int(0.6 * plt.rcParams.get('font.size', 12))),
                          color='0.15',
                          bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.8))

            def on_move(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                x = float(event.xdata)
                y = float(event.ydata)
                vline.set_xdata([x, x])
                hline.set_ydata([y, y])

                if use_Q:
                    Q = x
                    if Q != 0:
                        d = 2 * np.pi / Q
                        txt.set_text(f"Q={Q:.6g}\nd={d:.6g} Å\ny={y:.6g}")
                    else:
                        txt.set_text(f"Q={Q:.6g}\nd=∞\ny={y:.6g}")
                elif use_r:
                    txt.set_text(f"r={x:.6g} Å\ny={y:.6g}")
                else:
                    # 2θ mode
                    if crosshair['wavelength'] is not None:
                        lam = crosshair['wavelength']
                        theta_rad = np.radians(x / 2.0)
                        Q = 4 * np.pi * np.sin(theta_rad) / lam
                        if Q != 0:
                            d = 2 * np.pi / Q
                            txt.set_text(f"2θ={x:.6g}°\nQ={Q:.6g}\nd={d:.6g} Å\ny={y:.6g}")
                        else:
                            txt.set_text(f"2θ={x:.6g}°\nQ=0\nd=∞\ny={y:.6g}")
                    else:
                        txt.set_text(f"2θ={x:.6g}°\ny={y:.6g}")

                fig.canvas.draw_idle()

            cid = fig.canvas.mpl_connect('motion_notify_event', on_move)
            crosshair.update({'active': True, 'hline': hline, 'vline': vline,
                              'text': txt, 'cid_motion': cid})
            print("Crosshair ON. Move mouse over axes. Press 'n' again to turn off.")
        else:
            if crosshair['cid_motion'] is not None:
                fig.canvas.mpl_disconnect(crosshair['cid_motion'])
            for k in ('hline', 'vline', 'text'):
                art = crosshair[k]
                if art is not None:
                    try:
                        art.remove()
                    except Exception:
                        pass
            crosshair.update({'active': False, 'hline': None, 'vline': None,
                              'text': None, 'cid_motion': None})
            fig.canvas.draw_idle()
            print("Crosshair OFF.")
    # --- End crosshair additions (UPDATED) ---

    # -------- Session helper now provided by batplot.session (dump only here) --------

    
    # history management:
    state_history = []

    def push_state(note=""):
        """Snapshot current editable state (before a modifying action)."""
        try:
            # Helper to capture a representative tick line width
            def _tick_width(axis, which):
                try:
                    ticks = axis.get_major_ticks() if which=='major' else axis.get_minor_ticks()
                    for t in ticks:
                        ln = t.tick1line
                        if ln.get_visible():
                            return ln.get_linewidth()
                except Exception:
                    return None
                return None
            snap = {
                "note": note,
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
                "tick_state": tick_state.copy(),
                "font_size": plt.rcParams.get('font.size'),
                "font_chain": list(plt.rcParams.get('font.sans-serif', [])),
                "labels": list(labels),
                "delta": delta,
                "lines": [],
                "fig_size": list(fig.get_size_inches()),
                "fig_dpi": fig.dpi,
                "axes_bbox": [float(v) for v in ax.get_position().bounds],  # x0,y0,w,h
                "axis_labels": {"xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel()},
                "spines": {name: {"lw": sp.get_linewidth(), "color": sp.get_edgecolor(), "visible": sp.get_visible()} for name, sp in ax.spines.items()},
                "tick_widths": {
                    "x_major": _tick_width(ax.xaxis, 'major'),
                    "x_minor": _tick_width(ax.xaxis, 'minor'),
                    "y_major": _tick_width(ax.yaxis, 'major'),
                    "y_minor": _tick_width(ax.yaxis, 'minor')
                },
                "cif_tick_series": [tuple(entry) for entry in cif_tick_series] if 'cif_tick_series' in globals() else None,
                "show_cif_hkl": globals().get('show_cif_hkl', False)
            }
            # Line + data arrays
            for i, ln in enumerate(ax.lines):
                snap["lines"].append({
                    "index": i,
                    "x": np.array(ln.get_xdata(), copy=True),
                    "y": np.array(ln.get_ydata(), copy=True),
                    "color": ln.get_color(),
                    "lw": ln.get_linewidth(),
                    "ls": ln.get_linestyle(),
                    "marker": ln.get_marker(),
                    "markersize": getattr(ln, 'get_markersize', lambda: None)(),
                    "mfc": getattr(ln, 'get_markerfacecolor', lambda: None)(),
                    "mec": getattr(ln, 'get_markeredgecolor', lambda: None)(),
                    "alpha": ln.get_alpha()
                })
            # Data lists
            snap["x_data_list"] = [np.array(a, copy=True) for a in x_data_list]
            snap["y_data_list"] = [np.array(a, copy=True) for a in y_data_list]
            snap["orig_y"]      = [np.array(a, copy=True) for a in orig_y]
            snap["offsets"]     = list(offsets_list)
            # Label text content
            snap["label_texts"] = [t.get_text() for t in label_text_objects]
            state_history.append(snap)
            # Debug: show brief snapshot info
            try:
                last_note = snap.get('note', '')
                spine_vis = {n: s.get('visible') for n,s in snap.get('spines', {}).items()}
                print(f"[UNDO] push_state: '{last_note}' (spines: {spine_vis})")
            except Exception:
                pass
            # Cap history length
            if len(state_history) > 40:
                state_history.pop(0)
        except Exception as e:
            print(f"Warning: could not snapshot state: {e}")

    def restore_state():
        nonlocal delta
        if not state_history:
            print("No undo history.")
            return
        snap = state_history.pop()
        try:
            # Basic numeric state
            ax.set_xlim(*snap["xlim"])
            ax.set_ylim(*snap["ylim"])
            # Tick state
            for k, v in snap["tick_state"].items():
                if k in tick_state:
                    tick_state[k] = v
            update_tick_visibility()

            # Fonts
            if snap["font_chain"]:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = snap["font_chain"]
            if snap["font_size"]:
                try:
                    plt.rcParams['font.size'] = snap["font_size"]
                except Exception:
                    pass

            # Figure size & dpi
            if snap.get("fig_size") and isinstance(snap["fig_size"], (list, tuple)) and len(snap["fig_size"])==2:
                if not globals().get('keep_canvas_fixed', True):
                    try:
                        fig.set_size_inches(snap["fig_size"][0], snap["fig_size"][1], forward=True)
                    except Exception:
                        pass
                else:
                    print("(Canvas fixed) Ignoring undo figure size restore.")
            if snap.get("fig_dpi"):
                try:
                    fig.set_dpi(int(snap["fig_dpi"]))
                except Exception:
                    pass
            # Restore axes (plot frame) via stored bbox if present
            if snap.get("axes_bbox") and isinstance(snap["axes_bbox"], (list, tuple)) and len(snap["axes_bbox"])==4:
                try:
                    x0,y0,w,h = snap["axes_bbox"]
                    left = x0; bottom = y0; right = x0 + w; top = y0 + h
                    if 0 < left < right <=1 and 0 < bottom < top <=1:
                        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
                except Exception:
                    pass

            # Axis labels
            axis_labels = snap.get("axis_labels", {})
            if axis_labels.get("xlabel") is not None:
                ax.set_xlabel(axis_labels["xlabel"])
            if axis_labels.get("ylabel") is not None:
                ax.set_ylabel(axis_labels["ylabel"])

            # Spines (linewidth, color, visibility)
            for name, spec in snap.get("spines", {}).items():
                sp_obj = ax.spines.get(name)
                if sp_obj is None: continue
                try:
                    if "lw" in spec:
                        sp_obj.set_linewidth(spec["lw"])
                    if "color" in spec and spec["color"] is not None:
                        sp_obj.set_edgecolor(spec["color"])
                    if "visible" in spec:
                        try:
                            sp_obj.set_visible(bool(spec["visible"]))
                        except Exception:
                            pass
                except Exception:
                    pass
            # Debug: report restored spine visibilities
            try:
                restored = {name: bool(spec.get('visible')) for name, spec in snap.get('spines', {}).items()}
                print(f"[UNDO] restore_state: restored spines {restored}")
            except Exception:
                pass

            # Tick widths
            tw = snap.get("tick_widths", {})
            try:
                if tw.get("x_major") is not None:
                    ax.tick_params(axis='x', which='major', width=tw["x_major"])
                if tw.get("x_minor") is not None:
                    ax.tick_params(axis='x', which='minor', width=tw["x_minor"])
                if tw.get("y_major") is not None:
                    ax.tick_params(axis='y', which='major', width=tw["y_major"])
                if tw.get("y_minor") is not None:
                    ax.tick_params(axis='y', which='minor', width=tw["y_minor"])
            except Exception:
                pass

            # Labels list
            labels[:] = snap["labels"]

            # Data & lines
            if len(snap["lines"]) == len(ax.lines):
                for item in snap["lines"]:
                    i = item["index"]
                    ln = ax.lines[i]
                    ln.set_data(item["x"], item["y"])
                    ln.set_color(item["color"])
                    ln.set_linewidth(item["lw"])
                    ln.set_linestyle(item["ls"])
                    if item["marker"] is not None:
                        ln.set_marker(item["marker"])
                    if item.get("markersize") is not None:
                        try: ln.set_markersize(item["markersize"])
                        except Exception: pass
                    if item.get("mfc") is not None:
                        try: ln.set_markerfacecolor(item["mfc"])
                        except Exception: pass
                    if item.get("mec") is not None:
                        try: ln.set_markeredgecolor(item["mec"])
                        except Exception: pass
                    if item["alpha"] is not None:
                        ln.set_alpha(item["alpha"])

            # Replace lists
            x_data_list[:] = [np.array(a, copy=True) for a in snap["x_data_list"]]
            y_data_list[:] = [np.array(a, copy=True) for a in snap["y_data_list"]]
            orig_y[:]      = [np.array(a, copy=True) for a in snap["orig_y"]]
            offsets_list[:] = list(snap["offsets"])
            delta = snap.get("delta", delta)

            # CIF tick sets & label visibility
            if snap.get("cif_tick_series") is not None and 'cif_tick_series' in globals():
                try:
                    cif_tick_series[:] = [tuple(t) for t in snap["cif_tick_series"]]
                except Exception:
                    pass
            if 'show_cif_hkl' in snap:
                try:
                    globals()['show_cif_hkl'] = snap['show_cif_hkl']
                except Exception:
                    pass
            # Redraw CIF ticks after restoration if available
            if hasattr(ax, '_cif_draw_func'):
                try:
                    ax._cif_draw_func()
                except Exception:
                    pass

            # Restore label texts (keep numbering style)
            for i, txt in enumerate(label_text_objects):
                base = labels[i] if i < len(labels) else ""
                txt.set_text(f"{i+1}: {base}")

            update_labels(ax, y_data_list, label_text_objects, args.stack)
            try:
                globals()['tick_state'] = tick_state
            except Exception:
                pass
            try:
                fig.canvas.draw()
            except Exception:
                try: fig.canvas.draw_idle()
                except Exception: pass
            print("Undo: restored previous state.")
        except Exception as e:
            print(f"Error restoring state: {e}")


    while True:
        print_main_menu()
        key = input("Press a key: ")


        # NEW: disable 'y' and 'd' in stack mode
        if args.stack and key in ('y', 'd'):
            print("Option disabled in --stack mode.")
            continue

        if key == 'q':
            confirm = input("Quit interactive session? Remember to save the plot! (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                continue
        elif key == 'z':  # toggle hkl labels on CIF ticks (non-blocking)
            try:
                if not cif_tick_series:
                    print("No CIF ticks loaded.")
                    continue
                global show_cif_hkl, cif_extend_suspended
                # Flip visibility flag
                show_cif_hkl = not globals().get('show_cif_hkl', False)
                # Avoid re-entrant extension while redrawing
                prev_ext = cif_extend_suspended
                cif_extend_suspended = True
                if hasattr(ax, '_cif_draw_func'):
                    ax._cif_draw_func()
                cif_extend_suspended = prev_ext
                # Count visible labels (quick heuristic: text objects containing '(' )
                n_labels = 0
                if show_cif_hkl and hasattr(ax, '_cif_tick_art'):
                    for art in ax._cif_tick_art:
                        try:
                            if hasattr(art, 'get_text') and '(' in art.get_text():
                                n_labels += 1
                        except Exception:
                            pass
                print(f"CIF hkl labels {'ON' if show_cif_hkl else 'OFF'} (visible labels: {n_labels}).")
            except Exception as e:
                print(f"Error toggling hkl labels: {e}")
            continue
        elif key == 'b':  # <-- UNDO
            restore_state()
            continue
        elif key == 'n':
            if not is_diffraction:
                print("Crosshair disabled for non-diffraction data (allowed only for 2θ or Q).")
                continue
            try:
                toggle_crosshair()
            except Exception as e:
                print(f"Error toggling crosshair: {e}")
            continue
        elif key == 's':
            # Save current interactive session
            fname = input("Save session filename (default 'batplot_session.pkl', q=cancel): ").strip()
            if not fname or fname.lower() == 'q':
                print("Canceled.")
                continue
            if not os.path.splitext(fname)[1]:
                fname += '.pkl'
            # delegate to session dumper
            try:
                _bp_dump_session(
                    fname,
                    fig=fig,
                    ax=ax,
                    x_data_list=x_data_list,
                    y_data_list=y_data_list,
                    orig_y=orig_y,
                    offsets_list=offsets_list,
                    labels=labels,
                    delta=delta,
                    args=args,
                    tick_state=tick_state,
                    cif_tick_series=(cif_tick_series if 'cif_tick_series' in globals() else None),
                    cif_hkl_map=(cif_hkl_map if 'cif_hkl_map' in globals() else None),
                    cif_hkl_label_map=(cif_hkl_label_map if 'cif_hkl_label_map' in globals() else None),
                    show_cif_hkl=globals().get('show_cif_hkl', False),
                )
            except Exception as e:
                print(f"Error saving session: {e}")
            continue
        elif key == 'w':  # hidden game remains on 'i'
            play_jump_game(); continue
        elif key == 'c':
            try:
                has_cif = False
                try:
                    has_cif = any(f.lower().endswith('.cif') for f in args.files)
                except Exception:
                    pass
                while True:
                    print("Color menu:")
                    print("  m : manual color mapping  (e.g., 1:red 2:#00B006)")
                    print("  p : apply colormap palette to a range (e.g., 1-3 viridis)")
                    if has_cif and cif_tick_series:
                        print("  t : change CIF tick set color (e.g., 1:red 2:#888888)")
                    print("  q : return to main menu")
                    sub = input("Choose (m/p/t/q): ").strip().lower()
                    if sub == 'q':
                        break
                    if sub == '':
                        continue
                    if sub == 'm':
                        print("Current curves (q to cancel):")
                        for idx, label in enumerate(labels):
                            print(f"{idx+1}: {label}")
                        color_input = input("Enter colors (e.g., 1:red 2:#00B006) or q: ").strip()
                        if not color_input or color_input.lower() == 'q':
                            print("Canceled.")
                        else:
                            push_state("color-manual")
                            entries = color_input.split()
                            for entry in entries:
                                if ":" not in entry:
                                    print(f"Skip malformed token: {entry}")
                                    continue
                                idx_str, color = entry.split(":", 1)
                                try:
                                    i = int(idx_str) - 1
                                    if 0 <= i < len(ax.lines):
                                        ax.lines[i].set_color(color)
                                    else:
                                        print(f"Index out of range: {idx_str}")
                                except ValueError:
                                    print(f"Bad index: {idx_str}")
                        fig.canvas.draw()
                    elif sub == 't' and has_cif and cif_tick_series:
                        print("Current CIF tick sets:")
                        for i,(lab,_,_,_,_,color) in enumerate(cif_tick_series):
                            print(f"  {i+1}: {lab} (color {color})")
                        line = input("Enter mappings (e.g., 1:red 2:#555555) or q: ").strip()
                        if not line or line.lower()=='q':
                            print("Canceled.")
                        else:
                            mappings = line.split()
                            for token in mappings:
                                if ':' not in token:
                                    print(f"Skip malformed token: {token}")
                                    continue
                                idx_s, col = token.split(':',1)
                                try:
                                    idx_i = int(idx_s)-1
                                    if 0 <= idx_i < len(cif_tick_series):
                                        lab,fname,peaksQ,wl,qmax_sim,_c = cif_tick_series[idx_i]
                                        cif_tick_series[idx_i] = (lab,fname,peaksQ,wl,qmax_sim,col)
                                    else:
                                        print(f"Index out of range: {idx_s}")
                                except ValueError:
                                    print(f"Bad index: {idx_s}")
                            if hasattr(ax,'_cif_draw_func'):
                                ax._cif_draw_func()
                        fig.canvas.draw()
                    elif sub == 'p':
                        base_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'batlow']
                        extras = []
                        if 'turbo' in plt.colormaps():
                            extras.append('turbo')
                        if 'batlowK' in plt.colormaps():
                            extras.append('batlowK')
                        print("Common perceptually uniform palettes:")
                        print("  " + ", ".join(base_palettes + extras[:2]))
                        print("Example: 1-4 viridis   or: all magma_r   or: 1-3,5 plasma, _r for reverse")
                        line = input("Enter range(s) and palette (e.g., '1-3 viridis') or q: ").strip()
                        if not line or line.lower() == 'q':
                            print("Canceled.")
                        else:
                            parts = line.split()
                            if len(parts) < 2:
                                print("Need range(s) and palette.")
                            else:
                                palette_name = parts[-1]
                                range_part = " ".join(parts[:-1]).replace(" ", "")
                                def parse_ranges(spec, total):
                                    spec = spec.lower()
                                    if spec == 'all':
                                        return list(range(total))
                                    result = set()
                                    tokens = spec.split(',')
                                    for tok in tokens:
                                        if not tok:
                                            continue
                                        if '-' in tok:
                                            try:
                                                a, b = tok.split('-', 1)
                                                start = int(a) - 1
                                                end = int(b) - 1
                                                if start > end:
                                                    start, end = end, start
                                                for i in range(start, end + 1):
                                                    if 0 <= i < total:
                                                        result.add(i)
                                            except ValueError:
                                                print(f"Bad range token: {tok}")
                                        else:
                                            try:
                                                i = int(tok) - 1
                                                if 0 <= i < total:
                                                    result.add(i)
                                                else:
                                                    print(f"Index out of range: {tok}")
                                            except ValueError:
                                                print(f"Bad index token: {tok}")
                                    return sorted(result)
                                indices = parse_ranges(range_part, len(ax.lines))
                                if not indices:
                                    print("No valid indices parsed.")
                                else:
                                    try:
                                        cmap = plt.get_cmap(palette_name)
                                    except ValueError:
                                        cmap = None
                                    if cmap is None and palette_name.lower().startswith("batlow"):
                                        try:
                                            import importlib
                                            cmc = importlib.import_module('cmcrameri.cm')
                                            attr = palette_name.lower()
                                            if hasattr(cmc, attr):
                                                cmap = getattr(cmc, attr)
                                            elif hasattr(cmc, 'batlow'):
                                                cmap = getattr(cmc, 'batlow')
                                        except Exception:
                                            pass
                                    if cmap is None:
                                        print(f"Unknown colormap '{palette_name}'.")
                                    else:
                                        push_state("color-palette")
                                        nsel = len(indices)
                                        low_clip = 0.08
                                        high_clip = 0.85
                                        if nsel == 1:
                                            colors = [cmap(0.55)]
                                        elif nsel == 2:
                                            colors = [cmap(low_clip), cmap(high_clip)]
                                        else:
                                            positions = np.linspace(low_clip, high_clip, nsel)
                                            colors = [cmap(p) for p in positions]
                                        for c_idx, line_idx in enumerate(indices):
                                            ax.lines[line_idx].set_color(colors[c_idx])
                                        fig.canvas.draw()
                                        print(f"Applied '{palette_name}' to curves: " +
                                              ", ".join(str(i+1) for i in indices))
                    else:
                        print("Unknown color submenu option.")
            except Exception as e:
                print(f"Error in color menu: {e}")
        elif key == 'r':
            try:
                has_cif = False
                try:
                    has_cif = any(f.lower().endswith('.cif') for f in args.files)
                except Exception:
                    pass
                while True:
                    rename_opts = "c=curve"
                    if has_cif:
                        rename_opts += ", t=cif tick label"
                    rename_opts += ", x=x-axis, y=y-axis, q=return"
                    mode = input(f"Rename ({rename_opts}): ").strip().lower()
                    if mode == 'q':
                        break
                    if mode == '':
                        continue
                    if mode == 'c':
                        idx_in = input("Curve number to rename (q=cancel): ").strip()
                        if not idx_in or idx_in.lower() == 'q':
                            print("Canceled.")
                            continue
                        try:
                            idx = int(idx_in) - 1
                        except ValueError:
                            print("Invalid index.")
                            continue
                        if not (0 <= idx < len(labels)):
                            print("Invalid index.")
                            continue
                        new_label = input("New curve label (q=cancel): ").strip()
                        if not new_label or new_label.lower() == 'q':
                            print("Canceled.")
                            continue
                        push_state("rename-curve")
                        labels[idx] = new_label
                        label_text_objects[idx].set_text(f"{idx+1}: {new_label}")
                        fig.canvas.draw()
                    elif mode == 't':
                        if not cif_tick_series:
                            print("No CIF tick sets to rename.")
                            continue
                        for i,(lab, fname, *_rest) in enumerate(cif_tick_series):
                            print(f"  {i+1}: {lab} ({os.path.basename(fname)})")
                        s = input("CIF tick number to rename (q=cancel): ").strip()
                        if not s or s.lower()=='q':
                            print("Canceled.")
                            continue
                        try:
                            idx = int(s)-1
                            if not (0 <= idx < len(cif_tick_series)):
                                print("Index out of range."); continue
                        except ValueError:
                            print("Bad index."); continue
                        new_name = input("New CIF tick label (q=cancel): ").strip()
                        if not new_name or new_name.lower()=='q':
                            print("Canceled."); continue
                        lab,fname,peaksQ,wl,qmax_sim,color = cif_tick_series[idx]
                        cif_extend_suspended = True
                        if hasattr(ax, '_cif_tick_art'):
                            try:
                                for art in list(getattr(ax, '_cif_tick_art', [])):
                                    try:
                                        art.remove()
                                    except Exception:
                                        pass
                                ax._cif_tick_art = []
                            except Exception:
                                pass
                        cif_tick_series[idx] = (new_name, fname, peaksQ, wl, qmax_sim, color)
                        if hasattr(ax,'_cif_draw_func'): ax._cif_draw_func()
                        fig.canvas.draw()
                        cif_extend_suspended = False
                    elif mode in ('x','y'):
                        print("Enter new axis label (q=cancel). Prefer mathtext for superscripts:")
                        new_axis = input("New axis label: ").strip()
                        if not new_axis or new_axis.lower() == 'q':
                            print("Canceled.")
                            continue
                        new_axis = normalize_label_text(new_axis)
                        push_state("rename-axis")
                        if mode == 'x':
                            ax.set_xlabel(new_axis)
                        else:
                            ax.set_ylabel(new_axis)
                        sync_fonts()
                        fig.canvas.draw()
                    else:
                        print("Invalid choice.")
                    # loop continues until q
            except Exception as e:
                print(f"Error: {e}")
        elif key == 'a':
            try:
                if not args.stack:
                    print('Be careful, changing the arrangement may lead to a mess! If you want to rearrange the curves, use "--stack".')
                print("Current curve order:")
                for idx, label in enumerate(labels):
                    print(f"{idx+1}: {label}")
                new_order_str = input("Enter new order (space-separated indices, q=cancel): ").strip()
                if not new_order_str or new_order_str.lower() == 'q':
                    print("Canceled.")
                    continue
                new_order = [int(i)-1 for i in new_order_str.strip().split()]
                if len(new_order) != len(labels):
                    print("Error: Number of indices does not match number of curves.")
                    continue
                if any(i < 0 or i >= len(labels) for i in new_order):
                    print("Error: Invalid index in order list.")
                    continue

                push_state("rearrange")

                original_styles = []
                for ln in ax.lines:
                    original_styles.append({
                        "color": ln.get_color(),
                        "linewidth": ln.get_linewidth(),
                        "linestyle": ln.get_linestyle(),
                        "alpha": ln.get_alpha(),
                        "marker": ln.get_marker(),
                        "markersize": ln.get_markersize(),
                        "markerfacecolor": ln.get_markerfacecolor(),
                        "markeredgecolor": ln.get_markeredgecolor()
                    })
                reordered_styles = [original_styles[i] for i in new_order]
                xlim_current = ax.get_xlim()

                x_data_list[:]      = [x_data_list[i] for i in new_order]
                orig_y[:]           = [orig_y[i] for i in new_order]
                y_data_list[:]      = [y_data_list[i] for i in new_order]
                labels[:]           = [labels[i] for i in new_order]
                label_text_objects[:] = [label_text_objects[i] for i in new_order]
                x_full_list[:]      = [x_full_list[i] for i in new_order]
                raw_y_full_list[:]  = [raw_y_full_list[i] for i in new_order]
                offsets_list[:]     = [offsets_list[i] for i in new_order]

                if args.stack:
                    offset_local = 0.0
                    for i, (x_plot, y_norm, style) in enumerate(zip(x_data_list, orig_y, reordered_styles)):
                        y_plot_offset = y_norm + offset_local
                        y_data_list[i] = y_plot_offset
                        offsets_list[i] = offset_local
                        ln = ax.lines[i]
                        ln.set_data(x_plot, y_plot_offset)
                        ln.set_color(style["color"])
                        ln.set_linewidth(style["linewidth"])
                        ln.set_linestyle(style["linestyle"])
                        ln.set_alpha(style["alpha"])
                        ln.set_marker(style["marker"])
                        ln.set_markersize(style["markersize"])
                        ln.set_markerfacecolor(style["markerfacecolor"])
                        ln.set_markeredgecolor(style["markeredgecolor"])
                        y_range = (y_norm.max() - y_norm.min()) if y_norm.size else 0.0
                        gap = y_range + (delta * (y_range if args.autoscale else 1.0))
                        offset_local -= gap
                else:
                    offset_local = 0.0
                    for i, (x_plot, y_norm, style) in enumerate(zip(x_data_list, orig_y, reordered_styles)):
                        y_plot_offset = y_norm + offset_local
                        y_data_list[i] = y_plot_offset
                        offsets_list[i] = offset_local
                        ln = ax.lines[i]
                        ln.set_data(x_plot, y_plot_offset)
                        ln.set_color(style["color"])
                        ln.set_linewidth(style["linewidth"])
                        ln.set_linestyle(style["linestyle"])
                        ln.set_alpha(style["alpha"])
                        ln.set_marker(style["marker"])
                        ln.set_markersize(style["markersize"])
                        ln.set_markerfacecolor(style["markerfacecolor"])
                        ln.set_markeredgecolor(style["markeredgecolor"])
                        increment = (y_norm.max() - y_norm.min()) * delta if (args.autoscale and y_norm.size) else delta
                        offset_local += increment

                for i, (txt, lab) in enumerate(zip(label_text_objects, labels)):
                    txt.set_text(f"{i+1}: {lab}")
                # Preserve current axis titles (respect 't' menu toggles like bt/lt)
                ax.set_xlim(xlim_current)
                # Do not reset xlabel/ylabel here; rearrange should not change title visibility
                update_labels(ax, y_data_list, label_text_objects, args.stack)
                fig.canvas.draw()
            except Exception as e:
                print(f"Error rearranging curves: {e}")
        elif key == 'x':
            try:
                rng = input("Enter new X range (min max) or 'full' (q=cancel): ").strip()
                if not rng or rng.lower() == 'q':
                    print("Canceled.")
                    continue
                push_state("xrange")
                if rng.lower() == 'full':
                    new_min = min(xf.min() for xf in x_full_list if xf.size)
                    new_max = max(xf.max() for xf in x_full_list if xf.size)
                else:
                    new_min, new_max = map(float, rng.split())
                ax.set_xlim(new_min, new_max)
                for i in range(len(labels)):
                    xf = x_full_list[i]; yf_raw = raw_y_full_list[i]
                    mask = (xf>=new_min) & (xf<=new_max)
                    x_sub = xf[mask]; y_sub_raw = yf_raw[mask]
                    if x_sub.size == 0:
                        ax.lines[i].set_data([], [])
                        y_data_list[i] = np.array([]); orig_y[i] = np.array([]); continue
                    if not args.raw:
                        if y_sub_raw.size:
                            y_min = float(y_sub_raw.min())
                            y_max = float(y_sub_raw.max())
                            span = y_max - y_min
                            if span > 0:
                                y_sub_norm = (y_sub_raw - y_min) / span
                            else:
                                y_sub_norm = np.zeros_like(y_sub_raw)
                        else:
                            y_sub_norm = y_sub_raw
                    else:
                        y_sub_norm = y_sub_raw
                    offset_val = offsets_list[i]
                    y_with_offset = y_sub_norm + offset_val
                    ax.lines[i].set_data(x_sub, y_with_offset)
                    x_data_list[i] = x_sub
                    y_data_list[i] = y_with_offset
                    orig_y[i] = y_sub_norm
                ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
                update_labels(ax, y_data_list, label_text_objects, args.stack)
                # Extend CIF ticks after x-range change
                try:
                    if hasattr(ax, '_cif_extend_func'):
                        ax._cif_extend_func(ax.get_xlim()[1])
                except Exception:
                    pass
                try:
                    if hasattr(ax, '_cif_draw_func'):
                        ax._cif_draw_func()
                except Exception:
                    pass
                #ensure_text_visibility()
                fig.canvas.draw()
            except Exception as e:
                print(f"Error setting X-axis range: {e}")
        elif key == 'y':  # <-- Y-RANGE HANDLER (now only reachable if not args.stack)
            try:
                rng = input("Enter new Y range (min max), 'auto', or 'full' (q=cancel): ").strip().lower()
                if not rng or rng == 'q':
                    print("Canceled.")
                    continue
                push_state("yrange")
                if rng == 'auto':
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
                else:
                    if rng == 'full':
                        all_min = None
                        all_max = None
                        for arr in y_data_list:
                            if arr.size:
                                mn = float(arr.min())
                                mx = float(arr.max())
                                all_min = mn if all_min is None else min(all_min, mn)
                                all_max = mx if all_max is None else max(all_max, mx)
                        if all_min is None or all_max is None:
                            print("No data to compute full Y range.")
                            continue
                        y_min, y_max = all_min, all_max
                    else:
                        parts = rng.split()
                        if len(parts) != 2:
                            print("Need exactly two numbers for Y range.")
                            continue
                        y_min, y_max = map(float, parts)
                        if y_min == y_max:
                            print("Warning: min == max; expanding slightly.")
                            eps = abs(y_min)*1e-6 if y_min != 0 else 1e-6
                            y_min -= eps
                            y_max += eps
                    ax.set_ylim(y_min, y_max)
                update_labels(ax, y_data_list, label_text_objects, args.stack)
                fig.canvas.draw_idle()
                print(f"Y range set to {ax.get_ylim()}")
            except Exception as e:
                print(f"Error setting Y-axis range: {e}")
        elif key == 'd':  # <-- DELTA / OFFSET HANDLER (now only reachable if not args.stack)
            if len(labels) <= 1:
                print("Warning: Only one curve loaded; applying an offset is not recommended.")
            try:
                new_delta_str = input(f"Enter new offset spacing (current={delta}): ").strip()
                new_delta = float(new_delta_str)
                delta = new_delta
                offsets_list[:] = []
                if args.stack:
                    # (Should not occur because disabled, but keep safe path)
                    current_offset = 0.0
                    for i, y_norm in enumerate(orig_y):
                        y_with_offset = y_norm + current_offset
                        y_data_list[i] = y_with_offset
                        offsets_list.append(current_offset)
                        ax.lines[i].set_data(x_data_list[i], y_with_offset)
                        y_range = (y_norm.max() - y_norm.min()) if y_norm.size else 0.0
                        gap = y_range + (delta * (y_range if args.autoscale else 1.0))
                        current_offset -= gap
                else:
                    current_offset = 0.0
                    for i, y_norm in enumerate(orig_y):
                        y_with_offset = y_norm + current_offset
                        y_data_list[i] = y_with_offset
                        offsets_list.append(current_offset)
                        ax.lines[i].set_data(x_data_list[i], y_with_offset)
                        increment = (y_norm.max() - y_norm.min()) * delta if (args.autoscale and y_norm.size) else delta
                        current_offset += increment
                update_labels(ax, y_data_list, label_text_objects, args.stack)
                ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
                fig.canvas.draw()
                print(f"Offsets updated with delta={delta}")
            except Exception as e:
                print(f"Error updating offsets: {e}")
        elif key == 'l':
            try:
                while True:
                    print("Line submenu:")
                    print("  c  : change curve line widths")
                    print("  f  : change frame (axes spines) and tick widths")
                    print("  ld : show line and dots (markers) for all curves")
                    print("  d  : show only dots (no connecting line) for all curves")
                    print("  q  : return")
                    sub = input("Choose (c/f/ld/d/q): ").strip().lower()
                    if sub == 'q':
                        break
                    if sub == '':
                        continue
                    if sub == 'c':
                        spec = input("Curve widths (single value OR mappings like '1:1.2 3:2', q=cancel): ").strip()
                        if not spec or spec.lower() == 'q':
                            print("Canceled.")
                        else:
                            push_state("linewidth")
                            if ":" in spec:
                                parts = spec.split()
                                for p in parts:
                                    if ":" not in p:
                                        print(f"Skip malformed token: {p}")
                                        continue
                                    idx_str, lw_str = p.split(":", 1)
                                    try:
                                        idx = int(idx_str) - 1
                                        lw = float(lw_str)
                                        if 0 <= idx < len(ax.lines):
                                            ax.lines[idx].set_linewidth(lw)
                                        else:
                                            print(f"Index out of range: {idx+1}")
                                    except ValueError:
                                        print(f"Bad token: {p}")
                            else:
                                try:
                                    lw = float(spec)
                                    for ln in ax.lines:
                                        ln.set_linewidth(lw)
                                except ValueError:
                                    print("Invalid width value.")
                            fig.canvas.draw()
                    elif sub == 'f':
                        fw_in = input("Enter frame/tick width (e.g., 1.5) or 'm M' (major minor) or q: ").strip()
                        if not fw_in or fw_in.lower() == 'q':
                            print("Canceled.")
                        else:
                            push_state("framewidth")
                            parts = fw_in.split()
                            try:
                                if len(parts) == 1:
                                    frame_w = float(parts[0])
                                    tick_major = frame_w
                                    tick_minor = frame_w * 0.6
                                else:
                                    frame_w = float(parts[0])
                                    tick_major = float(parts[1])
                                    tick_minor = tick_major * 0.7
                                for sp in ax.spines.values():
                                    sp.set_linewidth(frame_w)
                                ax.tick_params(which='major', width=tick_major)
                                ax.tick_params(which='minor', width=tick_minor)
                                fig.canvas.draw()
                                print(f"Set frame width={frame_w}, major tick width={tick_major}, minor tick width={tick_minor}")
                            except ValueError:
                                print("Invalid numeric value(s).")
                    elif sub == 'ld':
                        push_state("line+dots")
                        try:
                            msize_in = input("Marker size (blank=auto ~3*lw): ").strip()
                            custom_msize = float(msize_in) if msize_in else None
                        except ValueError:
                            custom_msize = None
                        for ln in ax.lines:
                            lw = ln.get_linewidth() or 1.0
                            ln.set_linestyle('-')
                            ln.set_marker('o')
                            msize = custom_msize if custom_msize is not None else max(3.0, lw * 3.0)
                            ln.set_markersize(msize)
                            col = ln.get_color()
                            try:
                                ln.set_markerfacecolor(col)
                                ln.set_markeredgecolor(col)
                            except Exception:
                                pass
                        fig.canvas.draw()
                        print("Applied line+dots style to all curves.")
                    elif sub == 'd':
                        push_state("dots-only")
                        try:
                            msize_in = input("Marker size (blank=auto ~3*lw): ").strip()
                            custom_msize = float(msize_in) if msize_in else None
                        except ValueError:
                            custom_msize = None
                        for ln in ax.lines:
                            lw = ln.get_linewidth() or 1.0
                            ln.set_linestyle('None')
                            ln.set_marker('o')
                            msize = custom_msize if custom_msize is not None else max(3.0, lw * 3.0)
                            ln.set_markersize(msize)
                            col = ln.get_color()
                            try:
                                ln.set_markerfacecolor(col)
                                ln.set_markeredgecolor(col)
                            except Exception:
                                pass
                        fig.canvas.draw()
                        print("Applied dots-only style to all curves.")
                    else:
                        print("Unknown submenu option.")
            except Exception as e:
                print(f"Error setting widths: {e}")
        elif key == 'f':
            while True:
                subkey = input("Font submenu (s=size, f=family, q=return): ").strip().lower()
                if subkey == 'q':
                    break
                if subkey == '':
                    continue
                if subkey == 's':
                    try:
                        fs = input("Enter new font size (q=cancel): ").strip()
                        if not fs or fs.lower() == 'q':
                            print("Canceled.")
                        else:
                            push_state("font-change")
                            fs_val = float(fs)
                            apply_font_changes(new_size=fs_val)
                    except Exception as e:
                        print(f"Error changing font size: {e}")
                elif subkey == 'f':
                    try:
                        print("Common publication fonts:")
                        print("  1) Arial")
                        print("  2) Helvetica")
                        print("  3) Times New Roman")
                        print("  4) STIXGeneral")
                        print("  5) DejaVu Sans")
                        ft_raw = input("Enter font number or family name (q=cancel): ").strip()
                        if not ft_raw or ft_raw.lower() == 'q':
                            print("Canceled.")
                        else:
                            font_map = {
                                '1': 'Arial',
                                '2': 'Helvetica',
                                '3': 'Times New Roman',
                                '4': 'STIXGeneral',
                                '5': 'DejaVu Sans'
                            }
                            ft = font_map.get(ft_raw, ft_raw)
                            push_state("font-change")
                            print(f"Setting font family to: {ft}")
                            apply_font_changes(new_family=ft)
                    except Exception as e:
                        print(f"Error changing font family: {e}")
                else:
                    print("Invalid font submenu option.")
        elif key == 'g':
            try:
                while True:
                    choice = input("Resize submenu: (p=plot frame, c=canvas, q=cancel): ").strip().lower()
                    if not choice:
                        continue
                    if choice == 'q':
                        break
                    if choice == 'p':
                        push_state("resize-frame")
                        resize_plot_frame()
                        update_labels(ax, y_data_list, label_text_objects, args.stack)
                    elif choice == 'c':
                        push_state("resize-canvas")
                        resize_canvas()
                    else:
                        print("Unknown option.")
            except Exception as e:
                print(f"Error in resize submenu: {e}")
        elif key == 't':
            try:
                while True:
                    print("Toggle codes:")
                    print("  bx  bottom X major ticks & labels")
                    print("  tx  top    X major ticks & labels")
                    print("  ly  left   Y major ticks & labels")
                    print("  ry  right  Y major ticks & labels")
                    print("  mbx bottom X minor ticks")
                    print("  mtx top    X minor ticks")
                    print("  mly left   Y minor ticks")
                    print("  mry right  Y minor ticks")
                    print("  bt  bottom X axis title")
                    print("  tt  top    X axis title")
                    print("  lt  left   Y axis title")
                    print("  rt  right  Y axis title")
                    print("  bl  bottom plot frame line (spine)")
                    print("  tl  top    plot frame line (spine)")
                    print("  ll  left   plot frame line (spine)")
                    print("  rl  right  plot frame line (spine)")
                    print("  list show state   q return")
                    cmd = input("Enter code(s): ").strip().lower()
                    if not cmd:
                        continue
                    if cmd == 'q':
                        break
                    parts = cmd.split()
                    if parts == ['list']:
                        print_tick_state()
                        # Print spine (frame) visibility
                        print("Spine (frame) visibility:")
                        for spine in ['bottom','top','left','right']:
                            vis = get_spine_visible(spine)
                            print(f"  {spine:<6}: {'ON ' if vis else 'off'}")
                        continue
                    push_state("tick-toggle")
                    for p in parts:
                        # Axis title toggles
                        if p in ('bt','tt','lt','rt'):
                            if p == 'bt':
                                cur = ax.get_xlabel()
                                if cur:
                                    ax.set_xlabel("")
                                    print("Hid bottom X axis title")
                                else:
                                    ax.set_xlabel(ax._stored_xlabel if hasattr(ax,'_stored_xlabel') else ax.get_xlabel() or "")
                                    print("Shown bottom X axis title")
                            elif p == 'tt':
                                vis = getattr(ax, '_top_xlabel_on', False)
                                if not vis:
                                    lbl_text = ax.get_xlabel()
                                    if not lbl_text:
                                        print("No bottom X label to duplicate.")
                                    else:
                                        if not hasattr(ax,'_top_xlabel_artist') or ax._top_xlabel_artist is None:
                                            ax._top_xlabel_artist = ax.text(0.5,1.02,lbl_text,ha='center',va='bottom',transform=ax.transAxes)
                                        else:
                                            ax._top_xlabel_artist.set_text(lbl_text)
                                            ax._top_xlabel_artist.set_visible(True)
                                        ax._top_xlabel_on = True
                                        print("Shown duplicate top X axis title (bottom kept)")
                                else:
                                    if hasattr(ax,'_top_xlabel_artist') and ax._top_xlabel_artist is not None:
                                        ax._top_xlabel_artist.set_visible(False)
                                    ax._top_xlabel_on = False
                                    print("Hid top X axis title duplicate")
                            elif p == 'lt':
                                cur = ax.get_ylabel()
                                if cur:
                                    ax.set_ylabel("")
                                    print("Hid left Y axis title")
                                else:
                                    ax.set_ylabel(ax._stored_ylabel if hasattr(ax,'_stored_ylabel') else ax.get_ylabel() or "")
                                    print("Shown left Y axis title")
                            elif p == 'rt':
                                vis = getattr(ax, '_right_ylabel_on', False)
                                if not vis:
                                    base = ax.get_ylabel()
                                    if not base:
                                        print("No left Y label to duplicate.")
                                    else:
                                        if hasattr(ax,'_right_ylabel_artist') and ax._right_ylabel_artist is not None:
                                            try: ax._right_ylabel_artist.remove()
                                            except Exception: pass
                                        ax._right_ylabel_artist = ax.text(1.02,0.5,base, rotation=90, va='center', ha='left', transform=ax.transAxes)
                                        ax._right_ylabel_on = True
                                        print("Shown duplicate right Y axis title")
                                        position_right_ylabel()
                                else:
                                    if hasattr(ax,'_right_ylabel_artist') and ax._right_ylabel_artist is not None:
                                        try:
                                            ax._right_ylabel_artist.remove()
                                        except Exception:
                                            pass
                                        ax._right_ylabel_artist = None
                                    ax._right_ylabel_on = False
                                    print("Hid right Y axis title")
                            continue
                        # Plot frame (spine) toggles
                        if p in ('bl','tl','ll','rl'):
                            spine_map = {'bl':'bottom','tl':'top','ll':'left','rl':'right'}
                            spine = spine_map[p]
                            vis = get_spine_visible(spine)
                            set_spine_visible(spine, not vis)
                            print(f"Toggled {spine} spine -> {'ON' if not vis else 'off'}")
                            continue
                        # Tick toggles
                        if p in tick_state:
                            tick_state[p] = not tick_state[p]
                            print(f"Toggled {p} -> {'ON' if tick_state[p] else 'off'}")
                        else:
                            print(f"Unknown code: {p}")
                    update_tick_visibility(); update_labels(ax, y_data_list, label_text_objects, args.stack); sync_fonts(); position_top_xlabel(); position_right_ylabel()
            except Exception as e:
                print(f"Error in tick visibility menu: {e}")
        elif key == 'p':
            try:
                while True:
                    print_style_info()
                    sub = input("Style submenu: (e=export, q=return, r=refresh): ").strip().lower()
                    if sub == 'q':
                        break
                    if sub == 'r' or sub == '':
                        continue
                    if sub == 'e':
                        fname = input("Enter export filename (will add .bpcfg if missing, q=cancel): ").strip()
                        if not fname or fname.lower() == 'q':
                            print("Canceled.")
                        else:
                            export_style_config(fname)
                    else:
                        print("Unknown choice.")
            except Exception as e:
                print(f"Error in style submenu: {e}")
        elif key == 'i':
            try:
                fname = input("Enter style filename (.bpcfg, with or without extension, q=cancel): ").strip()
                if not fname or fname.lower() == 'q':
                    print("Canceled.")
                    continue
                if not os.path.isfile(fname):
                    root, ext = os.path.splitext(fname)
                    if ext == "":
                        alt = fname + ".bpcfg"
                        if os.path.isfile(alt):
                            fname = alt
                        else:
                            print("File not found.")
                            continue
                    else:
                        print("File not found.")
                        continue
                push_state("style-import")
                apply_style_config(fname)
            except Exception as e:
                print(f"Error importing style: {e}")
        elif key == 'e':
            try:
                filename = input("Enter filename (default SVG if no extension, q=cancel): ").strip()
                if not filename or filename.lower() == 'q':
                    print("Canceled.")
                else:
                    if not os.path.splitext(filename)[1]:
                        filename += ".svg"
                    # Confirm overwrite if file exists
                    export_target = _confirm_overwrite(filename)
                    if not export_target:
                        print("Export canceled.")
                    else:
                        # Temporarily remove numbering for export
                        for i, txt in enumerate(label_text_objects):
                            txt.set_text(labels[i])
                        fig.savefig(export_target, dpi=300)
                        print(f"Figure saved to {export_target}")
                        for i, txt in enumerate(label_text_objects):
                            txt.set_text(f"{i+1}: {labels[i]}")
                        fig.canvas.draw()
            except Exception as e:
                print(f"Error saving figure: {e}")
        # (Add delta 'd' branch here if present; ensure push_state at start)
        elif key == 'v':
            try:
                rng_in = input("Peak X range (min max, 'current' for axes limits, q=cancel): ").strip().lower()
                if not rng_in or rng_in == 'q':
                    print("Canceled.")
                    continue
                if rng_in == 'current':
                    x_min, x_max = ax.get_xlim()
                else:
                    parts = rng_in.split()
                    if len(parts) != 2:
                        print("Need exactly two numbers or 'current'.")
                        continue
                    x_min, x_max = map(float, parts)
                    if x_min > x_max:
                        x_min, x_max = x_max, x_min

                frac_in = input("Min relative peak height (0–1, default 0.1): ").strip()
                min_frac = float(frac_in) if frac_in else 0.1
                if min_frac < 0: min_frac = 0.0
                if min_frac > 1: min_frac = 1.0

                swin = input("Smoothing window (odd int >=3, blank=none): ").strip()
                if swin:
                    try:
                        win = int(swin)
                        if win < 3 or win % 2 == 0:
                            print("Invalid window; disabling smoothing.")
                            win = 0
                        else:
                            print(f"Using moving-average smoothing (window={win}).")
                    except ValueError:
                        print("Bad window value; no smoothing.")
                        win = 0
                else:
                    win = 0

                print("\n--- Peak Report ---")
                print(f"X range used: {x_min} .. {x_max}  (relative height threshold={min_frac})")
                for i, (x_arr, y_off) in enumerate(zip(x_data_list, y_data_list)):
                    # Recover original curve (remove vertical offset)
                    if i < len(offsets_list):
                        y_arr = y_off - offsets_list[i]
                    else:
                        y_arr = y_off.copy()

                    # Restrict to selected window
                    mask = (x_arr >= x_min) & (x_arr <= x_max)
                    x_sel = x_arr[mask]
                    y_sel = y_arr[mask]

                    label = labels[i] if i < len(labels) else f"Curve {i+1}"
                    print(f"\nCurve {i+1}: {label}")
                    if x_sel.size < 3:
                        print("  (Insufficient points)")
                        continue

                    # Optional smoothing
                    if win >= 3 and x_sel.size >= win:
                        kernel = np.ones(win, dtype=float) / win
                        y_sm = np.convolve(y_sel, kernel, mode='same')
                    else:
                        y_sm = y_sel

                    # Determine threshold
                    ymax = float(np.max(y_sm))
                    if ymax <= 0:
                        print("  (Non-positive data)")
                        continue
                    min_height = ymax * min_frac

                    # Simple local maxima detection
                    y_prev = y_sm[:-2]
                    y_mid  = y_sm[1:-1]
                    y_next = y_sm[2:]
                    core_mask = (y_mid > y_prev) & (y_mid >= y_next) & (y_mid >= min_height)
                    if not np.any(core_mask):
                        print("  (No peaks)")
                        continue
                    peak_indices = np.where(core_mask)[0] + 1  # shift because we looked at 1..n-2

                    # Optional refine: keep only distinct peaks (skip adjacent equal plateau)
                    peaks = []
                    last_idx = -10
                    for pi in peak_indices:
                        if pi - last_idx == 1 and y_sm[pi] == y_sm[last_idx]:
                            # same plateau, keep first
                            continue
                        peaks.append(pi)
                        last_idx = pi

                    print("  Peaks (x, y):")
                    for pi in peaks:
                        print(f"    x={x_sel[pi]:.6g}, y={y_sel[pi]:.6g}")
                print("\n--- End Peak Report ---\n")
            except Exception as e:
                print(f"Error finding peaks: {e}")
#
# ---------------- Argument Parsing ----------------
# Use shared parser from batplot.args to avoid large duplicate help text here
args = _bp_parse_args()

# batch_process moved to batplot.batch

# Detect batch invocation: exactly one positional argument and it's a dir OR 'all'
if len(args.files) == 1:
    sole = args.files[0]
    if sole.lower() == 'all':
        batch_process(os.getcwd(), args)
        exit()
    elif os.path.isdir(sole):
        batch_process(os.path.abspath(sole), args)
        exit()

# ---------------- Normal (multi-file) path continues below ----------------
# Apply conditional default for delta (normal mode only)
if args.delta is None:
    args.delta = 0.1 if args.stack else 0.0

# ---------------- Automatic session (.pkl) load shortcut ----------------
# If user invokes: batplot session.pkl [--interactive]
if len(args.files) == 1 and args.files[0].lower().endswith('.pkl'):
    sess_path = args.files[0]
    if not os.path.isfile(sess_path):
        print(f"Session file not found: {sess_path}")
        exit(1)
    try:
        with open(sess_path, 'rb') as f:
            sess = pickle.load(f)
        if not isinstance(sess, dict) or 'version' not in sess:
            print("Not a valid batplot session file.")
            exit(1)
    except Exception as e:
        print(f"Failed to load session: {e}")
        exit(1)
    # Reconstruct minimal state and go to interactive if requested
    plt.ion() if args.interactive else None
    fig, ax = plt.subplots(figsize=(8,6))
    y_data_list = []
    x_data_list = []
    labels_list = []
    orig_y = []
    label_text_objects = []
    x_full_list = []
    raw_y_full_list = []
    offsets_list = []
    tick_state = {
        'bx': True,'tx': False,'ly': True,'ry': False,
        'mbx': False,'mtx': False,'mly': False,'mry': False
    }
    saved_stack = bool(sess.get('args_subset', {}).get('stack', False))
    # Pull data
    # --- Robust reconstruction of stored curves ---
    x_loaded = sess.get('x_data', [])
    y_loaded = sess.get('y_data', [])  # stored plotted (baseline+offset) values
    orig_loaded = sess.get('orig_y', [])  # stored baseline (normalized/raw w/out offsets)
    offsets_saved = sess.get('offsets', [])
    n_curves = len(x_loaded)
    for i in range(n_curves):
        x_arr = np.array(x_loaded[i])
        off = offsets_saved[i] if i < len(offsets_saved) else 0.0
        if orig_loaded and i < len(orig_loaded):
            base = np.array(orig_loaded[i])
        else:
            # Fallback: derive baseline by subtracting offset from stored y (handles legacy sessions)
            y_arr_full = np.array(y_loaded[i]) if i < len(y_loaded) else np.array([])
            base = y_arr_full - off
        y_plot = base + off
        x_data_list.append(x_arr)
        orig_y.append(base)
        y_data_list.append(y_plot)
        ax.plot(x_arr, y_plot, lw=1)
        x_full_list.append(x_arr.copy())
        raw_y_full_list.append(base.copy())
    offsets_list[:] = offsets_saved if offsets_saved else [0.0]*n_curves
    # Apply stored line styles (if any)
    try:
        stored_styles = sess.get('line_styles', [])
        for ln, st in zip(ax.lines, stored_styles):
            if 'color' in st: ln.set_color(st['color'])
            if 'linewidth' in st: ln.set_linewidth(st['linewidth'])
            if 'linestyle' in st:
                try: ln.set_linestyle(st['linestyle'])
                except Exception: pass
            if 'alpha' in st and st['alpha'] is not None: ln.set_alpha(st['alpha'])
            if 'marker' in st and st['marker'] is not None:
                try: ln.set_marker(st['marker'])
                except Exception: pass
            if 'markersize' in st and st['markersize'] is not None:
                try: ln.set_markersize(st['markersize'])
                except Exception: pass
            if 'markerfacecolor' in st and st['markerfacecolor'] is not None:
                try: ln.set_markerfacecolor(st['markerfacecolor'])
                except Exception: pass
            if 'markeredgecolor' in st and st['markeredgecolor'] is not None:
                try: ln.set_markeredgecolor(st['markeredgecolor'])
                except Exception: pass
    except Exception:
        pass
    labels_list[:] = sess.get('labels', [f"Curve {i+1}" for i in range(len(y_data_list))])
    delta = sess.get('delta', 0.0)
    ax.set_xlabel(sess.get('axis', {}).get('xlabel', 'X'))
    ax.set_ylabel(sess.get('axis', {}).get('ylabel', 'Intensity'))
    if 'xlim' in sess.get('axis', {}):
        ax.set_xlim(*sess['axis']['xlim'])
    if 'ylim' in sess.get('axis', {}):
        ax.set_ylim(*sess['axis']['ylim'])
    # Apply figure size & dpi if stored
    fig_cfg = sess.get('figure', {})
    try:
        if fig_cfg.get('size') and isinstance(fig_cfg['size'], (list, tuple)) and len(fig_cfg['size']) == 2:
            fw, fh = fig_cfg['size']
            if not globals().get('keep_canvas_fixed', True):
                fig.set_size_inches(float(fw), float(fh), forward=True)
            else:
                print("(Canvas fixed) Ignoring session figure size restore.")
        if 'dpi' in fig_cfg:
            try: fig.set_dpi(int(fig_cfg['dpi']))
            except Exception: pass
    except Exception:
        pass
    # Restore spines (linewidth, color, visibility) and subplot margins/tick widths (for CLI .pkl load)
    try:
        spine_specs = fig_cfg.get('spines', {})
        if spine_specs:
            for name, spec in spine_specs.items():
                spn = ax.spines.get(name)
                if not spn: continue
                if 'linewidth' in spec: spn.set_linewidth(spec['linewidth'])
                if 'color' in spec and spec['color'] is not None: spn.set_edgecolor(spec['color'])
                if 'visible' in spec: spn.set_visible(bool(spec['visible']))
        else:
            # legacy fallback
            legacy_vis = fig_cfg.get('spine_vis', {})
            for name, vis in legacy_vis.items():
                spn = ax.spines.get(name)
                if spn:
                    spn.set_visible(bool(vis))
        spm = fig_cfg.get('subplot_margins')
        if spm and all(k in spm for k in ('left','right','bottom','top')):
            fig.subplots_adjust(left=spm['left'], right=spm['right'], bottom=spm['bottom'], top=spm['top'])
    except Exception:
        pass
    # Font
    font_cfg = sess.get('font', {})
    if font_cfg.get('chain'):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = font_cfg['chain']
    if font_cfg.get('size'):
        plt.rcParams['font.size'] = font_cfg['size']
    # Tick state restore
    saved_tick = sess.get('tick_state', {})
    for k,v in saved_tick.items():
        if k in tick_state: tick_state[k] = v
    # Persist on axes for interactive menu initialization
    try:
        ax._saved_tick_state = dict(tick_state)
    except Exception:
        pass
    # Tick widths restore
    try:
        tw = sess.get('tick_widths', {})
        if tw.get('x_major') is not None:
            ax.tick_params(axis='x', which='major', width=tw['x_major'])
        if tw.get('x_minor') is not None:
            ax.tick_params(axis='x', which='minor', width=tw['x_minor'])
        if tw.get('y_major') is not None:
            ax.tick_params(axis='y', which='major', width=tw['y_major'])
        if tw.get('y_minor') is not None:
            ax.tick_params(axis='y', which='minor', width=tw['y_minor'])
    except Exception:
        pass
    # Rebuild label texts
    for i, lab in enumerate(labels_list):
        txt = ax.text(1.0, 1.0, f"{i+1}: {lab}", ha='right', va='top', transform=ax.transAxes,
                      fontsize=plt.rcParams.get('font.size', 12))
        label_text_objects.append(txt)
    # CIF tick series (optional)
    cif_tick_series = sess.get('cif_tick_series') or []
    cif_hkl_map = {k: [tuple(v) for v in val] for k,val in sess.get('cif_hkl_map', {}).items()}
    cif_hkl_label_map = {k: dict(v) for k,v in sess.get('cif_hkl_label_map', {}).items()}
    cif_numbering_enabled = True
    cif_extend_suspended = False
    show_cif_hkl = sess.get('show_cif_hkl', False)
    # Provide minimal stubs to satisfy interactive menu dependencies
    # Axis mode restoration informs downstream toggles (e.g., CIF conversions, crosshair availability)
    axis_mode_restored = sess.get('axis_mode', 'unknown')
    use_Q = axis_mode_restored == 'Q'
    use_r = axis_mode_restored == 'r'
    use_E = axis_mode_restored == 'energy'
    use_k = axis_mode_restored == 'k'
    use_rft = axis_mode_restored == 'rft'
    use_2th = axis_mode_restored == '2theta'
    x_label = ax.get_xlabel() or 'X'
    def update_tick_visibility_local():
        # Major ticks/labels
        ax.tick_params(axis='x', bottom=tick_state['bx'], top=tick_state['tx'], labelbottom=tick_state['bx'], labeltop=tick_state['tx'])
        ax.tick_params(axis='y', left=tick_state['ly'], right=tick_state['ry'], labelleft=tick_state['ly'], labelright=tick_state['ry'])
        # Minor ticks
        if tick_state.get('mbx') or tick_state.get('mtx'):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', which='minor', bottom=tick_state.get('mbx', False), top=tick_state.get('mtx', False), labelbottom=False, labeltop=False)
        else:
            ax.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False, labeltop=False)
        if tick_state.get('mly') or tick_state.get('mry'):
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='y', which='minor', left=tick_state.get('mly', False), right=tick_state.get('mry', False), labelleft=False, labelright=False)
        else:
            ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
    update_tick_visibility_local()
    # Ensure label positions correct
    update_labels(ax, y_data_list, label_text_objects, saved_stack)
    if cif_tick_series:
        # Provide draw/extend helpers compatible with interactive menu using original placement logic
        def _session_q_to_2theta(peaksQ, wl):
            if wl is None:
                return []
            out = []
            for q in peaksQ:
                s = q*wl/(4*np.pi)
                if 0 <= s < 1:
                    out.append(np.degrees(2*np.arcsin(s)))
            return out

        def _session_ensure_wavelength(default_wl=1.5406):
            # Prefer any stored wl, else args.wl, else provided default
            for _lab,_fname,_peaks,_wl,_qmax,_color in cif_tick_series:
                if _wl is not None:
                    return _wl
            return getattr(args, 'wl', None) or default_wl

        def _session_cif_extend(xmax_domain):
            # Minimal extension: do nothing (could replicate original if needed)
            return

        def _session_cif_draw():
            if not cif_tick_series:
                return
            try:
                cur_ylim = ax.get_ylim(); yr = cur_ylim[1]-cur_ylim[0]
                if yr <= 0: yr = 1.0
                if saved_stack or len(y_data_list) > 1:
                    global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else cur_ylim[0]
                    base = global_min - 0.08*yr; spacing = 0.05*yr
                else:
                    global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else 0.0
                    base = global_min - 0.06*yr; spacing = 0.04*yr
                needed_min = base - (len(cif_tick_series)-1)*spacing - 0.04*yr
                if needed_min < cur_ylim[0]:
                    ax.set_ylim(needed_min, cur_ylim[1]); cur_ylim = ax.get_ylim(); yr = cur_ylim[1]-cur_ylim[0]
                # Clear previous artifacts
                for art in getattr(ax, '_cif_tick_art', []):
                    try: art.remove()
                    except Exception: pass
                new_art = []
                show_hkl_local = bool(show_cif_hkl)
                wl_any = _session_ensure_wavelength()
                # Draw each series
                for i,(lab,fname,peaksQ,wl,qmax_sim,color) in enumerate(cif_tick_series):
                    y_line = base - i*spacing
                    # Convert peaks to axis domain
                    if use_2th:
                        wl_use = wl if wl is not None else wl_any
                        domain_peaks = _session_q_to_2theta(peaksQ, wl_use)
                    else:
                        domain_peaks = peaksQ
                    # Clip to visible x-range
                    xlow,xhigh = ax.get_xlim()
                    domain_peaks = [p for p in domain_peaks if xlow <= p <= xhigh]
                    # Build hkl label map (keys are Q values, not 2θ)
                    label_map = cif_hkl_label_map.get(fname, {}) if show_hkl_local else {}
                    if show_hkl_local and len(domain_peaks) > 4000:
                        show_hkl_local = False  # safety
                    for p in domain_peaks:
                        ln, = ax.plot([p,p],[y_line, y_line+0.02*yr], color=color, lw=1.0, alpha=0.9, zorder=3)
                        new_art.append(ln)
                        if show_hkl_local:
                            # When axis is 2θ convert back to Q to look up hkl label
                            if use_2th and (wl or wl_any):
                                theta = np.radians(p/2.0)
                                Qp = 4*np.pi*np.sin(theta)/(wl if wl is not None else wl_any)
                            else:
                                Qp = p
                            lbl = label_map.get(round(Qp,6))
                            if lbl:
                                t_hkl = ax.text(p, y_line+0.022*yr, lbl, ha='center', va='bottom', fontsize=7, rotation=90, color=color)
                                new_art.append(t_hkl)
                    # Removed numbering prefix; keep one leading space for padding from axis
                    label_text = f" {lab}"
                    txt = ax.text(ax.get_xlim()[0], y_line+0.005*yr, label_text,
                                  ha='left', va='bottom', fontsize=max(8,int(0.55*plt.rcParams.get('font.size',12))), color=color)
                    new_art.append(txt)
                ax._cif_tick_art = new_art
                fig.canvas.draw_idle()
            except Exception:
                pass
        ax._cif_extend_func = _session_cif_extend
        ax._cif_draw_func = _session_cif_draw
        ax._cif_draw_func()

    # Restore axis title duplicates/visibility exactly as saved
    titles = sess.get('axis_titles', {})
    try:
        # Bottom X title
        if titles.get('has_bottom_x') is False:
            ax.set_xlabel("")
        # Left Y title
        if titles.get('has_left_y') is False:
            ax.set_ylabel("")
        # Top X duplicate
        if titles.get('top_x'):
            lbl_text = ax.get_xlabel()
            if lbl_text:
                if not hasattr(ax,'_top_xlabel_artist') or ax._top_xlabel_artist is None:
                    ax._top_xlabel_artist = ax.text(0.5, 1.02, lbl_text, ha='center', va='bottom', transform=ax.transAxes)
                else:
                    ax._top_xlabel_artist.set_text(lbl_text)
                    ax._top_xlabel_artist.set_visible(True)
                ax._top_xlabel_on = True
                # Position based on tick_state
                try:
                    _ui_position_top_xlabel(ax, fig, tick_state)
                except Exception:
                    pass
        else:
            if hasattr(ax,'_top_xlabel_artist') and ax._top_xlabel_artist is not None:
                try:
                    ax._top_xlabel_artist.set_visible(False)
                except Exception:
                    pass
            ax._top_xlabel_on = False
        # Right Y duplicate
        if titles.get('right_y'):
            base = ax.get_ylabel()
            if base:
                if hasattr(ax,'_right_ylabel_artist') and ax._right_ylabel_artist is not None:
                    try: ax._right_ylabel_artist.remove()
                    except Exception: pass
                ax._right_ylabel_artist = ax.text(1.02, 0.5, base, rotation=90, va='center', ha='left', transform=ax.transAxes)
                ax._right_ylabel_on = True
                try:
                    _ui_position_right_ylabel(ax, fig, tick_state)
                except Exception:
                    pass
        else:
            if hasattr(ax,'_right_ylabel_artist') and ax._right_ylabel_artist is not None:
                try:
                    ax._right_ylabel_artist.remove()
                except Exception:
                    pass
                ax._right_ylabel_artist = None
            ax._right_ylabel_on = False
    except Exception:
        pass
    if args.interactive:
        try:
            args.stack = saved_stack
        except Exception:
            pass
        interactive_menu(fig, ax, y_data_list, x_data_list, labels_list,
                         orig_y, label_text_objects, delta, x_label, args,
                         x_full_list, raw_y_full_list, offsets_list,
                         use_Q, use_r, use_E, use_k, use_rft)
    else:
        plt.show()
    exit()

# ---------------- Handle conversion ----------------
if args.convert:
    if args.wl is None:
        print("Error: --wl is required for --convert")
       
        exit(1)
    convert_to_qye(args.convert, args.wl)
    exit()

# ---------------- Plotting ----------------
offset = 0.0
direction = -1 if args.stack else 1  # stack downward
if args.interactive:
    plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

y_data_list = []
x_data_list = []
labels_list = []
orig_y = []
label_text_objects = []
# New lists to preserve full data & offsets
x_full_list = []
raw_y_full_list = []
offsets_list = []

# ---------------- Determine X-axis type ----------------
def _ext_token(path):
    return os.path.splitext(path)[1].lower()  # includes leading dot
any_qye = any(f.lower().endswith(".qye") for f in args.files)
any_gr  = any(f.lower().endswith(".gr")  for f in args.files)
any_nor = any(f.lower().endswith(".nor") for f in args.files)
any_chik = any("chik" in _ext_token(f) for f in args.files)
any_chir = any("chir" in _ext_token(f) for f in args.files)
any_txt = any(f.lower().endswith(".txt") for f in args.files)
any_cif = any(f.lower().endswith(".cif") for f in args.files)
non_cif_count = sum(0 if f.lower().endswith('.cif') else 1 for f in args.files)
cif_only = any_cif and non_cif_count == 0
any_lambda = any(":" in f for f in args.files) or args.wl is not None

# Incompatibilities (no mixing of fundamentally different axis domains)
if sum(bool(x) for x in (any_gr, any_nor, any_chik, any_chir, (any_qye or any_lambda or any_cif))) > 1:
    raise ValueError("Cannot mix .gr (r), .nor (energy), .chik (k), .chir (FT-EXAFS R), and Q/2θ/CIF data together. Split runs.")

# Automatic axis selection based on file extensions
if any_qye:
    axis_mode = "Q"
elif any_gr:
    axis_mode = "r"
elif any_nor:
    axis_mode = "energy"
elif any_chik:
    axis_mode = "k"
elif any_chir:
    axis_mode = "rft"
elif any_txt:
    # .txt is generic, require --xaxis
    if args.xaxis:
        axis_mode = args.xaxis
    else:
        raise ValueError("Cannot determine X-axis type for .txt files. Please specify --xaxis (Q, 2theta, r, k, energy, rft, or 'user defined').")
elif any_qye or any_lambda or any_cif:
    if args.xaxis and args.xaxis.lower() in ("2theta","two_theta","tth"):
        axis_mode = "2theta"
    else:
        axis_mode = "Q"
elif args.xaxis:
    axis_mode = args.xaxis
else:
    raise ValueError("Cannot determine X-axis type (need .qye / .gr / .nor / .chik / .chir / .cif / wavelength / --xaxis). For .txt or unknown file types, use --xaxis Q, 2theta, r, k, energy, rft, or 'user defined'.")

use_Q   = axis_mode == "Q"
use_2th = axis_mode == "2theta"
use_r   = axis_mode == "r"
use_E   = axis_mode == "energy"
use_k   = axis_mode == "k"      # NEW
use_rft = axis_mode == "rft"    # NEW

# ---------------- Read and plot files ----------------
# Helper to extract discrete peak positions from a simulated CIF pattern by local maxima picking
def _extract_peak_positions(Q_array, I_array, min_rel_height=0.05):
    if Q_array.size == 0 or I_array.size == 0:
        return []
    Imax = I_array.max() if I_array.size else 0
    if Imax <= 0:
        return []
    thr = Imax * min_rel_height
    peaks = []
    for i in range(1, len(I_array)-1):
        if I_array[i] >= thr and I_array[i] >= I_array[i-1] and I_array[i] >= I_array[i+1]:
            # simple peak refine by local quadratic (optional)
            y1,y2,y3 = I_array[i-1], I_array[i], I_array[i+1]
            x1,x2,x3 = Q_array[i-1], Q_array[i], Q_array[i+1]
            denom = (y1 - 2*y2 + y3)
            if abs(denom) > 1e-12:
                dx = 0.5*(y1 - y3)/denom
                if -0.6 < dx < 0.6:
                    xc = x2 + dx*(x3 - x1)/2.0
                    if Q_array[0] <= xc <= Q_array[-1]:
                        peaks.append(xc)
                        continue
            peaks.append(Q_array[i])
    return peaks

# Will accumulate CIF tick series to render after main curves
cif_tick_series = []  # list of (label, filename, peak_positions_Q, wavelength_or_None, qmax_simulated, color)
cif_hkl_map = {}      # filename -> list of (Q,h,k,l)
cif_hkl_label_map = {}  # filename -> dict of Q -> label string
cif_numbering_enabled = True  # show numbering for CIF tick sets (mixed mode only)
cif_extend_suspended = False  # guard flag to prevent auto extension during certain operations
QUIET_CIF_EXTEND = True  # suppress extension debug output

# Cached wavelength for CIF tick conversion (prevents interactive blocking prompts)
cif_cached_wavelength = None
show_cif_hkl = False

for idx_file, file_entry in enumerate(args.files):
    parts = file_entry.split(":")
    fname = parts[0]
    wavelength_file = float(parts[-1]) if len(parts) > 1 else args.wl
    if not os.path.isfile(fname):
        print(f"File not found: {fname}")
        continue
    file_ext = os.path.splitext(fname)[1].lower()
    is_chik = "chik" in file_ext
    is_chir = "chir" in file_ext
    is_cif  = file_ext == '.cif'
    label = os.path.basename(fname)
    if wavelength_file and not use_r and not use_E and file_ext not in (".gr", ".nor", ".cif"):
        label += f" (λ={wavelength_file:.5f} Å)"

    # ---- Read data (added .nor branch) ----
    if is_cif:
        try:
            # Simulate pattern directly in Q space regardless of current axis_mode
            Q_sim, I_sim = simulate_cif_pattern_Q(fname)
            x = Q_sim
            y = I_sim
            e = None
            # Force axis mode if needed
            if not (use_Q or use_2th):
                use_Q = True
            # Reflection list and per-Q hkl labels (no wavelength cutoff in pure Q domain)
            qmax_sim = float(Q_sim[-1]) if len(Q_sim) else 0.0
            refl = cif_reflection_positions(fname, Qmax=qmax_sim, wavelength=None)
            hkl_list = list_reflections_with_hkl(fname, Qmax=qmax_sim, wavelength=None)
            cif_hkl_label_map[fname] = build_hkl_label_map_from_list(hkl_list)
            # default tick color black
            cif_tick_series.append((label, fname, refl, None, qmax_sim, 'k'))
            # If CIF mixed with other data types, do NOT plot intensity curve (ticks only)
            if not cif_only:
                continue  # skip rest of loop so curve isn't added
        except Exception as e_read:
            print(f"Error simulating CIF {fname}: {e_read}")
            continue
    elif file_ext == ".gr":
        try:
            x, y = read_gr_file(fname); e = None
        except Exception as e_read:
            print(f"Error reading {fname}: {e_read}"); continue
    elif file_ext in [".nor", ".xy", ".xye", ".qye", ".dat", ".csv"] or is_chik or is_chir:
        try:
            data = robust_loadtxt_skipheader(fname)
        except Exception as e_read:
            print(f"Error reading {fname}: {e_read}"); continue
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 2:
            print(f"Invalid data format in {fname}"); continue
        x, y = data[:, 0], data[:, 1]
        e = data[:, 2] if data.shape[1] >= 3 else None
        # For .csv, .dat, .xy, .xye, .qye, .nor, .chik, .chir, this robustly skips headers
    elif args.fullprof and file_ext == ".dat":
        try:
            y_plot, n_rows = read_fullprof_rowwise(fname)
            xstart, xend, xstep = args.fullprof[0], args.fullprof[1], args.fullprof[2]
            x_plot = np.linspace(xstart, xend, len(y_plot))
            wavelength = args.fullprof[3] if len(args.fullprof)>=4 else wavelength_file
            if use_Q and wavelength:
                theta_rad = np.radians(x_plot / 2)
                x_plot = 4*np.pi*np.sin(theta_rad)/wavelength
            e_plot = None
        except Exception as e:
            print(f"Error reading FullProf-style {fname}: {e}")
            continue

    # ---- X-axis conversion logic updated (no conversion for energy) ----
    if use_Q and file_ext not in (".qye", ".gr", ".nor"):
        if wavelength_file:
            theta_rad = np.radians(x/2)
            x_plot = 4*np.pi*np.sin(theta_rad)/wavelength_file
        else:
            x_plot = x
    else:
        # r, energy, or already Q: direct
        x_plot = x

    # ---- Store full (converted) arrays BEFORE cropping ----
    x_full = x_plot.copy()
    y_full_raw = y.copy()
    raw_y_full_list.append(y_full_raw)
    x_full_list.append(x_full)

    # ---- Apply xrange (for initial display only; full data kept above) ----
    y_plot = y_full_raw
    e_plot = e
    if args.xrange:
        mask = (x_full>=args.xrange[0]) & (x_full<=args.xrange[1])
        ax.set_xlim(args.xrange[0], args.xrange[1])
        x_plot = x_full[mask]
        y_plot = y_full_raw[mask]
        if e_plot is not None:
            e_plot = e_plot[mask]
    else:
        x_plot = x_full

    # ---- Normalize (display subset) ----
    if not args.raw:
        # Min–max normalization to 0..1 within the currently displayed (cropped) segment
        if y_plot.size:
            y_min = float(y_plot.min())
            y_max = float(y_plot.max())
            span = y_max - y_min
            if span > 0:
                y_norm = (y_plot - y_min) / span
            else:
                # Flat line -> all zeros
                y_norm = np.zeros_like(y_plot)
        else:
            y_norm = y_plot
    else:
        y_norm = y_plot

    # ---- Apply offset (waterfall vs stack) ----
    if args.stack:
        y_plot_offset = y_norm + offset
        y_range = (y_norm.max() - y_norm.min()) if y_norm.size else 0.0
        gap = y_range + (args.delta * (y_range if args.autoscale else 1.0))
        offsets_list.append(offset)
        offset -= gap
    else:
        increment = (y_norm.max() - y_norm.min()) * args.delta if (args.autoscale and y_norm.size) else args.delta
        y_plot_offset = y_norm + offset
        offsets_list.append(offset)
        offset += increment

    # ---- Plot curve ----
    ax.plot(x_plot, y_plot_offset, "-", lw=1, alpha=0.8)
    y_data_list.append(y_plot_offset.copy())
    x_data_list.append(x_plot)
    labels_list.append(label)
    # Store current normalized (subset) (used by rearrange logic)
    orig_y.append(y_norm.copy())

# ---------------- Force axis to fit all data before labels ----------------
ax.relim()
ax.autoscale_view()
fig.canvas.draw()

# Define a sample_tick safely (may be None if no labels yet)
sample_tick = None
xt_lbls = ax.get_xticklabels()
if xt_lbls:
    sample_tick = xt_lbls[0]

else:
    yt_lbls = ax.get_yticklabels()
    if yt_lbls:
        sample_tick = yt_lbls[0]

# ---------------- Initial label creation (REPLACED BLOCK) ----------------
# Remove the old simple per-curve placement loop and use:
label_text_objects = []
tick_fs = sample_tick.get_fontsize() if sample_tick else plt.rcParams.get('font.size', 12)
# get_fontname() may not exist on some backends; use family from rcParams if missing
try:
    tick_fn = sample_tick.get_fontname() if sample_tick else plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])[0]
except Exception:
    tick_fn = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])[0]

if args.stack:
    x_max = ax.get_xlim()[1]
    for i, y_plot_offset in enumerate(y_data_list):
        y_max_curve = y_plot_offset.max() if len(y_plot_offset) else ax.get_ylim()[1]
        txt = ax.text(x_max, y_max_curve,
                      f"{i+1}: {labels_list[i]}",
                      va='top', ha='right',
                      fontsize=tick_fs, fontname=tick_fn,
                      transform=ax.transData)
        label_text_objects.append(txt)
else:
    n = len(y_data_list)
    top_pad = 0.02
    start_y = 0.98
    spacing = min(0.08, max(0.025, 0.90 / max(n, 1)))
    for i in range(n):
        y_pos = start_y - i * spacing
        if y_pos < 0.02:
            y_pos = 0.02
        txt = ax.text(1.0, y_pos,
                      f"{i+1}: {labels_list[i]}",
                      va='top', ha='right',
                      fontsize=tick_fs, fontname=tick_fn,
                      transform=ax.transAxes)
        label_text_objects.append(txt)

# Ensure consistent initial placement (especially for stacked mode)
update_labels(ax, y_data_list, label_text_objects, args.stack)

# ---------------- CIF tick overlay (after labels placed) ----------------
def _ensure_wavelength_for_2theta():
    """Ensure wavelength assigned to all CIF tick sets without prompting.

    Order of preference:
      1. Existing wavelength already stored in any series.
      2. args.wl if provided by user.
      3. Previously cached value (cif_cached_wavelength).
      4. Default 1.5406 Å.
    """
    global cif_cached_wavelength
    if not cif_tick_series:
        return None
    # If any entry already has wavelength, use and cache it
    for _lab,_fname,_peaks,_wl,_qmax,_color in cif_tick_series:
        if _wl is not None:
            cif_cached_wavelength = _wl
            return _wl
    wl = getattr(args, 'wl', None)
    if wl is None:
        wl = cif_cached_wavelength if cif_cached_wavelength is not None else 1.5406
    cif_cached_wavelength = wl
    for i,(lab, fname, peaksQ, w0, qmax_sim, color) in enumerate(cif_tick_series):
        cif_tick_series[i] = (lab, fname, peaksQ, wl, qmax_sim, color)
    return wl

def _Q_to_2theta(peaksQ, wl):
    out = []
    if wl is None:
        return out
    for q in peaksQ:
        s = q*wl/(4*np.pi)
        if 0 <= s < 1:
            out.append(np.degrees(2*np.arcsin(s)))
    return out

def extend_cif_tick_series(xmax_domain):
    """Extend CIF peak list if x-range upper bound increases beyond simulated Qmax.
    xmax_domain: upper x limit in current axis units (Q or 2θ).
    """
    if globals().get('cif_extend_suspended', False):
        return
    if not cif_tick_series:
        return
    # Determine target Q for extension depending on axis
    wl_any = None
    if use_2th:
        # Ensure wavelength known
        for _,_,_,wl_,_ in cif_tick_series:
            if wl_ is not None:
                wl_any = wl_
                break
        if wl_any is None:
            wl_any = _ensure_wavelength_for_2theta()
    updated = False
    for i,(lab,fname,peaksQ,wl,qmax_sim,color) in enumerate(cif_tick_series):
        if use_2th:
            wl_use = wl if wl is not None else wl_any
            theta_rad = np.radians(min(xmax_domain, 179.9)/2.0)
            Q_target = 4*np.pi*np.sin(theta_rad)/wl_use if wl_use else qmax_sim
        else:
            Q_target = xmax_domain
        if not QUIET_CIF_EXTEND:
            try:
                print(f"[CIF extend check] {lab}: current Qmax={qmax_sim:.3f}, target Q={Q_target:.3f}")
            except Exception:
                pass
        if Q_target > qmax_sim + 1e-6:
            new_Qmax = Q_target + 0.25
            try:
                # Only apply wavelength constraint for 2θ axis; in Q axis enumerate freely
                refl = cif_reflection_positions(fname, Qmax=new_Qmax, wavelength=(wl if (wl and use_2th) else None))
                cif_tick_series[i] = (lab, fname, refl, wl, float(new_Qmax), color)
                if not QUIET_CIF_EXTEND:
                    print(f"Extended CIF ticks for {lab} to Qmax={new_Qmax:.2f} (count={len(refl)})")
                updated = True
            except Exception as e:
                print(f"Warning: could not extend CIF peaks for {lab}: {e}")
    if updated:
        # After update, redraw ticks
        draw_cif_ticks()

def draw_cif_ticks():
    if not cif_tick_series:
        return
    cur_ylim = ax.get_ylim(); yr = cur_ylim[1]-cur_ylim[0]
    if yr <= 0: yr = 1.0
    if args.stack or len(y_data_list) > 1:
        global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else cur_ylim[0]
        base = global_min - 0.08*yr; spacing = 0.05*yr
    else:
        global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else 0.0
        base = global_min - 0.06*yr; spacing = 0.04*yr
    needed_min = base - (len(cif_tick_series)-1)*spacing - 0.04*yr
    if needed_min < cur_ylim[0]:
        ax.set_ylim(needed_min, cur_ylim[1]); cur_ylim = ax.get_ylim(); yr = cur_ylim[1]-cur_ylim[0]
    # Clear previous
    for art in getattr(ax, '_cif_tick_art', []):
        try: art.remove()
        except Exception: pass
    new_art = []
    mixed_mode = (not cif_only)  # cif_only variable defined earlier in script context
    show_hkl = globals().get('show_cif_hkl', False)
    for i,(lab, fname, peaksQ, wl, qmax_sim, color) in enumerate(cif_tick_series):
        y_line = base - i*spacing
        if use_2th:
            if wl is None: wl = _ensure_wavelength_for_2theta()
            domain_peaks = _Q_to_2theta(peaksQ, wl)
        else:
            domain_peaks = peaksQ
        # --- NEW: restrict to current visible x-range for performance ---
        xlow, xhigh = ax.get_xlim()
        if domain_peaks:
            # domain_peaks may be numpy array or list; create filtered list
            domain_peaks = [p for p in domain_peaks if xlow <= p <= xhigh]
        if not domain_peaks:
            # No peaks in current window; still write label row and continue
            # Removed numbering; keep space padding
            label_text = f" {lab}"
            txt = ax.text(ax.get_xlim()[0], y_line + 0.005*yr, label_text,
                          ha='left', va='bottom', fontsize=max(8,int(0.55*plt.rcParams.get('font.size',12))), color=color)
            new_art.append(txt)
            continue
        # Build map for quick hkl lookup by Q
        hkl_entries = cif_hkl_map.get(fname, [])
        # dictionary keyed by Q value
        hkl_by_q = {}
        for qval,h,k,l in hkl_entries:
            hkl_by_q.setdefault(qval, []).append((h,k,l))
        label_map = cif_hkl_label_map.get(fname, {})
        # --- Optimized tick & hkl label drawing ---
        if show_hkl and peaksQ and label_map:
            # Guard against pathological large peak lists (can freeze UI)
            if len(peaksQ) > 4000 or len(domain_peaks) > 4000:
                print(f"[hkl] Too many peaks in {lab} (>{len(peaksQ)}) – skipping hkl labels. Press 'z' again to toggle off.")
                # still draw ticks below without labels
                effective_show_hkl = False
            else:
                effective_show_hkl = True
        else:
            effective_show_hkl = False

        # Precompute rounding function once
        if effective_show_hkl:
            # For 2θ axis we convert back to Q then round; otherwise Q directly
            for p in domain_peaks:
                ln, = ax.plot([p,p],[y_line, y_line+0.02*yr], color=color, lw=1.0, alpha=0.9, zorder=3)
                new_art.append(ln)
                if use_2th and wl:
                    theta = np.radians(p/2.0)
                    Qp = 4*np.pi*np.sin(theta)/wl
                else:
                    Qp = p
                lbl = label_map.get(round(Qp,6))
                if lbl:
                    t_hkl = ax.text(p, y_line+0.022*yr, lbl, ha='center', va='bottom', fontsize=7, rotation=90, color=color)
                    new_art.append(t_hkl)
        else:
            # Just draw ticks (no hkl labels)
            for p in domain_peaks:
                ln, = ax.plot([p,p],[y_line, y_line+0.02*yr], color=color, lw=1.0, alpha=0.9, zorder=3)
                new_art.append(ln)
        # Removed numbering; keep space padding (placed per CIF row)
        label_text = f" {lab}"
        txt = ax.text(ax.get_xlim()[0], y_line + 0.005*yr, label_text,
                      ha='left', va='bottom', fontsize=max(8,int(0.55*plt.rcParams.get('font.size',12))), color=color)
        new_art.append(txt)
    ax._cif_tick_art = new_art
    # Store simplified metadata for hover: list of dicts with 'x','y','label'
    hover_meta = []
    show_hkl = globals().get('show_cif_hkl', False)
    # Build mapping from Q to label text if available
    for i,(lab, fname, peaksQ, wl, qmax_sim, color) in enumerate(cif_tick_series):
        if use_2th and wl is None:
            wl = getattr(ax, '_cif_hover_wl', None)
        # Recreate domain peaks consistent with those drawn (limit to view)
        if use_2th:
            if wl is None: continue
            domain_peaks = _Q_to_2theta(peaksQ, wl)
        else:
            domain_peaks = peaksQ
        xlow, xhigh = ax.get_xlim()
        domain_peaks = [p for p in domain_peaks if xlow <= p <= xhigh]
        if not domain_peaks:
            continue
        # y baseline for this series (same logic as above)
        if args.stack or len(y_data_list) > 1:
            global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else ax.get_ylim()[0]
            base = global_min - 0.08*yr; spacing = 0.05*yr
        else:
            global_min = min(float(a.min()) for a in y_data_list if len(a)) if y_data_list else 0.0
            base = global_min - 0.06*yr; spacing = 0.04*yr
        y_line = base - i*spacing
        label_map = cif_hkl_label_map.get(fname, {}) if show_hkl else {}
        for p in domain_peaks:
            if use_2th and wl:
                theta = np.radians(p/2.0); Qp = 4*np.pi*np.sin(theta)/wl
            else:
                Qp = p
            lbl = label_map.get(round(Qp,6), None)
            hover_meta.append({'x': p, 'y': y_line, 'hkl': lbl, 'series': lab})
    ax._cif_tick_hover_meta = hover_meta
    fig.canvas.draw_idle()

    # Install hover handler once
    if not hasattr(ax, '_cif_hover_cid'):
        tooltip = ax.text(0,0,"", va='bottom', ha='left', fontsize=8,
                          color='black', bbox=dict(boxstyle='round,pad=0.2', fc='1.0', ec='0.7', alpha=0.85),
                          visible=False)
        ax._cif_hover_tooltip = tooltip
        def _on_move(event):
            if event.inaxes != ax:
                if tooltip.get_visible():
                    tooltip.set_visible(False); fig.canvas.draw_idle()
                return
            meta = getattr(ax, '_cif_tick_hover_meta', None)
            if not meta:
                if tooltip.get_visible():
                    tooltip.set_visible(False); fig.canvas.draw_idle()
                return
            x = event.xdata; y = event.ydata
            # Find nearest tick within pixel tolerance
            trans = ax.transData
            best = None; best_d2 = 25  # squared pixel distance threshold (5 px)
            for entry in meta:
                px, py = trans.transform((entry['x'], entry['y']))
                ex, ey = trans.transform((x, y))
                d2 = (px-ex)**2 + (py-ey)**2
                if d2 < best_d2:
                    best_d2 = d2; best = entry
            if best is None:
                if tooltip.get_visible():
                    tooltip.set_visible(False); fig.canvas.draw_idle()
                return
            # Compose text
            hkl_txt = best['hkl'] if best.get('hkl') else ''
            tip = f"{best['series']}\nQ={best['x']:.4f}" if use_Q else (f"{best['series']}\n2θ={best['x']:.4f}" if use_2th else f"{best['series']} {best['x']:.4f}")
            if hkl_txt:
                tip += f"\n{hkl_txt}"
            tooltip.set_text(tip)
            tooltip.set_position((best['x'], best['y'] + 0.025*yr))
            if not tooltip.get_visible():
                tooltip.set_visible(True)
            fig.canvas.draw_idle()
        cid = fig.canvas.mpl_connect('motion_notify_event', _on_move)
        ax._cif_hover_cid = cid

if cif_tick_series:
    # Auto-assign distinct colors if all are default 'k'
    if len(cif_tick_series) > 1:
        if all(c[-1] == 'k' for c in cif_tick_series):
            try:
                cmap_name = 'tab10' if len(cif_tick_series) <= 10 else 'hsv'
                cmap = plt.get_cmap(cmap_name)
                new_series = []
                for i,(lab,fname,peaksQ,wl,qmax_sim,col) in enumerate(cif_tick_series):
                    color = cmap(i / max(1,(len(cif_tick_series)-1)))
                    new_series.append((lab,fname,peaksQ,wl,qmax_sim,color))
                cif_tick_series[:] = new_series
            except Exception:
                pass
    if use_2th:
        _ensure_wavelength_for_2theta()
    draw_cif_ticks()
    # expose helpers for interactive updates
    ax._cif_extend_func = extend_cif_tick_series
    ax._cif_draw_func = draw_cif_ticks

if use_E: x_label = "Energy (eV)"
elif use_r: x_label = r"r (Å)"
elif use_k: x_label = r"k ($\mathrm{\AA}^{-1}$)"
elif use_rft: x_label = "Radial distance (Å)"
elif use_Q: x_label = r"Q ($\mathrm{\AA}^{-1}$)"
elif use_2th: x_label = r"$2\theta$ (deg)"
elif args.xaxis:
    x_label = str(args.xaxis)
else:
    x_label = "X"
ax.set_xlabel(x_label, fontsize=16)
if args.raw:
    ax.set_ylabel("Intensity", fontsize=16)
else:
    ax.set_ylabel("Normalized intensity (a.u.)", fontsize=16)

# Store originals for axis-title toggle restoration (t menu bn/ln)
try:
    ax._stored_xlabel = ax.get_xlabel()
    ax._stored_ylabel = ax.get_ylabel()
except Exception:
    pass

# --- FINAL LABEL POSITION PASS ---
# Some downstream operations (e.g. CIF tick overlay extending y-limits or auto margin
# adjustments by certain backends) can occur after the initial label placement,
# leading to visibly misplaced curve labels on first show. We perform a final
# synchronous draw + update_labels here to lock them to the correct coordinates
# before any saving / interactive session starts. (Subsequent interactions still
# use the existing callbacks / update logic.)
try:
    fig.canvas.draw()  # ensure limits are finalized
    update_labels(ax, y_data_list, label_text_objects, args.stack)
except Exception:
    pass

# ---------------- Save figure object ----------------
if args.savefig:
    # Remove numbering for exported figure object (if ticks present)
    if cif_tick_series and 'cif_numbering_enabled' in globals() and cif_numbering_enabled:
        prev_num = cif_numbering_enabled
        cif_numbering_enabled = False
        if 'draw_cif_ticks' in globals():
            draw_cif_ticks()
        target = _confirm_overwrite(args.savefig)
        if target:
            with open(target, "wb") as f:
                pickle.dump(fig, f)
        cif_numbering_enabled = prev_num
        if 'draw_cif_ticks' in globals():
            draw_cif_ticks()
    else:
        target = _confirm_overwrite(args.savefig)
        if target:
            with open(target, "wb") as f:
                pickle.dump(fig, f)
    if target:
        print(f"Saved figure object to {target}")


def main():
    # ---------------- Show and interactive menu ----------------
    if args.interactive:
        # Increase default upper margin (more space): reduce 'top' value once and lock
        try:
            sp = fig.subplotpars
            if sp.top >= 0.88:  # only if near default
                fig.subplots_adjust(top=0.88)
                fig._interactive_top_locked = True
        except Exception:
            pass
        interactive_menu(
            fig, ax, y_data_list, x_data_list, labels_list,
            orig_y, label_text_objects, args.delta, x_label, args,
            x_full_list, raw_y_full_list, offsets_list,
            use_Q, use_r, use_E, use_k, use_rft,
        )
    elif args.out:
        out_file = args.out
        if not os.path.splitext(out_file)[1]:
            out_file += ".svg"
        # Confirm overwrite for export path
        export_target = _confirm_overwrite(out_file)
        if not export_target:
            print("Export canceled.")
        else:
            for i, txt in enumerate(label_text_objects):
                txt.set_text(labels_list[i])
            # Temporarily disable numbering for export
            if cif_tick_series and 'cif_numbering_enabled' in globals() and cif_numbering_enabled:
                prev_num = cif_numbering_enabled
                cif_numbering_enabled = False
                if 'draw_cif_ticks' in globals():
                    draw_cif_ticks()
                fig.savefig(export_target, dpi=300)
                cif_numbering_enabled = prev_num
                if 'draw_cif_ticks' in globals():
                    draw_cif_ticks()
            else:
                fig.savefig(export_target, dpi=300)
            print(f"Saved plot to {export_target}")
    else:
        # Default: show the plot in non-interactive, non-save mode
        plt.show()


# Entry point for CLI
if __name__ == "__main__":
    main()
