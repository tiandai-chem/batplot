"""Interactive menu for Capacity-Per-Cycle (CPC) plots.

Controls focus on style/geometry, and print/export/import of a .bpcfg style.
"""
from __future__ import annotations

from typing import Dict, Optional
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullFormatter
import random as _random

from .ui import (
    resize_plot_frame, resize_canvas,
    update_tick_visibility as _ui_update_tick_visibility,
    position_top_xlabel as _ui_position_top_xlabel,
    position_right_ylabel as _ui_position_right_ylabel,
    position_bottom_xlabel as _ui_position_bottom_xlabel,
    position_left_ylabel as _ui_position_left_ylabel,
)
from .utils import _confirm_overwrite


def _print_menu():
    col1 = [
        " f: font",
        " l: line",
        " m: marker sizes",
        " c: colors",
        "ry: show/hide efficiency",
        " t: toggle axes",
        " h: legend",
        " g: size",
    ]
    col2 = [
        "r: rename titles",
        "x: x range",
        "y: y ranges",
    ]
    col3 = [
        "p: print(export) style",
        "i: import style",
        "e: export figure",
        "s: save project",
        "b: undo",
        "q: quit",
    ]
    w1 = max(18, *(len(s) for s in col1))
    w2 = max(18, *(len(s) for s in col2))
    w3 = max(12, *(len(s) for s in col3))
    rows = max(len(col1), len(col2), len(col3))
    print("\nCPC interactive menu:")
    print(f"  {'(Styles)':<{w1}} {'(Geometries)':<{w2}} {'(Options)':<{w3}}")
    for i in range(rows):
        p1 = col1[i] if i < len(col1) else ""
        p2 = col2[i] if i < len(col2) else ""
        p3 = col3[i] if i < len(col3) else ""
        print(f"  {p1:<{w1}} {p2:<{w2}} {p3:<{w3}}")


def _style_snapshot(fig, ax, ax2, sc_charge, sc_discharge, sc_eff) -> Dict:
    try:
        fig_w, fig_h = map(float, fig.get_size_inches())
    except Exception:
        fig_w = fig_h = None

    def _color_of(artist) -> Optional[str]:
        try:
            if hasattr(artist, 'get_color'):
                c = artist.get_color()
                # scatter returns array sometimes; pick first
                if isinstance(c, (list, tuple)) and c and not isinstance(c, str):
                    return c[0]
                return c
            if hasattr(artist, 'get_facecolors'):
                arr = artist.get_facecolors()
                if arr is not None and len(arr):
                    from matplotlib.colors import to_hex
                    return to_hex(arr[0])
        except Exception:
            pass
        return None

    fam = plt.rcParams.get('font.sans-serif', [''])
    fam0 = fam[0] if fam else ''
    fsize = plt.rcParams.get('font.size', None)
    # Tick widths helper
    def _tick_width(axis, which: str):
        try:
            ticks = axis.get_major_ticks() if which == 'major' else axis.get_minor_ticks()
            for t in ticks:
                ln = t.tick1line
                if ln.get_visible():
                    return ln.get_linewidth()
        except Exception:
            return None
        return None

    # Current tick visibility (best-effort from axes)
    tick_vis = {
        'bx': True,
        'tx': False,
        'ly': True,
        'ry': True,
        'mbx': False,
        'mtx': False,
        'mly': False,
        'mry': False,
    }
    try:
        # Infer from current axes state
        import matplotlib as _mpl
        tick_vis['bx'] = any(lbl.get_visible() for lbl in ax.get_xticklabels())
        tick_vis['tx'] = False  # CPC doesn't duplicate top labels by default
        tick_vis['ly'] = any(lbl.get_visible() for lbl in ax.get_yticklabels())
        tick_vis['ry'] = any(lbl.get_visible() for lbl in ax2.get_yticklabels())
    except Exception:
        pass

    # Plot frame size
    ax_bbox = ax.get_position()
    frame_w_in = ax_bbox.width * fig_w if fig_w else None
    frame_h_in = ax_bbox.height * fig_h if fig_h else None

    # Build WASD-style state (20 parameters: 4 sides × 5 properties)
    # CPC: bottom/top are X-axis, left is primary Y (capacity), right is twin Y (efficiency)
    def _get_spine_visible(ax_obj, which: str) -> bool:
        sp = ax_obj.spines.get(which)
        try:
            return bool(sp.get_visible()) if sp is not None else False
        except Exception:
            return False
    
    wasd_state = {
        'bottom': {
            'spine': _get_spine_visible(ax, 'bottom'),
            'ticks': bool(tick_vis.get('bx', True)),
            'minor': bool(tick_vis.get('mbx', False)),
            'labels': bool(tick_vis.get('bx', True)),  # bottom x labels
            'title': bool(ax.get_xlabel())  # bottom x title
        },
        'top': {
            'spine': _get_spine_visible(ax, 'top'),
            'ticks': bool(tick_vis.get('tx', False)),
            'minor': bool(tick_vis.get('mtx', False)),
            'labels': bool(tick_vis.get('tx', False)),
            'title': bool(getattr(ax, '_top_xlabel_text', None) and getattr(ax._top_xlabel_text, 'get_visible', lambda: False)())
        },
        'left': {
            'spine': _get_spine_visible(ax, 'left'),
            'ticks': bool(tick_vis.get('ly', True)),
            'minor': bool(tick_vis.get('mly', False)),
            'labels': bool(tick_vis.get('ly', True)),  # left y labels (capacity)
            'title': bool(ax.get_ylabel())  # left y title
        },
        'right': {
            'spine': _get_spine_visible(ax2, 'right'),
            'ticks': bool(tick_vis.get('ry', True)),
            'minor': bool(tick_vis.get('mry', False)),
            'labels': bool(tick_vis.get('ry', True)),  # right y labels (efficiency)
            'title': bool(ax2.get_ylabel())  # right y title
        },
    }

    cfg = {
        'kind': 'cpc_style',
        'version': 2,
        'figure': {
            'canvas_size': [fig_w, fig_h],
            'frame_size': [frame_w_in, frame_h_in]
        },
        'font': {'family': fam0, 'size': fsize},
        'ticks': {
            'visibility': tick_vis,
            'widths': {
                'x_major': _tick_width(ax.xaxis, 'major'),
                'x_minor': _tick_width(ax.xaxis, 'minor'),
                'ly_major': _tick_width(ax.yaxis, 'major'),
                'ly_minor': _tick_width(ax.yaxis, 'minor'),
                'ry_major': _tick_width(ax2.yaxis, 'major'),
                'ry_minor': _tick_width(ax2.yaxis, 'minor'),
            }
        },
        'wasd_state': wasd_state,
        'spines': {
            'bottom': {'linewidth': ax.spines.get('bottom').get_linewidth() if ax.spines.get('bottom') else None,
                       'visible': ax.spines.get('bottom').get_visible() if ax.spines.get('bottom') else None},
            'top':    {'linewidth': ax.spines.get('top').get_linewidth() if ax.spines.get('top') else None,
                       'visible': ax.spines.get('top').get_visible() if ax.spines.get('top') else None},
            'left':   {'linewidth': ax.spines.get('left').get_linewidth() if ax.spines.get('left') else None,
                       'visible': ax.spines.get('left').get_visible() if ax.spines.get('left') else None},
            'right':  {'linewidth': ax2.spines.get('right').get_linewidth() if ax2.spines.get('right') else None,
                       'visible': ax2.spines.get('right').get_visible() if ax2.spines.get('right') else None},
        },
        'series': {
            'charge': {
                'color': _color_of(sc_charge),
                'markersize': float(getattr(sc_charge, 'get_sizes', lambda: [32])()[0]) if hasattr(sc_charge, 'get_sizes') else 32.0,
                'alpha': float(sc_charge.get_alpha()) if sc_charge.get_alpha() is not None else 1.0,
            },
            'discharge': {
                'color': _color_of(sc_discharge),
                'markersize': float(getattr(sc_discharge, 'get_sizes', lambda: [32])()[0]) if hasattr(sc_discharge, 'get_sizes') else 32.0,
                'alpha': float(sc_discharge.get_alpha()) if sc_discharge.get_alpha() is not None else 1.0,
            },
            'efficiency': {
                'color': (sc_eff.get_facecolors()[0].tolist() if hasattr(sc_eff, 'get_facecolors') and len(sc_eff.get_facecolors()) else '#2ca02c'),
                'markersize': float(getattr(sc_eff, 'get_sizes', lambda: [40])()[0]) if hasattr(sc_eff, 'get_sizes') else 40.0,
                'alpha': float(sc_eff.get_alpha()) if sc_eff.get_alpha() is not None else 1.0,
                'visible': bool(getattr(sc_eff, 'get_visible', lambda: True)()),
            }
        }
    }
    return cfg


def _apply_style(fig, ax, ax2, sc_charge, sc_discharge, sc_eff, cfg: Dict):
    try:
        font = cfg.get('font', {})
        fam = font.get('family')
        size = font.get('size')
        if fam:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial', 'Helvetica']
        if size is not None:
            plt.rcParams['font.size'] = float(size)
        # Apply to current axes tick labels and duplicate artists, if present
        if fam or size is not None:
            fam0 = fam if fam else None
            sz = float(size) if size is not None else None
            for a in (ax, ax2):
                try:
                    if sz is not None:
                        a.xaxis.label.set_size(sz); a.yaxis.label.set_size(sz)
                    if fam0:
                        a.xaxis.label.set_family(fam0); a.yaxis.label.set_family(fam0)
                except Exception:
                    pass
                try:
                    labels = a.get_xticklabels() + a.get_yticklabels()
                    for t in labels:
                        if sz is not None: t.set_size(sz)
                        if fam0: t.set_family(fam0)
                except Exception:
                    pass
                # Top/right tick labels (label2)
                try:
                    for t in a.xaxis.get_major_ticks():
                        if hasattr(t, 'label2'):
                            if sz is not None: t.label2.set_size(sz)
                            if fam0: t.label2.set_family(fam0)
                    for t in a.yaxis.get_major_ticks():
                        if hasattr(t, 'label2'):
                            if sz is not None: t.label2.set_size(sz)
                            if fam0: t.label2.set_family(fam0)
                except Exception:
                    pass
            try:
                art = getattr(ax, '_top_xlabel_artist', None)
                if art is not None:
                    if sz is not None: art.set_fontsize(sz)
                    if fam0: art.set_fontfamily(fam0)
            except Exception:
                pass
            try:
                art = getattr(ax, '_right_ylabel_artist', None)
                if art is not None:
                    if sz is not None: art.set_fontsize(sz)
                    if fam0: art.set_fontfamily(fam0)
            except Exception:
                pass
    except Exception:
        pass
    # Apply canvas and frame size (from 'g' command: plot frame and canvas)
    try:
        fig_cfg = cfg.get('figure', {})
        canvas_size = fig_cfg.get('canvas_size')
        if canvas_size and isinstance(canvas_size, (list, tuple)) and len(canvas_size) == 2:
            fig.set_size_inches(canvas_size[0], canvas_size[1], forward=True)
        
        # Frame size: calculate position from inches
        frame_size = fig_cfg.get('frame_size')
        if frame_size and isinstance(frame_size, (list, tuple)) and len(frame_size) == 2:
            fw_in, fh_in = frame_size
            canvas_w, canvas_h = fig.get_size_inches()
            if canvas_w > 0 and canvas_h > 0:
                # Keep current left/bottom position, adjust width/height
                current_pos = ax.get_position()
                new_w = fw_in / canvas_w
                new_h = fh_in / canvas_h
                ax.set_position([current_pos.x0, current_pos.y0, new_w, new_h])
    except Exception:
        pass
    try:
        s = cfg.get('series', {})
        ch = s.get('charge', {})
        if ch:
            if ch.get('color') is not None:
                sc_charge.set_color(ch['color'])
            if ch.get('markersize') is not None and hasattr(sc_charge, 'set_sizes'):
                sc_charge.set_sizes([float(ch['markersize'])])
            if ch.get('alpha') is not None:
                sc_charge.set_alpha(float(ch['alpha']))
        dh = s.get('discharge', {})
        if dh:
            if dh.get('color') is not None:
                sc_discharge.set_color(dh['color'])
            if dh.get('markersize') is not None and hasattr(sc_discharge, 'set_sizes'):
                sc_discharge.set_sizes([float(dh['markersize'])])
            if dh.get('alpha') is not None:
                sc_discharge.set_alpha(float(dh['alpha']))
        ef = s.get('efficiency', {})
        if ef:
            if ef.get('color') is not None:
                try:
                    sc_eff.set_color(ef['color'])
                except Exception:
                    pass
            if ef.get('markersize') is not None and hasattr(sc_eff, 'set_sizes'):
                sc_eff.set_sizes([float(ef['markersize'])])
            if ef.get('alpha') is not None:
                sc_eff.set_alpha(float(ef['alpha']))
            if 'visible' in ef:
                try:
                    sc_eff.set_visible(bool(ef['visible']))
                    # Keep right-axis title visibility consistent with efficiency visibility
                    try:
                        ax2.yaxis.label.set_visible(bool(ef['visible']))
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        pass
    # Apply tick visibility/widths and spines
    try:
        tk = cfg.get('ticks', {})
        # Try wasd_state first (version 2), fall back to visibility dict (version 1)
        wasd = cfg.get('wasd_state', {})
        if wasd:
            # Use WASD state (20 parameters)
            bx = bool(wasd.get('bottom', {}).get('labels', True))
            tx = bool(wasd.get('top', {}).get('labels', False))
            ly = bool(wasd.get('left', {}).get('labels', True))
            ry = bool(wasd.get('right', {}).get('labels', True))
            mbx = bool(wasd.get('bottom', {}).get('minor', False))
            mtx = bool(wasd.get('top', {}).get('minor', False))
            mly = bool(wasd.get('left', {}).get('minor', False))
            mry = bool(wasd.get('right', {}).get('minor', False))
        else:
            # Fall back to old visibility dict
            vis = tk.get('visibility', {})
            bx = bool(vis.get('bx', True))
            tx = bool(vis.get('tx', False))
            ly = bool(vis.get('ly', True))
            ry = bool(vis.get('ry', True))
            mbx = bool(vis.get('mbx', False))
            mtx = bool(vis.get('mtx', False))
            mly = bool(vis.get('mly', False))
            mry = bool(vis.get('mry', False))
        
        if True:  # Always apply
            ax.tick_params(axis='x', bottom=bx, labelbottom=bx, top=tx, labeltop=tx)
            ax.tick_params(axis='y', left=ly, labelleft=ly)
            ax2.tick_params(axis='y', right=ry, labelright=ry)
            # Minor ticks
            from matplotlib.ticker import AutoMinorLocator, NullFormatter
            if mbx or mtx:
                ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis='x', which='minor', bottom=mbx, top=mtx, labelbottom=False, labeltop=False)
            else:
                ax.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False, labeltop=False)
            if mly:
                ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis='y', which='minor', left=True, labelleft=False)
            else:
                ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
            if mry:
                ax2.yaxis.set_minor_locator(AutoMinorLocator()); ax2.yaxis.set_minor_formatter(NullFormatter())
                ax2.tick_params(axis='y', which='minor', right=True, labelright=False)
            else:
                ax2.tick_params(axis='y', which='minor', right=False, labelright=False)
        
        # Widths: support both version 2 (nested in 'widths') and version 1 (direct keys)
        widths = tk.get('widths', tk)  # Try nested first, fall back to tk itself
        if widths.get('x_major') is not None:
            ax.tick_params(axis='x', which='major', width=widths['x_major'])
        if widths.get('x_minor') is not None:
            ax.tick_params(axis='x', which='minor', width=widths['x_minor'])
        if widths.get('ly_major') is not None:
            ax.tick_params(axis='y', which='major', width=widths['ly_major'])
        if widths.get('ly_minor') is not None:
            ax.tick_params(axis='y', which='minor', width=widths['ly_minor'])
        if widths.get('ry_major') is not None:
            ax2.tick_params(axis='y', which='major', width=widths['ry_major'])
        if widths.get('ry_minor') is not None:
            ax2.tick_params(axis='y', which='minor', width=widths['ry_minor'])
    except Exception:
        pass
    try:
        sp = cfg.get('spines', {})
        for name, spec in sp.items():
            if name in ('bottom','top','left') and name in ax.spines:
                spn = ax.spines.get(name)
                if spn is None: continue
                if spec.get('linewidth') is not None:
                    try: spn.set_linewidth(float(spec['linewidth']))
                    except Exception: pass
                if spec.get('visible') is not None:
                    try: spn.set_visible(bool(spec['visible']))
                    except Exception: pass
            if name == 'right' and ax2.spines.get('right') is not None:
                spn = ax2.spines.get('right')
                if spec.get('linewidth') is not None:
                    try: spn.set_linewidth(float(spec['linewidth']))
                    except Exception: pass
                if spec.get('visible') is not None:
                    try: spn.set_visible(bool(spec['visible']))
                    except Exception: pass
    except Exception:
        pass
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def cpc_interactive_menu(fig, ax, ax2, sc_charge, sc_discharge, sc_eff):
    # Tick state for CPC (primary ax + twin right ax2)
    tick_state = {
        'bx': True,  # bottom x
        'tx': False, # top x
        'ly': True,  # left y (primary)
        'ry': True,  # right y (twin)
        'mbx': False,
        'mtx': False,
        'mly': False,
        'mry': False,
    }

    # --- Undo stack using style snapshots ---
    state_history = []  # list of cfg dicts

    def push_state(note: str = ""):
        try:
            snap = _style_snapshot(fig, ax, ax2, sc_charge, sc_discharge, sc_eff)
            snap['__note__'] = note
            # Also persist current tick_state explicitly
            snap.setdefault('ticks', {}).setdefault('visibility', dict(tick_state))
            state_history.append(snap)
            if len(state_history) > 40:
                state_history.pop(0)
        except Exception:
            pass

    def restore_state():
        if not state_history:
            print("No undo history.")
            return
        cfg = state_history.pop()
        try:
            _apply_style(fig, ax, ax2, sc_charge, sc_discharge, sc_eff, cfg)
            # Restore local tick_state from cfg
            vis = (cfg.get('ticks') or {}).get('visibility') or {}
            for k, v in vis.items():
                if k in tick_state:
                    tick_state[k] = bool(v)
            _update_ticks()
            try:
                fig.canvas.draw()
            except Exception:
                fig.canvas.draw_idle()
            print("Undo: restored previous state.")
        except Exception as e:
            print(f"Undo failed: {e}")

    def _update_ticks():
        try:
            # Apply shared visibility to primary ax; then adjust twin for right side
            _ui_update_tick_visibility(ax, tick_state)
            # Right axis tick params follow r_* keys
            ax2.tick_params(axis='y',
                            right=tick_state.get('r_ticks', tick_state.get('ry', False)),
                            labelright=tick_state.get('r_labels', tick_state.get('ry', False)))
            # Minor right-y consistency
            if tick_state.get('mry'):
                ax2.yaxis.set_minor_locator(AutoMinorLocator()); ax2.yaxis.set_minor_formatter(NullFormatter())
                ax2.tick_params(axis='y', which='minor', right=True, labelright=False)
            else:
                ax2.tick_params(axis='y', which='minor', right=False, labelright=False)
            # Position label spacings (bottom/left) for consistency
            _ui_position_bottom_xlabel(ax, fig, tick_state)
            _ui_position_left_ylabel(ax, fig, tick_state)
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _toggle_spine(code: str):
        # Map bl/tl/ll to ax; rl to ax2
        try:
            if code == 'bl':
                sp = ax.spines.get('bottom'); sp.set_visible(not sp.get_visible())
            elif code == 'tl':
                sp = ax.spines.get('top'); sp.set_visible(not sp.get_visible())
            elif code == 'll':
                sp = ax.spines.get('left'); sp.set_visible(not sp.get_visible())
            elif code == 'rl':
                sp = ax2.spines.get('right'); sp.set_visible(not sp.get_visible())
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _apply_legend_position():
        """Reapply legend position using stored inches offset relative to canvas center."""
        try:
            xy_in = getattr(fig, '_cpc_legend_xy_in', None)
            leg = ax.get_legend()
            if xy_in is None or leg is None:
                return
            # Compute figure-fraction anchor from inches
            fw, fh = fig.get_size_inches()
            if fw <= 0 or fh <= 0:
                return
            fx = 0.5 + float(xy_in[0]) / float(fw)
            fy = 0.5 + float(xy_in[1]) / float(fh)
            # Use current handles/labels
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            if h1 or h2:
                ax.legend(h1 + h2, l1 + l2, loc='center', bbox_to_anchor=(fx, fy), bbox_transform=fig.transFigure, borderaxespad=1.0)
        except Exception:
            pass

    # Ensure resize re-applies legend position in inches
    try:
        if not hasattr(fig, '_cpc_legpos_cid') or getattr(fig, '_cpc_legpos_cid') is None:
            def _on_resize(event):
                _apply_legend_position()
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
            fig._cpc_legpos_cid = fig.canvas.mpl_connect('resize_event', _on_resize)
    except Exception:
        pass

    _print_menu()
    while True:
        key = input("Press a key: ").strip().lower()
        if not key:
            continue
        if key == 'q':
            try:
                confirm = input("Quit CPC interactive? Remember to save! Quit now? (y/n): ").strip().lower()
            except Exception:
                confirm = 'y'
            if confirm == 'y':
                break
            else:
                _print_menu(); continue
        elif key == 'b':
            restore_state()
            _print_menu(); continue
        elif key == 'c':
            # Colors submenu: ly (left Y series) and ry (right Y efficiency)
            try:
                while True:
                    print("Colors: ly=left Y (capacity dots), ry=right Y (efficiency triangles), q=back")
                    sub = input("Colors> ").strip().lower()
                    if not sub:
                        continue
                    if sub == 'q':
                        break
                    if sub == 'ly':
                        push_state("colors-ly")
                        spec = input("Enter colors: 'c:<color> d:<color>' or 'r' for random (q=cancel): ").strip()
                        if not spec or spec.lower() == 'q':
                            continue
                        if spec.strip().lower() == 'r':
                            palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                            try:
                                sc_charge.set_color(_random.choice(palette))
                                sc_discharge.set_color(_random.choice(palette))
                            except Exception:
                                pass
                        else:
                            parts = spec.split()
                            for p in parts:
                                if ':' not in p:
                                    continue
                                role, col = p.split(':', 1)
                                role = role.strip().lower(); col = col.strip()
                                try:
                                    if role == 'c':
                                        sc_charge.set_color(col)
                                    elif role == 'd':
                                        sc_discharge.set_color(col)
                                except Exception:
                                    print(f"Bad color token: {p}")
                        try:
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    elif sub == 'ry':
                        push_state("colors-ry")
                        val = input("Enter color for efficiency triangles (hex/name) or 'r' random (q=cancel): ").strip()
                        if not val or val.lower() == 'q':
                            continue
                        if val.lower() == 'r':
                            palette = ['#2ca02c','#1f77b4','#ff7f0e','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                            col = _random.choice(palette)
                        else:
                            col = val
                        try:
                            sc_eff.set_color(col)
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    else:
                        print("Unknown option.")
            except Exception as e:
                print(f"Error in colors menu: {e}")
            _print_menu(); continue
        elif key == 'e':
            try:
                fname = input("Export filename (default .svg if no extension, q=cancel): ").strip()
                if not fname or fname.lower() == 'q':
                    _print_menu(); continue
                root, ext = os.path.splitext(fname)
                if ext == '':
                    fname = fname + '.svg'
                target = _confirm_overwrite(fname)
                if target:
                    fig.savefig(target, bbox_inches='tight')
                    print(f"Exported figure to {target}")
            except Exception as e:
                print(f"Export failed: {e}")
            _print_menu(); continue
        elif key == 's':
            # Save CPC session (.pkl) with all data and styles
            try:
                from .session import dump_cpc_session
                folder = os.getcwd()
                try:
                    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.pkl')])
                except Exception:
                    files = []
                if files:
                    print("Existing .pkl files:")
                    for i, f in enumerate(files, 1):
                        print(f"  {i}: {f}")
                choice = input("Enter new filename or number to overwrite (q=cancel): ").strip()
                if not choice or choice.lower() == 'q':
                    _print_menu(); continue
                if choice.isdigit() and files:
                    idx = int(choice)
                    if 1 <= idx <= len(files):
                        name = files[idx-1]
                        yn = input(f"Overwrite '{name}'? (y/n): ").strip().lower()
                        if yn != 'y':
                            _print_menu(); continue
                        target = os.path.join(folder, name)
                    else:
                        print("Invalid number.")
                        _print_menu(); continue
                else:
                    name = choice
                    root, ext = os.path.splitext(name)
                    if ext == '':
                        name = name + '.pkl'
                    target = name if os.path.isabs(name) else os.path.join(folder, name)
                    if os.path.exists(target):
                        yn = input(f"'{os.path.basename(target)}' exists. Overwrite? (y/n): ").strip().lower()
                        if yn != 'y':
                            _print_menu(); continue
                dump_cpc_session(target, fig=fig, ax=ax, ax2=ax2, sc_charge=sc_charge, sc_discharge=sc_discharge, sc_eff=sc_eff, skip_confirm=True)
            except Exception as e:
                print(f"Save failed: {e}")
            _print_menu(); continue
        elif key == 'p':
            try:
                snap = _style_snapshot(fig, ax, ax2, sc_charge, sc_discharge, sc_eff)
                print("\n--- CPC Style (Styles column only) ---")
                
                # Figure size (g command)
                fig_cfg = snap.get('figure', {})
                canvas = fig_cfg.get('canvas_size')
                frame = fig_cfg.get('frame_size')
                if canvas and all(v is not None for v in canvas):
                    print(f"Canvas size (inches): {canvas[0]:.3f} x {canvas[1]:.3f}")
                if frame and all(v is not None for v in frame):
                    print(f"Plot frame size (inches): {frame[0]:.3f} x {frame[1]:.3f}")
                
                # Font (f command)
                ft = snap.get('font', {})
                print(f"Font: family='{ft.get('family', '')}', size={ft.get('size', '')}")
                
                # Line widths (l command)
                spines = snap.get('spines', {})
                if spines:
                    print("Spines:")
                    for name in ('bottom', 'top', 'left', 'right'):
                        props = spines.get(name, {})
                        lw = props.get('linewidth', '?')
                        vis = props.get('visible', False)
                        print(f"  {name:<6} lw={lw} visible={vis}")
                
                ticks = snap.get('ticks', {})
                print(f"Tick widths: x_major={ticks.get('x_major_width')}, x_minor={ticks.get('x_minor_width')}")
                print(f"             ly_major={ticks.get('ly_major_width')}, ly_minor={ticks.get('ly_minor_width')}")
                print(f"             ry_major={ticks.get('ry_major_width')}, ry_minor={ticks.get('ry_minor_width')}")
                
                # Marker sizes (m command) and Colors (c command)
                s = snap.get('series', {})
                ch = s.get('charge', {}); dh = s.get('discharge', {}); ef = s.get('efficiency', {})
                print(f"Charge: color={ch.get('color')}, markersize={ch.get('markersize')}, alpha={ch.get('alpha')}")
                print(f"Discharge: color={dh.get('color')}, markersize={dh.get('markersize')}, alpha={dh.get('alpha')}")
                print(f"Efficiency: color={ef.get('color')}, markersize={ef.get('markersize')}, alpha={ef.get('alpha')}, visible={ef.get('visible')}")
                
                # Toggle axes (t command) - Per-side matrix (20 parameters)
                def _onoff(v):
                    return 'ON ' if bool(v) else 'off'
                
                wasd = snap.get('wasd_state', {})
                if wasd:
                    print("Per-side: spine, major, minor, labels, title")
                    for side in ('bottom', 'top', 'left', 'right'):
                        s = wasd.get(side, {})
                        spine_val = _onoff(s.get('spine', False))
                        major_val = _onoff(s.get('ticks', False))
                        minor_val = _onoff(s.get('minor', False))
                        labels_val = _onoff(s.get('labels', False))
                        title_val = _onoff(s.get('title', False))
                        print(f"  {side:<6}: spine={spine_val} major={major_val} minor={minor_val} labels={labels_val} title={title_val}")
                
                print("--- End Style ---\n")
                # List existing .bpcfg and allow numeric overwrite on export
                try:
                    files = sorted([f for f in os.listdir(os.getcwd()) if f.lower().endswith('.bpcfg')])
                except Exception:
                    files = []
                if files:
                    print("Existing .bpcfg files:")
                    for i, f in enumerate(files, 1):
                        print(f"  {i}: {f}")
                sub = input("Style submenu: (e=export, q=return): ").strip().lower()
                if sub == 'e':
                    choice = input("Enter new filename or number to overwrite (q=cancel): ").strip()
                    if not choice or choice.lower() == 'q':
                        _print_menu(); continue
                    target = None
                    if choice.isdigit() and files:
                        idx = int(choice)
                        if 1 <= idx <= len(files):
                            name = files[idx-1]
                            yn = input(f"Overwrite '{name}'? (y/n): ").strip().lower()
                            if yn == 'y':
                                target = os.path.join(os.getcwd(), name)
                        else:
                            print("Invalid number."); _print_menu(); continue
                    else:
                        name = choice
                        root, ext = os.path.splitext(name)
                        if ext == '':
                            name = name + '.bpcfg'
                        target = name if os.path.isabs(name) else os.path.join(os.getcwd(), name)
                        if os.path.exists(target):
                            yn = input(f"'{os.path.basename(target)}' exists. Overwrite? (y/n): ").strip().lower()
                            if yn != 'y':
                                target = None
                    if target:
                        with open(target, 'w', encoding='utf-8') as f:
                            json.dump(snap, f, indent=2)
                        print(f"Exported CPC style to {target}")
            except Exception as e:
                print(f"Error printing/exporting style: {e}")
            _print_menu(); continue
        elif key == 'i':
            try:
                push_state("import-style")
                try:
                    files = sorted([f for f in os.listdir(os.getcwd()) if f.lower().endswith('.bpcfg')])
                except Exception:
                    files = []
                if files:
                    print("Available .bpcfg files:")
                    for i, f in enumerate(files, 1):
                        print(f"  {i}: {f}")
                inp = input("Enter number to open or filename (.bpcfg; q=cancel): ").strip()
                if not inp or inp.lower() == 'q':
                    _print_menu(); continue
                if inp.isdigit() and files:
                    idx = int(inp)
                    if 1 <= idx <= len(files):
                        path = os.path.join(os.getcwd(), files[idx-1])
                    else:
                        print("Invalid number."); _print_menu(); continue
                else:
                    path = inp
                    if not os.path.isfile(path):
                        root, ext = os.path.splitext(path)
                        if ext == '':
                            alt = path + '.bpcfg'
                            if os.path.isfile(alt):
                                path = alt
                            else:
                                print("File not found."); _print_menu(); continue
                        else:
                            print("File not found."); _print_menu(); continue
                with open(path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict) or cfg.get('kind') != 'cpc_style':
                    print("Not a CPC style file."); _print_menu(); continue
                _apply_style(fig, ax, ax2, sc_charge, sc_discharge, sc_eff, cfg)
            except Exception as e:
                print(f"Error importing style: {e}")
            _print_menu(); continue
        elif key == 'ry':
            # Toggle efficiency visibility on the right axis
            try:
                push_state("toggle-eff")
                vis = bool(sc_eff.get_visible()) if hasattr(sc_eff, 'get_visible') else True
                new_vis = not vis
                
                # 1. Hide/show efficiency points
                sc_eff.set_visible(new_vis)
                
                # 2. Hide/show right y-axis title
                try:
                    ax2.yaxis.label.set_visible(new_vis)
                except Exception:
                    pass
                
                # 3. Hide/show right y-axis ticks and labels (only affect ax2, don't touch ax)
                try:
                    ax2.tick_params(axis='y', right=new_vis, labelright=new_vis)
                    # Update tick_state
                    tick_state['ry'] = bool(new_vis)
                except Exception:
                    pass
                
                # 4. Rebuild legend to remove/add efficiency entry
                try:
                    h1, l1 = ax.get_legend_handles_labels()
                except Exception:
                    h1, l1 = [], []
                try:
                    h2, l2 = ax2.get_legend_handles_labels()
                except Exception:
                    h2, l2 = [], []
                
                # Filter out efficiency entry if hidden
                eff_label = None
                try:
                    eff_label = sc_eff.get_label()
                except Exception:
                    pass
                
                pairs1 = list(zip(h1, l1))
                pairs2 = list(zip(h2, l2))
                
                def _keep(pair):
                    h, l = pair
                    # Drop invisible handles
                    try:
                        if hasattr(h, 'get_visible') and not h.get_visible():
                            return False
                    except Exception:
                        pass
                    # Drop the efficiency label when hidden
                    if eff_label and (l == eff_label) and not new_vis:
                        return False
                    return True
                
                vis_pairs1 = [p for p in pairs1 if _keep(p)]
                vis_pairs2 = [p for p in pairs2 if _keep(p)]
                H = [h for h, _ in vis_pairs1 + vis_pairs2]
                L = [l for _, l in vis_pairs1 + vis_pairs2]
                
                if H:
                    try:
                        # Honor stored inch-based anchor if present; else fallback to 'best'
                        xy_in = getattr(fig, '_cpc_legend_xy_in', None)
                        if xy_in is not None:
                            try:
                                fw, fh = fig.get_size_inches()
                                fx = 0.5 + float(xy_in[0]) / float(fw)
                                fy = 0.5 + float(xy_in[1]) / float(fh)
                                ax.legend(H, L, loc='center', bbox_to_anchor=(fx, fy), bbox_transform=fig.transFigure, borderaxespad=1.0)
                            except Exception:
                                ax.legend(H, L, loc='best', borderaxespad=1.0)
                        else:
                            ax.legend(H, L, loc='best', borderaxespad=1.0)
                    except Exception:
                        pass
                else:
                    # No visible series: hide legend if present
                    try:
                        leg = ax.get_legend()
                        if leg is not None:
                            leg.set_visible(False)
                    except Exception:
                        pass
                
                fig.canvas.draw_idle()
            except Exception:
                pass
            _print_menu(); continue
        elif key == 'h':
            # Legend submenu: toggle visibility, set position in inches relative to canvas center (0,0)
            try:
                # If no stored inches yet, try computing from the current legend bbox
                try:
                    if not hasattr(fig, '_cpc_legend_xy_in') or getattr(fig, '_cpc_legend_xy_in') is None:
                        leg0 = ax.get_legend()
                        if leg0 is not None:
                            try:
                                # Ensure renderer exists
                                try:
                                    renderer = fig.canvas.get_renderer()
                                except Exception:
                                    fig.canvas.draw()
                                    renderer = fig.canvas.get_renderer()
                                bb = leg0.get_window_extent(renderer=renderer)
                                cx = 0.5 * (bb.x0 + bb.x1)
                                cy = 0.5 * (bb.y0 + bb.y1)
                                # Convert display -> figure fraction
                                fx, fy = fig.transFigure.inverted().transform((cx, cy))
                                fw, fh = fig.get_size_inches()
                                fig._cpc_legend_xy_in = ((fx - 0.5) * fw, (fy - 0.5) * fh)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Show current status and position
                leg = ax.get_legend()
                vis = bool(leg.get_visible()) if leg is not None else False
                fw, fh = fig.get_size_inches()
                xy_in = getattr(fig, '_cpc_legend_xy_in', (0.0, 0.0))
                print(f"Legend is {'ON' if vis else 'off'}; position (inches from center): x={xy_in[0]:.2f}, y={xy_in[1]:.2f}")
                while True:
                    sub = input("Legend: t=toggle, m=set position (x y inches), q=back: ").strip().lower()
                    if not sub:
                        continue
                    if sub == 'q':
                        break
                    if sub == 't':
                        try:
                            push_state("legend-toggle")
                            leg = ax.get_legend()
                            if leg is not None and leg.get_visible():
                                leg.set_visible(False)
                            else:
                                # Ensure a legend exists at the stored position
                                h1, l1 = ax.get_legend_handles_labels()
                                h2, l2 = ax2.get_legend_handles_labels()
                                if h1 or h2:
                                    if hasattr(fig, '_cpc_legend_xy_in') and getattr(fig, '_cpc_legend_xy_in') is not None:
                                        _apply_legend_position()
                                    else:
                                        ax.legend(h1 + h2, l1 + l2, loc='best', borderaxespad=1.0)
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    elif sub == 'm':
                        push_state("legend-move")
                        xy_in = getattr(fig, '_cpc_legend_xy_in', (0.0, 0.0))
                        print(f"Current position: x={xy_in[0]:.2f}, y={xy_in[1]:.2f}")
                        vals = input("Enter legend position x y (inches from center; e.g., 0.0 0.0): ").strip()
                        parts = vals.replace(',', ' ').split()
                        if len(parts) != 2:
                            print("Need two numbers."); continue
                        try:
                            x_in = float(parts[0]); y_in = float(parts[1])
                        except Exception:
                            print("Invalid numbers."); continue
                        # Store and apply
                        try:
                            fig._cpc_legend_xy_in = (x_in, y_in)
                            _apply_legend_position()
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    else:
                        print("Unknown option.")
            except Exception:
                pass
            _print_menu(); continue
        elif key == 'f':
            sub = input("Font: f=family, s=size, q=back: ").strip().lower()
            if sub == 'q' or not sub:
                _print_menu(); continue
            if sub == 'f':
                fam = input("Enter font family (e.g., Arial, DejaVu Sans): ").strip()
                if fam:
                    try:
                        push_state("font-family")
                        plt.rcParams['font.family'] = 'sans-serif'
                        plt.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial', 'Helvetica']
                        # Apply to labels, ticks, and duplicate artists immediately
                        for a in (ax, ax2):
                            try:
                                a.xaxis.label.set_family(fam); a.yaxis.label.set_family(fam)
                            except Exception:
                                pass
                            try:
                                for t in a.get_xticklabels() + a.get_yticklabels():
                                    t.set_family(fam)
                            except Exception:
                                pass
                            # Update top and right tick labels (label2)
                            try:
                                for tick in a.xaxis.get_major_ticks():
                                    if hasattr(tick, 'label2'):
                                        tick.label2.set_family(fam)
                                for tick in a.yaxis.get_major_ticks():
                                    if hasattr(tick, 'label2'):
                                        tick.label2.set_family(fam)
                            except Exception:
                                pass
                        try:
                            art = getattr(ax, '_top_xlabel_artist', None)
                            if art is not None:
                                art.set_fontfamily(fam)
                            # Also update the new text artist
                            txt = getattr(ax, '_top_xlabel_text', None)
                            if txt is not None:
                                txt.set_fontfamily(fam)
                        except Exception:
                            pass
                        try:
                            # Right ylabel artist is on ax2, not ax
                            art = getattr(ax2, '_right_ylabel_artist', None)
                            if art is not None:
                                art.set_fontfamily(fam)
                        except Exception:
                            pass
                        # Update legend font
                        try:
                            leg = ax.get_legend()
                            if leg is not None:
                                for txt in leg.get_texts():
                                    txt.set_fontfamily(fam)
                        except Exception:
                            pass
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
            elif sub == 's':
                val = input("Enter font size (number): ").strip()
                try:
                    size = float(val)
                    push_state("font-size")
                    plt.rcParams['font.size'] = size
                    # Apply to labels, ticks, and duplicate artists immediately
                    for a in (ax, ax2):
                        try:
                            a.xaxis.label.set_size(size)
                            a.yaxis.label.set_size(size)
                        except Exception:
                            pass
                        try:
                            for t in a.get_xticklabels() + a.get_yticklabels():
                                t.set_size(size)
                        except Exception:
                            pass
                        # Update top and right tick labels (label2)
                        try:
                            for tick in a.xaxis.get_major_ticks():
                                if hasattr(tick, 'label2'):
                                    tick.label2.set_size(size)
                            for tick in a.yaxis.get_major_ticks():
                                if hasattr(tick, 'label2'):
                                    tick.label2.set_size(size)
                        except Exception:
                            pass
                    try:
                        art = getattr(ax, '_top_xlabel_artist', None)
                        if art is not None:
                            art.set_fontsize(size)
                        # Also update the new text artist
                        txt = getattr(ax, '_top_xlabel_text', None)
                        if txt is not None:
                            txt.set_fontsize(size)
                    except Exception:
                        pass
                    try:
                        # Right ylabel artist is on ax2, not ax
                        art = getattr(ax2, '_right_ylabel_artist', None)
                        if art is not None:
                            art.set_fontsize(size)
                    except Exception:
                        pass
                    # Update legend font size
                    try:
                        leg = ax.get_legend()
                        if leg is not None:
                            for txt in leg.get_texts():
                                txt.set_fontsize(size)
                    except Exception:
                        pass
                    fig.canvas.draw_idle()
                except Exception:
                    print("Invalid size.")
            _print_menu(); continue
        elif key == 'l':
            # Line widths: frame and ticks
            try:
                fw_in = input("Enter frame/tick width (e.g., 1.5) or 'm M' (major minor) or q: ").strip()
                if not fw_in or fw_in.lower() == 'q':
                    print("Canceled.")
                    _print_menu(); continue
                parts = fw_in.split()
                push_state("framewidth")
                if len(parts) == 1:
                    frame_w = float(parts[0])
                    tick_major = frame_w
                    tick_minor = frame_w * 0.6
                else:
                    frame_w = float(parts[0])
                    tick_major = float(parts[1])
                    tick_minor = float(tick_major) * 0.7
                # Set frame width for all spines (ax and ax2)
                for sp in ax.spines.values():
                    sp.set_linewidth(frame_w)
                for sp in ax2.spines.values():
                    sp.set_linewidth(frame_w)
                # Set tick widths for both axes
                ax.tick_params(which='major', width=tick_major)
                ax.tick_params(which='minor', width=tick_minor)
                ax2.tick_params(which='major', width=tick_major)
                ax2.tick_params(which='minor', width=tick_minor)
                fig.canvas.draw()
                print(f"Set frame width={frame_w}, major tick width={tick_major}, minor tick width={tick_minor}")
            except ValueError:
                print("Invalid numeric value(s).")
            except Exception as e:
                print(f"Error setting line widths: {e}")
            _print_menu(); continue
        elif key == 'm':
            try:
                print("Current marker sizes:")
                try:
                    c_ms = getattr(sc_charge, 'get_sizes', lambda: [32])()[0]
                except Exception:
                    c_ms = 32
                try:
                    d_ms = getattr(sc_discharge, 'get_sizes', lambda: [32])()[0]
                except Exception:
                    d_ms = 32
                try:
                    e_ms = getattr(sc_eff, 'get_sizes', lambda: [40])()[0]
                except Exception:
                    e_ms = 40
                print(f"  charge ms={c_ms}, discharge ms={d_ms}, efficiency ms={e_ms}")
                spec = input("Set marker size: 'c <ms>', 'd <ms>', 'e <ms>' (q=cancel): ").strip().lower()
                if not spec or spec == 'q':
                    _print_menu(); continue
                parts = spec.split()
                if len(parts) != 2:
                    print("Need two tokens."); _print_menu(); continue
                role, val = parts[0], parts[1]
                try:
                    num = float(val)
                    push_state("marker-size")
                    if role == 'c' and hasattr(sc_charge, 'set_sizes'):
                        sc_charge.set_sizes([num])
                    elif role == 'd' and hasattr(sc_discharge, 'set_sizes'):
                        sc_discharge.set_sizes([num])
                    elif role == 'e' and hasattr(sc_eff, 'set_sizes'):
                        sc_eff.set_sizes([num])
                    fig.canvas.draw_idle()
                except Exception:
                    print("Invalid value.")
            except Exception as e:
                print(f"Error: {e}")
            _print_menu(); continue
        elif key == 't':
            # Unified WASD toggles for spines/ticks/minor/labels/title per side
            try:
                # Local WASD state stored on figure to persist across openings
                wasd = getattr(fig, '_cpc_wasd_state', None)
                if not isinstance(wasd, dict):
                    wasd = {
                        'top':    {'spine': bool(ax.spines.get('top').get_visible()) if ax.spines.get('top') else False,
                                   'ticks': bool(tick_state.get('t_ticks', tick_state.get('tx', False))),
                                   'minor': bool(tick_state.get('mtx', False)),
                                   'labels': bool(tick_state.get('t_labels', tick_state.get('tx', False))),
                                   'title': bool(getattr(ax, '_top_xlabel_on', False))},
                        'bottom': {'spine': bool(ax.spines.get('bottom').get_visible()) if ax.spines.get('bottom') else True,
                                   'ticks': bool(tick_state.get('b_ticks', tick_state.get('bx', True))),
                                   'minor': bool(tick_state.get('mbx', False)),
                                   'labels': bool(tick_state.get('b_labels', tick_state.get('bx', True))),
                                   'title': bool(ax.get_xlabel())},
                        'left':   {'spine': bool(ax.spines.get('left').get_visible()) if ax.spines.get('left') else True,
                                   'ticks': bool(tick_state.get('l_ticks', tick_state.get('ly', True))),
                                   'minor': bool(tick_state.get('mly', False)),
                                   'labels': bool(tick_state.get('l_labels', tick_state.get('ly', True))),
                                   'title': bool(ax.get_ylabel())},
                        'right':  {'spine': bool(ax2.spines.get('right').get_visible()) if ax2.spines.get('right') else True,
                                   'ticks': bool(tick_state.get('r_ticks', tick_state.get('ry', True))),
                                   'minor': bool(tick_state.get('mry', False)),
                                   'labels': bool(tick_state.get('r_labels', tick_state.get('ry', True))),
                                   'title': bool(ax2.yaxis.get_label().get_text()) and bool(sc_eff.get_visible())},
                    }
                    setattr(fig, '_cpc_wasd_state', wasd)

                def _apply_wasd():
                    # Apply spines
                    # Note: top and bottom spines are shared between ax and ax2
                    try:
                        ax.spines['top'].set_visible(bool(wasd['top']['spine']))
                        ax.spines['bottom'].set_visible(bool(wasd['bottom']['spine']))
                        ax.spines['left'].set_visible(bool(wasd['left']['spine']))
                        # Also control top/bottom on ax2 since they're shared
                        ax2.spines['top'].set_visible(bool(wasd['top']['spine']))
                        ax2.spines['bottom'].set_visible(bool(wasd['bottom']['spine']))
                    except Exception:
                        pass
                    try:
                        ax2.spines['right'].set_visible(bool(wasd['right']['spine']))
                    except Exception:
                        pass
                    # Major ticks and tick labels
                    try:
                        ax.tick_params(axis='x', top=bool(wasd['top']['ticks']), bottom=bool(wasd['bottom']['ticks']),
                                       labeltop=bool(wasd['top']['labels']), labelbottom=bool(wasd['bottom']['labels']))
                        ax.tick_params(axis='y', left=bool(wasd['left']['ticks']), labelleft=bool(wasd['left']['labels']))
                        ax2.tick_params(axis='y', right=bool(wasd['right']['ticks']), labelright=bool(wasd['right']['labels']))
                    except Exception:
                        pass
                    # Minor ticks
                    try:
                        if wasd['top']['minor'] or wasd['bottom']['minor']:
                            ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
                        ax.tick_params(axis='x', which='minor', top=bool(wasd['top']['minor']), bottom=bool(wasd['bottom']['minor']),
                                       labeltop=False, labelbottom=False)
                    except Exception:
                        pass
                    try:
                        if wasd['left']['minor']:
                            ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
                        ax.tick_params(axis='y', which='minor', left=bool(wasd['left']['minor']), labelleft=False)
                    except Exception:
                        pass
                    try:
                        if wasd['right']['minor']:
                            ax2.yaxis.set_minor_locator(AutoMinorLocator()); ax2.yaxis.set_minor_formatter(NullFormatter())
                        ax2.tick_params(axis='y', which='minor', right=bool(wasd['right']['minor']), labelright=False)
                    except Exception:
                        pass
                    # Titles
                    try:
                        # Bottom X title
                        if bool(wasd['bottom']['title']):
                            # Restore stored xlabel if present
                            if hasattr(ax, '_stored_xlabel') and isinstance(ax._stored_xlabel, str) and ax._stored_xlabel:
                                ax.set_xlabel(ax._stored_xlabel)
                        else:
                            # Store once
                            if not hasattr(ax, '_stored_xlabel'):
                                try:
                                    ax._stored_xlabel = ax.get_xlabel()
                                except Exception:
                                    ax._stored_xlabel = ''
                            ax.set_xlabel("")
                    except Exception:
                        pass
                    try:
                        # Top X title - create a text artist positioned at the top
                        # First ensure we have the original xlabel text stored
                        if not hasattr(ax, '_stored_top_xlabel') or not ax._stored_top_xlabel:
                            # Try to get from current xlabel first
                            current_xlabel = ax.get_xlabel()
                            if current_xlabel:
                                ax._stored_top_xlabel = current_xlabel
                            # If still empty, try from stored bottom xlabel
                            elif hasattr(ax, '_stored_xlabel') and ax._stored_xlabel:
                                ax._stored_top_xlabel = ax._stored_xlabel
                            else:
                                ax._stored_top_xlabel = ''
                        
                        if bool(wasd['top']['title']) and ax._stored_top_xlabel:
                            # Get or create the top xlabel artist
                            if not hasattr(ax, '_top_xlabel_text') or ax._top_xlabel_text is None:
                                # Create a new text artist at the top center
                                ax._top_xlabel_text = ax.text(0.5, 1.0, '', transform=ax.transAxes,
                                                              ha='center', va='bottom',
                                                              fontsize=ax.xaxis.label.get_fontsize(),
                                                              fontfamily=ax.xaxis.label.get_fontfamily())
                            # Update text and make visible
                            ax._top_xlabel_text.set_text(ax._stored_top_xlabel)
                            ax._top_xlabel_text.set_visible(True)
                            
                            # Dynamic positioning based on top tick labels visibility
                            try:
                                # Get renderer for measurements
                                renderer = fig.canvas.get_renderer()
                                
                                # Base padding
                                labelpad = ax.xaxis.labelpad if hasattr(ax.xaxis, 'labelpad') else 4.0
                                fig_h = fig.get_size_inches()[1]
                                ax_bbox = ax.get_position()
                                ax_h_inches = ax_bbox.height * fig_h
                                base_pad_axes = (labelpad / 72.0) / ax_h_inches if ax_h_inches > 0 else 0.02
                                
                                # If top tick labels are visible, measure their height and add spacing
                                extra_offset = 0.0
                                if bool(wasd['top']['labels']) and renderer is not None:
                                    try:
                                        max_h_px = 0.0
                                        for t in ax.xaxis.get_major_ticks():
                                            lab = getattr(t, 'label2', None)  # Top labels are label2
                                            if lab is not None and lab.get_visible():
                                                bb = lab.get_window_extent(renderer=renderer)
                                                if bb is not None:
                                                    max_h_px = max(max_h_px, float(bb.height))
                                        # Convert pixels to axes coordinates
                                        if max_h_px > 0 and ax_h_inches > 0:
                                            dpi = float(fig.dpi) if hasattr(fig, 'dpi') else 100.0
                                            max_h_inches = max_h_px / dpi
                                            extra_offset = max_h_inches / ax_h_inches
                                    except Exception:
                                        # Fallback to fixed offset if labels are on
                                        extra_offset = 0.05
                                
                                total_offset = 1.0 + base_pad_axes + extra_offset
                                ax._top_xlabel_text.set_position((0.5, total_offset))
                            except Exception:
                                # Fallback positioning
                                if bool(wasd['top']['labels']):
                                    ax._top_xlabel_text.set_position((0.5, 1.07))
                                else:
                                    ax._top_xlabel_text.set_position((0.5, 1.02))
                        else:
                            # Hide top label
                            if hasattr(ax, '_top_xlabel_text') and ax._top_xlabel_text is not None:
                                ax._top_xlabel_text.set_visible(False)
                    except Exception:
                        pass
                    try:
                        # Left Y title
                        if bool(wasd['left']['title']):
                            if hasattr(ax, '_stored_ylabel') and isinstance(ax._stored_ylabel, str) and ax._stored_ylabel:
                                ax.set_ylabel(ax._stored_ylabel)
                        else:
                            if not hasattr(ax, '_stored_ylabel'):
                                try:
                                    ax._stored_ylabel = ax.get_ylabel()
                                except Exception:
                                    ax._stored_ylabel = ''
                            ax.set_ylabel("")
                    except Exception:
                        pass
                    try:
                        # Right Y title - simple approach like left/bottom
                        if bool(wasd['right']['title']) and bool(sc_eff.get_visible()):
                            if hasattr(ax2, '_stored_ylabel') and isinstance(ax2._stored_ylabel, str) and ax2._stored_ylabel:
                                ax2.set_ylabel(ax2._stored_ylabel)
                        else:
                            if not hasattr(ax2, '_stored_ylabel'):
                                try:
                                    ax2._stored_ylabel = ax2.get_ylabel()
                                except Exception:
                                    ax2._stored_ylabel = ''
                            ax2.set_ylabel("")
                    except Exception:
                        pass
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass

                def _print_wasd():
                    def b(v):
                        return 'ON ' if bool(v) else 'off'
                    print("State (top/bottom/left/right):")
                    print(f"  top    w1:{b(wasd['top']['spine'])} w2:{b(wasd['top']['ticks'])} w3:{b(wasd['top']['minor'])} w4:{b(wasd['top']['labels'])} w5:{b(wasd['top']['title'])}")
                    print(f"  bottom s1:{b(wasd['bottom']['spine'])} s2:{b(wasd['bottom']['ticks'])} s3:{b(wasd['bottom']['minor'])} s4:{b(wasd['bottom']['labels'])} s5:{b(wasd['bottom']['title'])}")
                    print(f"  left   a1:{b(wasd['left']['spine'])} a2:{b(wasd['left']['ticks'])} a3:{b(wasd['left']['minor'])} a4:{b(wasd['left']['labels'])} a5:{b(wasd['left']['title'])}")
                    print(f"  right  d1:{b(wasd['right']['spine'])} d2:{b(wasd['right']['ticks'])} d3:{b(wasd['right']['minor'])} d4:{b(wasd['right']['labels'])} d5:{b(wasd['right']['title'])}")

                print("WASD toggles: direction (w/a/s/d) x action (1..5)")
                print("  1=spine   2=ticks   3=minor ticks   4=tick labels   5=axis title")
                print("Examples: 'w2 w5' to toggle top ticks and top title; 'd2 d5' for right.")
                print("Type 'list' to show current state, 'q' to go back.")
                while True:
                    cmd = input("t> ").strip().lower()
                    if not cmd:
                        continue
                    if cmd == 'q':
                        break
                    if cmd == 'list':
                        _print_wasd(); continue
                    parts = cmd.split()
                    changed = False
                    for p in parts:
                        if len(p) != 2:
                            print(f"Unknown code: {p}"); continue
                        d, n = p[0], p[1]
                        side = {'w':'top','a':'left','s':'bottom','d':'right'}.get(d)
                        if side is None or n not in '12345':
                            print(f"Unknown code: {p}"); continue
                        key = { '1':'spine', '2':'ticks', '3':'minor', '4':'labels', '5':'title' }[n]
                        wasd[side][key] = not bool(wasd[side][key])
                        changed = True
                        # Keep tick_state in sync with new separate keys + legacy combined flags
                        if side == 'top' and key == 'ticks':
                            tick_state['t_ticks'] = bool(wasd['top']['ticks'])
                            tick_state['tx'] = bool(wasd['top']['ticks'] and wasd['top']['labels'])
                        if side == 'top' and key == 'labels':
                            tick_state['t_labels'] = bool(wasd['top']['labels'])
                            tick_state['tx'] = bool(wasd['top']['ticks'] and wasd['top']['labels'])
                        if side == 'bottom' and key == 'ticks':
                            tick_state['b_ticks'] = bool(wasd['bottom']['ticks'])
                            tick_state['bx'] = bool(wasd['bottom']['ticks'] and wasd['bottom']['labels'])
                        if side == 'bottom' and key == 'labels':
                            tick_state['b_labels'] = bool(wasd['bottom']['labels'])
                            tick_state['bx'] = bool(wasd['bottom']['ticks'] and wasd['bottom']['labels'])
                        if side == 'left' and key == 'ticks':
                            tick_state['l_ticks'] = bool(wasd['left']['ticks'])
                            tick_state['ly'] = bool(wasd['left']['ticks'] and wasd['left']['labels'])
                        if side == 'left' and key == 'labels':
                            tick_state['l_labels'] = bool(wasd['left']['labels'])
                            tick_state['ly'] = bool(wasd['left']['ticks'] and wasd['left']['labels'])
                        if side == 'right' and key == 'ticks':
                            tick_state['r_ticks'] = bool(wasd['right']['ticks'])
                            tick_state['ry'] = bool(wasd['right']['ticks'] and wasd['right']['labels'])
                        if side == 'right' and key == 'labels':
                            tick_state['r_labels'] = bool(wasd['right']['labels'])
                            tick_state['ry'] = bool(wasd['right']['ticks'] and wasd['right']['labels'])
                        if side == 'top' and key == 'minor':
                            tick_state['mtx'] = bool(wasd['top']['minor'])
                        if side == 'bottom' and key == 'minor':
                            tick_state['mbx'] = bool(wasd['bottom']['minor'])
                        if side == 'left' and key == 'minor':
                            tick_state['mly'] = bool(wasd['left']['minor'])
                        if side == 'right' and key == 'minor':
                            tick_state['mry'] = bool(wasd['right']['minor'])
                    if changed:
                        push_state("wasd-toggle")
                        _apply_wasd()
                        # Draw canvas to ensure tick labels are rendered before positioning top/right labels
                        try:
                            fig.canvas.draw()
                        except Exception:
                            try:
                                fig.canvas.draw_idle()
                            except Exception:
                                pass
            except Exception as e:
                print(f"Error in WASD tick menu: {e}")
            _print_menu(); continue
        elif key == 'g':
            while True:
                print("Geometry: p=plot frame, c=canvas, q=back")
                sub = input("Geom> ").strip().lower()
                if not sub:
                    continue
                if sub == 'q':
                    break
                if sub == 'p':
                    # We don't have y_data_list/labels; we just trigger a redraw after resize
                    try:
                        push_state("resize-frame")
                        resize_plot_frame(fig, ax, [], [], type('Args', (), {'stack': False})(), lambda *_: None)
                    except Exception as e:
                        print(f"Resize failed: {e}")
                elif sub == 'c':
                    try:
                        push_state("resize-canvas")
                        resize_canvas(fig)
                    except Exception as e:
                        print(f"Resize failed: {e}")
            _print_menu(); continue
        elif key == 'r':
            # Rename axis titles
            while True:
                print("Rename titles: x=x-axis, ly=left y-axis, ry=right y-axis, q=back")
                sub = input("Rename> ").strip().lower()
                if not sub:
                    continue
                if sub == 'q':
                    break
                if sub == 'x':
                    current = ax.get_xlabel()
                    print(f"Current x-axis title: '{current}'")
                    new_title = input("Enter new x-axis title (q=cancel): ").strip()
                    if new_title and new_title.lower() != 'q':
                        try:
                            push_state("rename-x")
                            ax.set_xlabel(new_title)
                            # Update stored titles for top/bottom
                            ax._stored_xlabel = new_title
                            ax._stored_top_xlabel = new_title
                            # If top title is visible, update it
                            if hasattr(ax, '_top_xlabel_text') and ax._top_xlabel_text is not None:
                                if ax._top_xlabel_text.get_visible():
                                    ax._top_xlabel_text.set_text(new_title)
                            fig.canvas.draw_idle()
                            print(f"X-axis title updated to: '{new_title}'")
                        except Exception as e:
                            print(f"Error: {e}")
                elif sub == 'ly':
                    current = ax.get_ylabel()
                    print(f"Current left y-axis title: '{current}'")
                    new_title = input("Enter new left y-axis title (q=cancel): ").strip()
                    if new_title and new_title.lower() != 'q':
                        try:
                            push_state("rename-ly")
                            ax.set_ylabel(new_title)
                            # Update stored title
                            ax._stored_ylabel = new_title
                            fig.canvas.draw_idle()
                            print(f"Left y-axis title updated to: '{new_title}'")
                        except Exception as e:
                            print(f"Error: {e}")
                elif sub == 'ry':
                    current = ax2.get_ylabel()
                    print(f"Current right y-axis title: '{current}'")
                    new_title = input("Enter new right y-axis title (q=cancel): ").strip()
                    if new_title and new_title.lower() != 'q':
                        try:
                            push_state("rename-ry")
                            ax2.set_ylabel(new_title)
                            # Update stored title
                            if not hasattr(ax2, '_stored_ylabel'):
                                ax2._stored_ylabel = ''
                            ax2._stored_ylabel = new_title
                            fig.canvas.draw_idle()
                            print(f"Right y-axis title updated to: '{new_title}'")
                        except Exception as e:
                            print(f"Error: {e}")
                else:
                    print("Unknown option.")
            _print_menu(); continue
        elif key == 'x':
            rng = input("Enter x-range: min max (q=cancel): ").strip()
            if rng and rng.lower() != 'q':
                parts = rng.replace(',', ' ').split()
                if len(parts) != 2:
                    print("Need two numbers.")
                else:
                    try:
                        lo = float(parts[0]); hi = float(parts[1])
                        if lo == hi:
                            print("Min and max cannot be equal.")
                        else:
                            push_state("x-range")
                            ax.set_xlim(min(lo, hi), max(lo, hi))
                            fig.canvas.draw_idle()
                    except Exception:
                        print("Invalid numbers.")
            _print_menu(); continue
        elif key == 'y':
            while True:
                print("Y-ranges: ly=left axis, ry=right axis, q=back")
                ycmd = input("Y> ").strip().lower()
                if not ycmd:
                    continue
                if ycmd == 'q':
                    break
                if ycmd == 'ly':
                    rng = input("Enter left y-range: min max (q=cancel): ").strip()
                    if not rng or rng.lower() == 'q':
                        continue
                    parts = rng.replace(',', ' ').split()
                    if len(parts) != 2:
                        print("Need two numbers."); continue
                    try:
                        lo = float(parts[0]); hi = float(parts[1])
                        if lo == hi:
                            print("Min and max cannot be equal."); continue
                        push_state("y-left-range")
                        ax.set_ylim(min(lo, hi), max(lo, hi))
                        fig.canvas.draw_idle()
                    except Exception:
                        print("Invalid numbers.")
                elif ycmd == 'ry':
                    try:
                        eff_on = bool(sc_eff.get_visible())
                    except Exception:
                        eff_on = True
                    if not eff_on:
                        print("Right Y is not shown; enable efficiency with 'ry' first.")
                        continue
                    rng = input("Enter right y-range: min max (q=cancel): ").strip()
                    if not rng or rng.lower() == 'q':
                        continue
                    parts = rng.replace(',', ' ').split()
                    if len(parts) != 2:
                        print("Need two numbers."); continue
                    try:
                        lo = float(parts[0]); hi = float(parts[1])
                        if lo == hi:
                            print("Min and max cannot be equal."); continue
                        push_state("y-right-range")
                        ax2.set_ylim(min(lo, hi), max(lo, hi))
                        fig.canvas.draw_idle()
                    except Exception:
                        print("Invalid numbers.")
            _print_menu(); continue
        else:
            print("Unknown key.")
            _print_menu(); continue


__all__ = ["cpc_interactive_menu"]
