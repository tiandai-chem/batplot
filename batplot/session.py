"""Session helpers for batplot interactive mode.

This module provides functions to persist and restore interactive plotting
state. The save function is wired into the interactive menu; load is available
for future integration.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import _confirm_overwrite


def dump_session(
    filename: str,
    *,
    fig,
    ax,
    x_data_list: Sequence[np.ndarray],
    y_data_list: Sequence[np.ndarray],
    orig_y: Sequence[np.ndarray],
    offsets_list: Sequence[float],
    labels: Sequence[str],
    delta: float,
    args,
    tick_state: Dict[str, bool],
    cif_tick_series: Iterable[Tuple[str, str, List[float], float | None, float, Any]] | None = None,
    cif_hkl_map: Dict[str, List[Tuple[float, int, int, int]]] | None = None,
    cif_hkl_label_map: Dict[str, Dict[float, str]] | None = None,
    show_cif_hkl: bool | None = None,
) -> None:
    """Serialize the current interactive session to a pickle file.

    Parameters mirror the state captured in the original inline helper.
    """

    # Infer axis mode string
    if getattr(args, 'xaxis', None) in ("Q", "2theta", "r", "energy", "k", "rft"):
        axis_mode_session = args.xaxis
    else:
        # Best-effort inference from labels/units already set on axes
        xl = (ax.get_xlabel() or "").lower()
        if "q (" in xl:
            axis_mode_session = "Q"
        elif "$2\\theta$" in xl or "2" in xl and "theta" in xl:
            axis_mode_session = "2theta"
        elif xl.startswith("r ") or xl.startswith("r ("):
            axis_mode_session = "r"
        elif "energy" in xl:
            axis_mode_session = "energy"
        elif xl.startswith("k ") or xl.startswith("k ("):
            axis_mode_session = "k"
        elif "radial" in xl:
            axis_mode_session = "rft"
        else:
            axis_mode_session = "unknown"

    label_layout = 'stack' if getattr(args, 'stack', False) else 'block'

    # Axes frame size (in inches) to complement the canvas size
    bbox = ax.get_position()
    fw, fh = fig.get_size_inches()
    frame_w_in = bbox.width * fw
    frame_h_in = bbox.height * fh

    # Save spines state
    spines_state = {
        name: {
            'linewidth': sp.get_linewidth(),
            'color': sp.get_edgecolor(),
            'visible': sp.get_visible(),
        } for name, sp in ax.spines.items()
    }

    # Helper to capture a representative tick line width
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

    tick_widths = {
        'x_major': _tick_width(ax.xaxis, 'major'),
        'x_minor': _tick_width(ax.xaxis, 'minor'),
        'y_major': _tick_width(ax.yaxis, 'major'),
        'y_minor': _tick_width(ax.yaxis, 'minor'),
    }

    sp = fig.subplotpars
    subplot_margins = {
        'left': float(sp.left),
        'right': float(sp.right),
        'bottom': float(sp.bottom),
        'top': float(sp.top),
    }

    try:
        sess = {
            'version': 2,
            'x_data': [np.array(a) for a in x_data_list],
            'y_data': [np.array(a) for a in y_data_list],
            'orig_y': [np.array(a) for a in orig_y],
            'offsets': list(offsets_list),
            'labels': list(labels),
            'line_styles': [
                {
                    'color': ln.get_color(),
                    'linewidth': ln.get_linewidth(),
                    'linestyle': ln.get_linestyle(),
                    'alpha': ln.get_alpha(),
                    'marker': ln.get_marker(),
                    'markersize': ln.get_markersize(),
                    'markerfacecolor': ln.get_markerfacecolor(),
                    'markeredgecolor': ln.get_markeredgecolor(),
                } for ln in ax.lines
            ],
            'delta': float(delta),
            'label_layout': label_layout,
            'axis_mode': axis_mode_session,
            'axis': {
                'xlabel': ax.get_xlabel(),
                'ylabel': ax.get_ylabel(),
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
            },
            'figure': {
                'size': tuple(map(float, fig.get_size_inches())),
                'dpi': int(fig.dpi),
                'frame_size': (frame_w_in, frame_h_in),
                'axes_bbox': {
                    'left': float(bbox.x0),
                    'bottom': float(bbox.y0),
                    'right': float(bbox.x0 + bbox.width),
                    'top': float(bbox.y0 + bbox.height),
                },
                'subplot_margins': subplot_margins,
                'spines': spines_state,
            },
            'tick_state': dict(tick_state),
            'tick_widths': tick_widths,
            'font': {
                'size': plt.rcParams.get('font.size'),
                'chain': list(plt.rcParams.get('font.sans-serif', [])),
            },
            'args_subset': {
                'stack': bool(getattr(args, 'stack', False)),
                'autoscale': bool(getattr(args, 'autoscale', False)),
                'raw': bool(getattr(args, 'raw', False)),
            },
            'cif_tick_series': [tuple(t) for t in (cif_tick_series or [])],
            'cif_hkl_map': {k: [tuple(v) for v in val] for k, val in (cif_hkl_map or {}).items()},
            'cif_hkl_label_map': {k: dict(v) for k, v in (cif_hkl_label_map or {}).items()},
            'show_cif_hkl': bool(show_cif_hkl),
        }
        sess['axis_titles'] = {
            'top_x': bool(getattr(ax, '_top_xlabel_on', False)),
            'right_y': bool(getattr(ax, '_right_ylabel_on', False)),
            'has_bottom_x': bool(ax.get_xlabel()),
            'has_left_y': bool(ax.get_ylabel()),
        }
        target = _confirm_overwrite(filename)
        if not target:
            print("Session save canceled.")
            return
        with open(target, 'wb') as f:
            pickle.dump(sess, f)
        print(f"Session saved to {target}")
    except Exception as e:  # pragma: no cover - defensive path
        print(f"Error saving session: {e}")


__all__ = ["dump_session"]
