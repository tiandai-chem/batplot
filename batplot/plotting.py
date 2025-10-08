"""Plotting helpers for batplot."""

from __future__ import annotations

from typing import List
import numpy as np


def update_labels(ax, y_data_list: List, label_text_objects: List, stack_mode: bool):
    """
    stack_mode True:
        Each label at (xmax, curve_ymax) in data coordinates.
    stack_mode False:
        Labels form a fixed vertical list at top-right in axes coordinates.
    """
    if not label_text_objects:
        return

    if stack_mode:
        x_max = ax.get_xlim()[1]
        for i, txt in enumerate(label_text_objects):
            if i < len(y_data_list) and len(y_data_list[i]) > 0:
                y_max_curve = float(np.max(y_data_list[i]))
            else:
                y_max_curve = ax.get_ylim()[1]
            txt.set_transform(ax.transData)
            txt.set_position((x_max, y_max_curve))
    else:
        n = len(label_text_objects)
        top_pad = 0.02
        start_y = 1.0 - top_pad
        spacing = min(0.08, max(0.025, 0.90 / max(n, 1)))
        for i, txt in enumerate(label_text_objects):
            y_pos = start_y - i * spacing
            if y_pos < 0.02:
                y_pos = 0.02
            txt.set_transform(ax.transAxes)
            txt.set_position((1.0, y_pos))
    ax.figure.canvas.draw_idle()
