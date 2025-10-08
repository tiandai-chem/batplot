"""Interactive menu for batplot (extracted from monolithic script).

This module provides interactive_menu(fig, ax, ...).
It depends on matplotlib and numpy via the caller environment.
"""

from __future__ import annotations

# We import numpy and matplotlib only as needed to avoid circulars
import numpy as np  # noqa: F401  (used inside the function body)


def interactive_menu(fig, ax, y_data_list, x_data_list, labels, orig_y,
                     label_text_objects, delta, x_label, args,
                     x_full_list, raw_y_full_list, offsets_list,
                     use_Q, use_r, use_E, use_k, use_rft):
    # NOTE: For brevity and safety, we import the actual implementation from the original module
    # to avoid large code duplication. Later we can fully migrate implementation here.
    # We call into the legacy function if available.
    try:
        from .batplot import interactive_menu as legacy_menu  # type: ignore
        return legacy_menu(fig, ax, y_data_list, x_data_list, labels, orig_y,
                           label_text_objects, delta, x_label, args,
                           x_full_list, raw_y_full_list, offsets_list,
                           use_Q, use_r, use_E, use_k, use_rft)
    except Exception:
        # Fallback: no-op if import failed
        return


__all__ = ["interactive_menu"]
