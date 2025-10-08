"""Utility helpers for batplot."""

import os
import sys


def normalize_label_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("Å⁻¹", "Å$^{-1}$")
    text = text.replace("Å ^-1", "Å$^{-1}$")
    text = text.replace("Å^-1", "Å$^{-1}$")
    text = text.replace(r"\AA⁻¹", r"\AA$^{-1}$")
    return text


def _confirm_overwrite(path: str, auto_suffix: bool = True):
    """Ask user before overwriting an existing file.

    Returns a path (possibly suffixed) or None to cancel.
    In non-interactive input, auto-suffix existing filenames if allowed.
    """
    try:
        if not os.path.exists(path):
            return path
        if not sys.stdin.isatty():
            if not auto_suffix:
                return None
            base, ext = os.path.splitext(path)
            k = 1
            new_path = f"{base}_{k}{ext}"
            while os.path.exists(new_path) and k < 1000:
                k += 1
                new_path = f"{base}_{k}{ext}"
            return new_path
        ans = input(f"File '{path}' exists. Overwrite? [y/N]: ").strip().lower()
        if ans == 'y':
            return path
        alt = input("Enter new filename (blank=cancel): ").strip()
        if not alt:
            return None
        if not os.path.splitext(alt)[1] and os.path.splitext(path)[1]:
            alt += os.path.splitext(path)[1]
        if os.path.exists(alt):
            print("Chosen alternative also exists; action canceled.")
            return None
        return alt
    except Exception:
        return path
