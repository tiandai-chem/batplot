"""Utility helpers for batplot."""

import os
import sys


def ensure_subdirectory(subdir_name: str, base_path: str = None) -> str:
    """Ensure subdirectory exists and return its path.
    
    Args:
        subdir_name: Name of subdirectory ('Figures', 'Styles', or 'Projects')
        base_path: Base directory (defaults to current working directory)
    
    Returns:
        Full path to the subdirectory
    """
    if base_path is None:
        base_path = os.getcwd()
    subdir_path = os.path.join(base_path, subdir_name)
    try:
        os.makedirs(subdir_path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create {subdir_name} directory: {e}")
        return base_path  # Fallback to base directory
    return subdir_path


def get_organized_path(filename: str, file_type: str, base_path: str = None) -> str:
    """Get the appropriate path for a file based on its type.
    
    Args:
        filename: The filename (can include path)
        file_type: 'figure', 'style', or 'project'
        base_path: Base directory (defaults to current working directory)
    
    Returns:
        Full path with appropriate subdirectory
    """
    # If filename already has a directory component, use it as-is
    if os.path.dirname(filename):
        return filename
    
    # Determine subdirectory based on file type
    subdir_map = {
        'figure': 'Figures',
        'style': 'Styles',
        'project': 'Projects'
    }
    
    subdir_name = subdir_map.get(file_type)
    if not subdir_name:
        # Unknown type, use current directory
        if base_path is None:
            base_path = os.getcwd()
        return os.path.join(base_path, filename)
    
    subdir_path = ensure_subdirectory(subdir_name, base_path)
    return os.path.join(subdir_path, filename)


def list_files_in_subdirectory(extensions: tuple, file_type: str, base_path: str = None) -> list:
    """List files with given extensions in the appropriate subdirectory.
    
    Args:
        extensions: Tuple of file extensions (e.g., ('.svg', '.png'))
        file_type: 'figure', 'style', or 'project'
        base_path: Base directory (defaults to current working directory)
    
    Returns:
        List of (filename, full_path) tuples sorted by filename
    """
    if base_path is None:
        base_path = os.getcwd()
    
    # Determine subdirectory based on file type
    subdir_map = {
        'figure': 'Figures',
        'style': 'Styles',
        'project': 'Projects'
    }
    
    subdir_name = subdir_map.get(file_type)
    if not subdir_name:
        # Unknown type, list from current directory
        folder = base_path
    else:
        folder = os.path.join(base_path, subdir_name)
        # Also create the directory if it doesn't exist
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            folder = base_path
    
    files = []
    try:
        all_files = os.listdir(folder)
        for f in all_files:
            if f.lower().endswith(extensions):
                files.append((f, os.path.join(folder, f)))
    except Exception:
        pass
    
    return sorted(files, key=lambda x: x[0])


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
