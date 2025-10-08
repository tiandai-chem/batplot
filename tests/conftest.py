"""Pytest configuration to ensure local package is imported during tests.

This avoids conflicts with any globally installed package named 'batplot'.
"""

import os
import sys

# Insert the repository root (parent of tests/) at the front of sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
