"""CLI entry for batplot.

Executes the existing monolithic implementation as a script to avoid
import-time side effects (argument parsing during import).
"""

from __future__ import annotations

import runpy
import sys

def main() -> int:
	runpy.run_module("batplot.batplot", run_name="__main__")
	return 0

__all__ = ["main"]
