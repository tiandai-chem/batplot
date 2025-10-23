"""Argument parsing for batplot CLI."""

from __future__ import annotations

import argparse
import sys


def _print_general_help() -> None:
    msg = (
        "batplot — quick plotting for lab data\n\n"
        "What it does:\n"
        "  • XY: XRD/PDF/XAS/... curves\n"
        "  • EC: GC/CPC/dQdV/CV (from .csv or .mpt)\n"
        "  • Operando: contour maps from a folder of XY and .mpt files (use --raw for raw intensity)\n"
        "  • Batch: export SVG plots for all files in a directory\n\n"
        "  • Interactive mode: --interactive flag opens a menu for styling, ranges, fonts, export, sessions\n\n"
    "How to run (basics):\n"
    "  [XY curves]\n"
    "    batplot file1.xy file2.qye [option1] [option2]             # XY curves\n"
    "    batplot allfiles                       # Plot all XY files in current directory on same figure\n"
    "    batplot allfiles --interactive         # Plot all files with interactive menu\n"
    "    batplot --all                          # Batch mode: all XY files in current directory\n"
    "    batplot --all --xaxis 2theta --xrange 10 80  # Batch mode with custom axis and range\n\n"
    "  [Electrochemistry]\n"
    "    batplot --gc FILE.mpt --mass 7.0       # EC GC from .mpt (requires --mass mg)\n"
    "    batplot --gc FILE.csv                  # EC GC from supported .csv (no mass required)\n"
    "    batplot --gc --all --mass 7.0          # Batch: all .mpt/.csv in directory (.mpt needs --mass)\n"
    "    batplot --all --gc style.bps --mass 7  # Batch with style: apply style.bps to all GC files\n"
    "    batplot --all --cv config.bpsg         # Batch with style+geom: apply to all CV files\n"
    "    batplot --dqdv FILE.csv                # EC dQ/dV from supported .csv\n"
    "    batplot --dqdv --all                   # Batch: all .csv in directory (dQdV mode)\n"
    "    batplot --cv FILE.mpt                  # EC CV (cyclic voltammetry) from .mpt\n"
    "    batplot --cv FILE.txt                  # EC CV (cyclic voltammetry) from .txt\n"
    "    batplot --cv --all                     # Batch: all .mpt/.txt in directory (CV mode)\n\n"
    "  [Operando]\n"
    "    batplot --operando [FOLDER]            # Operando contour from folder (normalized by default)\n"
    "    batplot --operando --raw               # Operando contour with raw intensity (no normalization)\n\n"
        "Features:\n"
        "  • Quick plotting with sensible defaults, no config files needed\n"
        "  • Supports many common file formats (see -h xy/ec/op)\n"
        "  • Interactive menus (--interactive): styling, ranges, fonts, export, sessions\n"
        "  • Batch processing: use 'allfiles' to plot all on same figure, or --all for separate SVGs\n"
        "  • Batch styling: apply .bps/.bpsg files to all exports (use --all flag)\n\n"
    
        "More help:\n"
        "  batplot -h xy   # XY file plotting guide\n"
        "  batplot -h ec   # Electrochemistry (GC/dQdV/CV/CPC) guide\n"
        "  batplot -h op   # Operando guide\n\n"
        "Contact:\n"
        "  GitHub: https://github.com/tiandai-chem/batplot\n"
        "  Email: tianda@uio.no\n"
    )
    print(msg)


def _print_xy_help() -> None:
    msg = (
        "XY plots (diffraction/PDF/XAS)\n\n"
        "Supported files: .xye .xy .qye .dat .csv .gr .nor .chik .chir .txt (2-col). CIF overlays supported.\n\n"
        "Axis detection: .qye→Q, .gr→r, .nor→energy, .chik→k, .chir→r, else use --xaxis (Q, 2theta, r, k, energy, rft).\n"
        "If mixing 2θ data in Q, give wavelength per-file (file.xye:1.5406) or global --wl.\n\n"
        "Examples:\n"
        "  batplot a.xye:1.5406 b.qye --stack --interactive\n"
        "  batplot a.dat b.xy --wl 1.54 --out fig.svg\n"
        "  batplot pattern.qye ticks.cif --interactive\n\n"
        "Plot all files together:\n"
        "  batplot allfiles                       # Plot all XY files on same figure\n"
        "  batplot allfiles --stack --interactive # Stack all files with interactive menu\n"
        "  batplot allfiles --xaxis 2theta --xrange 10 80  # All files with custom axis and range\n"
        "  batplot allfiles --wl 1.5406 --delta 0.2        # All files with wavelength and spacing\n\n"
        "Batch mode (separate SVG for each file):\n"
        "  batplot --all                          # Export all XY files to SVG\n"
        "  batplot --all --xaxis 2theta           # Batch mode with custom axis type\n"
        "  batplot --all --xrange 10 80           # Batch mode with X-axis range\n"
        "  batplot --all --wl 1.5406 --raw        # Batch mode with wavelength and raw intensity\n\n"
        "Tips and options:\n"
        "[XY plot]\n"
    "  --interactive             : open interactive menu for styling, ranges, fonts, export, sessions\n"
    "  --delta/-d <float>        : spacing between curves, e.g. --delta 0.1\n"
    "  --raw                     : plot raw intensity (normalized by default)\n"
    "  --xrange/-r <min> <max>   : set x-axis range, e.g. --xrange 0 10\n"
    "  --out/-o <filename>       : save figure to file, e.g. --out file.svg\n"
    "  --xaxis <type>            : set x-axis type (Q, 2theta, r, k, energy, rft, or user defined), e.g. --xaxis 2theta\n"
    "  --wl <float>              : set wavelength for Q conversion for all files, e.g. --wl 1.5406\n"
    "  --fullprof <args>         : FullProf overlay options\n"
    "  --stack                   : stack curves vertically\n"
    )
    print(msg)


def _print_ec_help() -> None:
    msg = (
        "Electrochemistry (GC, dQ/dV, CV, and CPC)\n\n"
        "Use --interactive for styling, colors, line widths, axis scales, etc.\n"
        "GC from .mpt: requires active mass in mg to compute mAh g⁻¹.\n"
        "  batplot --gc file.mpt --mass 6.5 --interactive\n\n"
        "GC from supported .csv: specific capacity is read directly (no --mass).\n"
        "  batplot --gc file.csv\n\n"
        "dQ/dV from supported .csv:\n"
        "  batplot --dqdv file.csv\n\n"
        "Cyclic voltammetry (CV) from .mpt or .txt: plots voltage vs current for each cycle.\n"
        "  batplot --cv file.mpt\n"
        "  batplot --cv file.txt\n\n"
        "Capacity-per-cycle (CPC) with coulombic efficiency from .csv or .mpt.\n"
        "Supports multiple files with individual color customization:\n"
        "  batplot --cpc file.csv\n"
        "  batplot --cpc file.mpt --mass 1.2\n"
        "  batplot --cpc file1.csv file2.csv file3.mpt --mass 1.2 --interactive\n"
        "Batch mode: Process all files in a directory and export to SVG.\n"
        "  batplot --gc --all --mass 7.0          # All .mpt/.csv files (.mpt requires --mass)\n"
        "  batplot --cv --all                     # All .mpt/.txt files (CV mode)\n"
        "  batplot --dqdv --all                   # All .csv files (dQdV mode)\n"
        "  batplot --cpc --all --mass 5.4         # All .mpt/.csv files (.mpt requires --mass)\n"
        "  batplot --gc /path/to/folder --mass 6  # Process specific directory\n\n"
        "Batch mode with style/geometry: Apply .bps/.bpsg files to all batch exports.\n"
        "  batplot --all --gc style.bps --mass 7  # Apply style to all GC plots\n"
        "  batplot --all --cv config.bpsg         # Apply style+geometry to all CV plots\n"
        "  batplot --all --dqdv my.bps            # Apply style to all dQdV plots\n"
        "  batplot --all --cpc geom.bpsg --mass 6 # Apply style+geom to all CPC plots\n\n"
        "Interactive (--interactive): choose cycles, colors/palettes, line widths, axis scales (linear/log/symlog),\n"
        "rename axes, toggle ticks/titles/spines, print/export/import style (.bps/.bpsg), save session (.pkl).\n"
        "Note: Batch mode (--all) exports SVG files automatically; --interactive is for single-file plotting only.\n"
    )
    print(msg)


def _print_op_help() -> None:
    msg = (
        "Operando contour plots\n\n"
        "Example usage:\n"
        "  batplot --operando --interactive --wl 0.25995  # Interactive mode with Q conversion\n"
        "  batplot --operando --raw --interactive         # Raw intensity (no normalization)\n"
        "  batplot --operando --xaxis 2theta              # Using 2theta axis\n\n"
        "  • Folder should contain XY files (.xy/.xye/.qye/.dat).\n"
        "  • By default, data is normalized (0-1). Use --raw to plot raw intensity.\n"
        "  • If no .qye present, provide --xaxis 2theta or set --wl for Q conversion.\n"
        "  • If a BioLogic .mpt is present, an EC side panel will be added automatically.\n\n"
        "Interactive (--interactive): resize axes/canvas, change colormap, set intensity range (oi),\n"
        "EC y-axis options (time ↔ ions), geometry tweaks, print/export/import style, save session.\n"
    )
    print(msg)


def build_parser() -> argparse.ArgumentParser:
    # We use a custom help so users can request topic help via `-h xy|ec|op`.
    parser = argparse.ArgumentParser(add_help=False)
    # Topic-aware help flag (optional argument)
    parser.add_argument("--help", "-h", nargs="?", const="", metavar="topic",
                        help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--delta", "-d", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--autoscale", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--xrange", "-r", nargs=2, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--out", "-o", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--errors", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--xaxis", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--convert", "-c", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--wl", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--fullprof", nargs="+", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--raw", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--interactive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--savefig", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--stack", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operando", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gc", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mass", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--dqdv", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cv", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cpc", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--all", type=str, nargs='?', const='all', help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    # We need to parse known args first to handle our custom help without errors
    ns, _unknown = parser.parse_known_args(argv)
    topic = getattr(ns, 'help', None)
    if topic is not None:
        t = (topic or '').strip().lower()
        if t in ("", "help"):
            _print_general_help()
        elif t in ("xy",):
            _print_xy_help()
        elif t in ("ec", "gc", "dqdv"):
            _print_ec_help()
        elif t in ("op", "operando"):
            _print_op_help()
        else:
            _print_general_help()
            print("\nUnknown help topic. Use: xy, ec, op")
        sys.exit(0)
    # No help requested: parse fully
    return parser.parse_args(argv)


__all__ = ["build_parser", "parse_args"]
