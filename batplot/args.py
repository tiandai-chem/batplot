"""Argument parsing for batplot CLI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "batplot: Plot diffraction / PDF / XAS data (.xye, .xy, .qye, .dat, .csv, .gr, .nor, .chik, .chir, .txt)\n"
            "  --delta or -d : vertical offset between curves (default 0.0 if --stack)\n"
            "  --xrange min max : X-axis range (2θ or Q), Example: --xrange 2 10\n"
            "  --out or -o : output image filename (default SVG), Example: --out figure.svg\n"
            "  --xaxis : X-axis type override if the file extension is not recognized (choose from: 2theta, Q, r, k, energy, rft, or 'user defined')\n"
            "  --wl : global wavelength (Å) for Q conversion, Example: --wl 1.5406\n"
            "  --fullprof : FullProf matrix: xstart xend xstep [wavelength], Example: --fullprof 2 10 0.02 1.5406\n"
            "  --raw : plot raw intensity values instead of normalized\n"
            "  --stack : stack curves from top to bottom\n"
            "  --interactive : keep figure open for interactive editing\n\n"
            "File type and X-axis selection:\n"
            "  - .qye: X axis is Q\n"
            "  - .gr: X axis is r\n"
            "  - .nor: X axis is Energy (eV)\n"
            "  - .chik: X axis is k\n"
            "  - .chir: X axis is r\n"
            "  - .txt: Treated as generic 2-column data.\n"
            "  If none of the files have a recognized extension, you must either provide a wavelength to each file (if you want to plot everything in Q space) or specify --xaxis (Q, 2theta, r, k, energy, rft, or 'user defined').\n\n"
            "Example usages:\n"
            "  batplot file1.xye:1.5406 file2.qye --stack --interactive\n"
            "  batplot file1.dat file2.dat --wl 1.5406 --delta 1.0 --out figure.svg\n"
            "  batplot file1.dat file2.xy --xaxis 2theta --raw --xrange 2 10\n"
            "  batplot file1.qye file2.xye:1.54 structure1.cif structure2.cif --stack --interactive\n\n"
            "Extra usage:\n"
            "  batplot FOLDER    -> batch export all supported files in FOLDER to FOLDER/batplot_svg/*.svg\n"
            "  batplot all       -> batch export all supported files in current directory to ./batplot_svg/*.svg\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


__all__ = ["build_parser", "parse_args"]
