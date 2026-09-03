"""Small command-line entry point for project inspection and launch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from geometric_coder.project import GeometricCoder


def build_parser() -> argparse.ArgumentParser:
    """Build the GeCo CLI parser."""
    parser = argparse.ArgumentParser(prog="geco", description="Geometric Coder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show project metadata")
    info.add_argument("project")

    launch = subparsers.add_parser("launch", help="Launch the local web interface")
    launch.add_argument("project")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", default=8050, type=int)
    launch.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GeCo command-line interface."""
    args = build_parser().parse_args(argv)
    project = GeometricCoder.open(args.project)
    if args.command == "info":
        for key, value in project.metadata.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "launch":
        project.launch(host=args.host, port=args.port, debug=args.debug)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
