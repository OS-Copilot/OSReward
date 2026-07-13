"""Command-line entry point: ``python -m webtrail <subcommand>``."""

from __future__ import annotations

import argparse

from . import collect, doctor, judge, taskio, viewer
from .postprocess import cli as postprocess_cli
from .postprocess import export


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="webtrail",
        description="Collect, judge, and curate web-agent trajectories.",
    )
    subparsers = parser.add_subparsers(required=True)
    taskio.add_parser(subparsers)
    doctor.add_parser(subparsers)
    collect.add_parser(subparsers)
    judge.add_parser(subparsers)
    postprocess_cli.add_parser(subparsers)
    export.add_parser(subparsers)
    viewer.add_parser(subparsers)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
