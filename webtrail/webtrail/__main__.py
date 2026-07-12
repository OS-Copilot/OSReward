"""Command-line entry point: ``python -m webtrail <subcommand>``."""

from __future__ import annotations

import argparse

from . import collect, judge
from .postprocess import cli as postprocess_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="webtrail",
        description="Collect, judge, and curate web-agent trajectories.",
    )
    subparsers = parser.add_subparsers(required=True)
    collect.add_parser(subparsers)
    judge.add_parser(subparsers)
    postprocess_cli.add_parser(subparsers)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
