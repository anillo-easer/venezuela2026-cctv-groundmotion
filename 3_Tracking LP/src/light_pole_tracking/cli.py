"""Command-line interface for the tracking pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pole-track",
        description="Track a compact luminaire relative to local ground controls.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Run one configured video view.")
    run.add_argument(
        "--config", type=Path, required=True, help="Run configuration JSON."
    )
    validate = subcommands.add_parser(
        "validate-config", help="Validate paths and settings without tracking."
    )
    validate.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            config.validate_inputs()
            print(f"Valid configuration: {config.path}")
            return 0
        result = run_pipeline(config)
        print(result.output_directory)
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
