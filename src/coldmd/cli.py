"""Command-line interface for cold-compression molecular dynamics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from . import __version__
from .config import ConfigurationError


def build_parser() -> argparse.ArgumentParser:
    """Build the public ``coldmd`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="coldmd",
        description=(
            "Run reproducible DFT-trained-potential cold-compression and "
            "decompression molecular dynamics."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="show a Python traceback when a command fails",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser(
        "validate",
        help="validate YAML, CIF, and calculator compatibility",
        description=(
            "Validate the configuration schema and, by default, initialize the "
            "calculator for energy/force/stress checks."
        ),
    )
    validate_parser.add_argument("config", type=Path, help="YAML configuration")
    validate_parser.add_argument(
        "--skip-calculator-check",
        action="store_true",
        help="validate configuration and input only, without loading the potential",
    )
    _add_json_option(validate_parser)
    validate_parser.set_defaults(handler=_handle_validate)

    dryrun_parser = commands.add_parser(
        "dryrun",
        help="verify setup with a fixed-cell NVT smoke trajectory",
        description=(
            "Validate calculator/integrator readiness without constructing an "
            "artificial high-pressure state or estimating production wall time."
        ),
    )
    dryrun_parser.add_argument("config", type=Path, help="YAML configuration")
    dryrun_parser.add_argument(
        "--nvt-steps",
        type=_positive_int,
        default=100,
        help="fixed-cell NVT smoke steps (default: 100)",
    )
    dryrun_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write the dry-run report to this directory",
    )
    _add_json_option(dryrun_parser)
    dryrun_parser.set_defaults(handler=_handle_dryrun)

    run_parser = commands.add_parser(
        "run",
        help="start a configured cold-compression run",
    )
    run_parser.add_argument("config", type=Path, help="YAML configuration")
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override output.directory from the YAML file",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow the API's guarded overwrite policy for an existing run directory",
    )
    _add_json_option(run_parser)
    run_parser.set_defaults(handler=_handle_run)

    fork_parser = commands.add_parser(
        "fork",
        help="start a new run from a parent stage snapshot with momenta",
        description=(
            "Create an independent run from one canonical EXTXYZ file below a "
            "parent run's stage_structures directory. Mechanical state is inherited; "
            "step, time, V/V0, RNG, integrator, and safety history start anew."
        ),
    )
    fork_parser.add_argument("config", type=Path, help="new child YAML configuration")
    fork_parser.add_argument(
        "--from-run",
        type=Path,
        required=True,
        help="parent ColdMD run directory",
    )
    fork_parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help=(
            "canonical .extxyz snapshot (absolute path or filename relative to "
            "PARENT/stage_structures)"
        ),
    )
    fork_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override output.directory from the child YAML",
    )
    fork_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow the guarded backup policy for an existing child directory",
    )
    fork_parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the source mechanics and child calculator without running MD",
    )
    _add_json_option(fork_parser)
    fork_parser.set_defaults(handler=_handle_fork)

    resume_parser = commands.add_parser(
        "resume",
        help="resume from the latest complete checkpoint",
    )
    resume_parser.add_argument("run_dir", type=Path, help="existing run directory")
    _add_json_option(resume_parser)
    resume_parser.set_defaults(handler=_handle_resume)

    analyze_parser = commands.add_parser(
        "analyze",
        help="analyze a completed or stopped run",
    )
    analyze_parser.add_argument("run_dir", type=Path, help="run directory")
    analyze_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional analysis/configuration override YAML",
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing derived analysis artifacts",
    )
    stage_selection = analyze_parser.add_mutually_exclusive_group()
    stage_selection.add_argument(
        "--stage",
        action="append",
        dest="stages",
        metavar="NAME",
        help=(
            "analyze this structural stage; repeat the option to select multiple "
            "stages (thermodynamic summaries still include every stage)"
        ),
    )
    stage_selection.add_argument(
        "--all-stages",
        action="store_true",
        help="run structural analyses for every trajectory stage",
    )
    analyze_parser.add_argument(
        "--si-o-cutoff-A",
        type=_positive_float,
        default=None,
        metavar="ANGSTROM",
        help=(
            "fixed Si-O coordination cutoff for NBO/BO classification; by default "
            "each stage uses the first minimum of its Si-O RDF"
        ),
    )
    _add_json_option(analyze_parser)
    analyze_parser.set_defaults(handler=_handle_analyze)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        _emit_result(result, as_json=args.json_output)
        return 0
    except KeyboardInterrupt:
        print("coldmd: interrupted", file=sys.stderr)
        return 130
    except (ConfigurationError, ValidationError, FileNotFoundError) as exc:
        if args.traceback:
            raise
        print(f"coldmd: validation failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # The API owns domain-specific runtime exceptions.
        if args.traceback:
            raise
        print(f"coldmd: {args.command} failed: {exc}", file=sys.stderr)
        return 1


def _handle_validate(args: argparse.Namespace) -> Any:
    from .api import validate_config

    return validate_config(
        args.config,
        check_calculator=not args.skip_calculator_check,
    )


def _handle_dryrun(args: argparse.Namespace) -> Any:
    from .api import dryrun

    return dryrun(
        args.config,
        nvt_steps=args.nvt_steps,
        output_dir=args.output_dir,
    )


def _handle_run(args: argparse.Namespace) -> Any:
    from .api import run

    return run(
        args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


def _handle_fork(args: argparse.Namespace) -> Any:
    from .api import fork

    return fork(
        args.config,
        from_run=args.from_run,
        state=args.state,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        check_only=args.check_only,
    )


def _handle_resume(args: argparse.Namespace) -> Any:
    from .api import resume

    return resume(args.run_dir)


def _handle_analyze(args: argparse.Namespace) -> Any:
    from .api import analyze

    return analyze(
        args.run_dir,
        config_path=args.config,
        stages=args.stages,
        all_stages=args.all_stages,
        si_o_cutoff_A=args.si_o_cutoff_A,
        force=args.force,
    )


def _add_json_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print the command result as machine-readable JSON",
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _emit_result(result: Any, *, as_json: bool) -> None:
    serializable = _to_serializable(result)
    if as_json:
        print(json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=False))
        return

    if result is None:
        print("completed successfully")
        return
    if isinstance(result, BaseModel):
        print("configuration is valid")
        return
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(value, (dict, list, tuple)):
                rendered = json.dumps(
                    _to_serializable(value), ensure_ascii=False, sort_keys=True
                )
            else:
                rendered = str(value)
            print(f"{key}: {rendered}")
        return
    print(result)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_to_serializable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return str(value)


__all__ = ["build_parser", "main"]
