"""High-level, restart-safe API for cold-compression MD workflows.

The CLI delegates to this module.  Keeping orchestration here makes it possible
to use the same validated workflow from Python without copying command-line
implementation details::

    from coldmd.api import validate_config, dryrun, run, fork, resume, analyze

No function in this module silently changes a configured physical target or
clips an unsafe trajectory state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import tempfile
import time
from typing import Any, Mapping, Sequence
import warnings as python_warnings

import numpy as np

from .calculators import (
    CalculatorProbeReport,
    CalculatorSpec,
    build_calculator,
    probe_calculator,
    verify_model_sha256,
)
from .config import (
    ColdMDConfig,
    CompressionStage,
    DecompressionStage,
    FinalNVTStage,
    InitialNVTStage,
    NPTStage,
    NPTHoldStage,
    NVTStage,
    NVTHoldStage,
    PeakNVTStage,
    PressurePlateauStage,
    RecoveryNPTStage,
    VolumeRampControlStage,
    dump_config,
    load_config,
)
from .dynamics import initialize_momenta_once
from .engine import (
    FixedCellNVTStage,
    IsotropicNPTStage,
    ProtocolEngine,
    RunCursor,
    StageSpec,
    StepContext,
    VolumeRampStage,
)
from .safety import SafetyLimits, SafetyMonitor
from .storage import (
    CheckpointCompatibilityError,
    CheckpointManager,
    RunOutputHook,
    collect_version_metadata,
    sha256_file,
    sha256_mapping,
)
from .structure import StructureReport, read_cif_structure, validate_atoms


RESOLVED_CONFIG_FILENAME = "resolved-config.yaml"
RUN_MANIFEST_FILENAME = "run-manifest.json"


FOUNDATION_RESUME_WARNING_CODE = "foundation_model_not_strictly_resumable"


@dataclass(frozen=True, slots=True)
class _ForkState:
    """Validated mechanical state and immutable parent-run provenance."""

    atoms: Any
    structure_report: StructureReport
    template_structure_report: StructureReport
    provenance: dict[str, Any]


class FoundationModelResumeWarning(UserWarning):
    """A production run uses a mutable foundation-model alias."""


def configuration_warnings(config: ColdMDConfig) -> list[dict[str, str]]:
    """Return stable, machine-readable warnings for a resolved configuration.

    Foundation aliases are convenient for setup and dry-run validation, but the alias
    text does not identify immutable model bytes. Strict continuation therefore
    remains disabled until the model is supplied as a local, content-hashed
    checkpoint.
    """

    warnings: list[dict[str, str]] = []
    if config.calculator.kind == "mace" and config.calculator.foundation_model:
        alias = config.calculator.foundation_model
        warnings.append(
            {
                "code": FOUNDATION_RESUME_WARNING_CODE,
                "severity": "warning",
                "message": (
                    f"calculator.foundation_model={alias!r} uses a mutable alias; "
                    "strict resume is unavailable for this run. Before a long "
                    "production calculation, save the MACE checkpoint locally and "
                    "configure calculator.model_path. Adding calculator.model_sha256 "
                    "is optional but recommended as an independent expected-hash check."
                ),
            }
        )
    return warnings


def validate_config(
    config_path: str | Path,
    *,
    check_calculator: bool = True,
) -> dict[str, Any]:
    """Validate YAML, CIF, and optionally the configured ASE calculator.

    The calculator probe evaluates finite energy, forces, and stress and checks
    the hydrostatic stress sign against a central finite difference in volume.
    A MACE foundation-model download occurs only when the YAML explicitly sets
    ``allow_download: true``.
    """

    config = load_config(config_path)
    atoms, structure_report = _load_structure(config)
    result: dict[str, Any] = {
        "status": "valid",
        "config_path": config.source_path,
        "protocol_design": (
            "stage_explicit" if config.config_version == 2 else config.protocol.control_mode
        ),
        "total_steps": _total_steps(config),
        "simulated_time_ps": _total_steps(config)
        * config.protocol.timestep_fs
        / 1000.0,
        "structure": asdict(structure_report),
        "calculator_checked": bool(check_calculator),
        "resolved_stage_count": len(config.protocol.stages),
        "warnings": configuration_warnings(config),
    }
    model_hash, model_hash_kind = _calculator_identity(config)
    result["model_sha256"] = model_hash
    result["model_identity_kind"] = model_hash_kind
    result["strict_resume_available"] = _strict_resume_available(
        config,
        model_hash_kind,
    )
    if check_calculator:
        calculator, report = _construct_and_probe_calculator(config, atoms)
        # Do not leave a heavyweight calculator referenced after validation.
        atoms.calc = None
        del calculator
        result["calculator"] = asdict(report)
    return result


def dryrun(
    config_path: str | Path,
    *,
    nvt_steps: int = 100,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Perform a safety-neutral preflight with a fixed-cell NVT smoke run.

    This deliberately does *not* manufacture a high-pressure or small-volume
    representative state.  It verifies calculator properties and every stage
    integrator can be constructed, then advances only a fixed-cell NVT copy of
    the input structure.  It is a program/readiness check, not a wall-time
    estimate and not scientific equilibration.
    """

    requested_steps = int(nvt_steps)
    if requested_steps < 1:
        raise ValueError("dryrun nvt_steps must be at least one")

    config = load_config(config_path)
    atoms, structure_report = _load_structure(config)
    calculator, calculator_report = _construct_and_probe_calculator(config, atoms)
    atoms.calc = calculator
    rng = np.random.default_rng(config.protocol.random_seed)
    initialize_momenta_once(
        atoms,
        config.protocol.temperature_K,
        rng,
        remove_com=config.protocol.remove_center_of_mass_momentum,
    )
    reference_volume = float(atoms.get_volume())
    stages = _build_stage_specs(config)

    # Constructor coverage catches mismatched ASE APIs without allowing a
    # synthetic high-pressure state to evolve.  Each constructor receives a
    # fresh mechanical copy so it cannot affect the later smoke trajectory.
    constructed_kinds: list[str] = []
    for stage in stages:
        trial = atoms.copy()
        trial.calc = calculator
        trial_engine = ProtocolEngine(
            trial,
            timestep_fs=config.protocol.timestep_fs,
            rng=np.random.default_rng(config.protocol.random_seed),
            reference_volume_A3=reference_volume,
        )
        trial_engine._build_dynamics(  # noqa: SLF001 - controlled preflight probe
            stage,
            completed_steps=0,
            stage_reference_cell_A=tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(trial.get_cell(), dtype=float)
            ),
        )
        constructed_kinds.append(stage.kind)

    thermostat = config.protocol.thermostat
    smoke_stage = FixedCellNVTStage(
        name="dryrun_fixed_cell_nvt",
        steps=requested_steps,
        temperature_K=config.protocol.temperature_K,
        thermostat_tau_fs=thermostat.tau_fs,
        branch="other",
        thermostat_kind=thermostat.kind,
        thermostat_chain_length=thermostat.chain_length,
        thermostat_substeps=thermostat.substeps,
    )
    config_hash = _config_hash(config)
    model_hash, model_hash_kind = _calculator_identity(config)
    with tempfile.TemporaryDirectory(prefix="coldmd-dryrun-") as temporary:
        hook = RunOutputHook(
            temporary,
            trajectory_filename=config.output.trajectory_filename,
            thermo_filename=config.output.thermo_filename,
            trajectory_interval_steps=max(1, config.output.trajectory_interval_steps),
            thermo_interval_steps=max(1, config.output.thermo_interval_steps),
            msd_interval_steps=config.output.msd_interval_steps,
            checkpoint_interval_steps=requested_steps + 1,
            safety_monitor=_build_safety_monitor(config, reference_volume),
            rng=rng,
            config_hash=config_hash,
            model_hash=model_hash,
            reference_volume_A3=reference_volume,
            append=False,
            write_forces=config.output.write_forces,
            write_stress=config.output.write_stress,
            write_momenta=config.output.write_momenta,
            write_stage_structures=False,
            checkpoint_on_abort=True,
            provenance_metadata={"purpose": "fixed_cell_nvt_dryrun"},
            versions=collect_version_metadata(),
        )
        engine = ProtocolEngine(
            atoms,
            timestep_fs=config.protocol.timestep_fs,
            rng=rng,
            hooks=(hook,),
            reference_volume_A3=reference_volume,
        )
        with hook:
            cursor = engine.run([smoke_stage])

    result: dict[str, Any] = {
        "status": "passed",
        "dryrun_nvt_steps": requested_steps,
        "dryrun_state": "input_cell_fixed_nvt",
        "production_total_steps": _total_steps(config),
        "resolved_stage_count": len(stages),
        "constructed_stage_kinds": sorted(set(constructed_kinds)),
        "final_global_step": cursor.global_step,
        "structure": asdict(structure_report),
        "calculator": asdict(calculator_report),
        "strict_resume_available": _strict_resume_available(config, model_hash_kind),
        "warnings": configuration_warnings(config),
        "note": (
            "Dry run verifies configuration, calculator interfaces, integrator "
            "construction, and a fixed-cell NVT smoke trajectory. It does not "
            "estimate production wall time or establish scientific stability."
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "dryrun.json"
        _atomic_json(report_path, result)
        result["report"] = report_path
    atoms.calc = None
    return result


def benchmark(
    config_path: str | Path,
    *,
    steps: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Deprecated compatibility alias for a safety-neutral :func:`dryrun`.

    It deliberately returns no throughput or production-duration estimate.
    """

    python_warnings.warn(
        "benchmark() was replaced by dryrun(); no production-time estimate is produced.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dryrun(
        config_path,
        nvt_steps=100 if steps is None else int(steps),
        output_dir=output_dir,
    )


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Start a fresh validated protocol and return its completion summary."""

    config = load_config(config_path)
    for warning in configuration_warnings(config):
        python_warnings.warn(
            f"[{warning['code']}] {warning['message']}",
            FoundationModelResumeWarning,
            stacklevel=2,
        )
    destination = (
        config.output.directory
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    with _RunDirectoryLock(Path(destination)):
        return _run_locked(config, Path(destination), overwrite=overwrite)


def fork(
    config_path: str | Path,
    *,
    from_run: str | Path,
    state: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    """Start a new protocol from a parent run's stage EXTXYZ snapshot.

    Fork is deliberately distinct from :func:`resume`.  Only cell, positions,
    PBC, atomic order, and momenta are inherited.  The child uses its own YAML,
    random-number seed, calculator, integrator state, output directory, and a
    fresh ``step=0``, ``time=0``, ``V/V0=1`` run coordinate system. Parent
    cursor/time/reference-volume values are retained only as provenance.
    """

    config = load_config(config_path)
    parent = Path(from_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"parent run directory does not exist: {parent}")
    destination = (
        config.output.directory
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    destination = Path(destination).expanduser().resolve()
    _assert_separate_fork_directories(parent, destination)

    # A published stage snapshot is immutable. Fork never acquires a writable
    # parent lock or modifies the parent tree; repeated hashes below detect the
    # unlikely case of an externally edited source during validation.
    fork_state = _load_fork_state(config, parent, state)

    if check_only:
        calculator, calculator_report = _construct_and_probe_calculator(
            config, fork_state.atoms
        )
        fork_state.atoms.calc = calculator
        constructed_kinds: list[str] = []
        reference_volume = float(fork_state.atoms.get_volume())
        for stage_spec in _build_stage_specs(config):
            trial = fork_state.atoms.copy()
            trial.calc = calculator
            trial_engine = ProtocolEngine(
                trial,
                timestep_fs=config.protocol.timestep_fs,
                rng=np.random.default_rng(config.protocol.random_seed),
                reference_volume_A3=reference_volume,
            )
            trial_engine._build_dynamics(  # noqa: SLF001 - preflight probe
                stage_spec,
                completed_steps=0,
                stage_reference_cell_A=tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(trial.get_cell(), dtype=float)
                ),
            )
            constructed_kinds.append(stage_spec.kind)
        fork_state.atoms.calc = None
        del calculator
        model_hash, model_hash_kind = _calculator_identity(config)
        return {
            "status": "valid",
            "fork_ready": True,
            "config_path": config.source_path,
            "total_steps": _total_steps(config),
            "simulated_time_ps": (
                _total_steps(config) * config.protocol.timestep_fs / 1000.0
            ),
            "structure": asdict(fork_state.structure_report),
            "calculator": asdict(calculator_report),
            "constructed_stage_kinds": sorted(set(constructed_kinds)),
            "model_sha256": model_hash,
            "model_identity_kind": model_hash_kind,
            "strict_resume_available": _strict_resume_available(
                config, model_hash_kind
            ),
            "warnings": configuration_warnings(config),
            "fork_provenance": fork_state.provenance,
            "child_origin": {
                "global_step": 0,
                "time_fs": 0.0,
                "volume_ratio_from_engine_start": 1.0,
            },
        }

    for warning in configuration_warnings(config):
        python_warnings.warn(
            f"[{warning['code']}] {warning['message']}",
            FoundationModelResumeWarning,
            stacklevel=2,
        )
    with _RunDirectoryLock(destination):
        return _run_locked(
            config,
            destination,
            overwrite=overwrite,
            fork_state=fork_state,
        )


def _run_locked(
    config: ColdMDConfig,
    destination: Path,
    *,
    overwrite: bool,
    fork_state: _ForkState | None = None,
) -> dict[str, Any]:
    """Fresh-run implementation while the stable output lock is held."""

    config = _with_output_directory(config, destination)
    _assert_fresh_output_policy(destination, overwrite=overwrite)

    # Complete expensive preflight work before moving a pre-existing output to
    # its recoverable backup.  A bad CIF/model therefore leaves prior user data
    # and the requested destination untouched.
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    if fork_state is None:
        atoms, structure_report = _load_structure(config)
        template_structure_report = structure_report
    else:
        atoms = fork_state.atoms.copy()
        atoms.calc = None
        structure_report = fork_state.structure_report
        template_structure_report = fork_state.template_structure_report
    calculator, calculator_report = _construct_and_probe_calculator(config, atoms)
    atoms.calc = calculator

    rng = np.random.default_rng(config.protocol.random_seed)
    if fork_state is not None:
        _require_fork_momenta(atoms)
    elif _initialize_velocities_for_fresh_run(config):
        initialize_momenta_once(
            atoms,
            config.protocol.temperature_K,
            rng,
            remove_com=config.protocol.remove_center_of_mass_momentum,
        )
    elif float(atoms.get_kinetic_energy()) <= 0.0:
        raise ValueError(
            "velocity initialization is disabled but the input structure contains "
            "no non-zero momenta; a fresh thermostatted run cannot start"
        )

    stages = _build_stage_specs(config)
    config_hash = _config_hash(config)
    model_hash, model_hash_kind = _calculator_identity(config)
    config_warnings = configuration_warnings(config)
    input_hash = sha256_file(config.input.cif_path)
    versions = _collect_execution_versions(config)
    reference_volume = float(atoms.get_volume())
    safety_monitor = _build_safety_monitor(config, reference_volume)
    provenance = _provenance(
        config,
        structure_report,
        input_hash=input_hash,
        model_hash_kind=model_hash_kind,
        reference_volume_A3=reference_volume,
    )
    if fork_state is not None:
        provenance["fork"] = fork_state.provenance
    backup = _prepare_fresh_output_directory(destination, overwrite=overwrite)
    resolved_path = dump_config(config, destination / RESOLVED_CONFIG_FILENAME)
    manifest_path = destination / RUN_MANIFEST_FILENAME
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_mode": "fork" if fork_state is not None else "fresh",
        "status": "running",
        "started_utc": started_utc,
        "completed_utc": None,
        "config_path": resolved_path,
        "config_sha256": config_hash,
        "input_cif_sha256": input_hash,
        "model_sha256": model_hash,
        "model_identity_kind": model_hash_kind,
        "strict_resume_available": _strict_resume_available(
            config,
            model_hash_kind,
        ),
        "warnings": config_warnings,
        "protocol_design": (
            "stage_explicit" if config.config_version == 2 else config.protocol.control_mode
        ),
        "total_steps": _total_steps(config),
        "versions": versions,
        "structure": asdict(structure_report),
        "input_template_structure": asdict(template_structure_report),
        "calculator_probe": asdict(calculator_report),
        "backup_of_previous_output": backup,
    }
    if fork_state is not None:
        manifest["fork_provenance"] = fork_state.provenance
    _atomic_json(manifest_path, manifest)

    event_logger: _EventLogger | None = None
    output: RunOutputHook | None = None
    engine: ProtocolEngine | None = None
    try:
        event_logger = _EventLogger(destination / config.output.log_filename)
        for warning in config_warnings:
            event_logger("configuration_warning", warning)
        if fork_state is not None:
            event_logger(
                "fork_start",
                {
                    "fork_provenance": fork_state.provenance,
                    "child_origin": {
                        "global_step": 0,
                        "time_fs": 0.0,
                        "volume_ratio_from_engine_start": 1.0,
                    },
                },
            )
        engine = ProtocolEngine(
            atoms,
            timestep_fs=config.protocol.timestep_fs,
            rng=rng,
            reference_volume_A3=reference_volume,
        )
        output = RunOutputHook(
            destination,
            trajectory_filename=config.output.trajectory_filename,
            thermo_filename=config.output.thermo_filename,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            thermo_interval_steps=config.output.thermo_interval_steps,
            msd_interval_steps=config.output.msd_interval_steps,
            checkpoint_interval_steps=config.output.checkpoint_interval_steps,
            safety_monitor=safety_monitor,
            rng=rng,
            config_hash=config_hash,
            model_hash=model_hash,
            reference_volume_A3=reference_volume,
            append=False,
            write_forces=config.output.write_forces,
            write_stress=config.output.write_stress,
            write_momenta=config.output.write_momenta,
            write_stage_structures=config.output.write_stage_structures,
            checkpoint_on_abort=config.safety.checkpoint_on_abort,
            provenance_metadata=provenance,
            versions=versions,
            event_callback=event_logger,
        )
        engine.hooks = (output,)
        with output:
            cursor = engine.run(stages)
    except BaseException as exc:
        if output is not None and engine is not None:
            _attempt_failure_checkpoint(output, atoms, engine.cursor, exc)
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if event_logger is not None:
            event_logger.close()

    manifest.update(
        {
            "status": "completed",
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "cursor": cursor.as_dict(),
        }
    )
    _atomic_json(manifest_path, manifest)
    result = {
        "status": "completed",
        "output_directory": destination,
        "manifest": manifest_path,
        "final_global_step": cursor.global_step,
        "simulated_time_ps": cursor.time_fs / 1000.0,
        "backup_of_previous_output": backup,
        "warnings": config_warnings,
    }
    if fork_state is not None:
        result["fork_provenance"] = fork_state.provenance
    return result


def resume(run_dir: str | Path) -> dict[str, Any]:
    """Resume the latest exact checkpoint in an existing run directory.

    Interior Nosé–Hoover NVT and MTK thermostat/barostat-chain states are never
    advertised as exact checkpoints; those stages therefore resume from the
    latest completed stage boundary.
    """

    destination = Path(run_dir).expanduser().resolve()
    if not destination.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {destination}")
    with _RunDirectoryLock(destination):
        try:
            return _resume_locked(destination)
        except BaseException as exc:
            _record_resume_failure(destination, exc)
            raise


def _resume_locked(destination: Path) -> dict[str, Any]:
    """Implementation of :func:`resume` while the run lock is held."""

    from .reconcile import reconcile_outputs_to_checkpoint

    started = time.perf_counter()
    config_path = destination / RESOLVED_CONFIG_FILENAME
    config = load_config(config_path)
    config_hash = _config_hash(config)
    model_hash, model_hash_kind = _calculator_identity(config)
    if not _strict_resume_available(config, model_hash_kind):
        raise CheckpointCompatibilityError(
            "strict resume currently requires calculator.kind='mace' with a local, "
            "content-hashed model_path. Foundation aliases and arbitrary Python "
            "factories can change their executable implementation without changing "
            "the stored text configuration. Start a new run with a local MACE "
            "checkpoint."
        )
    versions = _collect_execution_versions(config)
    manager = CheckpointManager(destination / "checkpoints", basename="checkpoint")
    checkpoint = manager.load(
        expected_config_hash=config_hash,
        expected_model_hash=model_hash,
        expected_versions=_restart_version_requirements(versions),
        require_exact_version_set=True,
    )
    if not bool(checkpoint.protocol_state.get("exact_resume_supported", False)):
        raise CheckpointCompatibilityError(
            "latest checkpoint is not marked as an exact restart point"
        )
    cursor = RunCursor.from_mapping(checkpoint.cursor_state)
    reference_volume = checkpoint.reference_volume_A3
    if cursor.reference_volume_A3 is None or not math.isclose(
        float(cursor.reference_volume_A3),
        reference_volume,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise CheckpointCompatibilityError(
            "checkpoint cursor and protocol reference volumes disagree"
        )

    expected_input_hash = checkpoint.metadata.get("input_cif_sha256")
    if not isinstance(expected_input_hash, str) or len(expected_input_hash) != 64:
        raise CheckpointCompatibilityError(
            "checkpoint lacks the original CIF content hash required for resume"
        )
    observed_input_hash = sha256_file(config.input.cif_path)
    if observed_input_hash != expected_input_hash:
        raise CheckpointCompatibilityError(
            "input CIF content differs from the file used by the checkpoint"
        )

    # Reload the validated CIF to retain tags, custom masses/charges, and
    # Atoms.info required by generic calculators, then replace only the saved
    # mechanical state after strict atomic-number/order validation.
    atoms, structure_report = _load_structure(config)
    checkpoint.apply_to(atoms, strict_atomic_numbers=True, require_momenta=True)
    calculator = build_calculator(
        CalculatorSpec.from_mapping(config.calculator.to_spec_mapping())
    )
    probe_report = probe_calculator(
        calculator,
        atoms,
        required_properties=config.calculator.required_properties,
        check_stress_finite_difference=False,
    )
    atoms.calc = calculator
    rng = np.random.default_rng(config.protocol.random_seed)
    checkpoint.restore_rng(rng)

    trajectory_properties = ["energy"]
    if config.output.write_forces:
        trajectory_properties.append("forces")
    if config.output.write_stress:
        trajectory_properties.append("stress")
    reconciliation = reconcile_outputs_to_checkpoint(
        destination,
        checkpoint,
        trajectory_filename=config.output.trajectory_filename,
        thermo_filename=config.output.thermo_filename,
        event_filename=config.output.log_filename,
        trajectory_properties=trajectory_properties,
    )

    stages = _build_stage_specs(config)
    manifest_path = destination / RUN_MANIFEST_FILENAME
    manifest = _read_json_mapping(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "status": "resume_preparing",
            "last_resume_utc": _utc_now(),
            "last_resume_checkpoint": checkpoint.checkpoint_id,
            "last_resume_reconciliation": reconciliation.as_dict(),
            "last_resume_structure": asdict(structure_report),
            "last_resume_calculator_probe": asdict(probe_report),
        }
    )
    _atomic_json(manifest_path, manifest)

    if cursor.stage_index == len(stages):
        repaired_boundary = False
        if (
            reconciliation.last_thermo_step != cursor.global_step
            or reconciliation.last_trajectory_step != cursor.global_step
        ):
            _repair_completed_boundary(
                destination=destination,
                config=config,
                checkpoint=checkpoint,
                atoms=atoms,
                rng=rng,
                stages=stages,
                cursor=cursor,
                reference_volume_A3=reference_volume,
                config_hash=config_hash,
                model_hash=model_hash,
                versions=versions,
                reconciliation=reconciliation,
            )
            repaired_boundary = True
        manifest.update(
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "last_resume_elapsed_seconds": time.perf_counter() - started,
                "cursor": cursor.as_dict(),
                "completed_boundary_repaired": repaired_boundary,
            }
        )
        _atomic_json(manifest_path, manifest)
        if reconciliation.changed:
            logger = _EventLogger(
                destination / config.output.log_filename, append=True
            )
            try:
                logger("resume_reconciliation", reconciliation.as_dict())
            finally:
                logger.close()
        return {
            "status": "already_completed",
            "output_directory": destination,
            "manifest": manifest_path,
            "final_global_step": cursor.global_step,
            "simulated_time_ps": cursor.time_fs / 1000.0,
            "reconciliation": reconciliation.as_dict(),
            "completed_boundary_repaired": repaired_boundary,
        }

    safety_monitor = _build_safety_monitor(config, reference_volume)
    event_logger: _EventLogger | None = None
    output: RunOutputHook | None = None
    engine: ProtocolEngine | None = None
    try:
        event_logger = _EventLogger(
            destination / config.output.log_filename, append=True
        )
        event_logger(
            "resume_reconciliation" if reconciliation.changed else "resume_start",
            reconciliation.as_dict(),
        )
        engine = ProtocolEngine(
            atoms,
            timestep_fs=config.protocol.timestep_fs,
            rng=rng,
            reference_volume_A3=reference_volume,
        )
        output = RunOutputHook(
            destination,
            trajectory_filename=config.output.trajectory_filename,
            thermo_filename=config.output.thermo_filename,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            thermo_interval_steps=config.output.thermo_interval_steps,
            msd_interval_steps=config.output.msd_interval_steps,
            checkpoint_interval_steps=config.output.checkpoint_interval_steps,
            safety_monitor=safety_monitor,
            rng=rng,
            config_hash=config_hash,
            model_hash=model_hash,
            reference_volume_A3=reference_volume,
            append=True,
            write_forces=config.output.write_forces,
            write_stress=config.output.write_stress,
            write_momenta=config.output.write_momenta,
            write_stage_structures=config.output.write_stage_structures,
            checkpoint_on_abort=config.safety.checkpoint_on_abort,
            provenance_metadata={
                "reference_volume_A3": reference_volume,
                "model_identity_kind": model_hash_kind,
                "resumed_from_checkpoint": checkpoint.checkpoint_id,
                "output_reconciliation": reconciliation.as_dict(),
            },
            versions=versions,
            event_callback=event_logger,
        )
        output.prime_resume_from_checkpoint(
            checkpoint,
            last_thermo_step=reconciliation.last_thermo_step,
            last_trajectory_step=reconciliation.last_trajectory_step,
        )
        engine.hooks = (output,)
        manifest["status"] = "resuming"
        _atomic_json(manifest_path, manifest)
        with output:
            final_cursor = engine.run(stages, cursor=cursor)
    except BaseException as exc:
        if output is not None and engine is not None:
            _attempt_failure_checkpoint(output, atoms, engine.cursor, exc)
        manifest.update(
            {
                "status": (
                    "resume_interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "resume_failed"
                ),
                "last_resume_elapsed_seconds": time.perf_counter() - started,
                "last_resume_error_type": type(exc).__name__,
                "last_resume_error": str(exc),
                "cursor_at_resume_failure": (
                    None if engine is None else engine.cursor.as_dict()
                ),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if event_logger is not None:
            event_logger.close()

    manifest.update(
        {
            "status": "completed",
            "completed_utc": _utc_now(),
            "last_resume_elapsed_seconds": time.perf_counter() - started,
            "cursor": final_cursor.as_dict(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return {
        "status": "completed",
        "output_directory": destination,
        "manifest": manifest_path,
        "resumed_from_checkpoint": checkpoint.checkpoint_id,
        "final_global_step": final_cursor.global_step,
        "simulated_time_ps": final_cursor.time_fs / 1000.0,
        "reconciliation": reconciliation.as_dict(),
    }


def analyze(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    stages: Sequence[str] | None = None,
    all_stages: bool = False,
    si_o_cutoff_A: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate structural, transport, and hysteresis diagnostics."""

    if stages and all_stages:
        raise ValueError("stages and all_stages are mutually exclusive")
    if si_o_cutoff_A is not None and (
        not math.isfinite(float(si_o_cutoff_A)) or float(si_o_cutoff_A) <= 0.0
    ):
        raise ValueError("si_o_cutoff_A must be a finite positive distance")

    from .reporting import generate_analysis_report

    result = generate_analysis_report(
        run_dir,
        config=config_path,
        stages=stages,
        all_stages=all_stages,
        si_o_cutoff_A=si_o_cutoff_A,
        force=force,
    )
    return {
        "status": "completed",
        "analysis_directory": result["analysis_dir"],
        "report": result["report"],
        "json": result["json"],
        "artifact_count": len(result["artifacts"]),
        "warnings": list(result["warnings"]),
    }


def _load_structure(config: ColdMDConfig) -> tuple[Any, StructureReport]:
    return read_cif_structure(
        config.input.cif_path,
        block=config.input.cif_frame,
        supercell=config.input.supercell,
        allow_fractional_occupancy=not config.input.reject_partial_occupancy,
        expected_atom_count=config.input.expected_atom_count,
        expected_formula=config.input.expected_formula,
        minimum_distance_a=config.safety.min_distance_A,
    )


_STAGE_SNAPSHOT_RE = re.compile(
    r"^stage-(?P<stage_index>\d{3})-(?P<stage_name>.+)-"
    r"(?P<event>start|end|handoff)-step-(?P<boundary_step>\d{12})"
    r"(?:-resume-(?P<resume_index>\d{3}))?\.extxyz$"
)


def _assert_separate_fork_directories(parent: Path, child: Path) -> None:
    """Reject overlapping parent/child trees before any output mutation."""

    if parent == child or parent in child.parents or child in parent.parents:
        raise ValueError(
            "fork parent and child output directories must be separate, non-nested "
            f"paths (parent={parent}, child={child})"
        )


def _resolve_fork_snapshot(
    parent: Path, state: str | Path
) -> tuple[Path, re.Match[str]]:
    stage_directory = (parent / "stage_structures").resolve()
    if not stage_directory.is_dir():
        raise FileNotFoundError(
            f"parent run has no stage_structures directory: {stage_directory}"
        )
    requested = Path(state).expanduser()
    candidate = requested if requested.is_absolute() else stage_directory / requested
    snapshot = candidate.resolve()
    if snapshot.parent != stage_directory:
        raise ValueError(
            "fork --state must name a direct child of the parent run's "
            f"stage_structures directory: {stage_directory}"
        )
    if not snapshot.is_file():
        raise FileNotFoundError(f"fork stage snapshot does not exist: {snapshot}")
    match = _STAGE_SNAPSHOT_RE.fullmatch(snapshot.name)
    if match is None:
        raise ValueError(
            "fork --state must be a canonical ColdMD stage snapshot named "
            "stage-...-(start|end|handoff)-step-............extxyz"
        )
    return snapshot, match


def _require_fork_momenta(atoms: Any) -> np.ndarray:
    """Return the finite momentum array explicitly stored in a fork state."""

    has_momenta = bool(getattr(atoms, "has", lambda _name: False)("momenta"))
    if not has_momenta:
        raise CheckpointCompatibilityError(
            "fork source EXTXYZ has no momenta; CIF and position-only files cannot "
            "define a mechanically continuous branch"
        )
    momenta = np.asarray(atoms.get_momenta(), dtype=float)
    if momenta.shape != (len(atoms), 3):
        raise CheckpointCompatibilityError(
            f"fork source momenta have shape {momenta.shape}; expected ({len(atoms)}, 3)"
        )
    if not np.isfinite(momenta).all():
        raise CheckpointCompatibilityError("fork source contains non-finite momenta")
    return momenta


def _fork_event_metadata(
    parent: Path,
    log_filename: str,
    snapshot: Path,
    event: str,
) -> dict[str, Any]:
    """Find the event that published a stage snapshot, when the log is present."""

    log_path = parent / log_filename
    if not log_path.is_file():
        return {}
    expected_event = f"stage_{event}"
    matched: dict[str, Any] = {}
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping) or row.get("event") != expected_event:
                continue
            candidates: list[Any] = [row.get("structure")]
            structures = row.get("structures")
            if isinstance(structures, Mapping):
                candidates.extend(structures.values())
            if any(
                isinstance(value, str)
                and Path(value).name == snapshot.name
                for value in candidates
            ):
                matched = dict(row)
    return matched


def _load_fork_state(
    config: ColdMDConfig,
    parent: Path,
    state: str | Path,
) -> _ForkState:
    """Load one canonical parent snapshot onto the child CIF identity template."""

    manifest_path = parent / RUN_MANIFEST_FILENAME
    resolved_config_path = parent / RESOLVED_CONFIG_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"parent run manifest is missing: {manifest_path}")
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"parent resolved configuration is missing: {resolved_config_path}"
        )
    parent_manifest_hash = sha256_file(manifest_path)
    parent_config_file_hash = sha256_file(resolved_config_path)
    parent_manifest = _read_json_mapping(manifest_path)
    parent_config = load_config(resolved_config_path)
    observed_parent_config_hash = _config_hash(parent_config)
    recorded_parent_config_hash = parent_manifest.get("config_sha256")
    if (
        recorded_parent_config_hash is not None
        and recorded_parent_config_hash != observed_parent_config_hash
    ):
        raise CheckpointCompatibilityError(
            "parent resolved configuration no longer matches run-manifest.json"
        )

    snapshot_path, filename_match = _resolve_fork_snapshot(parent, state)
    snapshot_hash = sha256_file(snapshot_path)
    try:
        from ase.io import read as ase_read
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("ASE is required to read fork stage snapshots") from exc
    frames = ase_read(
        str(snapshot_path),
        index=":",
        format="extxyz",
        parallel=False,
    )
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise CheckpointCompatibilityError(
            f"fork source must contain exactly one EXTXYZ frame; found {len(frames)}"
        )
    source_atoms = frames[0]
    if sha256_file(snapshot_path) != snapshot_hash:
        raise CheckpointCompatibilityError(
            "fork source changed while it was being read; retry with an immutable "
            "published stage snapshot"
        )
    if sha256_file(manifest_path) != parent_manifest_hash:
        raise CheckpointCompatibilityError(
            "parent run manifest changed during fork validation; retry after the "
            "current parent update completes"
        )
    if sha256_file(resolved_config_path) != parent_config_file_hash:
        raise CheckpointCompatibilityError(
            "parent resolved configuration changed during fork validation"
        )
    momenta = _require_fork_momenta(source_atoms)
    validate_atoms(
        source_atoms,
        source=snapshot_path,
        expected_atom_count=config.input.expected_atom_count,
        expected_formula=config.input.expected_formula,
        minimum_distance_a=config.safety.min_distance_A,
    )

    template, template_report = _load_structure(config)
    template_numbers = np.asarray(template.get_atomic_numbers(), dtype=np.int64)
    source_numbers = np.asarray(source_atoms.get_atomic_numbers(), dtype=np.int64)
    if not np.array_equal(template_numbers, source_numbers):
        raise CheckpointCompatibilityError(
            "fork snapshot atomic numbers/order do not match the new YAML's CIF "
            "identity template"
        )
    template.set_cell(np.asarray(source_atoms.get_cell(), dtype=float), scale_atoms=False)
    template.set_pbc(np.asarray(source_atoms.get_pbc(), dtype=bool))
    template.set_positions(np.asarray(source_atoms.get_positions(), dtype=float))
    template.set_momenta(momenta.copy())
    structure_report = validate_atoms(
        template,
        source=snapshot_path,
        expected_atom_count=config.input.expected_atom_count,
        expected_formula=config.input.expected_formula,
        minimum_distance_a=config.safety.min_distance_A,
    )

    filename = filename_match.groupdict()
    boundary_step = int(filename["boundary_step"])
    event_row = _fork_event_metadata(
        parent,
        parent_config.output.log_filename,
        snapshot_path,
        filename["event"],
    )
    selected_step_value = event_row.get("selected_global_step")
    selected_step = (
        int(selected_step_value)
        if isinstance(selected_step_value, (int, np.integer))
        and not isinstance(selected_step_value, (bool, np.bool_))
        and int(selected_step_value) >= 0
        else None
    )
    mechanical_step = selected_step if selected_step is not None else boundary_step
    source_temperature = float(template.get_temperature())
    if not math.isfinite(source_temperature):
        raise CheckpointCompatibilityError(
            "fork source momenta produce a non-finite temperature"
        )
    provenance: dict[str, Any] = {
        "mode": "stage_snapshot",
        "parent_run": str(parent),
        "parent_manifest": str(manifest_path),
        "parent_manifest_sha256": parent_manifest_hash,
        "parent_manifest_status": parent_manifest.get("status"),
        "parent_config": str(resolved_config_path),
        "parent_config_sha256": observed_parent_config_hash,
        "parent_config_file_sha256": parent_config_file_hash,
        "source_state": str(snapshot_path),
        "source_state_sha256": snapshot_hash,
        "source_stage_index": int(filename["stage_index"]),
        "source_stage_name": event_row.get("stage", filename["stage_name"]),
        "source_event": filename["event"],
        "source_boundary_global_step": boundary_step,
        "source_selected_global_step": selected_step,
        "source_mechanical_state_global_step": mechanical_step,
        "source_mechanical_state_time_ps": (
            mechanical_step * parent_config.protocol.timestep_fs / 1000.0
        ),
        "source_volume_A3": float(template.get_volume()),
        "source_temperature_K": source_temperature,
        "source_momentum_norm_ase_units": float(np.linalg.norm(momenta)),
        "source_net_momentum_norm_ase_units": float(
            np.linalg.norm(np.sum(momenta, axis=0))
        ),
        "inheritance": ["atomic_numbers", "cell", "pbc", "positions", "momenta"],
        "reset_in_child": [
            "global_step",
            "time_fs",
            "volume_ratio_from_engine_start",
            "rng_state",
            "integrator_state",
            "safety_history",
        ],
    }
    return _ForkState(
        atoms=template,
        structure_report=structure_report,
        template_structure_report=template_report,
        provenance=provenance,
    )


def _construct_and_probe_calculator(
    config: ColdMDConfig,
    atoms: Any,
) -> tuple[Any, CalculatorProbeReport]:
    specification = CalculatorSpec.from_mapping(config.calculator.to_spec_mapping())
    calculator = build_calculator(specification)
    report = probe_calculator(
        calculator,
        atoms,
        required_properties=config.calculator.required_properties,
        check_supported_elements=True,
        check_stress_finite_difference=True,
    )
    return calculator, report


def _calculator_identity(config: ColdMDConfig) -> tuple[str, str]:
    if config.calculator.model_path is not None:
        return (
            verify_model_sha256(
                config.calculator.model_path,
                config.calculator.model_sha256,
            ),
            "checkpoint_content_sha256",
        )
    # A generic Python factory or downloaded foundation alias does not expose a
    # portable checkpoint path.  Hashing the fully resolved specification still
    # prevents accidental config drift, but production runs should use a local,
    # content-addressed model_path whenever the backend supports one.
    return (
        sha256_mapping({"calculator": config.calculator.model_dump(mode="json")}),
        "calculator_spec_sha256",
    )


def _strict_resume_available(
    config: ColdMDConfig,
    model_hash_kind: str,
) -> bool:
    """Return whether strict resume can substantiate calculator identity.

    A generic import factory can change when its Python package or source changes,
    even if a separate data file and YAML remain byte-identical.  Until factory
    source/distribution provenance is part of the checkpoint contract, only a
    local content-addressed MACE checkpoint is admitted for strict continuation.
    """

    return bool(
        config.calculator.kind == "mace"
        and config.calculator.model_path is not None
        and model_hash_kind == "checkpoint_content_sha256"
    )


def _build_stage_specs(config: ColdMDConfig) -> list[StageSpec]:
    result: list[StageSpec] = []
    protocol = config.protocol
    for stage in protocol.stages:
        steps = _steps_for_duration(stage.duration_ps, protocol.timestep_fs)
        handoff_window_steps = (
            _steps_for_duration(stage.sampling_window_ps, protocol.timestep_fs)
            if isinstance(stage, (NVTHoldStage, NPTHoldStage, VolumeRampControlStage))
            else None
        )
        handoff_sample_interval_steps = config.output.thermo_interval_steps
        if isinstance(stage, (NVTStage, NVTHoldStage)):
            result.append(
                FixedCellNVTStage(
                    name=stage.name,
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    thermostat_tau_fs=protocol.thermostat.tau_fs,
                    branch=(
                        _engine_branch(stage.branch)
                        if isinstance(stage, NVTHoldStage)
                        else stage.branch
                    ),
                    thermostat_kind=protocol.thermostat.kind,
                    thermostat_chain_length=protocol.thermostat.chain_length,
                    thermostat_substeps=protocol.thermostat.substeps,
                    handoff_window_steps=handoff_window_steps,
                    handoff_sample_interval_steps=handoff_sample_interval_steps,
                )
            )
        elif isinstance(stage, (NPTStage, NPTHoldStage)):
            barostat = protocol.barostat
            result.append(
                IsotropicNPTStage(
                    name=stage.name,
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    pressure_GPa=stage.target_pressure_GPa,
                    thermostat_tau_fs=barostat.thermostat_tau_fs,
                    barostat_tau_fs=barostat.barostat_tau_fs,
                    branch=(
                        _engine_branch(stage.branch)
                        if isinstance(stage, NPTHoldStage)
                        else stage.branch
                    ),
                    thermostat_chain_length=barostat.thermostat_chain_length,
                    barostat_chain_length=barostat.barostat_chain_length,
                    thermostat_substeps=barostat.thermostat_substeps,
                    barostat_substeps=barostat.barostat_substeps,
                    handoff_window_steps=handoff_window_steps,
                    handoff_sample_interval_steps=handoff_sample_interval_steps,
                )
            )
        elif isinstance(stage, (InitialNVTStage, PeakNVTStage, FinalNVTStage)):
            hold_branch = (
                "initial_hold"
                if isinstance(stage, InitialNVTStage)
                else (
                    "high_density_hold"
                    if isinstance(stage, PeakNVTStage)
                    else "recovered_hold"
                )
            )
            result.append(
                FixedCellNVTStage(
                    name=stage.name,
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    thermostat_tau_fs=protocol.thermostat.tau_fs,
                    branch=hold_branch,
                    thermostat_kind=protocol.thermostat.kind,
                    thermostat_chain_length=protocol.thermostat.chain_length,
                    thermostat_substeps=protocol.thermostat.substeps,
                )
            )
        elif isinstance(stage, CompressionStage):
            result.append(
                VolumeRampStage(
                    name=stage.name,
                    role="cold_compression",
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    thermostat_tau_fs=protocol.thermostat.tau_fs,
                    target_volume_ratio=stage.target_volume_ratio,
                    schedule=stage.schedule,
                )
            )
        elif isinstance(stage, DecompressionStage):
            result.append(
                VolumeRampStage(
                    name=stage.name,
                    role="decompression",
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    thermostat_tau_fs=protocol.thermostat.tau_fs,
                    target_volume_ratio=stage.target_volume_ratio,
                    schedule=stage.schedule,
                )
            )
        elif isinstance(stage, VolumeRampControlStage):
            result.append(
                VolumeRampStage(
                    name=stage.name,
                    role=(
                        "cold_compression"
                        if stage.target_volume_ratio < 1.0
                        else "decompression"
                    ),
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    thermostat_tau_fs=protocol.thermostat.tau_fs,
                    target_volume_ratio=stage.target_volume_ratio,
                    schedule=stage.schedule,
                    handoff_window_steps=handoff_window_steps,
                    handoff_sample_interval_steps=handoff_sample_interval_steps,
                )
            )
        elif isinstance(stage, (PressurePlateauStage, RecoveryNPTStage)):
            barostat = protocol.barostat
            branch = (
                stage.branch
                if isinstance(stage, PressurePlateauStage)
                else "recovery"
            )
            result.append(
                IsotropicNPTStage(
                    name=stage.name,
                    steps=steps,
                    temperature_K=protocol.temperature_K,
                    pressure_GPa=stage.target_pressure_GPa,
                    thermostat_tau_fs=barostat.thermostat_tau_fs,
                    barostat_tau_fs=barostat.barostat_tau_fs,
                    branch=branch,
                    thermostat_chain_length=barostat.thermostat_chain_length,
                    barostat_chain_length=barostat.barostat_chain_length,
                    thermostat_substeps=barostat.thermostat_substeps,
                    barostat_substeps=barostat.barostat_substeps,
                )
            )
        else:  # pragma: no cover - Pydantic discriminator makes this unreachable.
            raise TypeError(f"unsupported configured stage: {type(stage).__name__}")
    return result


def _engine_branch(branch: str) -> str:
    """Map v2 research metadata onto the engine's legacy analysis labels."""

    return {
        "compression": "cold_compression",
        "decompression": "decompression",
        "recovery": "recovery",
        "reference": "other",
        "other": "other",
    }[branch]


def _build_safety_monitor(
    config: ColdMDConfig,
    reference_volume_A3: float,
) -> SafetyMonitor:
    return SafetyMonitor(
        SafetyLimits.from_config(config.safety),
        reference_volume_A3=reference_volume_A3,
        maximum_consecutive_violations=config.safety.maximum_consecutive_violations,
    )


def _repair_completed_boundary(
    *,
    destination: Path,
    config: ColdMDConfig,
    checkpoint: Any,
    atoms: Any,
    rng: Any,
    stages: Sequence[StageSpec],
    cursor: RunCursor,
    reference_volume_A3: float,
    config_hash: str,
    model_hash: str,
    versions: Mapping[str, str],
    reconciliation: Any,
) -> None:
    """Write a missing final thermo/frame after a checkpoint-first crash."""

    if not stages:
        raise CheckpointCompatibilityError("completed protocol has no stages")
    stage = stages[-1]
    cell = np.asarray(atoms.get_cell(), dtype=float)
    cell_tuple = tuple(tuple(float(value) for value in row) for row in cell)
    target_volume_ratio = (
        float(stage.target_volume_ratio)
        if isinstance(stage, VolumeRampStage)
        else None
    )
    target_pressure = (
        float(stage.pressure_GPa) if isinstance(stage, IsotropicNPTStage) else None
    )
    branch = getattr(stage, "branch", getattr(stage, "role", None))
    thermostat_kind = (
        stage.thermostat_kind
        if isinstance(stage, FixedCellNVTStage)
        else (
            "bussi"
            if isinstance(stage, VolumeRampStage)
            else (
                "nose_hoover_chain"
                if isinstance(stage, IsotropicNPTStage)
                else None
            )
        )
    )
    context = StepContext(
        stage_index=len(stages) - 1,
        stage_name=stage.name,
        stage_kind=stage.kind,
        stage_step=stage.steps,
        stage_steps=stage.steps,
        global_step=cursor.global_step,
        time_fs=cursor.time_fs,
        timestep_fs=config.protocol.timestep_fs,
        progress=1.0,
        target_volume_ratio=target_volume_ratio,
        final_target_volume_ratio=target_volume_ratio,
        target_pressure_GPa=target_pressure,
        branch=branch,
        volume_ratio_from_engine_start=float(atoms.get_volume())
        / reference_volume_A3,
        reference_volume_A3=reference_volume_A3,
        stage_reference_cell_A=cell_tuple,
        resumed=True,
        thermostat_kind=thermostat_kind,
    )
    logger = _EventLogger(destination / config.output.log_filename, append=True)
    output: RunOutputHook | None = None
    try:
        output = RunOutputHook(
            destination,
            trajectory_filename=config.output.trajectory_filename,
            thermo_filename=config.output.thermo_filename,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            thermo_interval_steps=config.output.thermo_interval_steps,
            msd_interval_steps=config.output.msd_interval_steps,
            checkpoint_interval_steps=config.output.checkpoint_interval_steps,
            safety_monitor=_build_safety_monitor(config, reference_volume_A3),
            rng=rng,
            config_hash=config_hash,
            model_hash=model_hash,
            reference_volume_A3=reference_volume_A3,
            append=True,
            write_forces=config.output.write_forces,
            write_stress=config.output.write_stress,
            write_momenta=config.output.write_momenta,
            write_stage_structures=config.output.write_stage_structures,
            checkpoint_on_abort=config.safety.checkpoint_on_abort,
            provenance_metadata={
                "purpose": "completed_boundary_repair",
                "checkpoint_id": checkpoint.checkpoint_id,
            },
            versions=versions,
            event_callback=logger,
        )
        output.prime_resume_from_checkpoint(
            checkpoint,
            last_thermo_step=reconciliation.last_thermo_step,
            last_trajectory_step=reconciliation.last_trajectory_step,
        )
        logger(
            "completed_boundary_repair",
            {
                "step": cursor.global_step,
                "last_thermo_step": reconciliation.last_thermo_step,
                "last_trajectory_step": reconciliation.last_trajectory_step,
            },
        )
        with output:
            output.on_stage_end(atoms, context)
    finally:
        logger.close()


def _config_hash(config: ColdMDConfig) -> str:
    return sha256_mapping(config.model_dump(mode="json"))


def _total_steps(config: ColdMDConfig) -> int:
    return sum(
        _steps_for_duration(stage.duration_ps, config.protocol.timestep_fs)
        for stage in config.protocol.stages
    )


def _initialize_velocities_for_fresh_run(config: ColdMDConfig) -> bool:
    """Resolve the one-time fresh-run momentum initialization policy.

    Free-form sequences need a protocol-level default because their first stage
    may be NPT or a volume ramp.  Legacy protocols retain the original
    ``initial_nvt.initialize_velocities`` behavior when the new setting is left
    unspecified.  Resume never calls this helper and therefore never redraws
    momenta.
    """

    if config.config_version == 2:
        return (
            True
            if config.protocol.initialize_velocities is None
            else bool(config.protocol.initialize_velocities)
        )
    configured = config.protocol.initialize_velocities
    if configured is not None:
        return bool(configured)
    if config.protocol.control_mode == "stage_sequence":
        return True
    first_stage = config.protocol.stages[0]
    return bool(
        isinstance(first_stage, InitialNVTStage)
        and first_stage.initialize_velocities
    )


def _steps_for_duration(duration_ps: float, timestep_fs: float) -> int:
    return int(round(float(duration_ps) * 1000.0 / float(timestep_fs)))


def _with_output_directory(config: ColdMDConfig, directory: Path) -> ColdMDConfig:
    resolved = Path(directory).expanduser().resolve()
    return config.model_copy(
        update={"output": config.output.model_copy(update={"directory": resolved})},
        deep=True,
    )


def _assert_fresh_output_policy(path: Path, *, overwrite: bool) -> None:
    """Check output policy without mutating the requested destination."""

    destination = path.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(
            f"output path exists and is not a directory: {destination}"
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination}; choose a new directory "
            "or pass overwrite=True to move it to a timestamped backup"
        )


def _prepare_fresh_output_directory(path: Path, *, overwrite: bool) -> Path | None:
    path = path.expanduser().resolve()
    backup: Path | None = None
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {path}; choose a new directory "
                "or pass overwrite=True to move it to a timestamped backup"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_name(f"{path.name}.backup-{stamp}")
        sequence = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.backup-{stamp}-{sequence:03d}")
            sequence += 1
        shutil.move(str(path), str(backup))
    path.mkdir(parents=True, exist_ok=True)
    return backup


def _record_resume_failure(destination: Path, error: BaseException) -> None:
    """Best-effort lifecycle record for setup, reconciliation, or MD failures."""

    manifest_path = destination / RUN_MANIFEST_FILENAME
    try:
        manifest = _read_json_mapping(manifest_path) if manifest_path.exists() else {}
        already_interrupted = manifest.get("status") == "resume_interrupted"
        manifest.update(
            {
                "status": (
                    "resume_interrupted"
                    if isinstance(error, KeyboardInterrupt) or already_interrupted
                    else "resume_failed"
                ),
                "last_resume_failure_utc": _utc_now(),
                "last_resume_error_type": type(error).__name__,
                "last_resume_error": str(error),
            }
        )
        _atomic_json(manifest_path, manifest)
    except Exception as manifest_error:  # pragma: no cover - damaged filesystem
        if hasattr(error, "add_note"):
            error.add_note(f"Could not update run manifest: {manifest_error!r}")


class _RunDirectoryLock:
    """Process-level advisory lock held for the complete resume transaction."""

    def __init__(self, directory: Path) -> None:
        resolved = directory.expanduser().resolve()
        # A sibling lock remains stable if ``--overwrite`` moves the entire run
        # directory to a backup.  Both fresh run and resume use this same inode.
        self.path = resolved.parent / f".{resolved.name}.coldmd.lock"
        self._stream: Any | None = None

    def __enter__(self) -> "_RunDirectoryLock":
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Linux production target
            raise RuntimeError("safe resume locking requires a POSIX/Linux runtime") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip()
            stream.close()
            raise RuntimeError(
                "another coldmd run or resume process holds this output lock"
                + (f": {owner}" if owner else "")
            ) from exc
        self._stream = stream
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "acquired_utc": _utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self._stream is None:
            return
        import fcntl

        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None


def _provenance(
    config: ColdMDConfig,
    structure: StructureReport,
    *,
    input_hash: str,
    model_hash_kind: str,
    reference_volume_A3: float,
) -> dict[str, Any]:
    return {
        "input_cif": str(config.input.cif_path),
        "input_cif_sha256": input_hash,
        "model_identity_kind": model_hash_kind,
        "reference_volume_A3": reference_volume_A3,
        "control_mode": config.protocol.control_mode,
        "stage_protocol": [stage.model_dump(mode="json") for stage in config.protocol.stages],
        "structure": asdict(structure),
    }


def _restart_version_requirements(current: Mapping[str, str]) -> dict[str, str]:
    # Strict continuation is deliberately fail-closed: source bytes, platform,
    # runtime dependency closure, and accelerator identity must all match.
    return dict(current)


def _collect_execution_versions(config: ColdMDConfig) -> dict[str, str]:
    """Collect software plus the actually selected accelerator fingerprint."""

    metadata: dict[str, str] = {
        "configured-device": config.calculator.device,
        "configured-dtype": config.calculator.dtype,
    }
    torch_module = _optional_torch()
    if torch_module is None:
        metadata["execution-device"] = "torch-unavailable"
        return collect_version_metadata(metadata)

    configured = config.calculator.device
    if configured == "auto":
        actual = "cuda" if torch_module.cuda.is_available() else "cpu"
    else:
        actual = configured
    metadata["execution-device"] = actual
    metadata["torch-cuda-runtime"] = str(getattr(torch_module.version, "cuda", None))
    metadata["torch-hip-runtime"] = str(getattr(torch_module.version, "hip", None))

    if actual.startswith("cuda") and torch_module.cuda.is_available():
        suffix = actual.removeprefix("cuda")
        index = (
            int(suffix[1:])
            if suffix.startswith(":")
            else int(torch_module.cuda.current_device())
        )
        properties = torch_module.cuda.get_device_properties(index)
        metadata["accelerator-name"] = str(properties.name)
        metadata["accelerator-capability"] = (
            f"{int(properties.major)}.{int(properties.minor)}"
        )
        metadata["accelerator-total-memory-bytes"] = str(
            int(properties.total_memory)
        )
    elif actual == "mps":
        metadata["accelerator-name"] = "mps"
    else:
        metadata["accelerator-name"] = platform.processor() or platform.machine()
    return collect_version_metadata(metadata)


def _attempt_failure_checkpoint(
    output: RunOutputHook,
    atoms: Any,
    cursor: RunCursor,
    error: BaseException,
) -> None:
    """Best-effort forensic snapshot that never masks the original failure."""

    try:
        output.checkpoint_failure(atoms, cursor, error)
    except Exception as checkpoint_error:  # pragma: no cover - filesystem failure
        if hasattr(error, "add_note"):
            error.add_note(
                "Forensic emergency checkpoint also failed; the latest normal "
                f"checkpoint remains unchanged: {checkpoint_error!r}"
            )


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _EventLogger:
    """Small JSONL event sink used by :class:`RunOutputHook`."""

    def __init__(self, path: Path, *, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append or path.exists() else "x"
        self.path = path
        self._stream = path.open(mode, encoding="utf-8")

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        row = {"utc": _utc_now(), "event": event, **dict(payload)}
        self._stream.write(
            json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False, sort_keys=True)
            + "\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


__all__ = [
    "FoundationModelResumeWarning",
    "analyze",
    "configuration_warnings",
    "dryrun",
    "fork",
    "resume",
    "run",
    "validate_config",
]
