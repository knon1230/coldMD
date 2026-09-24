"""Recoverable output reconciliation before checkpoint resume.

An exact mechanical checkpoint can lag behind already written observables.  A
pressure-plateau failure is the common case: MTK internal state is not
serializable, so the latest resumable checkpoint is the plateau boundary while
thermo/trajectory output may extend well beyond it.  Appending a replay to that
tail would mix two divergent attempts.

This module detects that condition, rebuilds the canonical prefix through the
checkpoint in a staging directory, and moves the complete prior attempt into a
timestamped quarantine.  No prior output is deleted.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .storage import CheckpointData, SegmentedTrajectoryWriter, StorageError
from .observables import THERMO_FIELDS


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Canonical output state after inspecting or reconciling a resume."""

    changed: bool
    checkpoint_step: int
    last_thermo_step: int | None
    last_trajectory_step: int | None
    original_max_thermo_step: int | None
    original_max_trajectory_step: int | None
    original_max_event_step: int | None
    kept_thermo_rows: int
    quarantined_thermo_rows: int
    kept_trajectory_frames: int
    quarantined_trajectory_frames: int
    attempt_directory: Path | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ThermoScan:
    path: Path
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TrajectoryFrame:
    atoms: Any
    metadata: dict[str, Any]
    step: int


def reconcile_outputs_to_checkpoint(
    run_dir: str | Path,
    checkpoint: CheckpointData,
    *,
    trajectory_filename: str,
    thermo_filename: str,
    event_filename: str = "coldmd.log",
    trajectory_properties: Sequence[str] | None = None,
    frames_per_segment: int = 10_000,
) -> ReconciliationReport:
    """Make canonical output a monotone prefix ending no later than checkpoint.

    If no stored row/frame lies after the checkpoint, this is a read-only
    inspection.  Otherwise all affected canonical files and the complete old
    event log are moved under ``discarded_attempts`` and prefix replacements are
    installed only after they have been fully reconstructed in staging.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    cutoff = checkpoint.resume_step
    _reject_incomplete_prior_transaction(root)
    thermo_path = root / thermo_filename
    thermo = _scan_thermo(thermo_path)
    segment_paths = _trajectory_segments(root, trajectory_filename)
    trajectory_steps = _scan_trajectory_steps(segment_paths)
    event_path = root / event_filename
    event_steps = _scan_event_steps(event_path)
    stage_structures_ahead = _stage_structures_after(root, cutoff)

    last_thermo = thermo.steps[-1] if thermo.steps else None
    last_trajectory = trajectory_steps[-1] if trajectory_steps else None
    thermo_ahead = last_thermo is not None and last_thermo > cutoff
    trajectory_ahead = last_trajectory is not None and last_trajectory > cutoff
    last_event = event_steps[-1] if event_steps else None
    event_ahead = last_event is not None and last_event > cutoff
    structure_ahead = bool(stage_structures_ahead)

    kept_thermo = sum(step <= cutoff for step in thermo.steps)
    kept_trajectory = sum(step <= cutoff for step in trajectory_steps)
    if not any((thermo_ahead, trajectory_ahead, event_ahead, structure_ahead)):
        return ReconciliationReport(
            changed=False,
            checkpoint_step=cutoff,
            last_thermo_step=last_thermo,
            last_trajectory_step=last_trajectory,
            original_max_thermo_step=last_thermo,
            original_max_trajectory_step=last_trajectory,
            original_max_event_step=last_event,
            kept_thermo_rows=len(thermo.steps),
            quarantined_thermo_rows=0,
            kept_trajectory_frames=len(trajectory_steps),
            quarantined_trajectory_frames=0,
            attempt_directory=None,
        )

    staging = Path(
        tempfile.mkdtemp(prefix=".resume-reconcile-", dir=str(root))
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempts_root = root / "discarded_attempts"
    attempt = attempts_root / f"attempt-{stamp}-after-step-{cutoff:012d}"
    sequence = 1
    while attempt.exists():
        attempt = attempts_root / (
            f"attempt-{stamp}-after-step-{cutoff:012d}-{sequence:03d}"
        )
        sequence += 1

    try:
        if thermo.path.exists():
            _write_thermo_prefix(thermo, staging / thermo_filename, cutoff)
        if segment_paths:
            _write_trajectory_prefix(
                segment_paths,
                staging / trajectory_filename,
                cutoff,
                trajectory_properties=trajectory_properties,
                frames_per_segment=frames_per_segment,
            )

        staged_thermo = _scan_thermo(staging / thermo_filename)
        staged_segments = _trajectory_segments(staging, trajectory_filename)
        staged_trajectory_steps = _scan_trajectory_steps(staged_segments)
        _assert_expected_prefix(thermo.steps, staged_thermo.steps, cutoff, "thermo")
        _assert_expected_prefix(
            trajectory_steps,
            staged_trajectory_steps,
            cutoff,
            "trajectory",
        )

        _durable_mkdir(attempts_root)
        if attempt.exists():  # Defensive against a same-process race or bad clock.
            raise FileExistsError(attempt)
        _durable_mkdir(attempt)
        journal = {
            "schema_version": 1,
            "status": "commit_started",
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_step": cutoff,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "staging_directory": str(staging),
            "attempt_directory": str(attempt),
        }
        _atomic_json(root / "resume-reconciliation.json", journal)

        originals: list[Path] = []
        if thermo.path.exists():
            originals.append(thermo.path)
        originals.extend(segment_paths)
        originals.extend(_sidecar_path(path) for path in segment_paths)
        if event_path.exists():
            originals.append(event_path)
        originals.extend(stage_structures_ahead)
        moved: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        for source in originals:
            if not source.exists():
                continue
            relative = source.relative_to(root)
            destination = attempt / relative
            # Record rollback intent before the rename.  If rename succeeds but
            # a following fsync fails, the mutation is still recoverable.
            moved.append((source, destination))
            _durable_move(source, destination)

        staged_files = sorted(path for path in staging.rglob("*") if path.is_file())
        for source in staged_files:
            relative = source.relative_to(staging)
            destination = root / relative
            # As above, installation intent must precede the first mutation.
            installed.append(destination)
            _durable_replace(source, destination)

        result = ReconciliationReport(
            changed=True,
            checkpoint_step=cutoff,
            last_thermo_step=(
                staged_thermo.steps[-1] if staged_thermo.steps else None
            ),
            last_trajectory_step=(
                staged_trajectory_steps[-1] if staged_trajectory_steps else None
            ),
            original_max_thermo_step=last_thermo,
            original_max_trajectory_step=last_trajectory,
            original_max_event_step=last_event,
            kept_thermo_rows=len(staged_thermo.steps),
            quarantined_thermo_rows=len(thermo.steps) - len(staged_thermo.steps),
            kept_trajectory_frames=len(staged_trajectory_steps),
            quarantined_trajectory_frames=(
                len(trajectory_steps) - len(staged_trajectory_steps)
            ),
            attempt_directory=attempt,
        )
        _atomic_json(
            attempt / "reconciliation.json",
            {
                **result.as_dict(),
                "checkpoint_id": checkpoint.checkpoint_id,
                "note": (
                    "Complete prior canonical outputs are retained here because "
                    "they extended beyond the exact resume checkpoint."
                ),
            },
        )
        journal.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "report": result.as_dict(),
            }
        )
        _atomic_json(root / "resume-reconciliation.json", journal)
        return result
    except Exception:
        # Before commit, originals remain canonical.  During commit, every move
        # is recoverably reversed below (the bookkeeping lists are initialized
        # just before commit).
        if "installed" in locals() and "moved" in locals():
            rollback_errors: list[str] = []
            for destination in reversed(installed):
                if not destination.exists():
                    continue
                try:
                    relative = destination.relative_to(root)
                    rollback_target = staging / relative
                    _durable_move(destination, rollback_target)
                except Exception as rollback_error:  # pragma: no cover - I/O fault
                    rollback_errors.append(repr(rollback_error))
            for original, quarantined in reversed(moved):
                if not quarantined.exists():
                    continue
                try:
                    if original.exists():
                        raise StorageError(
                            f"Rollback target unexpectedly exists: {original}"
                        )
                    _durable_move(quarantined, original)
                except Exception as rollback_error:  # pragma: no cover - I/O fault
                    rollback_errors.append(repr(rollback_error))
            if (root / "resume-reconciliation.json").exists():
                try:
                    _atomic_json(
                        root / "resume-reconciliation.json",
                        {
                            "schema_version": 1,
                            "status": "rollback_failed" if rollback_errors else "rolled_back",
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "checkpoint_step": cutoff,
                            "attempt_directory": str(attempt),
                            "rollback_errors": rollback_errors,
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception:
                    pass
        raise
    finally:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def _scan_thermo(path: Path) -> _ThermoScan:
    if not path.exists():
        return _ThermoScan(path, (), (), ())
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise StorageError(f"Thermo CSV has no header: {path}")
            rows = tuple(dict(row) for row in reader)
            fieldnames = tuple(reader.fieldnames)
    except (OSError, csv.Error) as exc:
        raise StorageError(f"Could not inspect thermo CSV {path}: {exc}") from exc
    if "step" not in fieldnames:
        raise StorageError(f"Thermo CSV lacks required step column: {path}")
    if fieldnames != tuple(THERMO_FIELDS):
        raise StorageError(
            f"Thermo CSV schema differs from this coldmd version: {path}"
        )
    expected_keys = set(fieldnames)
    for index, row in enumerate(rows, start=2):
        if set(row) != expected_keys or any(value is None for value in row.values()):
            raise StorageError(f"Malformed thermo CSV row at {path}:{index}")
    steps = tuple(_strict_step(row.get("step"), f"{path} thermo row") for row in rows)
    _require_strictly_increasing(steps, f"thermo steps in {path}")
    return _ThermoScan(path, fieldnames, rows, steps)


def _write_thermo_prefix(scan: _ThermoScan, destination: Path, cutoff: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan.fieldnames))
        writer.writeheader()
        for row, step in zip(scan.rows, scan.steps, strict=True):
            if step <= cutoff:
                writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _trajectory_segments(root: Path, filename: str) -> list[Path]:
    base = Path(filename)
    if base.name != filename or base.suffix.lower() not in {".traj", ".extxyz", ".xyz"}:
        raise ValueError("trajectory_filename must be a plain ASE trajectory filename")
    pattern = re.compile(
        rf"^{re.escape(base.stem)}-(\d{{6}}){re.escape(base.suffix.lower())}$"
    )
    found: list[tuple[int, Path]] = []
    for path in root.glob(f"{base.stem}-*{base.suffix}"):
        match = pattern.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    if found:
        numbers = [number for number, _ in found]
        if numbers != list(range(len(numbers))):
            raise StorageError(f"Trajectory segments are not contiguous: {numbers}")
    return [path for _, path in found]


def _scan_trajectory_steps(paths: Sequence[Path]) -> tuple[int, ...]:
    steps: list[int] = []
    for path in paths:
        metadata = _read_sidecar(path)
        actual_frames = sum(1 for _ in _iter_frames(path))
        if actual_frames != len(metadata):
            raise StorageError(
                f"Sidecar/frame count mismatch for {path}: "
                f"metadata={len(metadata)}, frames={actual_frames}"
            )
        steps.extend(
            _strict_step(item.get("coldmd_step"), f"{path} frame metadata")
            for item in metadata
        )
    _require_strictly_increasing(steps, "trajectory metadata steps")
    return tuple(steps)


def _write_trajectory_prefix(
    paths: Sequence[Path],
    destination: Path,
    cutoff: int,
    *,
    trajectory_properties: Sequence[str] | None,
    frames_per_segment: int,
) -> None:
    writer: SegmentedTrajectoryWriter | None = None
    try:
        for path in paths:
            metadata = _read_sidecar(path)
            frame_count = 0
            for index, atoms in enumerate(_iter_frames(path)):
                frame_count += 1
                if index >= len(metadata):
                    raise StorageError(
                        f"Trajectory frame {index} has no sidecar metadata: {path}"
                    )
                item = metadata[index]
                step = _strict_step(
                    item.get("coldmd_step"), f"{path} frame {index} metadata"
                )
                if step > cutoff:
                    continue
                if writer is None:
                    writer = SegmentedTrajectoryWriter(
                        destination,
                        frames_per_segment=frames_per_segment,
                        mode="x",
                        properties=trajectory_properties,
                    )
                writer.write(atoms, metadata=item)
            if frame_count != len(metadata):
                raise StorageError(
                    f"Sidecar/frame count mismatch for {path}: "
                    f"metadata={len(metadata)}, frames={frame_count}"
                )
    finally:
        if writer is not None:
            writer.close()


def _iter_frames(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".traj":
        try:
            from ase.io.trajectory import Trajectory
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise StorageError("ASE is required to reconcile trajectories") from exc
        reader = Trajectory(str(path), mode="r")
        try:
            for atoms in reader:
                yield atoms
        finally:
            reader.close()
        return
    try:
        from ase.io import iread
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise StorageError("ASE is required to reconcile trajectories") from exc
    yield from iread(str(path), index=":", format="extxyz", parallel=False)


def _read_sidecar(path: Path) -> list[dict[str, Any]]:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        raise StorageError(f"Trajectory segment has no metadata sidecar: {sidecar}")
    indexed: dict[int, dict[str, Any]] = {}
    with sidecar.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                segment = payload["segment"]
                frame = payload["frame"]
                metadata = payload["metadata"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise StorageError(
                    f"Invalid metadata at {sidecar}:{line_number}: {exc}"
                ) from exc
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise StorageError(f"Invalid frame index at {sidecar}:{line_number}")
            segment_match = re.search(r"-(\d{6})$", path.stem)
            expected_segment = int(segment_match.group(1)) if segment_match else None
            if (
                isinstance(segment, bool)
                or not isinstance(segment, int)
                or segment != expected_segment
            ):
                raise StorageError(
                    f"Invalid segment index at {sidecar}:{line_number}: {segment!r}"
                )
            if not isinstance(metadata, Mapping):
                raise StorageError(f"Metadata is not a mapping at {sidecar}:{line_number}")
            if frame in indexed:
                raise StorageError(f"Duplicate frame {frame} in {sidecar}")
            indexed[frame] = dict(metadata)
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise StorageError(f"Metadata frame indices are not contiguous in {sidecar}")
    return [indexed[index] for index in expected]


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.meta.jsonl")


def _strict_step(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StorageError(f"{label} step cannot be boolean")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9]\d*", value.strip()):
        result = int(value)
    else:
        raise StorageError(f"{label} has no strict non-negative integer step: {value!r}")
    if result < 0:
        raise StorageError(f"{label} step cannot be negative")
    return result


def _require_strictly_increasing(steps: Sequence[int], label: str) -> None:
    for previous, current in zip(steps, steps[1:]):
        if current <= previous:
            raise StorageError(
                f"{label} are not strictly increasing: {previous} then {current}"
            )


def _assert_expected_prefix(
    original: Sequence[int], staged: Sequence[int], cutoff: int, label: str
) -> None:
    expected = tuple(step for step in original if step <= cutoff)
    if tuple(staged) != expected:
        raise StorageError(
            f"Rebuilt {label} prefix failed validation: {tuple(staged)!r} != {expected!r}"
        )


def _stage_structures_after(root: Path, cutoff: int) -> list[Path]:
    directory = root / "stage_structures"
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for suffix in ("extxyz", "cif"):
        for path in directory.glob(f"*.{suffix}"):
            match = re.search(
                rf"-step-(\d{{12}})(?:-resume-\d+)?\.{suffix}$", path.name
            )
            if match and int(match.group(1)) > cutoff:
                result.append(path)
    return result


def _scan_event_steps(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    steps: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StorageError(f"Invalid event JSON at {path}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise StorageError(f"Event row is not a mapping at {path}:{line_number}")
            if payload.get("step") is not None:
                steps.append(_strict_step(payload["step"], f"{path}:{line_number}"))
    for previous, current in zip(steps, steps[1:]):
        if current < previous:
            raise StorageError(
                f"Event steps are not monotone in {path}: {previous} then {current}"
            )
    return tuple(steps)


def _reject_incomplete_prior_transaction(root: Path) -> None:
    journal = root / "resume-reconciliation.json"
    if not journal.exists():
        return
    try:
        value = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Could not read reconciliation journal {journal}") from exc
    status = value.get("status") if isinstance(value, Mapping) else None
    if status in {"commit_started", "rollback_failed"}:
        raise StorageError(
            "A prior output-reconciliation transaction did not complete. "
            f"Inspect {journal} and its attempt/staging paths before retrying."
        )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _durable_mkdir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _durable_move(source: Path, destination: Path) -> None:
    source_parent = source.parent
    _durable_mkdir(destination.parent)
    # Staging, canonical output, and quarantine all live below the same run
    # directory, so a same-filesystem atomic rename is a required invariant.
    os.replace(source, destination)
    if destination.is_file():
        _fsync_file(destination)
    _fsync_directory(destination.parent)
    if source_parent != destination.parent:
        _fsync_directory(source_parent)


def _durable_replace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    _durable_mkdir(destination.parent)
    os.replace(source, destination)
    if destination.is_file():
        _fsync_file(destination)
    _fsync_directory(destination.parent)
    if source_parent != destination.parent:
        _fsync_directory(source_parent)


def _durable_mkdir(path: Path) -> None:
    """Create a directory tree and persist every newly linked directory entry."""

    target = Path(path)
    missing: list[Path] = []
    current = target
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        raise NotADirectoryError(target)
    # mkdir(parents=True) creates ancestors from outermost to innermost.  Fsync
    # each new directory and, critically, the parent containing its new entry.
    for directory in reversed(missing):
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _fsync_file(path: Path) -> None:
    if os.name == "nt":
        # CPython/Windows rejects fsync on an ``os.open(..., O_RDONLY)`` file
        # descriptor. A binary update handle provides the equivalent flush for
        # ColdMD-owned writable output files.
        with Path(path).open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    # Windows does not permit opening directory handles through ``os.open``.
    # The production/restart target is POSIX Linux, where directory fsync stays
    # mandatory; this guard only makes read-only Windows development tests use
    # the strongest durability primitive that the Python runtime exposes there.
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = ["ReconciliationReport", "reconcile_outputs_to_checkpoint"]
