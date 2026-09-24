"""Durable storage helpers for cold-compression molecular dynamics.

The low-level writers deliberately do not depend on the dynamics engine.
They can therefore be used by ASE callbacks and recovery/inspection commands
without creating import cycles.  :class:`RunOutputHook` composes those writers
with coldmd's observable and safety layers through the engine's duck-typed
``on_stage_start/on_step/on_stage_end`` hook protocol.

Checkpoint files never contain pickle data.  A checkpoint consists of an
atomically replaced JSON manifest and an immutable, generation-named NPZ
array file.  The NPZ file is committed first and the manifest is committed
last.  Consequently, interruption at any point leaves either the previous
checkpoint readable or the new checkpoint completely readable.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import threading
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .observables import (
    EV_A3_TO_GPA,
    THERMO_FIELDS,
    ThermoRecord,
    compute_thermo_record,
)
from .safety import SafetyMonitor, SafetyViolation


CHECKPOINT_FORMAT = "coldmd-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
_JSON_TYPE_TAG = "__coldmd_json_type__"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class StorageError(RuntimeError):
    """Base class for durable-storage failures."""


class CheckpointError(StorageError):
    """Raised when a checkpoint cannot be written or decoded safely."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint is incompatible with this run."""


@dataclass(frozen=True, slots=True)
class TrajectoryFrameRef:
    """Location of a just-written frame in a segmented trajectory."""

    segment: int
    frame: int
    path: Path


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Files committed by :meth:`CheckpointManager.save`."""

    manifest: Path
    arrays: Path


@dataclass(slots=True)
class CheckpointData:
    """Validated, calculator-independent checkpoint contents.

    A calculator and arbitrary ASE constraints are intentionally not restored
    from disk.  Use :meth:`apply_to` on the input ``Atoms`` object to preserve
    those run-time objects, or use :meth:`to_atoms` and attach them explicitly.
    """

    checkpoint_id: str
    created_utc: str
    atomic_numbers: NDArray[np.int64]
    positions: NDArray[np.float64]
    cell: NDArray[np.float64]
    pbc: NDArray[np.bool_]
    momenta: NDArray[np.float64] | None
    protocol_state: dict[str, Any]
    rng_state: dict[str, Any] | None
    config_hash: str
    model_hash: str
    versions: dict[str, str]
    metadata: dict[str, Any]
    arrays_path: Path
    manifest_path: Path
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    @property
    def natoms(self) -> int:
        """Number of atoms in the saved state."""

        return int(self.atomic_numbers.shape[0])

    @property
    def cursor_state(self) -> dict[str, Any]:
        """Return a copy of the validated protocol cursor mapping."""

        value = self.protocol_state.get("cursor")
        if not isinstance(value, Mapping):
            raise CheckpointCompatibilityError(
                "Checkpoint protocol state has no cursor mapping."
            )
        return dict(value)

    @property
    def reference_volume_A3(self) -> float:
        """Return the original engine reference volume needed for resume."""

        value = self.protocol_state.get("reference_volume_A3")
        if value is None:
            value = self.cursor_state.get("reference_volume_A3")
        try:
            reference = float(value)
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "Checkpoint has no valid reference_volume_A3."
            ) from exc
        if not np.isfinite(reference) or reference <= 0.0:
            raise CheckpointCompatibilityError(
                "Checkpoint reference_volume_A3 must be finite and positive."
            )
        return reference

    @property
    def resume_step(self) -> int:
        """Return the saved global step used to prime append de-duplication."""

        value = self.cursor_state.get("global_step")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CheckpointCompatibilityError(
                "Checkpoint cursor has no integral global_step."
            )
        step = int(value)
        if step < 0:
            raise CheckpointCompatibilityError(
                "Checkpoint cursor global_step cannot be negative."
            )
        return step

    def apply_to(
        self,
        atoms: Any,
        *,
        strict_atomic_numbers: bool = True,
        require_momenta: bool = True,
    ) -> Any:
        """Apply the saved mechanical state to an existing ASE ``Atoms``.

        The existing calculator and constraints are retained.  Atomic numbers
        are checked before any mutation occurs.  The same ``atoms`` object is
        returned for convenient use in recovery code.
        """

        current_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        if current_numbers.shape != self.atomic_numbers.shape:
            raise CheckpointCompatibilityError(
                "Checkpoint atom count does not match the target Atoms object "
                f"({self.natoms} != {current_numbers.shape[0]})."
            )
        if strict_atomic_numbers and not np.array_equal(
            current_numbers, self.atomic_numbers
        ):
            raise CheckpointCompatibilityError(
                "Checkpoint atomic numbers/order do not match the target "
                "Atoms object."
            )
        if require_momenta and self.momenta is None:
            raise CheckpointCompatibilityError(
                "Checkpoint has no momenta; exact MD continuation is not possible."
            )

        # Cell first, positions second: set_cell(..., scale_atoms=False) must not
        # affinely transform the already saved Cartesian coordinates.
        atoms.set_cell(self.cell.copy(), scale_atoms=False)
        atoms.set_pbc(self.pbc.copy())
        atoms.set_positions(self.positions.copy())
        if self.momenta is not None:
            atoms.set_momenta(self.momenta.copy())
        return atoms

    def to_atoms(self, *, require_momenta: bool = False) -> Any:
        """Construct a new ASE ``Atoms`` without calculator or constraints."""

        try:
            from ase import Atoms
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise StorageError("ASE is required to reconstruct Atoms.") from exc

        if require_momenta and self.momenta is None:
            raise CheckpointCompatibilityError(
                "Checkpoint has no momenta; exact MD continuation is not possible."
            )
        atoms = Atoms(
            numbers=self.atomic_numbers.copy(),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
        )
        if self.momenta is not None:
            atoms.set_momenta(self.momenta.copy())
        return atoms

    def restore_rng(self, rng: Any) -> Any:
        """Restore the saved state into a compatible RNG and return it."""

        if self.rng_state is None:
            raise CheckpointCompatibilityError("Checkpoint contains no RNG state.")
        _apply_rng_state(rng, self.rng_state)
        return rng


class ThermoCSVWriter:
    """Append validated records to a durable thermodynamic CSV stream.

    ``fieldnames`` may be omitted.  In that case an existing header is reused,
    or the key order of the first record defines the header.  Supplying a
    ``record_provider`` makes the object directly attachable as an ASE callback::

        writer = ThermoCSVWriter(path, record_provider=collect_thermo)
        dynamics.attach(writer, interval=10)
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        fieldnames: Sequence[str] | None = None,
        *,
        append: bool = True,
        strict: bool = True,
        fsync_interval: int = 1,
        record_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if fsync_interval < 1:
            raise ValueError("fsync_interval must be at least 1.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strict = bool(strict)
        self.fsync_interval = int(fsync_interval)
        self.record_provider = record_provider
        self._fieldnames = _validate_fieldnames(fieldnames)
        self._file: Any | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._writes_since_sync = 0
        self._lock = threading.RLock()
        self._closed = False

        exists_with_data = self.path.exists() and self.path.stat().st_size > 0
        if exists_with_data and append:
            existing_header = self._read_header()
            if self._fieldnames is None:
                self._fieldnames = existing_header
            elif self._fieldnames != existing_header:
                raise StorageError(
                    "Thermo CSV header does not match requested fieldnames: "
                    f"{existing_header!r} != {self._fieldnames!r}."
                )
        elif exists_with_data and not append:
            raise FileExistsError(
                f"Refusing to overwrite non-empty thermo CSV: {self.path}"
            )

        if self._fieldnames is not None:
            self._open_writer(append=exists_with_data)

    @property
    def fieldnames(self) -> tuple[str, ...] | None:
        """The established CSV columns, if a header has been written."""

        if self._fieldnames is None:
            return None
        return tuple(self._fieldnames)

    def _read_header(self) -> list[str]:
        try:
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream), None)
        except OSError as exc:
            raise StorageError(f"Could not read thermo CSV {self.path}.") from exc
        if not header:
            raise StorageError(f"Thermo CSV has no valid header: {self.path}")
        validated = _validate_fieldnames(header)
        assert validated is not None
        return validated

    def _open_writer(self, *, append: bool) -> None:
        if self._closed:
            raise StorageError("ThermoCSVWriter is closed.")
        if self._file is not None:
            return
        assert self._fieldnames is not None
        mode = "a" if append else "x"
        try:
            self._file = self.path.open(mode, encoding="utf-8", newline="")
        except FileExistsError:
            # A zero-length file is safe to initialize, but opening it with x
            # necessarily fails.  It contains no user data to overwrite.
            if not append and self.path.stat().st_size == 0:
                self._file = self.path.open("a", encoding="utf-8", newline="")
            else:
                raise
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            extrasaction="raise" if self.strict else "ignore",
        )
        if not append:
            self._writer.writeheader()
            self._sync()

    def write(self, record: Any) -> None:
        """Write one record, flushing it immediately and fsyncing periodically."""

        record_mapping = _record_mapping(record)
        with self._lock:
            if self._closed:
                raise StorageError("ThermoCSVWriter is closed.")
            if self._fieldnames is None:
                if not record_mapping:
                    raise ValueError("Cannot infer CSV columns from an empty record.")
                self._fieldnames = _validate_fieldnames(
                    [str(key) for key in record_mapping]
                )
                self._open_writer(append=False)
            assert self._fieldnames is not None
            assert self._writer is not None

            supplied = {str(key) for key in record_mapping}
            unknown = supplied.difference(self._fieldnames)
            missing = set(self._fieldnames).difference(supplied)
            if self.strict and (unknown or missing):
                raise StorageError(
                    "Thermo record does not match its established header; "
                    f"unknown={sorted(unknown)}, missing={sorted(missing)}."
                )
            normalized = {
                name: _csv_value(record_mapping.get(name))
                for name in self._fieldnames
            }
            try:
                self._writer.writerow(normalized)
                self._file.flush()
            except (OSError, csv.Error) as exc:
                raise StorageError(f"Could not write thermo CSV {self.path}.") from exc
            self._writes_since_sync += 1
            if self._writes_since_sync >= self.fsync_interval:
                self._sync()

    def __call__(self, record: Any | None = None) -> None:
        """Callback alias for :meth:`write`.

        ASE calls observers without arguments, in which case ``record_provider``
        is used.  Tests and drivers may pass a record directly.
        """

        if record is None:
            if self.record_provider is None:
                raise StorageError(
                    "ThermoCSVWriter callback needs a record_provider."
                )
            record = self.record_provider()
        self.write(record)

    def _sync(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._writes_since_sync = 0

    def flush(self) -> None:
        """Flush Python buffers and fsync the CSV file."""

        with self._lock:
            self._sync()

    def close(self) -> None:
        """Synchronize and close the CSV file; safe to call repeatedly."""

        with self._lock:
            if self._closed:
                return
            if self._file is not None:
                self._sync()
                self._file.close()
                self._file = None
                self._writer = None
            self._closed = True

    def __enter__(self) -> ThermoCSVWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class SegmentedTrajectoryWriter:
    """Write restart-friendly, size-bounded ASE trajectory segments.

    The first argument may be either an output directory or a base filename
    ending in ``.traj``, ``.extxyz``, or ``.xyz``.  A filename such as
    ``trajectory.traj`` produces ``trajectory-000000.traj``, then
    ``trajectory-000001.traj``.  ``.xyz`` is intentionally written in extended
    XYZ format so lattice and periodic-boundary information are retained.

    Existing contiguous segments are discovered in ``mode='a'``.  Per-frame
    metadata, when supplied, is stored in a JSONL sidecar with the same segment
    number.  It is deliberately separate from ``Atoms.info`` so the writer
    never mutates the running object or triggers extra calculator evaluations.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        prefix: str | None = None,
        frames_per_segment: int = 10_000,
        mode: Literal["a", "x"] = "a",
        atoms: Any | None = None,
        properties: Sequence[str] | None = None,
        sync_interval: int = 100,
    ) -> None:
        if frames_per_segment < 1:
            raise ValueError("frames_per_segment must be at least 1.")
        if sync_interval < 1:
            raise ValueError("sync_interval must be at least 1.")
        if mode not in {"a", "x"}:
            raise ValueError("mode must be 'a' or 'x'.")

        location = Path(directory)
        supported_suffixes = {".traj", ".extxyz", ".xyz"}
        if location.suffix.lower() in supported_suffixes:
            if prefix is not None:
                raise ValueError(
                    "prefix must be omitted when directory is a trajectory filename."
                )
            self.directory = location.parent
            self.prefix = location.stem
            self.suffix = location.suffix.lower()
        else:
            self.directory = location
            self.prefix = "trajectory" if prefix is None else prefix
            self.suffix = ".traj"
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise ValueError("prefix must be a non-empty filename component.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_per_segment = int(frames_per_segment)
        self.mode = mode
        self.atoms = atoms
        self.properties = None if properties is None else list(properties)
        self.sync_interval = int(sync_interval)
        self._trajectory: Any | None = None
        self._metadata_file: Any | None = None
        self._segment = 0
        self._frame = 0
        self._writes_since_sync = 0
        self._closed = False
        self._lock = threading.RLock()

        existing = self._discover_segments()
        if existing and mode == "x":
            raise FileExistsError(
                f"Trajectory segments already exist for prefix {self.prefix!r}."
            )
        if existing:
            self._validate_contiguous(existing)
            self._segment = existing[-1][0]
            self._frame = self._trajectory_length(existing[-1][1])
            if self._frame >= self.frames_per_segment:
                self._segment += 1
                self._frame = 0

    @property
    def current_segment(self) -> int:
        return self._segment

    @property
    def current_frame(self) -> int:
        """Next frame index within the current segment."""

        return self._frame

    def _segment_path(self, segment: int | None = None) -> Path:
        number = self._segment if segment is None else segment
        return self.directory / f"{self.prefix}-{number:06d}{self.suffix}"

    def _metadata_path(self, segment: int | None = None) -> Path:
        number = self._segment if segment is None else segment
        return self.directory / f"{self.prefix}-{number:06d}.meta.jsonl"

    def _discover_segments(self) -> list[tuple[int, Path]]:
        pattern = re.compile(
            rf"^{re.escape(self.prefix)}-(\d{{6}}){re.escape(self.suffix)}$"
        )
        found: list[tuple[int, Path]] = []
        for path in self.directory.glob(f"{self.prefix}-*{self.suffix}"):
            match = pattern.fullmatch(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return sorted(found)

    @staticmethod
    def _validate_contiguous(segments: Sequence[tuple[int, Path]]) -> None:
        numbers = [number for number, _ in segments]
        expected = list(range(numbers[0], numbers[-1] + 1))
        if numbers[0] != 0 or numbers != expected:
            raise StorageError(
                f"Trajectory segments are not contiguous from zero: {numbers}"
            )

    @staticmethod
    def _trajectory_class() -> Any:
        try:
            from ase.io.trajectory import Trajectory
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise StorageError("ASE is required to write trajectories.") from exc
        return Trajectory

    def _trajectory_length(self, path: Path) -> int:
        if self.suffix != ".traj":
            try:
                from ase.io import iread
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise StorageError("ASE is required to read trajectories.") from exc
            try:
                return sum(
                    1
                    for _ in iread(
                        str(path), index=":", format="extxyz", parallel=False
                    )
                )
            except Exception as exc:
                raise StorageError(
                    f"Could not read trajectory segment {path}."
                ) from exc
        Trajectory = self._trajectory_class()
        try:
            reader = Trajectory(str(path), mode="r")
            try:
                return int(len(reader))
            finally:
                reader.close()
        except Exception as exc:
            raise StorageError(f"Could not read trajectory segment {path}.") from exc

    def _ensure_open(self, atoms: Any) -> None:
        if self.suffix != ".traj":
            return
        if self._trajectory is not None:
            return
        Trajectory = self._trajectory_class()
        path = self._segment_path()
        ase_mode = "a" if path.exists() else "w"
        try:
            self._trajectory = Trajectory(
                str(path),
                mode=ase_mode,
                atoms=atoms,
                properties=self.properties,
            )
        except Exception as exc:
            raise StorageError(f"Could not open trajectory segment {path}.") from exc

    def _rotate(self) -> None:
        self._close_handles()
        self._segment += 1
        self._frame = 0
        self._writes_since_sync = 0

    def write(
        self,
        atoms: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryFrameRef:
        """Write one frame and return its segment/frame reference."""

        with self._lock:
            if self._closed:
                raise StorageError("SegmentedTrajectoryWriter is closed.")
            target = self.atoms if atoms is None else atoms
            if target is None:
                raise StorageError("No Atoms object was supplied to trajectory writer.")
            if self._frame >= self.frames_per_segment:
                self._rotate()
            encoded_metadata = (
                None
                if metadata is None
                else self._encode_metadata(self._frame, metadata)
            )
            self._ensure_open(target)
            path = self._segment_path()
            frame = self._frame
            try:
                if self.suffix == ".traj":
                    self._trajectory.write(target)
                else:
                    from ase.io import write as ase_write

                    ase_write(
                        str(path),
                        target,
                        format="extxyz",
                        append=path.exists(),
                        parallel=False,
                    )
            except Exception as exc:
                raise StorageError(f"Could not write trajectory frame to {path}.") from exc

            if encoded_metadata is not None:
                self._write_metadata(encoded_metadata)

            self._frame += 1
            self._writes_since_sync += 1
            ref = TrajectoryFrameRef(self._segment, frame, path)
            if self._writes_since_sync >= self.sync_interval:
                self.flush()
            return ref

    def __call__(self) -> TrajectoryFrameRef:
        """Write the bound ``atoms`` object when used as an ASE observer."""

        return self.write()

    def _encode_metadata(self, frame: int, metadata: Mapping[str, Any]) -> str:
        if not isinstance(metadata, Mapping):
            raise TypeError("Trajectory metadata must be a mapping.")
        payload = {
            "segment": self._segment,
            "frame": frame,
            "metadata": _to_jsonable(dict(metadata)),
        }
        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise StorageError("Trajectory metadata is not JSON serializable.") from exc

    def _write_metadata(self, line: str) -> None:
        if self._metadata_file is None:
            self._metadata_file = self._metadata_path().open(
                "a", encoding="utf-8", newline="\n"
            )
        try:
            self._metadata_file.write(line + "\n")
            self._metadata_file.flush()
        except OSError as exc:
            raise StorageError(
                f"Could not write trajectory metadata {self._metadata_path()}."
            ) from exc

    def flush(self) -> None:
        """Close/synchronize current handles; the next write reopens in append."""

        with self._lock:
            if self._closed:
                return
            self._close_handles()
            self._writes_since_sync = 0

    def _close_handles(self) -> None:
        if self._trajectory is not None:
            try:
                self._trajectory.close()
            finally:
                self._trajectory = None
            _fsync_file(self._segment_path())
        elif self._segment_path().is_file():
            _fsync_file(self._segment_path())
        if self._metadata_file is not None:
            try:
                self._metadata_file.flush()
                os.fsync(self._metadata_file.fileno())
            finally:
                self._metadata_file.close()
                self._metadata_file = None
        _fsync_directory(self.directory)

    def close(self) -> None:
        """Synchronize and close the writer; safe to call repeatedly."""

        with self._lock:
            if self._closed:
                return
            self._close_handles()
            self._closed = True

    def __enter__(self) -> SegmentedTrajectoryWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class CheckpointManager:
    """Atomically save and safely load complete MD checkpoints.

    ``<basename>.json`` is the sole commit pointer.  Each save writes a new
    generation-named NPZ, fsyncs and commits it, and only then atomically
    replaces the JSON manifest.  Old generation files are retained by default,
    which also makes the previous state recoverable after a user-level mistake.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        basename: str = "checkpoint",
        max_json_bytes: int = 16 * 1024 * 1024,
        max_array_file_bytes: int = 512 * 1024 * 1024,
        max_uncompressed_array_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        if not basename or Path(basename).name != basename:
            raise ValueError("basename must be a non-empty filename component.")
        for name, value in {
            "max_json_bytes": max_json_bytes,
            "max_array_file_bytes": max_array_file_bytes,
            "max_uncompressed_array_bytes": max_uncompressed_array_bytes,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive.")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.basename = basename
        self.manifest_path = self.directory / f"{basename}.json"
        self.max_json_bytes = int(max_json_bytes)
        self.max_array_file_bytes = int(max_array_file_bytes)
        self.max_uncompressed_array_bytes = int(max_uncompressed_array_bytes)
        self._lock = threading.RLock()

    @property
    def exists(self) -> bool:
        """Whether the committed manifest exists."""

        return self.manifest_path.is_file()

    def save(
        self,
        atoms: Any,
        protocol_state: Mapping[str, Any],
        *,
        rng: Any | Mapping[str, Any] | None,
        config_hash: str,
        model_hash: str,
        metadata: Mapping[str, Any] | None = None,
        versions: Mapping[str, str] | None = None,
    ) -> CheckpointPaths:
        """Atomically commit an MD state and return the committed paths.

        ``protocol_state`` must include all integrator/thermostat/barostat state
        needed by the selected dynamics implementation.  The storage layer
        cannot infer those algorithm-specific variables from ASE.
        """

        if not isinstance(protocol_state, Mapping):
            raise TypeError("protocol_state must be a mapping.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        config_hash = _validate_hash(config_hash, "config_hash")
        model_hash = _validate_hash(model_hash, "model_hash")

        arrays, has_momenta = _atoms_to_arrays(atoms)
        natoms = int(arrays["atomic_numbers"].shape[0])
        checkpoint_id = uuid.uuid4().hex
        arrays_name = f"{self.basename}-{checkpoint_id}.npz"
        arrays_path = self.directory / arrays_name
        rng_state = _capture_rng_state(rng)
        merged_versions = collect_version_metadata(versions)
        created = _utc_now()

        with self._lock:
            temporary_npz = self.directory / (
                f".{arrays_name}.tmp-{uuid.uuid4().hex}"
            )
            try:
                _write_npz_and_fsync(temporary_npz, arrays)
                if temporary_npz.stat().st_size > self.max_array_file_bytes:
                    raise CheckpointError(
                        "Checkpoint array file exceeds configured size limit."
                    )
                arrays_hash = sha256_file(temporary_npz)
                os.replace(temporary_npz, arrays_path)
                _fsync_directory(self.directory)

                manifest = {
                    "format": CHECKPOINT_FORMAT,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_id": checkpoint_id,
                    "created_utc": created,
                    "arrays": {
                        "file": arrays_name,
                        "sha256": arrays_hash,
                    },
                    "atoms": {
                        "natoms": natoms,
                        "has_momenta": has_momenta,
                    },
                    "protocol_state": _to_jsonable(dict(protocol_state)),
                    "rng_state": _to_jsonable(rng_state),
                    "provenance": {
                        "config_sha256": config_hash,
                        "model_sha256": model_hash,
                        "versions": merged_versions,
                        "metadata": _to_jsonable(dict(metadata or {})),
                    },
                }
                encoded = json.dumps(
                    manifest,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8") + b"\n"
                if len(encoded) > self.max_json_bytes:
                    raise CheckpointError(
                        "Checkpoint manifest exceeds configured JSON size limit."
                    )
                _atomic_write_bytes(self.manifest_path, encoded)
            except Exception:
                _unlink_if_exists(temporary_npz)
                # If manifest commit failed, this generation is unreachable and
                # may be removed.  The prior manifest and NPZ remain untouched.
                if not _manifest_references(self.manifest_path, arrays_name):
                    _unlink_if_exists(arrays_path)
                raise

        return CheckpointPaths(self.manifest_path, arrays_path)

    def load(
        self,
        *,
        expected_config_hash: str | None = None,
        expected_model_hash: str | None = None,
        expected_versions: Mapping[str, str] | None = None,
        require_exact_version_set: bool = False,
        expected_schema_version: int = CHECKPOINT_SCHEMA_VERSION,
    ) -> CheckpointData:
        """Load and validate the committed checkpoint without using pickle."""

        if expected_schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                "This loader only implements checkpoint schema "
                f"{CHECKPOINT_SCHEMA_VERSION}; requested {expected_schema_version}."
            )
        if expected_config_hash is not None:
            expected_config_hash = _validate_hash(
                expected_config_hash, "expected_config_hash"
            )
        if expected_model_hash is not None:
            expected_model_hash = _validate_hash(
                expected_model_hash, "expected_model_hash"
            )
        with self._lock:
            manifest = self._read_manifest()
            schema_version = manifest.get("schema_version")
            if schema_version != expected_schema_version:
                raise CheckpointCompatibilityError(
                    "Checkpoint schema is incompatible: "
                    f"saved={schema_version!r}, expected={expected_schema_version}."
                )
            if manifest.get("format") != CHECKPOINT_FORMAT:
                raise CheckpointCompatibilityError(
                    f"Unsupported checkpoint format: {manifest.get('format')!r}."
                )

            try:
                checkpoint_id = str(manifest["checkpoint_id"])
                created_utc = str(manifest["created_utc"])
                arrays_info = manifest["arrays"]
                arrays_name = arrays_info["file"]
                expected_arrays_hash = _validate_hash(
                    arrays_info["sha256"], "arrays.sha256"
                )
                atoms_info = manifest["atoms"]
                natoms = int(atoms_info["natoms"])
                has_momenta = bool(atoms_info["has_momenta"])
                provenance = manifest["provenance"]
                config_hash = _validate_hash(
                    provenance["config_sha256"], "config_sha256"
                )
                model_hash = _validate_hash(
                    provenance["model_sha256"], "model_sha256"
                )
                versions = _string_mapping(provenance["versions"], "versions")
                metadata_value = _from_jsonable(provenance.get("metadata", {}))
                protocol_value = _from_jsonable(manifest["protocol_state"])
                rng_value = _from_jsonable(manifest.get("rng_state"))
            except (KeyError, TypeError, ValueError) as exc:
                raise CheckpointError("Checkpoint manifest is incomplete or invalid.") from exc

            if natoms < 1:
                raise CheckpointError("Checkpoint natoms must be positive.")
            if not isinstance(metadata_value, dict):
                raise CheckpointError("Checkpoint metadata must decode to a mapping.")
            if not isinstance(protocol_value, dict):
                raise CheckpointError("Protocol state must decode to a mapping.")
            if rng_value is not None and not isinstance(rng_value, dict):
                raise CheckpointError("RNG state must decode to a mapping or null.")
            if expected_config_hash is not None and config_hash != expected_config_hash:
                raise CheckpointCompatibilityError(
                    "Configuration hash differs from checkpoint."
                )
            if expected_model_hash is not None and model_hash != expected_model_hash:
                raise CheckpointCompatibilityError("Model hash differs from checkpoint.")
            if expected_versions is not None:
                normalized_expected = _string_mapping(
                    expected_versions,
                    "expected_versions",
                )
                if require_exact_version_set and set(versions) != set(normalized_expected):
                    saved_only = sorted(set(versions).difference(normalized_expected))
                    current_only = sorted(set(normalized_expected).difference(versions))
                    raise CheckpointCompatibilityError(
                        "Restart fingerprint key set differs from checkpoint: "
                        f"saved_only={saved_only}, current_only={current_only}."
                    )
                for package, expected in normalized_expected.items():
                    saved = versions.get(str(package))
                    if saved != str(expected):
                        raise CheckpointCompatibilityError(
                            f"Version mismatch for {package}: "
                            f"saved={saved!r}, expected={str(expected)!r}."
                        )

            arrays_path = self._resolve_arrays_path(arrays_name)
            if not arrays_path.is_file():
                raise CheckpointError(
                    f"Checkpoint array file does not exist: {arrays_path.name}"
                )
            if arrays_path.stat().st_size > self.max_array_file_bytes:
                raise CheckpointError(
                    "Checkpoint array file exceeds configured size limit."
                )
            if sha256_file(arrays_path) != expected_arrays_hash:
                raise CheckpointError("Checkpoint array-file SHA-256 mismatch.")
            arrays = self._read_arrays(arrays_path, natoms, has_momenta)

        return CheckpointData(
            checkpoint_id=checkpoint_id,
            created_utc=created_utc,
            atomic_numbers=arrays["atomic_numbers"],
            positions=arrays["positions"],
            cell=arrays["cell"],
            pbc=arrays["pbc"],
            momenta=arrays.get("momenta"),
            protocol_state=protocol_value,
            rng_state=rng_value,
            config_hash=config_hash,
            model_hash=model_hash,
            versions=versions,
            metadata=metadata_value,
            arrays_path=arrays_path,
            manifest_path=self.manifest_path,
            schema_version=int(schema_version),
        )

    def load_into(
        self,
        atoms: Any,
        *,
        strict_atomic_numbers: bool = True,
        require_momenta: bool = True,
        **load_kwargs: Any,
    ) -> CheckpointData:
        """Load a checkpoint, apply it to ``atoms``, and return its metadata."""

        checkpoint = self.load(**load_kwargs)
        checkpoint.apply_to(
            atoms,
            strict_atomic_numbers=strict_atomic_numbers,
            require_momenta=require_momenta,
        )
        return checkpoint

    def _read_manifest(self) -> dict[str, Any]:
        try:
            size = self.manifest_path.stat().st_size
        except FileNotFoundError as exc:
            raise CheckpointError(
                f"Checkpoint manifest does not exist: {self.manifest_path}"
            ) from exc
        if size > self.max_json_bytes:
            raise CheckpointError("Checkpoint manifest exceeds configured size limit.")
        try:
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream, parse_constant=_reject_json_constant)
        except (OSError, UnicodeError, ValueError) as exc:
            raise CheckpointError("Checkpoint manifest is not valid UTF-8 JSON.") from exc
        if not isinstance(value, dict):
            raise CheckpointError("Checkpoint manifest root must be a JSON object.")
        return value

    def _resolve_arrays_path(self, name: Any) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise CheckpointError("Unsafe checkpoint array filename.")
        expected_prefix = f"{self.basename}-"
        if not name.startswith(expected_prefix) or not name.endswith(".npz"):
            raise CheckpointError("Unexpected checkpoint array filename.")
        resolved_dir = self.directory.resolve()
        candidate = (self.directory / name).resolve()
        if candidate.parent != resolved_dir:
            raise CheckpointError("Checkpoint array path escapes its directory.")
        return candidate

    def _read_arrays(
        self, path: Path, natoms: int, has_momenta: bool
    ) -> dict[str, NDArray[Any]]:
        expected_keys = {"atomic_numbers", "positions", "cell", "pbc"}
        if has_momenta:
            expected_keys.add("momenta")
        try:
            with zipfile.ZipFile(path, mode="r") as container:
                members = container.infolist()
                member_names = [member.filename for member in members]
                if len(member_names) != len(set(member_names)):
                    raise CheckpointError("Checkpoint NPZ has duplicate members.")
                if any(member.flag_bits & 0x1 for member in members):
                    raise CheckpointError("Encrypted checkpoint NPZ is unsupported.")
                declared_size = sum(member.file_size for member in members)
                if declared_size > self.max_uncompressed_array_bytes:
                    raise CheckpointError(
                        "Checkpoint arrays exceed configured uncompressed size limit."
                    )
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys != expected_keys:
                    raise CheckpointError(
                        f"Checkpoint arrays have unexpected keys: {sorted(keys)}"
                    )
                arrays = {key: np.array(archive[key], copy=True) for key in keys}
        except CheckpointError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise CheckpointError("Checkpoint NPZ is invalid or unsafe.") from exc

        total_bytes = sum(array.nbytes for array in arrays.values())
        if total_bytes > self.max_uncompressed_array_bytes:
            raise CheckpointError(
                "Checkpoint arrays exceed configured uncompressed size limit."
            )
        _validate_checkpoint_arrays(arrays, natoms, has_momenta)
        return {
            "atomic_numbers": np.asarray(arrays["atomic_numbers"], dtype=np.int64),
            "positions": np.asarray(arrays["positions"], dtype=np.float64),
            "cell": np.asarray(arrays["cell"], dtype=np.float64),
            "pbc": np.asarray(arrays["pbc"], dtype=np.bool_),
            **(
                {"momenta": np.asarray(arrays["momenta"], dtype=np.float64)}
                if has_momenta
                else {}
            ),
        }


EventCallback = Callable[[str, Mapping[str, Any]], Any]


class RunOutputHook:
    """Integrated output, safety, and checkpoint hook for ``ProtocolEngine``.

    One :class:`~coldmd.observables.ThermoRecord` is sampled for each observed
    physical state and shared by the safety monitor, CSV stream, trajectory
    metadata, and event reporter.  Safety is checked at every MD step even
    when durable outputs use a coarser cadence.

    Normal checkpoints are committed at every stage boundary and at the
    configured interval only when ``StepContext.exact_resume_supported`` is
    true.  Interior extended-system states such as Nose-Hoover-chain NVT and
    MTK NPT are therefore never advertised as exact restart points unless their
    complete integrator state is available.  A separate ``emergency.json``
    checkpoint is attempted on failure for forensic use without replacing the
    most recent safe fallback.
    """

    def __init__(
        self,
        output_directory: str | os.PathLike[str],
        *,
        trajectory_filename: str = "trajectory.traj",
        thermo_filename: str = "thermo.csv",
        trajectory_interval_steps: int = 100,
        thermo_interval_steps: int = 10,
        msd_interval_steps: int | None = None,
        checkpoint_interval_steps: int = 10_000,
        safety_monitor: SafetyMonitor,
        rng: Any,
        config_hash: str,
        model_hash: str,
        reference_volume_A3: float | None = None,
        append: bool = False,
        resume_step: int | None = None,
        last_thermo_step: int | None = None,
        last_trajectory_step: int | None = None,
        frames_per_segment: int = 10_000,
        write_forces: bool = True,
        write_stress: bool = True,
        write_momenta: bool = True,
        write_stage_structures: bool = True,
        checkpoint_on_abort: bool = True,
        provenance_metadata: Mapping[str, Any] | None = None,
        versions: Mapping[str, str] | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        for name, value in {
            "trajectory_interval_steps": trajectory_interval_steps,
            "thermo_interval_steps": thermo_interval_steps,
            "msd_interval_steps": (
                trajectory_interval_steps
                if msd_interval_steps is None
                else msd_interval_steps
            ),
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "frames_per_segment": frames_per_segment,
        }.items():
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")
        if not isinstance(safety_monitor, SafetyMonitor):
            raise TypeError("safety_monitor must be a SafetyMonitor instance.")
        if provenance_metadata is not None and not isinstance(
            provenance_metadata, Mapping
        ):
            raise TypeError("provenance_metadata must be a mapping or None.")
        if (
            resume_step is not None
            or last_thermo_step is not None
            or last_trajectory_step is not None
        ) and not append:
            raise ValueError("Resume step arguments require append=True.")
        if resume_step is None and (
            last_thermo_step is not None or last_trajectory_step is not None
        ):
            raise ValueError(
                "last output steps require resume_step to establish their upper bound."
            )

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.trajectory_interval_steps = int(trajectory_interval_steps)
        self.thermo_interval_steps = int(thermo_interval_steps)
        self.msd_interval_steps = int(
            self.trajectory_interval_steps
            if msd_interval_steps is None
            else msd_interval_steps
        )
        # A single trajectory stream avoids duplicate raw data.  Only fixed-cell
        # NVT stages need the optional tighter cadence for a full-stage MSD;
        # barostatted and driven stages retain the ordinary structural cadence.
        self._effective_nvt_trajectory_interval_steps = min(
            self.trajectory_interval_steps, self.msd_interval_steps
        )
        self.checkpoint_interval_steps = int(checkpoint_interval_steps)
        self.safety_monitor = safety_monitor
        self.rng = rng
        self.config_hash = _validate_hash(config_hash, "config_hash")
        self.model_hash = _validate_hash(model_hash, "model_hash")
        if reference_volume_A3 is not None and (
            not np.isfinite(reference_volume_A3) or reference_volume_A3 <= 0.0
        ):
            raise ValueError("reference_volume_A3 must be finite and positive.")
        self.reference_volume_A3 = (
            None if reference_volume_A3 is None else float(reference_volume_A3)
        )
        monitor_reference = safety_monitor.reference_volume_A3
        if (
            self.reference_volume_A3 is not None
            and monitor_reference is not None
            and not np.isclose(
                self.reference_volume_A3,
                monitor_reference,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "reference_volume_A3 disagrees with SafetyMonitor reference volume."
            )
        if self.reference_volume_A3 is not None and monitor_reference is None:
            safety_monitor.reset(reference_volume_A3=self.reference_volume_A3)
        self.append = bool(append)
        self.write_forces = bool(write_forces)
        self.write_stress = bool(write_stress)
        self.write_momenta = bool(write_momenta)
        self.write_stage_structures = bool(write_stage_structures)
        self.checkpoint_on_abort = bool(checkpoint_on_abort)
        self.provenance_metadata = dict(provenance_metadata or {})
        self.versions = None if versions is None else dict(versions)
        self.event_callback = event_callback

        thermo_path = self.output_directory / thermo_filename
        trajectory_path = self.output_directory / trajectory_filename
        trajectory_properties = ["energy"]
        if self.write_forces:
            trajectory_properties.append("forces")
        if self.write_stress:
            trajectory_properties.append("stress")

        self.thermo_writer = ThermoCSVWriter(
            thermo_path,
            fieldnames=THERMO_FIELDS,
            append=self.append,
            strict=True,
        )
        try:
            self.trajectory_writer = SegmentedTrajectoryWriter(
                trajectory_path,
                frames_per_segment=int(frames_per_segment),
                mode="a" if self.append else "x",
                properties=trajectory_properties,
            )
        except Exception:
            self.thermo_writer.close()
            raise
        self.checkpoint_manager = CheckpointManager(
            self.output_directory / "checkpoints", basename="checkpoint"
        )
        self.emergency_checkpoint_manager = CheckpointManager(
            self.output_directory / "checkpoints", basename="emergency"
        )
        self.stage_structure_directory = self.output_directory / "stage_structures"
        if self.write_stage_structures:
            self.stage_structure_directory.mkdir(parents=True, exist_ok=True)

        self._last_record: ThermoRecord | None = None
        self._last_record_key: tuple[int, str] | None = None
        # Safety follows physical MD states, not stage labels.  Adjacent stages
        # share one boundary step and must not count the same warning twice.
        self._last_safety_step: int | None = None
        self._last_thermo_step: int | None = None
        self._last_trajectory_step: int | None = None
        self._last_checkpoint_step: int | None = None
        self._last_emergency_error_id: int | None = None
        self._closed = False
        self._lock = threading.RLock()
        if resume_step is not None:
            self.prime_resume(
                resume_step,
                reference_volume_A3=self.reference_volume_A3,
                last_thermo_step=last_thermo_step,
                last_trajectory_step=last_trajectory_step,
            )

    @property
    def last_record(self) -> ThermoRecord | None:
        """Most recently sampled record, if any."""

        return self._last_record

    def prime_resume(
        self,
        step: int,
        *,
        reference_volume_A3: float | None = None,
        safety_state: Mapping[str, Any] | None = None,
        last_thermo_step: int | None = None,
        last_trajectory_step: int | None = None,
    ) -> None:
        """Prime boundary de-duplication from a loaded checkpoint.

        Call this before handing an ``append=True`` hook to the engine.  The
        engine emits ``on_stage_start`` for the restored boundary; marking the
        saved global step prevents that identical physical state from being
        counted twice by safety or checkpointing.  The API must separately
        supply the actual last retained thermo and trajectory steps after
        reconciling outputs.  If either is omitted, that output is conservatively
        written again at the resume boundary rather than risking a missing row
        or frame.  ``safety_state`` may be supplied directly from checkpoint
        protocol state when it has not already been restored by the API.
        """

        if not self.append:
            raise StorageError("prime_resume requires append=True.")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, np.integer))
            or int(step) < 0
        ):
            raise ValueError("Resume step must be a non-negative integer.")
        if self._last_record is not None:
            raise StorageError("Cannot prime resume after observations have begun.")
        resume_step = int(step)
        thermo_step = _validated_retained_step(
            last_thermo_step,
            name="last_thermo_step",
            resume_step=resume_step,
        )
        trajectory_step = _validated_retained_step(
            last_trajectory_step,
            name="last_trajectory_step",
            resume_step=resume_step,
        )
        if safety_state is not None:
            self.safety_monitor.load_state_dict(safety_state)
        restored_step = self.safety_monitor.state_dict().get("previous_step")
        if restored_step is not None and int(restored_step) != resume_step:
            raise StorageError(
                "Safety checkpoint step disagrees with the resume cursor."
            )
        if reference_volume_A3 is not None:
            reference = float(reference_volume_A3)
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError("reference_volume_A3 must be finite and positive.")
            if self.reference_volume_A3 is not None and not np.isclose(
                self.reference_volume_A3,
                reference,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise StorageError(
                    "Resume reference volume disagrees with RunOutputHook."
                )
            self.reference_volume_A3 = reference
        self._last_safety_step = resume_step
        self._last_thermo_step = thermo_step
        self._last_trajectory_step = trajectory_step
        self._last_checkpoint_step = resume_step

    def prime_resume_from_checkpoint(
        self,
        checkpoint: CheckpointData,
        *,
        last_thermo_step: int | None = None,
        last_trajectory_step: int | None = None,
    ) -> None:
        """Restore safety/reference state and de-duplication from a checkpoint."""

        if not isinstance(checkpoint, CheckpointData):
            raise TypeError("checkpoint must be CheckpointData.")
        safety_state = checkpoint.protocol_state.get("safety_monitor")
        if not isinstance(safety_state, Mapping):
            raise CheckpointCompatibilityError(
                "Checkpoint has no safety_monitor state mapping."
            )
        self.prime_resume(
            checkpoint.resume_step,
            reference_volume_A3=checkpoint.reference_volume_A3,
            safety_state=safety_state,
            last_thermo_step=last_thermo_step,
            last_trajectory_step=last_trajectory_step,
        )

    def _context_value(self, context: Any, name: str, default: Any = None) -> Any:
        if isinstance(context, Mapping):
            return context.get(name, default)
        return getattr(context, name, default)

    def _record_key(self, context: Any) -> tuple[int, str]:
        return (
            int(self._context_value(context, "global_step")),
            str(self._context_value(context, "stage_name")),
        )

    def _sample(self, atoms: Any, context: Any) -> ThermoRecord:
        key = self._record_key(context)
        if self._last_record_key == key and self._last_record is not None:
            return self._last_record
        record = compute_thermo_record(
            atoms,
            step=key[0],
            time_fs=float(self._context_value(context, "time_fs")),
            stage=key[1],
            stage_step=int(self._context_value(context, "stage_step", 0)),
            target_volume_ratio=_optional_context_float(
                self._context_value(context, "target_volume_ratio")
            ),
            target_pressure_GPa=_optional_context_float(
                self._context_value(context, "target_pressure_GPa")
            ),
            volume_ratio_from_engine_start=_optional_context_float(
                self._context_value(context, "volume_ratio_from_engine_start")
            ),
            branch=_optional_context_string(
                self._context_value(context, "branch")
            ),
        )
        self._last_record = record
        self._last_record_key = key
        return record

    def _initialize_reference_volume(self, record: ThermoRecord, context: Any) -> None:
        context_reference = _optional_context_float(
            self._context_value(context, "reference_volume_A3")
        )
        if context_reference is not None and (
            not np.isfinite(context_reference) or context_reference <= 0.0
        ):
            raise StorageError(
                "Step context reference_volume_A3 must be finite and positive."
            )
        if self.reference_volume_A3 is None:
            if context_reference is not None:
                self.reference_volume_A3 = context_reference
            else:
                ratio = _optional_context_float(
                    self._context_value(context, "volume_ratio_from_engine_start")
                )
                if ratio is None or not np.isfinite(ratio) or ratio <= 0.0:
                    self.reference_volume_A3 = record.volume_A3
                else:
                    self.reference_volume_A3 = record.volume_A3 / ratio
        elif context_reference is not None and not np.isclose(
            self.reference_volume_A3,
            context_reference,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise StorageError(
                "RunOutputHook and engine reference_volume_A3 values disagree."
            )

        monitor_reference = self.safety_monitor.reference_volume_A3
        if monitor_reference is None:
            # No prior state exists when this path is taken; reset establishes
            # both the scientifically meaningful V0 and strain-monitor history.
            self.safety_monitor.reset(
                reference_volume_A3=self.reference_volume_A3
            )
        elif not np.isclose(
            monitor_reference,
            self.reference_volume_A3,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise StorageError(
                "SafetyMonitor and RunOutputHook reference volumes disagree."
            )

    def _check_safety(
        self, atoms: Any, context: Any, record: ThermoRecord
    ) -> tuple[Any, ...]:
        step = int(self._context_value(context, "global_step"))
        if self._last_safety_step == step:
            return ()
        self._initialize_reference_volume(record, context)
        try:
            issues = self.safety_monitor.check(atoms, record)
        except SafetyViolation as violation:
            self._emergency_checkpoint(atoms, context, record, violation)
            raise
        self._last_safety_step = step
        return issues

    def _trajectory_snapshot(self, atoms: Any, record: ThermoRecord) -> Any:
        """Copy a frame and attach only explicitly requested cached results."""

        snapshot = atoms.copy()
        if self.write_momenta:
            snapshot.set_momenta(np.asarray(atoms.get_momenta(), dtype=float).copy())
        else:
            snapshot.arrays.pop("momenta", None)

        results: dict[str, Any] = {"energy": record.potential_energy_eV}
        if self.write_forces:
            results["forces"] = np.asarray(atoms.get_forces(), dtype=float).copy()
        if self.write_stress:
            results["stress"] = np.array(
                [
                    record.stress_xx_GPa,
                    record.stress_yy_GPa,
                    record.stress_zz_GPa,
                    record.stress_yz_GPa,
                    record.stress_xz_GPa,
                    record.stress_xy_GPa,
                ],
                dtype=float,
            ) / EV_A3_TO_GPA
        try:
            from ase.calculators.singlepoint import SinglePointCalculator
        except ImportError as exc:  # pragma: no cover - dependency declaration
            raise StorageError("ASE is required to write trajectory snapshots.") from exc
        snapshot.calc = SinglePointCalculator(snapshot, **results)
        return snapshot

    def _trajectory_metadata(
        self, record: ThermoRecord, context: Any
    ) -> dict[str, Any]:
        metadata = record.trajectory_metadata()
        metadata.update(
            {
                "coldmd_stage_kind": str(
                    self._context_value(context, "stage_kind", "unknown")
                ),
                "coldmd_stage_progress": float(
                    self._context_value(context, "progress", 0.0)
                ),
                "coldmd_volume_ratio_from_engine_start": (
                    record.volume_ratio_from_engine_start
                ),
                "coldmd_resumed": bool(
                    self._context_value(context, "resumed", False)
                ),
                "coldmd_exact_resume_supported": self._exact_resume_supported(
                    context
                ),
                "coldmd_branch": record.branch,
            }
        )
        return metadata

    def _write_thermo(self, record: ThermoRecord, *, force: bool = False) -> None:
        if not force and record.step % self.thermo_interval_steps != 0:
            return
        if self._last_thermo_step == record.step:
            return
        self.thermo_writer.write(record)
        self._last_thermo_step = record.step

    def _write_trajectory(
        self,
        atoms: Any,
        context: Any,
        record: ThermoRecord,
        *,
        force: bool = False,
    ) -> None:
        stage_kind = str(self._context_value(context, "stage_kind", ""))
        interval = (
            self._effective_nvt_trajectory_interval_steps
            if stage_kind == "fixed_nvt"
            else self.trajectory_interval_steps
        )
        if not force and record.step % interval != 0:
            return
        if self._last_trajectory_step == record.step:
            return
        snapshot = self._trajectory_snapshot(atoms, record)
        self.trajectory_writer.write(
            snapshot, metadata=self._trajectory_metadata(record, context)
        )
        self._last_trajectory_step = record.step

    @staticmethod
    def _exact_resume_supported(context: Any) -> bool:
        explicit = getattr(context, "exact_resume_supported", None)
        if explicit is not None:
            return bool(explicit)
        stage_kind = str(
            context.get("stage_kind", "")
            if isinstance(context, Mapping)
            else getattr(context, "stage_kind", "")
        )
        stage_step = int(
            context.get("stage_step", 0)
            if isinstance(context, Mapping)
            else getattr(context, "stage_step", 0)
        )
        stage_steps = int(
            context.get("stage_steps", 0)
            if isinstance(context, Mapping)
            else getattr(context, "stage_steps", 0)
        )
        thermostat_kind = str(
            context.get("thermostat_kind", "")
            if isinstance(context, Mapping)
            else getattr(context, "thermostat_kind", "")
        )
        at_stage_boundary = stage_step in {0, stage_steps}
        has_unstored_extended_state = (
            stage_kind == "isotropic_npt_plateau"
            or (
                stage_kind == "fixed_nvt"
                and thermostat_kind == "nose_hoover_chain"
            )
        )
        return at_stage_boundary or not has_unstored_extended_state

    def _cursor_mapping(self, context: Any) -> dict[str, Any]:
        if hasattr(context, "resume_cursor"):
            cursor = context.resume_cursor()
            if hasattr(cursor, "as_dict"):
                return dict(cursor.as_dict())
            if dataclasses.is_dataclass(cursor):
                return dataclasses.asdict(cursor)
        if hasattr(context, "as_dict"):
            cursor_value = context.as_dict()
            if isinstance(cursor_value, Mapping):
                return dict(cursor_value)
        if dataclasses.is_dataclass(context) and not isinstance(context, type):
            return dataclasses.asdict(context)
        if isinstance(context, Mapping) and "cursor" in context:
            cursor_value = context["cursor"]
            if isinstance(cursor_value, Mapping):
                return dict(cursor_value)
        return {
            "stage_index": int(self._context_value(context, "stage_index", 0)),
            "stage_step": int(self._context_value(context, "stage_step", 0)),
            "global_step": int(self._context_value(context, "global_step", 0)),
            "time_fs": float(self._context_value(context, "time_fs", 0.0)),
            "stage_reference_cell_A": self._context_value(
                context, "stage_reference_cell_A"
            ),
        }

    def _protocol_state(
        self,
        context: Any,
        *,
        event: str,
        exact_resume_supported: bool | None = None,
    ) -> dict[str, Any]:
        resumable = (
            self._exact_resume_supported(context)
            if exact_resume_supported is None
            else bool(exact_resume_supported)
        )
        return {
            "checkpoint_schema_version": 1,
            "event": event,
            "exact_resume_supported": resumable,
            "reference_volume_A3": self.reference_volume_A3,
            "cursor": self._cursor_mapping(context),
            "safety_monitor": self.safety_monitor.state_dict(),
        }

    def _checkpoint(
        self,
        atoms: Any,
        context: Any,
        *,
        event: str,
        force: bool = False,
        allow_same_step: bool = False,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointPaths | None:
        step = int(self._context_value(context, "global_step", 0))
        if not force and step % self.checkpoint_interval_steps != 0:
            return None
        if not self._exact_resume_supported(context):
            self._emit_event(
                "checkpoint_skipped",
                {
                    "step": step,
                    "reason": (
                        "non-exact extended-system state or incomplete "
                        "sampling-window handoff pool (Nose-Hoover chain, MTK, or "
                        "tail selection)"
                    ),
                    "event": event,
                },
            )
            return None
        if self._last_checkpoint_step == step and not allow_same_step:
            return None
        paths = self.checkpoint_manager.save(
            atoms,
            self._protocol_state(context, event=event),
            rng=self.rng,
            config_hash=self.config_hash,
            model_hash=self.model_hash,
            metadata={
                **self.provenance_metadata,
                **dict(extra_metadata or {}),
                "checkpoint_event": event,
                "global_step": step,
                "reference_volume_A3": self.reference_volume_A3,
                "stage_name": str(
                    self._context_value(context, "stage_name", "unknown")
                ),
            },
            versions=self.versions,
        )
        self._last_checkpoint_step = step
        self._emit_event(
            "checkpoint_written",
            {
                "step": step,
                "event": event,
                "manifest": str(paths.manifest),
                "arrays": str(paths.arrays),
            },
        )
        return paths

    def _emergency_checkpoint(
        self,
        atoms: Any,
        context: Any,
        record: ThermoRecord | None,
        error: BaseException,
    ) -> CheckpointPaths | None:
        if not self.checkpoint_on_abort:
            return None
        error_id = id(error)
        if self._last_emergency_error_id == error_id:
            return None
        self._last_emergency_error_id = error_id
        step = int(self._context_value(context, "global_step", 0))
        metadata: dict[str, Any] = {
            **self.provenance_metadata,
            "checkpoint_event": "emergency",
            "global_step": step,
            "stage_name": str(
                self._context_value(context, "stage_name", "unknown")
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "normal_checkpoint_preserved": True,
        }
        if record is not None:
            metadata["last_thermo_record"] = {
                key: (
                    value
                    if value is None
                    or isinstance(value, (str, bool, int))
                    or (isinstance(value, float) and np.isfinite(value))
                    else repr(value)
                )
                for key, value in record.as_dict().items()
            }
        try:
            paths = self.emergency_checkpoint_manager.save(
                atoms,
                self._protocol_state(
                    context,
                    event="emergency",
                    exact_resume_supported=False,
                ),
                rng=self.rng,
                config_hash=self.config_hash,
                model_hash=self.model_hash,
                metadata=metadata,
                versions=self.versions,
            )
            self._emit_event(
                "emergency_checkpoint_written",
                {"step": step, "manifest": str(paths.manifest)},
            )
            return paths
        except Exception as checkpoint_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "Emergency checkpoint could not be written; the latest normal "
                    f"stage-boundary checkpoint remains intact: {checkpoint_error!r}"
                )
            return None

    def checkpoint_failure(
        self,
        atoms: Any,
        context_or_cursor: Any,
        error: BaseException,
    ) -> CheckpointPaths | None:
        """Attempt a separate forensic checkpoint after an engine failure.

        This public entry point is designed for exceptions raised inside
        ``dynamics.step()`` before the engine can emit ``on_step``.  It accepts
        either the last ``StepContext`` or ``engine.cursor``.  The checkpoint
        is always marked non-resumable and never replaces ``checkpoint.json``;
        the latest normal stage-boundary fallback remains intact.
        """

        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception instance.")
        with self._lock:
            cursor_reference = _optional_context_float(
                self._context_value(context_or_cursor, "reference_volume_A3")
            )
            if cursor_reference is not None:
                if not np.isfinite(cursor_reference) or cursor_reference <= 0.0:
                    raise StorageError(
                        "Failure cursor reference_volume_A3 is invalid."
                    )
                if self.reference_volume_A3 is None:
                    self.reference_volume_A3 = cursor_reference
                elif not np.isclose(
                    self.reference_volume_A3,
                    cursor_reference,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise StorageError(
                        "Failure cursor reference volume disagrees with output state."
                    )
            return self._emergency_checkpoint(
                atoms,
                context_or_cursor,
                self._last_record,
                error,
            )

    def _write_stage_structure(
        self, atoms: Any, context: Any, event: str
    ) -> tuple[Path, Path] | None:
        """Write a fork-capable EXTXYZ and a visualization-friendly CIF companion."""

        if not self.write_stage_structures:
            return None
        stage_index = int(self._context_value(context, "stage_index", 0))
        stage_name = _safe_component(
            str(self._context_value(context, "stage_name", "stage"))
        )
        step = int(self._context_value(context, "global_step", 0))
        stem = (
            f"stage-{stage_index:03d}-{stage_name}-{event}-step-{step:012d}"
        )
        extxyz_base = self.stage_structure_directory / f"{stem}.extxyz"
        extxyz_destination, cif_destination = _unused_stage_structure_pair(
            extxyz_base, allocate_suffix=self.append
        )
        if extxyz_destination.exists() or cif_destination.exists():
            raise FileExistsError(
                "Refusing to overwrite stage structure pair: "
                f"{extxyz_destination}, {cif_destination}"
            )
        token = uuid.uuid4().hex
        extxyz_temporary = extxyz_destination.parent / (
            f".{extxyz_destination.name}.tmp-{token}"
        )
        cif_temporary = cif_destination.parent / (
            f".{cif_destination.name}.tmp-{token}"
        )
        try:
            try:
                from ase.io import write as ase_write
            except ImportError as exc:  # pragma: no cover - dependency declaration
                raise StorageError("ASE is required to write stage structures.") from exc
            snapshot = atoms.copy()
            snapshot.calc = None
            ase_write(
                str(extxyz_temporary),
                snapshot,
                format="extxyz",
                parallel=False,
            )
            ase_write(
                str(cif_temporary),
                snapshot,
                format="cif",
                parallel=False,
            )
            _fsync_file(extxyz_temporary)
            _fsync_file(cif_temporary)
            # Publish the lossy visualization file first and the fork-capable
            # EXTXYZ last.  Seeing an EXTXYZ therefore normally implies its CIF
            # companion was already committed.
            os.replace(cif_temporary, cif_destination)
            os.replace(extxyz_temporary, extxyz_destination)
            _fsync_directory(extxyz_destination.parent)
        except Exception:
            _unlink_if_exists(extxyz_temporary)
            _unlink_if_exists(cif_temporary)
            raise
        return extxyz_destination, cif_destination

    def _emit_event(self, name: str, payload: Mapping[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(name, payload)

    def _observe(self, atoms: Any, context: Any) -> ThermoRecord:
        try:
            record = self._sample(atoms, context)
            issues = self._check_safety(atoms, context, record)
        except SafetyViolation:
            raise
        except Exception as error:
            self._emergency_checkpoint(atoms, context, self._last_record, error)
            raise
        if issues:
            self._emit_event(
                "safety_warning",
                {
                    "step": record.step,
                    "issues": [issue.as_dict() for issue in issues],
                },
            )
        return record

    def on_stage_start(self, atoms: Any, context: Any) -> None:
        """Record and checkpoint the initial boundary of a stage."""

        with self._lock:
            self._require_open()
            record = self._observe(atoms, context)
            # Commit the recoverable boundary before ancillary output I/O.
            self._checkpoint(atoms, context, event="stage_start", force=True)
            self._write_thermo(record, force=True)
            self._write_trajectory(atoms, context, record, force=True)
            structures = self._write_stage_structure(atoms, context, "start")
            self._emit_event(
                "stage_start",
                {
                    "step": record.step,
                    "stage": record.stage,
                    "structure": None if structures is None else str(structures[0]),
                    "structure_cif": None if structures is None else str(structures[1]),
                    "structures": (
                        None
                        if structures is None
                        else {"extxyz": str(structures[0]), "cif": str(structures[1])}
                    ),
                },
            )

    def on_step(self, atoms: Any, context: Any) -> None:
        """Apply per-step safety and configured durable-output cadences."""

        with self._lock:
            self._require_open()
            record = self._observe(atoms, context)
            self._write_thermo(record)
            self._write_trajectory(atoms, context, record)
            self._checkpoint(atoms, context, event="interval")

    def on_stage_end(self, atoms: Any, context: Any) -> None:
        """Force all scientific outputs and a resumable boundary checkpoint."""

        with self._lock:
            self._require_open()
            record = self._observe(atoms, context)
            # This checkpoint remains usable even if a later report/snapshot
            # write fails at the otherwise successfully completed boundary.
            self._checkpoint(atoms, context, event="stage_end", force=True)
            self._write_thermo(record, force=True)
            self._write_trajectory(atoms, context, record, force=True)
            structures = self._write_stage_structure(atoms, context, "end")
            self.flush()
            self._emit_event(
                "stage_end",
                {
                    "step": record.step,
                    "stage": record.stage,
                    "structure": None if structures is None else str(structures[0]),
                    "structure_cif": None if structures is None else str(structures[1]),
                    "structures": (
                        None
                        if structures is None
                        else {"extxyz": str(structures[0]), "cif": str(structures[1])}
                    ),
                },
            )

    def on_stage_handoff(
        self,
        atoms: Any,
        context: Any,
        handoff: Mapping[str, Any],
    ) -> None:
        """Commit the sampling-selected state that starts the next stage.

        The preceding ``on_stage_end`` has already preserved the actual final
        MD state and its thermodynamic record.  This second, same-step
        checkpoint intentionally replaces the commit pointer with the selected
        physically consistent sampling snapshot, whose cursor points at the
        following stage.
        """

        with self._lock:
            self._require_open()
            details = dict(handoff)
            # The engine has restored an actual sampling-tail snapshot after
            # observing the chronological final step.  Rebase the strain guard
            # to this selected cell at the boundary so the next real MD step is
            # compared to the state it truly inherits.  The same history is
            # included in the handoff checkpoint for an exact next-stage resume.
            step = int(self._context_value(context, "global_step", 0))
            self.safety_monitor.reset(
                reference_volume_A3=self.reference_volume_A3,
                previous_cell=np.asarray(atoms.get_cell(), dtype=np.float64),
                previous_step=step,
            )
            self._last_safety_step = step
            self._last_record = None
            self._last_record_key = None
            details["safety_history_rebased_to_selected_snapshot"] = True
            paths = self._checkpoint(
                atoms,
                context,
                event="sampling_window_handoff",
                force=True,
                allow_same_step=True,
                extra_metadata={"stage_handoff": details},
            )
            structures = self._write_stage_structure(atoms, context, "handoff")
            self.flush()
            self._emit_event(
                "stage_handoff",
                {
                    **details,
                    "checkpoint": None if paths is None else str(paths.manifest),
                    "structure": None if structures is None else str(structures[0]),
                    "structure_cif": None if structures is None else str(structures[1]),
                    "structures": (
                        None
                        if structures is None
                        else {"extxyz": str(structures[0]), "cif": str(structures[1])}
                    ),
                },
            )

    def load_latest_checkpoint(
        self, *, expected_versions: Mapping[str, str] | None = None
    ) -> CheckpointData:
        """Read the latest committed normal checkpoint with hash checks."""

        return self.checkpoint_manager.load(
            expected_config_hash=self.config_hash,
            expected_model_hash=self.model_hash,
            expected_versions=expected_versions,
        )

    def flush(self) -> None:
        """Synchronize open CSV and trajectory output."""

        with self._lock:
            if self._closed:
                return
            self.thermo_writer.flush()
            self.trajectory_writer.flush()

    def close(self) -> None:
        """Synchronize and close all writers; safe to call repeatedly."""

        with self._lock:
            if self._closed:
                return
            errors: list[BaseException] = []
            for writer in (self.thermo_writer, self.trajectory_writer):
                try:
                    writer.close()
                except BaseException as error:  # preserve both close failures
                    errors.append(error)
            self._closed = True
            if errors:
                failure = StorageError("One or more output writers failed to close.")
                for error in errors:
                    failure.add_note(repr(error))
                raise failure

    def _require_open(self) -> None:
        if self._closed:
            raise StorageError("RunOutputHook is closed.")

    def __enter__(self) -> RunOutputHook:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        del exc_type, traceback
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "Output close also failed while handling the original error: "
                    f"{close_error!r}"
                )


def _optional_context_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_context_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _validated_retained_step(
    value: int | None,
    *,
    name: str,
    resume_step: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer or None.")
    step = int(value)
    if step < 0:
        raise ValueError(f"{name} must be non-negative.")
    if step > resume_step:
        raise StorageError(
            f"{name}={step} lies beyond resume_step={resume_step}; "
            "reconcile or archive post-checkpoint output before resuming."
        )
    return step


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return normalized or "stage"


def _unused_stage_structure_pair(
    extxyz_path: Path, *, allocate_suffix: bool
) -> tuple[Path, Path]:
    """Allocate matching EXTXYZ/CIF names without splitting a resume pair."""

    if extxyz_path.suffix.lower() != ".extxyz":
        raise ValueError("stage structure base path must end in .extxyz")

    def pair(index: int | None) -> tuple[Path, Path]:
        suffix = "" if index is None else f"-resume-{index:03d}"
        stem = f"{extxyz_path.stem}{suffix}"
        return (
            extxyz_path.with_name(f"{stem}.extxyz"),
            extxyz_path.with_name(f"{stem}.cif"),
        )

    initial = pair(None)
    if not allocate_suffix or not any(path.exists() for path in initial):
        return initial
    for index in range(1, 1_000_000):
        candidate = pair(index)
        if not any(path.exists() for path in candidate):
            return candidate
    raise StorageError(
        f"Could not allocate an unused stage structure pair derived from {extxyz_path}."
    )


def collect_version_metadata(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Collect restart provenance without importing numerical backends.

    Besides the headline package versions, the metadata contains the installed
    dependency closure of the MD stack and a digest of every coldmd Python
    source file.  The latter makes an editable-install source change visible
    even when the distribution version string was not bumped.
    """

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "coldmd-source-sha256": _coldmd_source_sha256(),
    }
    for distribution in (
        "ase",
        "torch",
        "mace-torch",
        "cold-compression-md",
        "scipy",
        "pydantic",
        "PyYAML",
        "matplotlib",
        "packaging",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    versions.update(_runtime_dependency_versions())
    if extra is not None:
        versions.update(_string_mapping(extra, "extra versions"))
    return versions


def _coldmd_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = sorted(
        root.rglob("*.py"),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not sources:
        raise StorageError(f"No coldmd Python sources found below {root}")
    for source in sources:
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _runtime_dependency_versions() -> dict[str, str]:
    """Return installed versions in the transitive numerical-runtime closure."""

    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError as exc:  # Declared runtime dependency.
        raise StorageError(
            "packaging is required to fingerprint the restart environment"
        ) from exc

    pending = [
        "cold-compression-md",
        "mace-torch",
        "torch",
        "ase",
        "numpy",
        "scipy",
        "pydantic",
        "PyYAML",
        "matplotlib",
        "packaging",
    ]
    seen: set[str] = set()
    result: dict[str, str] = {}
    while pending:
        requested = pending.pop()
        key = canonicalize_name(requested)
        if key in seen:
            continue
        seen.add(key)
        try:
            distribution = importlib.metadata.distribution(requested)
        except importlib.metadata.PackageNotFoundError:
            continue
        canonical = canonicalize_name(distribution.metadata.get("Name", requested))
        result[f"dependency:{canonical}"] = str(distribution.version)
        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                continue
            try:
                enabled = requirement.marker is None or requirement.marker.evaluate(
                    {"extra": ""}
                )
            except Exception:
                enabled = False
            if enabled:
                pending.append(requirement.name)
    return result


def sha256_file(
    path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024
) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(value: Mapping[str, Any]) -> str:
    """Hash a mapping using coldmd's deterministic JSON representation."""

    payload = json.dumps(
        _to_jsonable(dict(value)),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atoms_to_arrays(atoms: Any) -> tuple[dict[str, NDArray[Any]], bool]:
    try:
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell_obj = atoms.get_cell()
        cell = np.asarray(getattr(cell_obj, "array", cell_obj), dtype=np.float64)
        pbc = np.asarray(atoms.get_pbc(), dtype=np.bool_)
    except Exception as exc:
        raise CheckpointError("Could not extract state from Atoms object.") from exc

    has_method = getattr(atoms, "has", None)
    if callable(has_method):
        has_momenta = bool(has_method("momenta"))
    else:
        atom_arrays = getattr(atoms, "arrays", {})
        has_momenta = isinstance(atom_arrays, Mapping) and "momenta" in atom_arrays

    arrays: dict[str, NDArray[Any]] = {
        "atomic_numbers": np.ascontiguousarray(atomic_numbers),
        "positions": np.ascontiguousarray(positions),
        "cell": np.ascontiguousarray(cell),
        "pbc": np.ascontiguousarray(pbc),
    }
    if has_momenta:
        try:
            arrays["momenta"] = np.ascontiguousarray(
                np.asarray(atoms.get_momenta(), dtype=np.float64)
            )
        except Exception as exc:
            raise CheckpointError("Atoms advertises invalid momenta.") from exc
    _validate_checkpoint_arrays(arrays, len(atomic_numbers), has_momenta)
    return arrays, has_momenta


def _validate_checkpoint_arrays(
    arrays: Mapping[str, NDArray[Any]], natoms: int, has_momenta: bool
) -> None:
    expected_shapes = {
        "atomic_numbers": (natoms,),
        "positions": (natoms, 3),
        "cell": (3, 3),
        "pbc": (3,),
    }
    if has_momenta:
        expected_shapes["momenta"] = (natoms, 3)
    for key, shape in expected_shapes.items():
        if key not in arrays or tuple(arrays[key].shape) != shape:
            actual = None if key not in arrays else tuple(arrays[key].shape)
            raise CheckpointError(
                f"Checkpoint array {key!r} has shape {actual}, expected {shape}."
            )
    if natoms < 1:
        raise CheckpointError("Cannot checkpoint an empty Atoms object.")
    numbers = np.asarray(arrays["atomic_numbers"])
    if not np.issubdtype(numbers.dtype, np.integer) or np.any(numbers < 1):
        raise CheckpointError("Atomic numbers must be positive integers.")
    for key in ("positions", "cell", "momenta"):
        if key in arrays and not np.all(np.isfinite(arrays[key])):
            raise CheckpointError(f"Checkpoint array {key!r} is not finite.")
    cell = np.asarray(arrays["cell"], dtype=np.float64)
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise CheckpointError(
            "Checkpoint cell must have a finite, positive determinant."
        )


def _capture_rng_state(rng: Any | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if rng is None:
        return None
    if isinstance(rng, Mapping):
        return {"kind": "mapping", "state": dict(rng)}
    if isinstance(rng, np.random.Generator):
        return {
            "kind": "numpy_generator",
            "bit_generator": type(rng.bit_generator).__name__,
            "state": rng.bit_generator.state,
        }
    if isinstance(rng, np.random.RandomState):
        return {"kind": "numpy_randomstate", "state": rng.get_state()}
    if isinstance(rng, random.Random):
        return {"kind": "python_random", "state": rng.getstate()}
    raise TypeError(
        "rng must be numpy.random.Generator, numpy.random.RandomState, "
        "random.Random, a state mapping, or None."
    )


def _apply_rng_state(rng: Any, saved: Mapping[str, Any]) -> None:
    kind = saved.get("kind")
    state = saved.get("state")
    if kind == "numpy_generator":
        if not isinstance(rng, np.random.Generator):
            raise CheckpointCompatibilityError("Saved RNG requires numpy Generator.")
        expected = saved.get("bit_generator")
        actual = type(rng.bit_generator).__name__
        if expected != actual:
            raise CheckpointCompatibilityError(
                f"Bit-generator mismatch: saved={expected!r}, target={actual!r}."
            )
        rng.bit_generator.state = state
        return
    if kind == "numpy_randomstate":
        if not isinstance(rng, np.random.RandomState):
            raise CheckpointCompatibilityError("Saved RNG requires numpy RandomState.")
        rng.set_state(state)
        return
    if kind == "python_random":
        if not isinstance(rng, random.Random):
            raise CheckpointCompatibilityError("Saved RNG requires random.Random.")
        rng.setstate(state)
        return
    if kind == "mapping":
        raise CheckpointCompatibilityError(
            "An opaque mapping RNG state cannot be applied automatically."
        )
    raise CheckpointCompatibilityError(f"Unknown saved RNG kind: {kind!r}.")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite floating-point values are not valid JSON state.")
        return value
    if isinstance(value, np.generic):
        return {
            _JSON_TYPE_TAG: "numpy_scalar",
            "dtype": str(value.dtype),
            "value": _to_jsonable(value.item()),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype arrays cannot be serialized safely.")
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            raise ValueError("Non-finite arrays are not valid JSON state.")
        return {
            _JSON_TYPE_TAG: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, tuple):
        return {
            _JSON_TYPE_TAG: "tuple",
            "items": [_to_jsonable(item) for item in value],
        }
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON-state mapping keys must be strings.")
            if key == _JSON_TYPE_TAG:
                raise ValueError(f"Reserved JSON-state key used: {_JSON_TYPE_TAG}")
            result[key] = _to_jsonable(item)
        return result
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported checkpoint JSON value: {type(value).__name__}")


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_JSON_TYPE_TAG)
    if tag is None:
        return {str(key): _from_jsonable(item) for key, item in value.items()}
    if tag == "tuple":
        if set(value) != {_JSON_TYPE_TAG, "items"} or not isinstance(
            value["items"], list
        ):
            raise CheckpointError("Malformed tuple in checkpoint JSON.")
        return tuple(_from_jsonable(item) for item in value["items"])
    if tag == "numpy_scalar":
        if set(value) != {_JSON_TYPE_TAG, "dtype", "value"}:
            raise CheckpointError("Malformed numpy scalar in checkpoint JSON.")
        try:
            return np.asarray(value["value"], dtype=np.dtype(value["dtype"]))[()]
        except (TypeError, ValueError) as exc:
            raise CheckpointError("Invalid numpy scalar dtype/value.") from exc
    if tag == "ndarray":
        if set(value) != {_JSON_TYPE_TAG, "dtype", "shape", "data"}:
            raise CheckpointError("Malformed ndarray in checkpoint JSON.")
        try:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise TypeError("object dtype")
            array = np.asarray(value["data"], dtype=dtype)
            shape = tuple(int(item) for item in value["shape"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise CheckpointError("Invalid ndarray dtype/data in JSON.") from exc
        if array.shape != shape:
            raise CheckpointError(
                f"JSON ndarray shape mismatch: data={array.shape}, declared={shape}."
            )
        return array
    raise CheckpointError(f"Unknown checkpoint JSON type tag: {tag!r}.")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(
        _to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_mapping(record: Any) -> Mapping[str, Any]:
    """Adapt mappings and record dataclasses without importing observables."""

    if isinstance(record, Mapping):
        result = record
    else:
        as_dict = getattr(record, "as_dict", None)
        if callable(as_dict):
            result = as_dict()
        elif dataclasses.is_dataclass(record) and not isinstance(record, type):
            result = dataclasses.asdict(record)
        else:
            raise TypeError(
                "Thermo record must be a mapping, expose as_dict(), or be a "
                "dataclass instance."
            )
    if not isinstance(result, Mapping):
        raise TypeError("Thermo record adapter did not return a mapping.")
    if any(not isinstance(key, str) for key in result):
        raise TypeError("Thermo record keys must be strings.")
    return result


def _validate_fieldnames(fieldnames: Sequence[str] | None) -> list[str] | None:
    if fieldnames is None:
        return None
    result = [str(item) for item in fieldnames]
    if not result or any(not item for item in result):
        raise ValueError("CSV fieldnames must be non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError("CSV fieldnames must be unique.")
    return result


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256.")
    return value.lower()


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"{name} keys and values must be strings.")
        result[key] = item
    return result


def _write_npz_and_fsync(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        _unlink_if_exists(path)
        raise


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        _unlink_if_exists(temporary)
        raise


def _manifest_references(path: Path, arrays_name: str) -> bool:
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream, parse_constant=_reject_json_constant)
        return manifest.get("arrays", {}).get("file") == arrays_name
    except (OSError, AttributeError, ValueError):
        return False


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    """Persist directory entries on POSIX; best effort on other platforms."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some platforms/filesystems do not support fsync on directories.
        pass
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    """Best-effort file-content synchronization after third-party writers close."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON numeric constant is forbidden: {value}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointData",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointPaths",
    "EventCallback",
    "RunOutputHook",
    "SegmentedTrajectoryWriter",
    "StorageError",
    "ThermoCSVWriter",
    "TrajectoryFrameRef",
    "collect_version_metadata",
    "sha256_file",
    "sha256_mapping",
]
