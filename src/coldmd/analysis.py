"""Offline structural and transport analysis for cold-compression trajectories.

The routines in this module intentionally operate on stored trajectories.  They do
not alter an MD run and they do not infer a thermodynamic phase from loss of
crystallinity alone.  Transport classifications are limited to what is observable
on the supplied trajectory time scale.

ASE ``.traj`` files written by :mod:`coldmd` contain the following frame metadata::

    coldmd_step, coldmd_time_fs, coldmd_stage, coldmd_stage_step,
    coldmd_target_volume_ratio, coldmd_target_pressure_GPa

The loader is deliberately tolerant of older files using ``step`` or
``coldmd_stage``.  Segment-boundary duplicates are removed when their global step
is available.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

try:
    from ase import Atoms
    from ase.geometry import find_mic
    from ase.io import iread
except ImportError as exc:  # pragma: no cover - exercised by packaging smoke tests
    raise ImportError(
        "coldmd.analysis requires ASE. Install the project dependencies first."
    ) from exc


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TransportStatus = Literal[
    "diffusive", "nondiffusive_on_timescale", "undetermined"
]


class AnalysisError(RuntimeError):
    """Raised when stored data cannot support the requested analysis."""


def _json_value(value: Any) -> Any:
    """Convert NumPy/dataclass values into strict JSON-compatible objects."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(slots=True)
class FrameRecord:
    """One ASE frame together with provenance needed for stage selection."""

    atoms: Atoms
    segment: Path
    segment_index: int
    frame_index: int
    step: int | None
    time_fs: float | None
    stage: str
    stage_step: int | None = None
    stage_kind: str | None = None


@dataclass(slots=True)
class TrajectoryDataset:
    """In-memory view of one or more trajectory segments."""

    records: list[FrameRecord]
    files: list[Path]
    thermo_rows: list[dict[str, Any]] = field(default_factory=list)

    def stages(self) -> list[str]:
        return list(dict.fromkeys(record.stage for record in self.records))

    def select(self, stages: str | Sequence[str] | None = None) -> list[FrameRecord]:
        return select_stage_records(self.records, stages)


@dataclass(slots=True)
class RDFResult:
    """Element-pair partial radial distribution functions."""

    r_A: FloatArray
    bin_edges_A: FloatArray
    g: dict[str, FloatArray]
    counts: dict[str, FloatArray]
    ideal_counts: dict[str, FloatArray]
    n_frames: int
    rmax_A: float
    stages: list[str]
    pair_symbols: dict[str, tuple[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class CoordinationPairResult:
    """Coordination-number distribution for a directed center-neighbor pair."""

    center: str
    neighbor: str
    cutoff_A: float
    coordination: IntArray
    count: IntArray
    probability: FloatArray
    mean: float
    std: float
    n_centers: int
    per_frame_mean: FloatArray
    cutoff_source: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class CoordinationResult:
    pairs: dict[str, CoordinationPairResult]
    n_frames: int
    stages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class DiffusionEstimate:
    """Block-estimated 3-D diffusion coefficient and classification."""

    status: TransportStatus
    coefficient_A2_ps: float | None
    coefficient_cm2_s: float | None
    standard_error_A2_ps: float | None
    confidence95_A2_ps: tuple[float, float] | None
    upper_bound_A2_ps: float | None
    slope_A2_ps: float | None
    intercept_A2: float | None
    r_squared: float | None
    fit_start_ps: float | None
    fit_end_ps: float | None
    block_coefficients_A2_ps: list[float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class MSDResult:
    lag_time_ps: FloatArray
    msd_A2: dict[str, FloatArray]
    diffusion: dict[str, DiffusionEstimate]
    n_frames: int
    observation_time_ps: float
    stages: list[str]
    cell_relative_variation: float
    coordinate_method: str
    origin_mode: str
    diffusion_estimator: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class StructureFactorEstimate:
    k_inv_A: float
    q_inv_A: FloatArray
    S_q: FloatArray
    method: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class SelfIntermediateScatteringResult:
    lag_time_ps: FloatArray
    k_inv_A: float
    Fs: dict[str, FloatArray]
    n_frames: int
    observation_time_ps: float
    stages: list[str]
    k_source: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class HysteresisResult:
    time_ps: FloatArray
    pressure_GPa: FloatArray
    volume_A3: FloatArray
    volume_ratio: FloatArray
    density_g_cm3: FloatArray
    stage: list[str]
    branch: list[str]
    pv_path_integral_GPa_A3: float | None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class StageThermoSummary:
    """Tail-window thermodynamic statistics for one configured stage.

    Standard deviations are *sample* standard deviations (``ddof=1``).  A
    standard deviation is ``None`` rather than zero when fewer than two finite
    observations exist, so a one-point stage cannot be mistaken for a converged
    distribution.
    """

    stage_order: int
    stage: str
    kind: str | None
    branch: str
    target_pressure_GPa: float | None
    duration_ps: float | None
    sampling_window_ps: float | None
    selection_method: str
    sampling_start_stage_step: int | None
    n_samples: int
    n_pressure: int
    n_volume: int
    n_density: int
    n_temperature: int
    first_sample_time_ps: float | None
    last_sample_time_ps: float | None
    realized_sample_span_ps: float | None
    pressure_mean_GPa: float | None
    pressure_std_GPa: float | None
    volume_mean_A3: float | None
    volume_std_A3: float | None
    density_mean_g_cm3: float | None
    density_std_g_cm3: float | None
    temperature_mean_K: float | None
    temperature_std_K: float | None
    pressure_source: str | None
    available_rows: int
    dropped_invalid_rows: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class StageThermoSummaryResult:
    """All-stage thermodynamic summaries in resolved protocol order."""

    summaries: list[StageThermoSummary]
    source_row_count: int
    retained_row_count: int
    dropped_duplicate_rows: int
    protocol_stage_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class OxygenSpeciationFrame:
    """Instantaneous oxygen topology counts for one stored trajectory frame."""

    step: int | None
    time_fs: float | None
    time_ps: float | None
    stage: str
    stage_step: int | None
    si_o_cutoff_A: float
    cutoff_source: str
    n_oxygen: int
    n_nbo: int
    n_bo: int
    n_other: int
    fraction_nbo: float | None
    fraction_bo: float | None
    fraction_other: float | None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(slots=True)
class OxygenSpeciationResult:
    """Frame-wise NBO/BO/O-other classification and cutoff provenance."""

    frames: list[OxygenSpeciationFrame]
    n_frames: int
    stages: list[str]
    cutoffs_A_by_stage: dict[str, float]
    cutoff_source_by_stage: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def discover_trajectory_files(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve a file, directory, or explicit list into naturally sorted segments.

    A directory search prioritizes the canonical ``trajectory-000000.traj``
    segmented layout. Falling back to recursive ``*.traj``/``*.extxyz`` search
    makes the analyzer useful for imported ASE runs without silently reading
    generated analysis artifacts.
    """

    if isinstance(source, (str, Path)):
        sources: list[Path] = [Path(source)]
    else:
        sources = [Path(item) for item in source]

    files: list[Path] = []
    for item in sources:
        if item.is_file():
            files.append(item)
            continue
        if not item.exists():
            raise FileNotFoundError(f"Trajectory source does not exist: {item}")
        if not item.is_dir():
            raise AnalysisError(f"Unsupported trajectory source: {item}")

        segment_pattern = re.compile(r"^.+-\d{6}\.(?:traj|extxyz|xyz)$", re.IGNORECASE)
        canonical = [
            path
            for directory in (item, item / "trajectory")
            if directory.is_dir()
            for path in directory.iterdir()
            if path.is_file() and segment_pattern.fullmatch(path.name)
        ]
        if canonical:
            files.extend(canonical)
            continue

        discovered: list[Path] = []
        for pattern in ("*.traj", "*.extxyz", "*.xyz"):
            discovered.extend(
                path
                for path in item.rglob(pattern)
                if not {
                    "analysis",
                    "checkpoints",
                    "discarded_attempts",
                    "stage_structures",
                }.intersection(part.lower() for part in path.parts)
            )
        files.extend(discovered)

    unique = sorted({path.resolve() for path in files}, key=_natural_key)
    if not unique:
        raise FileNotFoundError(f"No ASE trajectory files found under {source!r}")
    return unique


def _typed_csv_value(key: str, value: str) -> Any:
    if value == "":
        return None
    if key in {"step", "stage_step"}:
        try:
            return int(value)
        except ValueError:
            return value
    if key == "stage":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def load_thermo_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load canonical ``thermo.csv`` while preserving unknown future columns."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisError(f"Thermo CSV has no header: {csv_path}")
        return [
            {key: _typed_csv_value(key, value or "") for key, value in row.items()}
            for row in reader
        ]


def _infer_stage_from_filename(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(?:stage[-_])([A-Za-z0-9_.-]+)", stem)
    if match:
        return match.group(1)
    for token in (
        "initial_nvt",
        "compression",
        "peak_nvt",
        "peak_hold",
        "decompression",
        "recovery_npt",
        "final_nvt",
        "final_aging",
    ):
        if token in stem.lower():
            return token
    return "unknown"


def _load_segment_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """Load a :class:`SegmentedTrajectoryWriter` JSONL sidecar when present."""

    sidecar = path.with_name(f"{path.stem}.meta.jsonl")
    if not sidecar.exists():
        return {}
    result: dict[int, dict[str, Any]] = {}
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                frame = int(payload["frame"])
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise TypeError("metadata is not an object")
                if frame in result:
                    raise AnalysisError(
                        f"Duplicate frame {frame} in metadata sidecar {sidecar}"
                    )
                result[frame] = dict(metadata)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError(
            f"Could not read trajectory metadata sidecar {sidecar}: {exc}"
        ) from exc
    return result


def _metadata_number(
    info: Mapping[str, Any], keys: Sequence[str], integer: bool = False
) -> int | float | None:
    for key in keys:
        if key not in info or info[key] is None:
            continue
        try:
            return int(info[key]) if integer else float(info[key])
        except (TypeError, ValueError):
            continue
    return None


def _frames_equal(left: Atoms, right: Atoms, atol: float = 1.0e-10) -> bool:
    return (
        len(left) == len(right)
        and left.get_chemical_symbols() == right.get_chemical_symbols()
        and np.array_equal(left.pbc, right.pbc)
        and np.allclose(left.cell.array, right.cell.array, rtol=0.0, atol=atol)
        and np.allclose(left.positions, right.positions, rtol=0.0, atol=atol)
    )


def _match_stage(stage: str, selectors: Sequence[str]) -> bool:
    stage_lower = stage.lower()
    return any(fnmatch(stage_lower, selector.lower()) for selector in selectors)


def select_stage_records(
    records: Sequence[FrameRecord], stages: str | Sequence[str] | None = None
) -> list[FrameRecord]:
    """Select records using exact names or shell-style patterns."""

    if stages is None:
        return list(records)
    selectors = [stages] if isinstance(stages, str) else list(stages)
    chosen = [record for record in records if _match_stage(record.stage, selectors)]
    if not chosen:
        available = list(dict.fromkeys(record.stage for record in records))
        raise AnalysisError(
            f"No trajectory frames match stage selector(s) {selectors}; "
            f"available stages: {available}"
        )
    return chosen


def load_segmented_trajectories(
    source: str | Path | Sequence[str | Path],
    *,
    stages: str | Sequence[str] | None = None,
    stride: int = 1,
    thermo_csv: str | Path | None = None,
    deduplicate: bool = True,
) -> TrajectoryDataset:
    """Load ASE trajectory segments and attach stage/time metadata.

    ``stride`` is applied globally after duplicate removal, so changing the segment
    boundaries does not change the sampled frame sequence.
    """

    if stride < 1:
        raise ValueError("stride must be at least 1")
    files = discover_trajectory_files(source)

    thermo_rows: list[dict[str, Any]] = []
    if thermo_csv is None and isinstance(source, (str, Path)):
        root = Path(source)
        if root.is_file():
            root = root.parent
        candidates = (root / "thermo.csv", root.parent / "thermo.csv")
        thermo_csv = next((path for path in candidates if path.exists()), None)
    if thermo_csv is not None and Path(thermo_csv).exists():
        thermo_rows = load_thermo_csv(thermo_csv)
    thermo_by_step = {
        int(row["step"]): row
        for row in thermo_rows
        if isinstance(row.get("step"), (int, float))
    }

    raw: list[FrameRecord] = []
    seen_steps: set[int] = set()
    for segment_index, path in enumerate(files):
        inferred_stage = _infer_stage_from_filename(path)
        segment_metadata = _load_segment_metadata(path)
        try:
            iterator = iread(str(path), index=":")
            for frame_index, atoms in enumerate(iterator):
                info = dict(atoms.info)
                info.update(segment_metadata.get(frame_index, {}))
                step_value = _metadata_number(
                    info, ("coldmd_step", "global_step", "step"), integer=True
                )
                step = int(step_value) if step_value is not None else None
                row = thermo_by_step.get(step) if step is not None else None
                stage = str(
                    info.get("coldmd_stage")
                    or info.get("stage")
                    or info.get("stage_name")
                    or (row or {}).get("stage")
                    or inferred_stage
                )
                time_value = _metadata_number(
                    info, ("coldmd_time_fs", "time_fs"), integer=False
                )
                if time_value is None and row is not None:
                    time_value = row.get("time_fs")
                time_fs = float(time_value) if time_value is not None else None
                stage_step_value = _metadata_number(
                    info, ("coldmd_stage_step", "stage_step"), integer=True
                )
                if stage_step_value is None and row is not None:
                    stage_step_value = row.get("stage_step")
                stage_step = (
                    int(stage_step_value) if stage_step_value is not None else None
                )
                stage_kind_value = info.get("coldmd_stage_kind") or info.get("stage_kind")
                stage_kind = (
                    None
                    if stage_kind_value is None or not str(stage_kind_value).strip()
                    else str(stage_kind_value).strip()
                )

                if deduplicate and step is not None and step in seen_steps:
                    continue
                if (
                    deduplicate
                    and step is None
                    and raw
                    and time_fs is not None
                    and raw[-1].time_fs == time_fs
                    and _frames_equal(raw[-1].atoms, atoms)
                ):
                    continue
                if step is not None:
                    seen_steps.add(step)
                raw.append(
                    FrameRecord(
                        atoms=atoms.copy(),
                        segment=path,
                        segment_index=segment_index,
                        frame_index=frame_index,
                        step=step,
                        time_fs=time_fs,
                        stage=stage,
                        stage_step=stage_step,
                        stage_kind=stage_kind,
                    )
                )
        except Exception as exc:
            raise AnalysisError(f"Failed to read trajectory segment {path}: {exc}") from exc

    selected = select_stage_records(raw, stages)
    records = selected[::stride]
    if not records:
        raise AnalysisError("No trajectory frames remain after selection/striding")
    return TrajectoryDataset(records=records, files=files, thermo_rows=thermo_rows)


def _records(
    value: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
) -> list[FrameRecord]:
    if isinstance(value, TrajectoryDataset):
        return list(value.records)
    items = list(value)
    if not items:
        raise AnalysisError("At least one trajectory frame is required")
    if isinstance(items[0], FrameRecord):
        return list(items)  # type: ignore[return-value]
    if isinstance(items[0], Atoms):
        records: list[FrameRecord] = []
        for index, atoms in enumerate(items):  # type: ignore[arg-type]
            info = atoms.info
            step_value = _metadata_number(
                info, ("coldmd_step", "global_step", "step"), integer=True
            )
            time_value = _metadata_number(
                info, ("coldmd_time_fs", "time_fs"), integer=False
            )
            stage_step_value = _metadata_number(
                info, ("coldmd_stage_step", "stage_step"), integer=True
            )
            records.append(
                FrameRecord(
                    atoms=atoms,
                    segment=Path("<memory>"),
                    segment_index=0,
                    frame_index=index,
                    step=int(step_value) if step_value is not None else index,
                    time_fs=float(time_value) if time_value is not None else None,
                    stage=str(
                        info.get("coldmd_stage")
                        or info.get("stage")
                        or info.get("stage_name")
                        or "unknown"
                    ),
                    stage_step=(
                        int(stage_step_value) if stage_step_value is not None else None
                    ),
                )
            )
        return records
    raise TypeError("Expected a TrajectoryDataset, FrameRecord sequence, or Atoms sequence")


def _validate_frame_compatibility(records: Sequence[FrameRecord]) -> list[str]:
    if not records:
        raise AnalysisError("No frames supplied")
    first = records[0].atoms
    symbols = first.get_chemical_symbols()
    if not first.pbc.all():
        raise AnalysisError("Cold-compression analysis requires periodicity in all 3 axes")
    if abs(first.cell.volume) <= 0.0:
        raise AnalysisError("Trajectory contains a singular cell")
    for record in records[1:]:
        atoms = record.atoms
        if atoms.get_chemical_symbols() != symbols:
            raise AnalysisError("Atom count/order/species changed within the selected frames")
        if not atoms.pbc.all():
            raise AnalysisError("All selected frames must be periodic in all 3 axes")
        if abs(atoms.cell.volume) <= 0.0:
            raise AnalysisError("Trajectory contains a singular cell")
    return symbols


def _cell_heights(cell: FloatArray) -> FloatArray:
    # Rows of inv(cell).T are reciprocal vectors without the 2*pi factor.
    reciprocal = np.linalg.inv(cell).T
    return 1.0 / np.linalg.norm(reciprocal, axis=1)


def safe_mic_radius(records: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms]) -> float:
    """Largest cutoff for which a single minimum-image distance is unambiguous."""

    recs = _records(records)
    _validate_frame_compatibility(recs)
    return 0.5 * min(
        float(np.min(_cell_heights(record.atoms.cell.array))) for record in recs
    )


def _canonical_pair(pair: Sequence[str]) -> tuple[str, str]:
    if len(pair) != 2:
        raise ValueError(f"Element pair must contain exactly two symbols: {pair}")
    left, right = str(pair[0]), str(pair[1])
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def _sample_records(records: Sequence[FrameRecord], maximum: int | None) -> list[FrameRecord]:
    if maximum is None or len(records) <= maximum:
        return list(records)
    if maximum < 1:
        raise ValueError("maximum frame count must be positive")
    indices = np.unique(np.linspace(0, len(records) - 1, maximum, dtype=int))
    return [records[index] for index in indices]


def partial_rdf(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    pairs: Sequence[Sequence[str]] | None = None,
    bins: int = 300,
    rmax_A: float | None = None,
    include_total: bool = True,
    max_frames: int | None = None,
) -> RDFResult:
    """Calculate normalized element-pair partial RDFs.

    Distances use ASE's triclinic minimum-image implementation.  ``rmax_A`` may
    not exceed half the smallest perpendicular cell height because this direct
    pair histogram includes only one image of each atom.
    """

    if bins < 10:
        raise ValueError("bins must be at least 10")
    recs = select_stage_records(_records(frames), stages)
    recs = _sample_records(recs, max_frames)
    symbols = np.asarray(_validate_frame_compatibility(recs), dtype=object)
    elements = sorted(set(symbols.tolist()))
    if pairs is None:
        requested = [
            (left, right)
            for left_index, left in enumerate(elements)
            for right in elements[left_index:]
        ]
    else:
        requested = list(dict.fromkeys(_canonical_pair(pair) for pair in pairs))
    unknown = sorted({item for pair in requested for item in pair} - set(elements))
    if unknown:
        raise AnalysisError(f"RDF pair contains species absent from trajectory: {unknown}")

    safe_radius = safe_mic_radius(recs)
    if rmax_A is None:
        rmax_A = 0.98 * safe_radius
    if rmax_A <= 0.0:
        raise ValueError("rmax_A must be positive")
    if rmax_A > safe_radius * (1.0 + 1.0e-10):
        raise AnalysisError(
            f"rmax_A={rmax_A:.6g} exceeds the safe minimum-image radius "
            f"{safe_radius:.6g} A"
        )

    edges = np.linspace(0.0, float(rmax_A), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    keys = [_pair_key(pair) for pair in requested]
    counts = {key: np.zeros(bins, dtype=float) for key in keys}
    ideal = {key: np.zeros(bins, dtype=float) for key in keys}
    if include_total:
        counts["ALL-ALL"] = np.zeros(bins, dtype=float)
        ideal["ALL-ALL"] = np.zeros(bins, dtype=float)

    atom_i, atom_j = np.triu_indices(len(symbols), k=1)
    pair_masks: dict[str, NDArray[np.bool_]] = {}
    for pair, key in zip(requested, keys, strict=True):
        left, right = pair
        if left == right:
            mask = (symbols[atom_i] == left) & (symbols[atom_j] == right)
        else:
            mask = (
                ((symbols[atom_i] == left) & (symbols[atom_j] == right))
                | ((symbols[atom_i] == right) & (symbols[atom_j] == left))
            )
        pair_masks[key] = mask

    species_counts = {element: int(np.count_nonzero(symbols == element)) for element in elements}
    for record in recs:
        atoms = record.atoms
        displacements = atoms.positions[atom_j] - atoms.positions[atom_i]
        _, distances = find_mic(displacements, atoms.cell, atoms.pbc)
        inverse_volume = 1.0 / float(atoms.get_volume())
        for pair, key in zip(requested, keys, strict=True):
            histogram, _ = np.histogram(distances[pair_masks[key]], bins=edges)
            counts[key] += histogram
            left, right = pair
            if left == right:
                possible_pairs = species_counts[left] * (species_counts[left] - 1) / 2.0
            else:
                possible_pairs = species_counts[left] * species_counts[right]
            ideal[key] += possible_pairs * inverse_volume * shell_volume
        if include_total:
            histogram, _ = np.histogram(distances, bins=edges)
            counts["ALL-ALL"] += histogram
            possible_pairs = len(symbols) * (len(symbols) - 1) / 2.0
            ideal["ALL-ALL"] += possible_pairs * inverse_volume * shell_volume

    curves = {
        key: np.divide(
            counts[key],
            ideal[key],
            out=np.zeros_like(counts[key]),
            where=ideal[key] > 0.0,
        )
        for key in counts
    }
    pair_symbols = {key: pair for key, pair in zip(keys, requested, strict=True)}
    if include_total:
        pair_symbols["ALL-ALL"] = ("ALL", "ALL")
    return RDFResult(
        r_A=centers,
        bin_edges_A=edges,
        g=curves,
        counts=counts,
        ideal_counts=ideal,
        n_frames=len(recs),
        rmax_A=float(rmax_A),
        stages=list(dict.fromkeys(record.stage for record in recs)),
        pair_symbols=pair_symbols,
        metadata={
            "normalization": "ideal-gas pair count per shell, accumulated per frame",
            "safe_mic_radius_A": safe_radius,
            "sampled_frames": len(recs),
        },
    )


def estimate_first_minimum(r_A: FloatArray, g_r: FloatArray) -> float:
    """Estimate the first-shell cutoff from a smoothed RDF.

    The estimate is intentionally conservative and raises when no defensible first
    minimum is present.  Callers should then provide a chemistry-informed cutoff.
    """

    r = np.asarray(r_A, dtype=float)
    g = np.asarray(g_r, dtype=float)
    finite = np.isfinite(r) & np.isfinite(g)
    r, g = r[finite], g[finite]
    if len(r) < 12 or float(np.max(g, initial=0.0)) <= 0.0:
        raise AnalysisError("RDF is insufficient to estimate a first minimum")

    window = min(21, len(g) - (1 - len(g) % 2))
    if window < 5:
        smooth = g
    else:
        smooth = signal.savgol_filter(g, window_length=window, polyorder=3)
    maximum = float(np.max(smooth))
    peaks, properties = signal.find_peaks(
        smooth,
        prominence=max(0.03 * maximum, 1.0e-8),
        height=max(0.10 * maximum, 1.0e-8),
    )
    if not len(peaks):
        raise AnalysisError("No significant RDF peak found for cutoff estimation")
    peak = int(peaks[0])
    minima, _ = signal.find_peaks(-smooth[peak + 1 :], prominence=0.01 * maximum)
    if len(minima):
        minimum = peak + 1 + int(minima[0])
    else:
        stop = min(len(smooth), peak + max(4, int(0.8 * peak) + 2))
        if stop <= peak + 2:
            raise AnalysisError("No RDF range remains beyond the first peak")
        minimum = peak + 1 + int(np.argmin(smooth[peak + 1 : stop]))
    if minimum <= peak or r[minimum] <= r[peak]:
        raise AnalysisError("Unable to locate a first RDF minimum")
    return float(r[minimum])


# The order is deliberate: it is the conventional order used in the report and
# produces stable CSV/figure column names.  These are instantaneous labels, not
# chemical elements; the classifications are recomputed for every frame.
DEFAULT_TOPOLOGY_RDF_PAIRS: tuple[tuple[str, str], ...] = (
    ("Ca", "NBO"),
    ("Ca", "BO"),
    ("Si", "NBO"),
    ("Si", "BO"),
    ("NBO", "NBO"),
    ("NBO", "BO"),
    ("BO", "BO"),
)
_OXYGEN_TOPOLOGY_LABELS = frozenset(("NBO", "BO", "O_other"))


def _finite_float(value: Any) -> float | None:
    """Return one finite scalar, treating booleans and malformed cells as absent."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _frame_time_ps(record: FrameRecord) -> float | None:
    return None if record.time_fs is None else _finite_float(record.time_fs / 1000.0)


def _stage_cutoff_map(
    records: Sequence[FrameRecord],
    *,
    si_o_cutoff_A: float | Mapping[str, float] | None,
    rdf_bins: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve either an explicit Si--O cutoff or a first minimum per stage."""

    symbols = set(_validate_frame_compatibility(records))
    missing = {"Si", "O"} - symbols
    if missing:
        raise AnalysisError(
            "Oxygen topology requires both Si and O in every selected frame; "
            f"missing {sorted(missing)}"
        )
    stages = list(dict.fromkeys(record.stage for record in records))
    grouped = {
        stage: [record for record in records if record.stage == stage]
        for stage in stages
    }
    cutoffs: dict[str, float] = {}
    sources: dict[str, str] = {}

    if isinstance(si_o_cutoff_A, Mapping):
        unknown = set(str(name) for name in si_o_cutoff_A) - set(stages)
        if unknown:
            raise AnalysisError(
                "Stage-specific Si-O cutoff mapping names stages absent from the "
                f"selection: {sorted(unknown)}"
            )
        for stage in stages:
            if stage not in si_o_cutoff_A:
                raise AnalysisError(
                    f"No fixed Si-O cutoff was supplied for selected stage {stage!r}"
                )
            cutoff = _finite_float(si_o_cutoff_A[stage])
            if cutoff is None or cutoff <= 0.0:
                raise ValueError(f"Si-O cutoff must be a positive finite value for {stage!r}")
            cutoffs[stage] = cutoff
            sources[stage] = "stage-specific si_o_cutoff_A override"
    elif si_o_cutoff_A is not None:
        cutoff = _finite_float(si_o_cutoff_A)
        if cutoff is None or cutoff <= 0.0:
            raise ValueError("si_o_cutoff_A must be a positive finite value")
        for stage in stages:
            cutoffs[stage] = cutoff
            sources[stage] = "fixed si_o_cutoff_A override"
    else:
        for stage, stage_records in grouped.items():
            # A stage-specific RDF is important during compression: a cutoff
            # inferred at 0 GPa is not automatically transferable at 10 GPa.
            rdf = partial_rdf(
                stage_records,
                pairs=[("Si", "O")],
                bins=rdf_bins,
                include_total=False,
            )
            key = _pair_key(_canonical_pair(("Si", "O")))
            try:
                cutoff = estimate_first_minimum(rdf.r_A, rdf.g[key])
            except AnalysisError as exc:
                raise AnalysisError(
                    f"Could not infer a stage-specific Si-O first minimum for "
                    f"{stage!r}; supply si_o_cutoff_A explicitly. {exc}"
                ) from exc
            cutoffs[stage] = cutoff
            sources[stage] = "first minimum of stage-specific Si-O RDF"

    for stage, stage_records in grouped.items():
        safe_radius = safe_mic_radius(stage_records)
        if cutoffs[stage] > safe_radius * (1.0 + 1.0e-10):
            raise AnalysisError(
                f"Si-O cutoff {cutoffs[stage]:.6g} A for stage {stage!r} exceeds "
                f"the safe minimum-image radius {safe_radius:.6g} A"
            )
    return cutoffs, sources


def _classify_oxygen_frame(
    record: FrameRecord,
    *,
    cutoff_A: float,
    cutoff_source: str,
) -> tuple[NDArray[np.object_], OxygenSpeciationFrame]:
    """Classify oxygen sites using triclinic minimum-image Si neighbours."""

    atoms = record.atoms
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    oxygen_indices = np.flatnonzero(symbols == "O")
    silicon_indices = np.flatnonzero(symbols == "Si")
    if not len(oxygen_indices) or not len(silicon_indices):
        raise AnalysisError("Oxygen topology requires at least one O and one Si atom")
    displacement = (
        atoms.positions[silicon_indices][None, :, :]
        - atoms.positions[oxygen_indices][:, None, :]
    )
    _, distances = find_mic(displacement.reshape(-1, 3), atoms.cell, atoms.pbc)
    neighbor_counts = np.count_nonzero(
        distances.reshape(len(oxygen_indices), len(silicon_indices)) < cutoff_A,
        axis=1,
    )
    n_nbo = int(np.count_nonzero(neighbor_counts == 1))
    n_bo = int(np.count_nonzero(neighbor_counts == 2))
    n_other = int(len(oxygen_indices) - n_nbo - n_bo)
    labels = symbols.copy()
    labels[oxygen_indices[neighbor_counts == 1]] = "NBO"
    labels[oxygen_indices[neighbor_counts == 2]] = "BO"
    labels[oxygen_indices[(neighbor_counts != 1) & (neighbor_counts != 2)]] = "O_other"
    n_oxygen = int(len(oxygen_indices))
    frame = OxygenSpeciationFrame(
        step=record.step,
        time_fs=record.time_fs,
        time_ps=_frame_time_ps(record),
        stage=record.stage,
        stage_step=record.stage_step,
        si_o_cutoff_A=float(cutoff_A),
        cutoff_source=cutoff_source,
        n_oxygen=n_oxygen,
        n_nbo=n_nbo,
        n_bo=n_bo,
        n_other=n_other,
        fraction_nbo=n_nbo / n_oxygen if n_oxygen else None,
        fraction_bo=n_bo / n_oxygen if n_oxygen else None,
        fraction_other=n_other / n_oxygen if n_oxygen else None,
    )
    return labels, frame


def _speciation_result(
    records: Sequence[FrameRecord],
    frames: Sequence[OxygenSpeciationFrame],
    cutoffs: Mapping[str, float],
    sources: Mapping[str, str],
) -> OxygenSpeciationResult:
    stages = list(dict.fromkeys(record.stage for record in records))
    return OxygenSpeciationResult(
        frames=list(frames),
        n_frames=len(frames),
        stages=stages,
        cutoffs_A_by_stage={stage: float(cutoffs[stage]) for stage in stages},
        cutoff_source_by_stage={stage: str(sources[stage]) for stage in stages},
        metadata={
            "classification_mode": "instantaneous Si-neighbor count under triclinic MIC",
            "definitions": {
                "NBO": "oxygen with exactly one Si neighbor",
                "BO": "oxygen with exactly two Si neighbors",
                "O_other": "oxygen with zero or at least three Si neighbors",
            },
            "safe_mic_radius_A": safe_mic_radius(records),
        },
    )


def oxygen_speciation(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    si_o_cutoff_A: float | Mapping[str, float] | None = None,
    rdf_bins: int = 300,
    max_frames: int | None = None,
) -> OxygenSpeciationResult:
    """Classify every O site as NBO, BO, or O-other for each selected frame.

    With ``si_o_cutoff_A=None`` (the default), a separate Si--O first minimum is
    estimated from the RDF of every selected stage.  Pass one positive scalar to
    enforce a fixed cutoff across stages, or a ``{stage: cutoff_A}`` mapping for
    explicit stage-specific values.
    """

    records = _sample_records(select_stage_records(_records(frames), stages), max_frames)
    _validate_frame_compatibility(records)
    cutoffs, sources = _stage_cutoff_map(
        records, si_o_cutoff_A=si_o_cutoff_A, rdf_bins=rdf_bins
    )
    classified = [
        _classify_oxygen_frame(
            record,
            cutoff_A=cutoffs[record.stage],
            cutoff_source=sources[record.stage],
        )[1]
        for record in records
    ]
    return _speciation_result(records, classified, cutoffs, sources)


def _topology_pairs(
    pairs: Sequence[Sequence[str]] | None,
    available_labels: set[str],
) -> list[tuple[str, str]]:
    # Ca is an optional cation in this reusable primitive.  Default mode should
    # still deliver Si--topology curves for a Si--O system without Ca.  An
    # explicitly requested absent label remains an input error below.
    requested = (
        [
            pair
            for pair in DEFAULT_TOPOLOGY_RDF_PAIRS
            if set(pair).issubset(available_labels)
        ]
        if pairs is None
        else list(pairs)
    )
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in requested:
        if len(pair) != 2:
            raise ValueError(f"Topology RDF pair must contain two labels: {pair}")
        left, right = str(pair[0]), str(pair[1])
        if "O" in {left, right}:
            raise AnalysisError(
                "Use NBO, BO, or O_other rather than O in topology-resolved RDF pairs"
            )
        unknown = {left, right} - available_labels
        if unknown:
            raise AnalysisError(
                "Topology RDF pair contains labels absent from the trajectory: "
                f"{sorted(unknown)}"
            )
        identity = tuple(sorted((left, right)))
        if identity not in seen:
            normalized.append((left, right))
            seen.add(identity)
    if not normalized:
        raise ValueError("At least one topology RDF pair is required")
    return normalized


def topology_resolved_rdf(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    pairs: Sequence[Sequence[str]] | None = None,
    si_o_cutoff_A: float | Mapping[str, float] | None = None,
    rdf_bins: int = 300,
    bins: int = 300,
    rmax_A: float | None = None,
    include_total: bool = True,
    max_frames: int | None = None,
) -> RDFResult:
    """RDFs resolved by frame-wise NBO/BO oxygen topology.

    The requested pair populations and their ideal-gas normalizers are recomputed
    in each frame.  This avoids bias when oxygen sites switch between NBO and BO
    during the selected trajectory window.
    """

    if bins < 10:
        raise ValueError("bins must be at least 10")
    records = _sample_records(select_stage_records(_records(frames), stages), max_frames)
    symbols = np.asarray(_validate_frame_compatibility(records), dtype=object)
    cutoffs, sources = _stage_cutoff_map(
        records, si_o_cutoff_A=si_o_cutoff_A, rdf_bins=rdf_bins
    )
    available_labels = (set(symbols.tolist()) - {"O"}) | set(_OXYGEN_TOPOLOGY_LABELS)
    requested = _topology_pairs(pairs, available_labels)

    safe_radius = safe_mic_radius(records)
    if rmax_A is None:
        rmax_A = 0.98 * safe_radius
    if rmax_A <= 0.0:
        raise ValueError("rmax_A must be positive")
    if rmax_A > safe_radius * (1.0 + 1.0e-10):
        raise AnalysisError(
            f"rmax_A={rmax_A:.6g} exceeds the safe minimum-image radius "
            f"{safe_radius:.6g} A"
        )
    edges = np.linspace(0.0, float(rmax_A), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    keys = [f"{left}-{right}" for left, right in requested]
    counts = {key: np.zeros(bins, dtype=float) for key in keys}
    ideal = {key: np.zeros(bins, dtype=float) for key in keys}
    if include_total:
        counts["ALL-ALL"] = np.zeros(bins, dtype=float)
        ideal["ALL-ALL"] = np.zeros(bins, dtype=float)

    atom_i, atom_j = np.triu_indices(len(symbols), k=1)
    classified_frames: list[OxygenSpeciationFrame] = []
    for record in records:
        labels, speciation = _classify_oxygen_frame(
            record,
            cutoff_A=cutoffs[record.stage],
            cutoff_source=sources[record.stage],
        )
        classified_frames.append(speciation)
        atoms = record.atoms
        displacement = atoms.positions[atom_j] - atoms.positions[atom_i]
        _, distances = find_mic(displacement, atoms.cell, atoms.pbc)
        inverse_volume = 1.0 / float(atoms.get_volume())
        for (left, right), key in zip(requested, keys, strict=True):
            if left == right:
                mask = (labels[atom_i] == left) & (labels[atom_j] == right)
                population = int(np.count_nonzero(labels == left))
                possible_pairs = population * (population - 1) / 2.0
            else:
                mask = (
                    ((labels[atom_i] == left) & (labels[atom_j] == right))
                    | ((labels[atom_i] == right) & (labels[atom_j] == left))
                )
                possible_pairs = float(np.count_nonzero(labels == left)) * float(
                    np.count_nonzero(labels == right)
                )
            histogram, _ = np.histogram(distances[mask], bins=edges)
            counts[key] += histogram
            ideal[key] += possible_pairs * inverse_volume * shell_volume
        if include_total:
            histogram, _ = np.histogram(distances, bins=edges)
            counts["ALL-ALL"] += histogram
            possible_pairs = len(symbols) * (len(symbols) - 1) / 2.0
            ideal["ALL-ALL"] += possible_pairs * inverse_volume * shell_volume

    curves = {
        key: np.divide(
            counts[key], ideal[key], out=np.zeros_like(counts[key]), where=ideal[key] > 0.0
        )
        for key in counts
    }
    pair_symbols = {key: pair for key, pair in zip(keys, requested, strict=True)}
    if include_total:
        pair_symbols["ALL-ALL"] = ("ALL", "ALL")
    speciation = _speciation_result(records, classified_frames, cutoffs, sources)
    return RDFResult(
        r_A=centers,
        bin_edges_A=edges,
        g=curves,
        counts=counts,
        ideal_counts=ideal,
        n_frames=len(records),
        rmax_A=float(rmax_A),
        stages=list(dict.fromkeys(record.stage for record in records)),
        pair_symbols=pair_symbols,
        metadata={
            "normalization": (
                "ideal-gas pair count per shell, accumulated per frame with "
                "instantaneous NBO/BO labels"
            ),
            "safe_mic_radius_A": safe_radius,
            "sampled_frames": len(records),
            "requested_pairs": [list(pair) for pair in requested],
            "oxygen_speciation": speciation.to_dict(),
        },
    )


# A readable alias for callers that discover this feature by oxygen topology.
oxygen_topology_rdf = topology_resolved_rdf


def _lookup_cutoff(
    center: str,
    neighbor: str,
    cutoffs_A: Mapping[Any, float] | None,
) -> float | None:
    if cutoffs_A is None:
        return None
    candidates: tuple[Any, ...] = (
        (center, neighbor),
        f"{center}-{neighbor}",
        (neighbor, center),
        f"{neighbor}-{center}",
    )
    for key in candidates:
        if key in cutoffs_A:
            return float(cutoffs_A[key])
    return None


def coordination_distributions(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    pairs: Sequence[Sequence[str]] | None = None,
    cutoffs_A: Mapping[Any, float] | None = None,
    rdf: RDFResult | None = None,
    rdf_bins: int = 300,
    max_frames: int | None = None,
) -> CoordinationResult:
    """Calculate directed center-neighbor coordination distributions.

    Distinct pairs are directed: ``("Si", "O")`` counts O around each Si,
    whereas ``("O", "Si")`` counts Si around each O.
    """

    recs = select_stage_records(_records(frames), stages)
    recs = _sample_records(recs, max_frames)
    symbols_list = _validate_frame_compatibility(recs)
    symbols = np.asarray(symbols_list, dtype=object)
    elements = sorted(set(symbols_list))
    if pairs is None:
        directed_pairs = [(center, neighbor) for center in elements for neighbor in elements]
    else:
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Every coordination pair must have two symbols")
        directed_pairs = [(str(pair[0]), str(pair[1])) for pair in pairs]
        directed_pairs = list(dict.fromkeys(directed_pairs))
    unknown = sorted({item for pair in directed_pairs for item in pair} - set(elements))
    if unknown:
        raise AnalysisError(f"Coordination pair contains absent species: {unknown}")

    if rdf is None:
        unordered = list(dict.fromkeys(_canonical_pair(pair) for pair in directed_pairs))
        rdf = partial_rdf(recs, pairs=unordered, bins=rdf_bins, include_total=False)

    resolved_cutoffs: dict[tuple[str, str], tuple[float, str]] = {}
    for center, neighbor in directed_pairs:
        explicit = _lookup_cutoff(center, neighbor, cutoffs_A)
        if explicit is not None:
            cutoff, source = explicit, "configured"
        else:
            key = _pair_key(_canonical_pair((center, neighbor)))
            if key not in rdf.g:
                raise AnalysisError(f"RDF curve required for {center}-{neighbor} is absent")
            cutoff = estimate_first_minimum(rdf.r_A, rdf.g[key])
            source = "first minimum of smoothed partial RDF"
        if cutoff <= 0.0:
            raise ValueError(f"Coordination cutoff must be positive for {center}-{neighbor}")
        if cutoff > safe_mic_radius(recs) * (1.0 + 1.0e-10):
            raise AnalysisError(
                f"Coordination cutoff {cutoff:.5g} A for {center}-{neighbor} "
                "exceeds the safe minimum-image radius"
            )
        resolved_cutoffs[(center, neighbor)] = (cutoff, source)

    all_counts: dict[tuple[str, str], list[IntArray]] = defaultdict(list)
    frame_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in recs:
        atoms = record.atoms
        for center, neighbor in directed_pairs:
            center_indices = np.flatnonzero(symbols == center)
            neighbor_indices = np.flatnonzero(symbols == neighbor)
            delta = (
                atoms.positions[neighbor_indices][None, :, :]
                - atoms.positions[center_indices][:, None, :]
            )
            _, distances = find_mic(delta.reshape(-1, 3), atoms.cell, atoms.pbc)
            distance_matrix = distances.reshape(len(center_indices), len(neighbor_indices))
            if center == neighbor:
                np.fill_diagonal(distance_matrix, np.inf)
            cutoff = resolved_cutoffs[(center, neighbor)][0]
            values = np.count_nonzero(distance_matrix < cutoff, axis=1).astype(np.int64)
            all_counts[(center, neighbor)].append(values)
            frame_means[(center, neighbor)].append(float(np.mean(values)))

    results: dict[str, CoordinationPairResult] = {}
    for center, neighbor in directed_pairs:
        values = np.concatenate(all_counts[(center, neighbor)])
        histogram = np.bincount(values)
        coordination = np.arange(len(histogram), dtype=np.int64)
        probability = histogram / max(1, int(np.sum(histogram)))
        cutoff, source = resolved_cutoffs[(center, neighbor)]
        key = f"{center}->{neighbor}"
        results[key] = CoordinationPairResult(
            center=center,
            neighbor=neighbor,
            cutoff_A=cutoff,
            coordination=coordination,
            count=histogram.astype(np.int64),
            probability=probability.astype(float),
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            n_centers=len(values),
            per_frame_mean=np.asarray(frame_means[(center, neighbor)], dtype=float),
            cutoff_source=source,
        )
    return CoordinationResult(
        pairs=results,
        n_frames=len(recs),
        stages=list(dict.fromkeys(record.stage for record in recs)),
    )


def cell_relative_variation(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
) -> float:
    recs = _records(frames)
    reference = recs[0].atoms.cell.array
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return max(
        float(np.linalg.norm(record.atoms.cell.array - reference) / scale)
        for record in recs
    )


def unwrap_scaled_positions(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    reference_cell: FloatArray | None = None,
    remove_affine_cell_motion: bool = True,
) -> FloatArray:
    """Unwrap a trajectory in fractional coordinates.

    When ``remove_affine_cell_motion`` is true, unwrapped fractional coordinates
    are mapped through one reference cell.  An atom retaining the same fractional
    coordinates during cell deformation then has zero non-affine displacement.
    This representation is suitable for visualizing deformation-relative motion,
    but transport classification is intentionally restricted to fixed-cell holds.
    """

    recs = _records(frames)
    _validate_frame_compatibility(recs)
    scaled = np.stack(
        [record.atoms.get_scaled_positions(wrap=True) for record in recs], axis=0
    ).astype(float)
    unwrapped = np.empty_like(scaled)
    unwrapped[0] = scaled[0]
    pbc = np.asarray(recs[0].atoms.pbc, dtype=bool)
    for index in range(1, len(recs)):
        increment = scaled[index] - scaled[index - 1]
        increment[:, pbc] -= np.rint(increment[:, pbc])
        unwrapped[index] = unwrapped[index - 1] + increment

    if remove_affine_cell_motion:
        cell = (
            np.asarray(reference_cell, dtype=float)
            if reference_cell is not None
            else recs[0].atoms.cell.array
        )
        if cell.shape != (3, 3):
            raise ValueError("reference_cell must have shape (3, 3)")
        return np.einsum("tni,ij->tnj", unwrapped, cell)
    cells = np.stack([record.atoms.cell.array for record in recs])
    return np.einsum("tni,tij->tnj", unwrapped, cells)


def _times_ps(
    records: Sequence[FrameRecord],
    *,
    frame_interval_fs: float | None = None,
    timestep_fs: float | None = None,
) -> FloatArray:
    times_fs = np.asarray(
        [np.nan if record.time_fs is None else record.time_fs for record in records],
        dtype=float,
    )
    if np.all(np.isfinite(times_fs)):
        times = (times_fs - times_fs[0]) / 1000.0
    elif timestep_fs is not None and all(record.step is not None for record in records):
        steps = np.asarray([record.step for record in records], dtype=float)
        times = (steps - steps[0]) * float(timestep_fs) / 1000.0
    elif frame_interval_fs is not None:
        times = np.arange(len(records), dtype=float) * float(frame_interval_fs) / 1000.0
    else:
        raise AnalysisError(
            "Frame times are absent; provide frame_interval_fs or timestep_fs with stored steps"
        )
    if len(times) > 1:
        differences = np.diff(times)
        if np.any(differences <= 0.0):
            raise AnalysisError("Trajectory frame times must be strictly increasing")
        median = float(np.median(differences))
        if not np.allclose(differences, median, rtol=2.0e-3, atol=1.0e-12):
            raise AnalysisError(
                "MSD/Fs analysis requires uniformly spaced stored frames; select one stage/segment"
            )
    return times


def _lag_indices(n_frames: int, maximum_lag: int, maximum_points: int) -> IntArray:
    maximum_lag = min(maximum_lag, n_frames - 1)
    if maximum_lag < 1:
        return np.asarray([0], dtype=np.int64)
    if maximum_lag + 1 <= maximum_points:
        return np.arange(maximum_lag + 1, dtype=np.int64)
    early_count = min(32, maximum_points // 3)
    early = np.arange(0, early_count, dtype=np.int64)
    geometric = np.geomspace(max(1, early_count), maximum_lag, maximum_points - len(early))
    return np.unique(np.concatenate((early, np.rint(geometric).astype(np.int64))))


def _sample_origin_indices(count: int, maximum_origins: int) -> IntArray:
    if count <= maximum_origins:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum_origins, dtype=np.int64))


def _msd_curve(
    coordinates: FloatArray,
    atom_indices: IntArray,
    lags: IntArray,
    maximum_origins: int,
) -> FloatArray:
    values = np.empty(len(lags), dtype=float)
    for output_index, lag in enumerate(lags):
        if lag == 0:
            values[output_index] = 0.0
            continue
        origins = _sample_origin_indices(len(coordinates) - int(lag), maximum_origins)
        delta = coordinates[origins + lag][:, atom_indices] - coordinates[origins][:, atom_indices]
        values[output_index] = float(np.mean(np.einsum("...i,...i->...", delta, delta)))
    return values


def _single_origin_msd_curve(
    coordinates: FloatArray,
    atom_indices: IntArray,
    lags: IntArray,
) -> FloatArray:
    """Return displacement from the first stored frame for selected atoms."""

    delta = (
        coordinates[lags][:, atom_indices]
        - coordinates[0][None, atom_indices, :]
    )
    return np.mean(np.einsum("...i,...i->...", delta, delta), axis=1)


def _linear_fit(
    time_ps: FloatArray,
    msd_A2: FloatArray,
    fit_fraction: tuple[float, float],
) -> tuple[float, float, float, float, float] | None:
    if not (0.0 <= fit_fraction[0] < fit_fraction[1] <= 1.0):
        raise ValueError("fit_fraction must satisfy 0 <= start < end <= 1")
    end_time = float(time_ps[-1])
    start = fit_fraction[0] * end_time
    end = fit_fraction[1] * end_time
    mask = (
        (time_ps >= start)
        & (time_ps <= end)
        & np.isfinite(time_ps)
        & np.isfinite(msd_A2)
    )
    if np.count_nonzero(mask) < 4 or np.ptp(time_ps[mask]) <= 0.0:
        return None
    fit = stats.linregress(time_ps[mask], msd_A2[mask])
    return (
        float(fit.slope),
        float(fit.intercept),
        float(fit.rvalue**2),
        float(time_ps[mask][0]),
        float(time_ps[mask][-1]),
    )


def _block_diffusion_coefficients(
    coordinates: FloatArray,
    atom_indices: IntArray,
    frame_dt_ps: float,
    *,
    n_blocks: int,
    fit_fraction: tuple[float, float],
    maximum_lag_fraction: float,
    maximum_lag_points: int,
    maximum_origins: int,
) -> list[float]:
    if n_blocks < 2:
        return []
    coefficients: list[float] = []
    for block in np.array_split(coordinates, n_blocks, axis=0):
        if len(block) < 12:
            continue
        max_lag = max(4, int((len(block) - 1) * maximum_lag_fraction))
        lags = _lag_indices(len(block), max_lag, min(100, maximum_lag_points))
        times = lags.astype(float) * frame_dt_ps
        curve = _msd_curve(block, atom_indices, lags, maximum_origins)
        fit = _linear_fit(times, curve, fit_fraction)
        if fit is not None and math.isfinite(fit[0]):
            coefficients.append(float(fit[0] / 6.0))
    return coefficients


def _classify_diffusion(
    *,
    fit: tuple[float, float, float, float, float] | None,
    block_coefficients: list[float],
    msd: FloatArray,
    resolved_lag_time_ps: float,
    min_observation_ps: float,
    nondiffusive_msd_A2: float,
    diffusive_growth_A2: float,
    min_r_squared: float,
) -> DiffusionEstimate:
    if fit is None:
        return DiffusionEstimate(
            status="undetermined",
            coefficient_A2_ps=None,
            coefficient_cm2_s=None,
            standard_error_A2_ps=None,
            confidence95_A2_ps=None,
            upper_bound_A2_ps=None,
            slope_A2_ps=None,
            intercept_A2=None,
            r_squared=None,
            fit_start_ps=None,
            fit_end_ps=None,
            block_coefficients_A2_ps=block_coefficients,
            reason="Insufficient finite points in the requested fit interval.",
        )

    slope, intercept, r_squared, fit_start, fit_end = fit
    coefficient = max(0.0, slope / 6.0)
    valid_blocks = np.asarray(
        [value for value in block_coefficients if math.isfinite(value)], dtype=float
    )
    standard_error = (
        float(np.std(valid_blocks, ddof=1) / np.sqrt(len(valid_blocks)))
        if len(valid_blocks) >= 2
        else None
    )
    confidence_multiplier = (
        float(stats.t.ppf(0.975, df=len(valid_blocks) - 1))
        if len(valid_blocks) >= 2
        else None
    )
    confidence = (
        (
            max(0.0, coefficient - confidence_multiplier * standard_error),
            max(0.0, coefficient + confidence_multiplier * standard_error),
        )
        if standard_error is not None and confidence_multiplier is not None
        else None
    )
    finite_msd = msd[np.isfinite(msd)]
    final_msd = float(finite_msd[-1]) if len(finite_msd) else math.nan
    upper_bound = (
        nondiffusive_msd_A2 / (6.0 * resolved_lag_time_ps)
        if resolved_lag_time_ps > 0.0
        else None
    )

    if resolved_lag_time_ps < min_observation_ps:
        status: TransportStatus = "undetermined"
        reason = (
            f"Resolved lag time {resolved_lag_time_ps:.4g} ps is shorter than the "
            f"configured {min_observation_ps:.4g} ps classification window."
        )
    elif final_msd <= nondiffusive_msd_A2:
        status = "nondiffusive_on_timescale"
        reason = (
            f"MSD remains below {nondiffusive_msd_A2:.4g} A^2 over the observed "
            "fixed-cell interval; this is a time-scale-qualified upper bound, not zero diffusion."
        )
    else:
        fit_mask_growth = max(0.0, slope * (fit_end - fit_start))
        positive_blocks = (
            float(np.mean(valid_blocks > 0.0)) if len(valid_blocks) else 0.0
        )
        lower_positive = confidence is None or confidence[0] > 0.0
        if (
            slope > 0.0
            and r_squared >= min_r_squared
            and fit_mask_growth >= diffusive_growth_A2
            and len(valid_blocks) >= 2
            and positive_blocks >= 0.75
            and lower_positive
        ):
            status = "diffusive"
            reason = (
                "Late-time MSD is linear with positive, block-consistent slope and "
                "sufficient displacement over the fit window."
            )
        else:
            status = "undetermined"
            reason = (
                "The trajectory is neither a resolved diffusive regime nor a bounded "
                "non-diffusive plateau under the configured criteria."
            )

    return DiffusionEstimate(
        status=status,
        coefficient_A2_ps=coefficient,
        coefficient_cm2_s=coefficient * 1.0e-4,
        standard_error_A2_ps=standard_error,
        confidence95_A2_ps=confidence,
        upper_bound_A2_ps=upper_bound if status == "nondiffusive_on_timescale" else None,
        slope_A2_ps=slope,
        intercept_A2=intercept,
        r_squared=r_squared,
        fit_start_ps=fit_start,
        fit_end_ps=fit_end,
        block_coefficients_A2_ps=[float(value) for value in valid_blocks],
        reason=reason,
    )


def species_msd(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    species: Sequence[str] | None = None,
    frame_interval_fs: float | None = None,
    timestep_fs: float | None = None,
    cell_tolerance: float = 1.0e-5,
    maximum_lag_fraction: float = 0.5,
    maximum_lag_points: int = 250,
    maximum_origins: int = 256,
    fit_fraction: tuple[float, float] = (0.5, 0.9),
    n_blocks: int = 4,
    min_observation_ps: float = 20.0,
    nondiffusive_msd_A2: float = 0.25,
    diffusive_growth_A2: float = 1.0,
    min_r_squared: float = 0.90,
) -> MSDResult:
    """Calculate single-origin species MSD and multi-origin diffusion estimates.

    Only fixed-cell stages are accepted.  This prevents affine compression from
    being mistaken for atomic transport.  Coordinates are first unwrapped in
    fractional space, which handles boundary crossings in triclinic cells.  The
    published MSD curve uses the first stage frame as its sole origin and spans
    the full stored stage.  Diffusion fitting remains a separate, explicitly
    multi-origin estimator with block uncertainty.
    """

    if not (0.0 < maximum_lag_fraction <= 1.0):
        raise ValueError("maximum_lag_fraction must be in (0, 1]")
    if maximum_lag_points < 8 or maximum_origins < 1:
        raise ValueError("maximum_lag_points >= 8 and maximum_origins >= 1 are required")
    recs = select_stage_records(_records(frames), stages)
    symbols_list = _validate_frame_compatibility(recs)
    if len(recs) < 12:
        raise AnalysisError("At least 12 fixed-cell frames are required for MSD analysis")
    variation = cell_relative_variation(recs)
    if variation > cell_tolerance:
        raise AnalysisError(
            f"Selected cell varies by {variation:.3g}, exceeding tolerance "
            f"{cell_tolerance:.3g}. Use a fixed-cell high-density or final hold."
        )
    times = _times_ps(
        recs, frame_interval_fs=frame_interval_fs, timestep_fs=timestep_fs
    )
    frame_dt = float(np.median(np.diff(times)))
    coordinates = unwrap_scaled_positions(recs, remove_affine_cell_motion=True)
    symbols = np.asarray(symbols_list, dtype=object)
    selected_species = sorted(set(symbols_list)) if species is None else list(species)
    absent = sorted(set(selected_species) - set(symbols_list))
    if absent:
        raise AnalysisError(f"MSD species absent from trajectory: {absent}")
    # A single origin can use the complete stage without the decreasing-origin
    # statistics that motivate truncating a multi-origin curve.
    lags = _lag_indices(len(recs), len(recs) - 1, maximum_lag_points)
    elapsed_times = times[lags] - times[0]
    diffusion_maximum_lag = max(
        1, int((len(recs) - 1) * maximum_lag_fraction)
    )
    diffusion_lags = _lag_indices(
        len(recs), diffusion_maximum_lag, maximum_lag_points
    )
    diffusion_times = diffusion_lags.astype(float) * frame_dt
    curves: dict[str, FloatArray] = {}
    estimates: dict[str, DiffusionEstimate] = {}
    for element in selected_species:
        indices = np.flatnonzero(symbols == element).astype(np.int64)
        curves[element] = _single_origin_msd_curve(coordinates, indices, lags)
        diffusion_curve = _msd_curve(
            coordinates, indices, diffusion_lags, maximum_origins
        )
        fit = _linear_fit(diffusion_times, diffusion_curve, fit_fraction)
        blocks = _block_diffusion_coefficients(
            coordinates,
            indices,
            frame_dt,
            n_blocks=n_blocks,
            fit_fraction=fit_fraction,
            maximum_lag_fraction=maximum_lag_fraction,
            maximum_lag_points=maximum_lag_points,
            maximum_origins=maximum_origins,
        )
        estimates[element] = _classify_diffusion(
            fit=fit,
            block_coefficients=blocks,
            msd=diffusion_curve,
            resolved_lag_time_ps=float(diffusion_times[-1]),
            min_observation_ps=min_observation_ps,
            nondiffusive_msd_A2=nondiffusive_msd_A2,
            diffusive_growth_A2=diffusive_growth_A2,
            min_r_squared=min_r_squared,
        )
    return MSDResult(
        lag_time_ps=elapsed_times,
        msd_A2=curves,
        diffusion=estimates,
        n_frames=len(recs),
        observation_time_ps=float(times[-1]),
        stages=list(dict.fromkeys(record.stage for record in recs)),
        cell_relative_variation=variation,
        coordinate_method="fractional unwrapping mapped through fixed reference cell",
        origin_mode="single origin: first stored frame of the stage",
        diffusion_estimator=(
            "separate multi-origin late-time linear fit with block uncertainty"
        ),
    )


def estimate_structure_factor_peak(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    q_range_inv_A: tuple[float, float] = (0.75, 12.0),
    q_points: int = 500,
    rdf_bins: int = 500,
    max_frames: int = 50,
) -> StructureFactorEstimate:
    """Estimate an unweighted isotropic S(q) peak from the total RDF.

    The transform uses a Lorch window to reduce finite-r cutoff ripples.  For
    element-specific experimental comparison, users should provide a scattering-
    weighted ``k_inv_A`` instead.
    """

    if q_range_inv_A[0] <= 0.0 or q_range_inv_A[1] <= q_range_inv_A[0]:
        raise ValueError("q_range_inv_A must be a positive increasing pair")
    if q_points < 20 or rdf_bins < 30:
        raise ValueError("q_points >= 20 and rdf_bins >= 30 are required")
    recs = select_stage_records(_records(frames), stages)
    recs = _sample_records(recs, max_frames)
    rdf = partial_rdf(recs, bins=rdf_bins, include_total=True)
    r = rdf.r_A
    g = rdf.g["ALL-ALL"]
    mean_density = float(
        np.mean([len(record.atoms) / record.atoms.get_volume() for record in recs])
    )
    q = np.linspace(q_range_inv_A[0], q_range_inv_A[1], q_points)
    lorch = np.sinc(r / rdf.rmax_A)
    integrand_base = (g - 1.0) * r**2 * lorch
    kernel = np.sinc(np.outer(q, r) / np.pi)
    sq = 1.0 + 4.0 * np.pi * mean_density * np.trapezoid(
        kernel * integrand_base[None, :], r, axis=1
    )
    smoothed = signal.savgol_filter(sq, min(31, q_points // 2 * 2 - 1), 3)
    peaks, properties = signal.find_peaks(
        smoothed, prominence=max(0.02, 0.03 * float(np.ptp(smoothed)))
    )
    if len(peaks):
        prominences = properties.get("prominences", np.zeros(len(peaks)))
        peak = int(peaks[int(np.argmax(prominences))])
    else:
        peak = int(np.argmax(smoothed))
    return StructureFactorEstimate(
        k_inv_A=float(q[peak]),
        q_inv_A=q,
        S_q=sq,
        method="unweighted total RDF transform with Lorch window; strongest resolved peak",
    )


def self_intermediate_scattering(
    frames: TrajectoryDataset | Sequence[FrameRecord] | Sequence[Atoms],
    *,
    stages: str | Sequence[str] | None = None,
    species: Sequence[str] | None = None,
    k_inv_A: float | None = None,
    frame_interval_fs: float | None = None,
    timestep_fs: float | None = None,
    cell_tolerance: float = 1.0e-5,
    maximum_lag_fraction: float = 0.5,
    maximum_lag_points: int = 250,
    maximum_origins: int = 256,
) -> SelfIntermediateScatteringResult:
    r"""Calculate orientationally averaged self-intermediate scattering.

    The isotropic average is
    :math:`F_s(k,t)=\langle\sin(k|\Delta r|)/(k|\Delta r|)\rangle`.
    As for MSD, only fixed-cell holds are accepted.
    """

    recs = select_stage_records(_records(frames), stages)
    symbols_list = _validate_frame_compatibility(recs)
    if len(recs) < 12:
        raise AnalysisError("At least 12 fixed-cell frames are required for Fs analysis")
    variation = cell_relative_variation(recs)
    if variation > cell_tolerance:
        raise AnalysisError(
            f"Selected cell varies by {variation:.3g}; Fs transport analysis is "
            "restricted to fixed-cell holds"
        )
    times = _times_ps(
        recs, frame_interval_fs=frame_interval_fs, timestep_fs=timestep_fs
    )
    frame_dt = float(np.median(np.diff(times)))
    if k_inv_A is None:
        estimate = estimate_structure_factor_peak(recs)
        k_inv_A = estimate.k_inv_A
        k_source = estimate.method
    else:
        if k_inv_A <= 0.0:
            raise ValueError("k_inv_A must be positive")
        k_source = "configured"

    coordinates = unwrap_scaled_positions(recs, remove_affine_cell_motion=True)
    symbols = np.asarray(symbols_list, dtype=object)
    selected_species = sorted(set(symbols_list)) if species is None else list(species)
    absent = sorted(set(selected_species) - set(symbols_list))
    if absent:
        raise AnalysisError(f"Fs species absent from trajectory: {absent}")
    maximum_lag = max(1, int((len(recs) - 1) * maximum_lag_fraction))
    lags = _lag_indices(len(recs), maximum_lag, maximum_lag_points)
    curves: dict[str, FloatArray] = {}
    for element in selected_species:
        indices = np.flatnonzero(symbols == element)
        curve = np.empty(len(lags), dtype=float)
        for output_index, lag in enumerate(lags):
            if lag == 0:
                curve[output_index] = 1.0
                continue
            origins = _sample_origin_indices(len(recs) - int(lag), maximum_origins)
            delta = coordinates[origins + lag][:, indices] - coordinates[origins][:, indices]
            radii = np.linalg.norm(delta, axis=-1)
            curve[output_index] = float(np.mean(np.sinc(k_inv_A * radii / np.pi)))
        curves[element] = curve
    return SelfIntermediateScatteringResult(
        lag_time_ps=lags.astype(float) * frame_dt,
        k_inv_A=float(k_inv_A),
        Fs=curves,
        n_frames=len(recs),
        observation_time_ps=float(times[-1]),
        stages=list(dict.fromkeys(record.stage for record in recs)),
        k_source=k_source,
    )


def _branch_from_stage(stage: str) -> str:
    name = stage.lower()
    if "recovery" in name:
        return "recovery"
    if "decompress" in name or "unload" in name:
        return "decompression"
    if "compress" in name or "load" in name:
        return "compression"
    if "peak" in name or "high" in name:
        return "high_density_hold"
    if "final" in name or "aging" in name or "recovered" in name:
        return "recovered_hold"
    if "initial" in name:
        return "initial_hold"
    return "other"


def _branch_from_row(row: Mapping[str, Any], stage: str) -> str:
    """Resolve a branch from explicit branch, stage kind, then legacy name."""

    explicit = row.get("branch")
    aliases = {
        "cold_compression": "compression",
        "compression": "compression",
        "loading": "compression",
        "decompression": "decompression",
        "unloading": "decompression",
        "recovery": "recovery",
        "initial": "initial_hold",
        "initial_hold": "initial_hold",
        "high_density_hold": "high_density_hold",
        "recovered_hold": "recovered_hold",
    }
    if explicit is not None and str(explicit).strip():
        name = str(explicit).strip().lower()
        return aliases.get(name, name)
    kind = row.get("kind")
    if kind is not None and str(kind).strip():
        kind_name = str(kind).strip().lower()
        kind_branches = {
            "initial_nvt": "initial_hold",
            "peak_nvt": "high_density_hold",
            "final_nvt": "recovered_hold",
            "recovery_npt": "recovery",
            "compression": "compression",
            "decompression": "decompression",
            # Generic sequence stages deliberately make no scientific claim
            # about loading direction.  Users who want them joined to a P-V
            # loading/unloading path set ``branch`` explicitly; otherwise they
            # remain neutral observations.
            "nvt": "other",
            "npt": "other",
            "fixed_nvt": "other",
            "isotropic_npt_plateau": "other",
            "pressure_plateau": "other",
        }
        if kind_name in kind_branches:
            return kind_branches[kind_name]
    return _branch_from_stage(stage)


def _row_stage(row: Mapping[str, Any]) -> str:
    value = row.get("stage")
    return str(value).strip() if value is not None and str(value).strip() else "unknown"


def _row_time_ps(row: Mapping[str, Any]) -> float | None:
    direct = _finite_float(row.get("time_ps"))
    if direct is not None:
        return direct
    time_fs = _finite_float(row.get("time_fs"))
    return None if time_fs is None else time_fs / 1000.0


def _thermo_duplicate_key(row: Mapping[str, Any]) -> tuple[str, str, int | float] | None:
    """Return a stage-local duplicate identity without erasing boundaries.

    A stage boundary may legitimately contain two records at one global step:
    an endpoint for the stage being left and a ``stage_step=0`` record for the
    stage being entered.  Those are distinct reporting observations.  Within a
    stage, a resumed run can re-write the same stage step; its newest row wins.
    """

    stage = _row_stage(row)
    stage_step = _finite_int(row.get("stage_step"))
    if stage_step is not None:
        return "stage_step", stage, stage_step
    step = _finite_int(row.get("step"))
    if step is not None:
        return "stage_global_step", stage, step
    time_fs = _finite_float(row.get("time_fs"))
    if time_fs is not None:
        return "stage_time_fs", stage, round(time_fs, 9)
    time_ps = _finite_float(row.get("time_ps"))
    if time_ps is not None:
        return "stage_time_ps", stage, round(time_ps, 12)
    return None


def _deduplicate_thermo_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    retained: list[Mapping[str, Any]] = []
    positions: dict[tuple[str, str, int | float], int] = {}
    dropped = 0
    for row in rows:
        key = _thermo_duplicate_key(row)
        if key is None or key not in positions:
            if key is not None:
                positions[key] = len(retained)
            retained.append(row)
            continue
        dropped += 1
        # Identical stage/stage-step observations originate from a resumed
        # output stream.  Retain the newest persisted row deterministically.
        retained[positions[key]] = row
    return retained, dropped


def _protocol_stage_rows(
    protocol_stages: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize config stages without making analysis depend on Pydantic models."""

    if protocol_stages is None:
        return []
    raw_stages: Any = protocol_stages
    if isinstance(raw_stages, Mapping):
        raw_stages = raw_stages.get("stages", raw_stages)
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        raise TypeError("protocol_stages must be a sequence of stage mappings")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, Mapping):
            if hasattr(raw_stage, "model_dump"):
                raw_stage = raw_stage.model_dump(mode="python")
            else:
                raise TypeError("Each protocol stage must be a mapping")
        name = raw_stage.get("name")
        if name is None or not str(name).strip():
            raise AnalysisError(f"Protocol stage {order} has no usable name")
        stage = str(name).strip()
        if stage in seen:
            raise AnalysisError(f"Protocol repeats stage name {stage!r}")
        seen.add(stage)
        sampling_window = _finite_float(raw_stage.get("sampling_window_ps"))
        if raw_stage.get("sampling_window_ps") is not None and (
            sampling_window is None or sampling_window <= 0.0
        ):
            raise ValueError(
                f"sampling_window_ps must be a positive finite value for {stage!r}"
            )
        normalized.append(
            {
                "stage_order": order,
                "stage": stage,
                "kind": (
                    str(raw_stage.get("kind"))
                    if raw_stage.get("kind") is not None
                    else None
                ),
                "branch": _branch_from_row(raw_stage, stage),
                "target_pressure_GPa": _finite_float(
                    raw_stage.get("target_pressure_GPa")
                ),
                "duration_ps": _finite_float(raw_stage.get("duration_ps")),
                "sampling_window_ps": sampling_window,
            }
        )
    return normalized


def _thermo_time_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_fs: float | None,
) -> tuple[FloatArray | None, str | None]:
    """Return one common, ordered time axis for a stage's thermo rows."""

    direct = np.asarray([_finite_float(row.get("time_ps")) for row in rows], dtype=object)
    if all(value is not None for value in direct):
        return np.asarray(direct, dtype=float), "stored_time_ps_tail"
    time_fs = np.asarray([_finite_float(row.get("time_fs")) for row in rows], dtype=object)
    if all(value is not None for value in time_fs):
        return np.asarray(time_fs, dtype=float) / 1000.0, "stored_time_fs_tail"
    timestep = _finite_float(timestep_fs)
    if timestep is None or timestep <= 0.0:
        return None, None
    stage_steps = np.asarray([_finite_int(row.get("stage_step")) for row in rows], dtype=object)
    if all(value is not None for value in stage_steps):
        return np.asarray(stage_steps, dtype=float) * timestep / 1000.0, "stored_stage_step_tail"
    steps = np.asarray([_finite_int(row.get("step")) for row in rows], dtype=object)
    if all(value is not None for value in steps):
        return np.asarray(steps, dtype=float) * timestep / 1000.0, "stored_global_step_tail"
    return None, None


def _select_thermo_tail(
    rows: Sequence[Mapping[str, Any]],
    *,
    sampling_window_ps: float | None,
    timestep_fs: float | None,
) -> tuple[list[Mapping[str, Any]], str, FloatArray | None]:
    available = list(rows)
    if not available:
        return [], "missing_stage", None
    times, method = _thermo_time_values(available, timestep_fs=timestep_fs)
    if sampling_window_ps is None:
        return available, "full_stage", times
    if times is None or method is None:
        raise AnalysisError(
            "sampling_window_ps is configured, but thermo rows lack one complete "
            "time/stage-step axis; provide stored time metadata or timestep_fs"
        )
    cutoff = float(times[-1] - sampling_window_ps)
    selected_indices = np.flatnonzero(times >= cutoff - 1.0e-12)
    return [available[index] for index in selected_indices], method, times[selected_indices]


def _mean_and_sample_std(values: Sequence[float | None]) -> tuple[int, float | None, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(finite):
        return 0, None, None
    return (
        int(len(finite)),
        float(np.mean(finite)),
        float(np.std(finite, ddof=1)) if len(finite) >= 2 else None,
    )


def _summary_pressure_values(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float | None], list[str]]:
    values: list[float | None] = []
    sources: list[str] = []
    for row in rows:
        total = _finite_float(row.get("total_pressure_GPa"))
        if total is not None:
            values.append(total)
            sources.append("total_pressure_GPa")
            continue
        configurational = _finite_float(row.get("configurational_pressure_GPa"))
        if configurational is not None:
            values.append(configurational)
            sources.append("configurational_pressure_GPa")
            continue
        values.append(None)
    return values, sources


def _pressure_source_label(sources: Sequence[str]) -> str | None:
    used = set(sources)
    if not used:
        return None
    if used == {"total_pressure_GPa"}:
        return "total_pressure_GPa"
    if used == {"configurational_pressure_GPa"}:
        return "configurational_pressure_GPa (legacy fallback)"
    return "total_pressure_GPa with configurational_pressure_GPa legacy fallback"


def _empty_stage_summary(metadata: Mapping[str, Any]) -> StageThermoSummary:
    return StageThermoSummary(
        **metadata,
        selection_method="missing_stage",
        sampling_start_stage_step=None,
        n_samples=0,
        n_pressure=0,
        n_volume=0,
        n_density=0,
        n_temperature=0,
        first_sample_time_ps=None,
        last_sample_time_ps=None,
        realized_sample_span_ps=None,
        pressure_mean_GPa=None,
        pressure_std_GPa=None,
        volume_mean_A3=None,
        volume_std_A3=None,
        density_mean_g_cm3=None,
        density_std_g_cm3=None,
        temperature_mean_K=None,
        temperature_std_K=None,
        pressure_source=None,
        available_rows=0,
        dropped_invalid_rows=0,
        status="missing_stage_data",
    )


def _stage_thermo_summary(
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_fs: float | None,
) -> StageThermoSummary:
    if not rows:
        return _empty_stage_summary(metadata)
    selected, method, selected_times = _select_thermo_tail(
        rows,
        sampling_window_ps=metadata["sampling_window_ps"],
        timestep_fs=timestep_fs,
    )
    pressure, pressure_sources = _summary_pressure_values(selected)
    volume = [_finite_float(row.get("volume_A3")) for row in selected]
    density = [_finite_float(row.get("density_g_cm3")) for row in selected]
    temperature = [_finite_float(row.get("temperature_K")) for row in selected]
    n_pressure, pressure_mean, pressure_std = _mean_and_sample_std(pressure)
    n_volume, volume_mean, volume_std = _mean_and_sample_std(volume)
    n_density, density_mean, density_std = _mean_and_sample_std(density)
    n_temperature, temperature_mean, temperature_std = _mean_and_sample_std(temperature)
    n_samples = len(selected)
    valid_all = sum(
        value_p is not None and value_v is not None and value_d is not None and value_t is not None
        for value_p, value_v, value_d, value_t in zip(
            pressure, volume, density, temperature, strict=True
        )
    )
    if not n_samples:
        status = "no_selected_samples"
    elif not any((n_pressure, n_volume, n_density, n_temperature)):
        status = "no_finite_thermo"
    elif valid_all == n_samples:
        status = "ok"
    else:
        status = "partial_finite_thermo"
    first_time = float(selected_times[0]) if selected_times is not None else None
    last_time = float(selected_times[-1]) if selected_times is not None else None
    return StageThermoSummary(
        **metadata,
        selection_method=method,
        sampling_start_stage_step=_finite_int(selected[0].get("stage_step")) if selected else None,
        n_samples=n_samples,
        n_pressure=n_pressure,
        n_volume=n_volume,
        n_density=n_density,
        n_temperature=n_temperature,
        first_sample_time_ps=first_time,
        last_sample_time_ps=last_time,
        realized_sample_span_ps=(last_time - first_time)
        if first_time is not None and last_time is not None and n_samples > 1
        else None,
        pressure_mean_GPa=pressure_mean,
        pressure_std_GPa=pressure_std,
        volume_mean_A3=volume_mean,
        volume_std_A3=volume_std,
        density_mean_g_cm3=density_mean,
        density_std_g_cm3=density_std,
        temperature_mean_K=temperature_mean,
        temperature_std_K=temperature_std,
        pressure_source=_pressure_source_label(pressure_sources),
        available_rows=len(rows),
        dropped_invalid_rows=n_samples - valid_all,
        status=status,
    )


def stage_thermo_statistics(
    thermo: str | Path | Sequence[Mapping[str, Any]],
    *,
    protocol_stages: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    timestep_fs: float | None = None,
) -> StageThermoSummaryResult:
    """Summarize P, V, density, and T over each configured stage's tail window.

    ``protocol_stages`` may be either the resolved ``protocol.stages`` list or a
    mapping containing a ``stages`` field.  Configured stages are emitted in that
    exact order, including stages with no stored rows; unconfigured observed
    stages follow in their first-observed order.  Total pressure is used whenever
    available, with configurational pressure used only as a legacy fallback.
    """

    source_rows = load_thermo_csv(thermo) if isinstance(thermo, (str, Path)) else list(thermo)
    if not source_rows:
        raise AnalysisError("Thermo data are empty")
    if not all(isinstance(row, Mapping) for row in source_rows):
        raise TypeError("Thermo data must be mappings or a canonical thermo CSV path")
    rows, dropped_duplicates = _deduplicate_thermo_rows(source_rows)
    configured = _protocol_stage_rows(protocol_stages)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_order: list[str] = []
    for row in rows:
        stage = _row_stage(row)
        if stage not in grouped:
            observed_order.append(stage)
        grouped[stage].append(row)

    summaries: list[StageThermoSummary] = []
    configured_names = {item["stage"] for item in configured}
    for metadata in configured:
        summaries.append(
            _stage_thermo_summary(metadata, grouped.get(metadata["stage"], []), timestep_fs=timestep_fs)
        )
    for stage in observed_order:
        if stage in configured_names:
            continue
        metadata = {
            "stage_order": len(summaries),
            "stage": stage,
            "kind": None,
            "branch": _branch_from_row(grouped[stage][0], stage),
            "target_pressure_GPa": _finite_float(grouped[stage][0].get("target_pressure_GPa")),
            "duration_ps": None,
            "sampling_window_ps": None,
        }
        summaries.append(_stage_thermo_summary(metadata, grouped[stage], timestep_fs=timestep_fs))
    return StageThermoSummaryResult(
        summaries=summaries,
        source_row_count=len(source_rows),
        retained_row_count=len(rows),
        dropped_duplicate_rows=dropped_duplicates,
        protocol_stage_count=len(configured),
        metadata={
            "pressure_policy": (
                "total_pressure_GPa primary; configurational_pressure_GPa legacy fallback"
            ),
            "standard_deviation": "sample standard deviation (ddof=1); null when n < 2",
            "duplicate_policy": (
                "deduplicate by (stage, stage_step) and retain the newest row; "
                "when stage_step is absent, use stage-local global step or time"
            ),
        },
    )


# An intentionally verbose synonym for callers that prefer report-oriented naming.
summarize_stage_thermo = stage_thermo_statistics


def sampling_window_thermo_series(
    thermo: str | Path | Sequence[Mapping[str, Any]],
    *,
    protocol_stages: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    timestep_fs: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return full-stage local-time T/P/V series with a sampling-window mask.

    The time coordinate begins at the first stored row of each stage.  Every
    stored row is retained; ``in_sampling_window`` and the cutoff fields mark
    the tail used for stage statistics.  Separate stages are never joined.
    """

    source_rows = load_thermo_csv(thermo) if isinstance(thermo, (str, Path)) else list(thermo)
    if not source_rows:
        raise AnalysisError("Thermo data are empty")
    rows, _ = _deduplicate_thermo_rows(source_rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_row_stage(row)].append(row)
    configured = _protocol_stage_rows(protocol_stages)
    metadata_by_stage = {item["stage"]: item for item in configured}
    output: dict[str, dict[str, Any]] = {}
    for stage, stage_rows in grouped.items():
        metadata = metadata_by_stage.get(
            stage,
            {
                "stage": stage,
                "kind": None,
                "sampling_window_ps": None,
                "branch": _branch_from_row(stage_rows[0], stage),
            },
        )
        times, method = _thermo_time_values(stage_rows, timestep_fs=timestep_fs)
        if times is None or method is None or not len(times):
            continue
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) < 0.0):
            raise AnalysisError(
                f"Thermo time decreases within stage {stage!r}; full-stage plotting "
                "requires chronological rows"
            )
        local_times = times - float(times[0])
        sampling_window = _finite_float(metadata.get("sampling_window_ps"))
        if sampling_window is None:
            sampling_cutoff = float(times[0])
            sampling_mask = np.ones(len(times), dtype=bool)
            selection_method = "full_stage"
        else:
            sampling_cutoff = float(times[-1] - sampling_window)
            sampling_mask = times >= sampling_cutoff - 1.0e-12
            selection_method = method
        pressure, sources = _summary_pressure_values(stage_rows)
        output[stage] = {
            "stage": stage,
            "kind": metadata.get("kind"),
            "branch": metadata.get("branch"),
            "selection_method": selection_method,
            "sampling_window_ps": sampling_window,
            "sampling_cutoff_ps": max(
                0.0, sampling_cutoff - float(times[0])
            ),
            "sampling_first_observation_ps": (
                float(local_times[np.flatnonzero(sampling_mask)[0]])
                if np.any(sampling_mask)
                else None
            ),
            "sampling_end_ps": float(local_times[-1]),
            "time_ps": [float(value) for value in local_times],
            "in_sampling_window": [bool(value) for value in sampling_mask],
            "temperature_K": [
                _finite_float(row.get("temperature_K")) for row in stage_rows
            ],
            "pressure_GPa": pressure,
            "volume_A3": [
                _finite_float(row.get("volume_A3")) for row in stage_rows
            ],
            "pressure_source": _pressure_source_label(sources),
        }
    return output


def thermo_hysteresis(
    thermo: str | Path | Sequence[Mapping[str, Any]],
) -> HysteresisResult:
    """Extract pressure-volume-density paths from canonical thermodynamic output."""

    rows = load_thermo_csv(thermo) if isinstance(thermo, (str, Path)) else list(thermo)
    if not rows:
        raise AnalysisError("Thermo data are empty")

    def numeric(row: Mapping[str, Any], *keys: str) -> float:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return math.nan

    filtered: list[Mapping[str, Any]] = []
    for row in rows:
        pressure = numeric(row, "total_pressure_GPa", "configurational_pressure_GPa")
        volume = numeric(row, "volume_A3")
        if math.isfinite(pressure) and math.isfinite(volume):
            filtered.append(row)
    if len(filtered) < 2:
        raise AnalysisError("At least two finite pressure-volume thermo rows are required")

    time = np.asarray(
        [numeric(row, "time_ps") for row in filtered], dtype=float
    )
    if not np.all(np.isfinite(time)):
        time_fs = np.asarray([numeric(row, "time_fs") for row in filtered], dtype=float)
        if np.all(np.isfinite(time_fs)):
            time = time_fs / 1000.0
        else:
            time = np.arange(len(filtered), dtype=float)
    pressure = np.asarray(
        [numeric(row, "total_pressure_GPa", "configurational_pressure_GPa") for row in filtered],
        dtype=float,
    )
    volume = np.asarray([numeric(row, "volume_A3") for row in filtered], dtype=float)
    density = np.asarray([numeric(row, "density_g_cm3") for row in filtered], dtype=float)
    # ``target_volume_ratio`` is stage-relative and must never be joined across
    # stages. New runs persist one engine-global ratio, which also remains valid
    # when a thermo log is trimmed. Use volume/first-volume only for wholly legacy
    # input; partially populated explicit provenance is ambiguous and rejected.
    persisted_ratio = np.asarray(
        [numeric(row, "volume_ratio_from_engine_start") for row in filtered],
        dtype=float,
    )
    explicit = np.isfinite(persisted_ratio)
    if np.all(explicit):
        if np.any(persisted_ratio <= 0.0):
            raise AnalysisError("Persisted global volume ratios must be positive")
        volume_ratio = persisted_ratio
    elif not np.any(explicit):
        volume_ratio = volume / volume[0]
    else:
        raise AnalysisError(
            "volume_ratio_from_engine_start is only partially populated; refusing "
            "to mix engine-global and reconstructed volume references"
        )
    stages = [str(row.get("stage") or "unknown") for row in filtered]
    branches = [
        _branch_from_row(row, stage)
        for row, stage in zip(filtered, stages, strict=True)
    ]
    path_integral = float(np.trapezoid(pressure, volume)) if len(volume) >= 2 else None
    return HysteresisResult(
        time_ps=time,
        pressure_GPa=pressure,
        volume_A3=volume,
        volume_ratio=volume_ratio,
        density_g_cm3=density,
        stage=stages,
        branch=branches,
        pv_path_integral_GPa_A3=path_integral,
    )


def write_json(data: Any, path: str | Path) -> Path:
    """Write an analysis object as standards-compliant UTF-8 JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_value(data)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


__all__ = [
    "AnalysisError",
    "CoordinationPairResult",
    "CoordinationResult",
    "DEFAULT_TOPOLOGY_RDF_PAIRS",
    "DiffusionEstimate",
    "FrameRecord",
    "HysteresisResult",
    "MSDResult",
    "OxygenSpeciationFrame",
    "OxygenSpeciationResult",
    "RDFResult",
    "SelfIntermediateScatteringResult",
    "StageThermoSummary",
    "StageThermoSummaryResult",
    "StructureFactorEstimate",
    "TrajectoryDataset",
    "cell_relative_variation",
    "coordination_distributions",
    "discover_trajectory_files",
    "estimate_first_minimum",
    "estimate_structure_factor_peak",
    "load_segmented_trajectories",
    "load_thermo_csv",
    "oxygen_speciation",
    "oxygen_topology_rdf",
    "partial_rdf",
    "safe_mic_radius",
    "sampling_window_thermo_series",
    "select_stage_records",
    "self_intermediate_scattering",
    "stage_thermo_statistics",
    "species_msd",
    "summarize_stage_thermo",
    "thermo_hysteresis",
    "topology_resolved_rdf",
    "unwrap_scaled_positions",
    "write_json",
]
