"""Stage-based execution engine for cold compression and decompression MD.

The engine owns protocol sequencing only.  It does not initialize velocities,
write files, silently clamp unsafe states, or classify the resulting structure.
Logging, safety checks, and checkpointing enter through small hooks that observe
immutable :class:`StepContext` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
)

import numpy as np

from ase import Atoms
from ase.md.md import MolecularDynamics

from .dynamics import (
    IsotropicVolumeRampBussi,
    build_fixed_cell_nvt,
    build_isotropic_mtk_npt,
    build_isotropic_volume_ramp,
    build_velocity_verlet,
)
from .schedules import RampSchedule, volume_ratio_at_step


VolumeRampRole = Literal["cold_compression", "decompression"]
ThermostatKind = Literal["bussi", "nose_hoover_chain"]
StageBranch = Literal[
    "cold_compression",
    "decompression",
    "recovery",
    "initial_hold",
    "high_density_hold",
    "recovered_hold",
    "other",
]
# Kept as public aliases for callers written against the original stage-specific
# metadata types.  Branches are analysis labels rather than integration rules,
# so an NVT or NPT stage may use any of them in a free-form stage sequence.
NPTPlateauRole: TypeAlias = StageBranch
FixedHoldRole: TypeAlias = StageBranch


_STAGE_BRANCHES = {
    "cold_compression",
    "decompression",
    "recovery",
    "initial_hold",
    "high_density_hold",
    "recovered_hold",
    "other",
}
_THERMOSTAT_KINDS = {"bussi", "nose_hoover_chain"}


def _stage_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("stage name must be a non-empty string")
    return value.strip()


def _steps(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("stage steps must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError("stage steps must be greater than zero")
    return result


def _positive(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero; got {value!r}")
    return result


def _finite(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _handoff_window(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _cell_tuple(atoms: Atoms) -> tuple[tuple[float, float, float], ...]:
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("atoms must have a finite 3x3 cell")
    if float(np.linalg.det(cell)) <= np.finfo(float).eps:
        raise ValueError("atoms cell must be right-handed and non-singular")
    return tuple(tuple(float(component) for component in row) for row in cell)


@dataclass(frozen=True, slots=True)
class FixedCellNVTStage:
    """A fixed-cell canonical stage using the selected thermostat."""

    name: str
    steps: int
    temperature_K: float
    thermostat_tau_fs: float
    branch: FixedHoldRole = "other"
    thermostat_kind: ThermostatKind = "bussi"
    thermostat_chain_length: int = 3
    thermostat_substeps: int = 1
    handoff_window_steps: int | None = None
    handoff_sample_interval_steps: int = 10

    kind: ClassVar[str] = "fixed_nvt"

    def __post_init__(self) -> None:
        _stage_name(self.name)
        _steps(self.steps)
        _positive(self.temperature_K, "temperature_K")
        _positive(self.thermostat_tau_fs, "thermostat_tau_fs")
        if self.branch not in _STAGE_BRANCHES:
            raise ValueError("invalid fixed-cell NVT branch")
        if self.thermostat_kind not in _THERMOSTAT_KINDS:
            raise ValueError(
                "thermostat_kind must be 'bussi' or 'nose_hoover_chain'"
            )
        _positive_integer(self.thermostat_chain_length, "thermostat_chain_length")
        _positive_integer(self.thermostat_substeps, "thermostat_substeps")
        window = _handoff_window(self.handoff_window_steps, "handoff_window_steps")
        if window is not None and window > self.steps:
            raise ValueError("handoff_window_steps cannot exceed stage steps")
        _positive_integer(
            self.handoff_sample_interval_steps, "handoff_sample_interval_steps"
        )


@dataclass(frozen=True, slots=True)
class NVEStage:
    """A fixed-cell velocity-Verlet microcanonical stage."""

    name: str
    steps: int

    kind: ClassVar[str] = "nve"

    def __post_init__(self) -> None:
        _stage_name(self.name)
        _steps(self.steps)


@dataclass(frozen=True, slots=True)
class VolumeRampStage:
    """A prescribed isotropic cold-compression or decompression stage.

    ``target_volume_ratio`` is ``V_end / V_stage_start``.  It is not cumulative
    with respect to the initial CIF unless this stage is the first volume ramp.
    """

    name: str
    role: VolumeRampRole
    steps: int
    temperature_K: float
    thermostat_tau_fs: float
    target_volume_ratio: float
    schedule: RampSchedule = "log_linear"
    handoff_window_steps: int | None = None
    handoff_sample_interval_steps: int = 10

    @property
    def kind(self) -> str:
        return self.role

    def __post_init__(self) -> None:
        _stage_name(self.name)
        _steps(self.steps)
        _positive(self.temperature_K, "temperature_K")
        _positive(self.thermostat_tau_fs, "thermostat_tau_fs")
        ratio = _positive(self.target_volume_ratio, "target_volume_ratio")
        # This call also validates the schedule string at runtime.
        volume_ratio_at_step(ratio, 0, self.steps, self.schedule)
        if self.role == "cold_compression":
            if ratio >= 1.0:
                raise ValueError(
                    "cold_compression requires 0 < target_volume_ratio < 1"
                )
        elif self.role == "decompression":
            if ratio <= 1.0:
                raise ValueError("decompression requires target_volume_ratio > 1")
        else:
            raise ValueError(
                "volume-ramp role must be 'cold_compression' or 'decompression'"
            )
        window = _handoff_window(self.handoff_window_steps, "handoff_window_steps")
        if window is not None and window > self.steps:
            raise ValueError("handoff_window_steps cannot exceed stage steps")
        _positive_integer(
            self.handoff_sample_interval_steps, "handoff_sample_interval_steps"
        )


@dataclass(frozen=True, slots=True)
class IsotropicNPTStage:
    """One fixed-pressure isotropic MTK plateau."""

    name: str
    steps: int
    temperature_K: float
    pressure_GPa: float
    thermostat_tau_fs: float
    barostat_tau_fs: float
    branch: NPTPlateauRole = "recovery"
    thermostat_chain_length: int = 3
    barostat_chain_length: int = 3
    thermostat_substeps: int = 1
    barostat_substeps: int = 1
    handoff_window_steps: int | None = None
    handoff_sample_interval_steps: int = 10

    kind: ClassVar[str] = "isotropic_npt_plateau"

    def __post_init__(self) -> None:
        _stage_name(self.name)
        _steps(self.steps)
        _positive(self.temperature_K, "temperature_K")
        _finite(self.pressure_GPa, "pressure_GPa")
        _positive(self.thermostat_tau_fs, "thermostat_tau_fs")
        _positive(self.barostat_tau_fs, "barostat_tau_fs")
        if self.branch not in _STAGE_BRANCHES:
            raise ValueError("invalid isotropic NPT branch")
        _positive_integer(self.thermostat_chain_length, "thermostat_chain_length")
        _positive_integer(self.barostat_chain_length, "barostat_chain_length")
        _positive_integer(self.thermostat_substeps, "thermostat_substeps")
        _positive_integer(self.barostat_substeps, "barostat_substeps")
        window = _handoff_window(self.handoff_window_steps, "handoff_window_steps")
        if window is not None and window > self.steps:
            raise ValueError("handoff_window_steps cannot exceed stage steps")
        _positive_integer(
            self.handoff_sample_interval_steps, "handoff_sample_interval_steps"
        )


StageSpec: TypeAlias = (
    FixedCellNVTStage | NVEStage | VolumeRampStage | IsotropicNPTStage
)


@dataclass(frozen=True, slots=True)
class RunCursor:
    """Serializable position in a protocol.

    A mid-stage Nosé–Hoover-chain NVT or isotropic MTK cursor is intentionally
    not resumable: ASE's extended thermostat/barostat state is not public.
    Checkpoint code should restart such work from the beginning of its stage.
    """

    stage_index: int = 0
    stage_step: int = 0
    global_step: int = 0
    time_fs: float = 0.0
    stage_reference_cell_A: tuple[tuple[float, float, float], ...] | None = None
    reference_volume_A3: float | None = None

    def __post_init__(self) -> None:
        for name in ("stage_index", "stage_step", "global_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must not be negative")
        if isinstance(self.time_fs, (bool, np.bool_)) or not isinstance(
            self.time_fs, Real
        ):
            raise TypeError("time_fs must be a real number")
        time = float(self.time_fs)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time_fs must be finite and non-negative")
        if self.stage_reference_cell_A is not None:
            cell = np.asarray(self.stage_reference_cell_A, dtype=float)
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                raise ValueError("stage_reference_cell_A must be a finite 3x3 cell")
            if float(np.linalg.det(cell)) <= np.finfo(float).eps:
                raise ValueError(
                    "stage_reference_cell_A must be right-handed and non-singular"
                )
        if self.reference_volume_A3 is not None:
            _positive(self.reference_volume_A3, "reference_volume_A3")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint representation."""

        return {
            "stage_index": int(self.stage_index),
            "stage_step": int(self.stage_step),
            "global_step": int(self.global_step),
            "time_fs": float(self.time_fs),
            "stage_reference_cell_A": (
                None
                if self.stage_reference_cell_A is None
                else [list(row) for row in self.stage_reference_cell_A]
            ),
            "reference_volume_A3": (
                None
                if self.reference_volume_A3 is None
                else float(self.reference_volume_A3)
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunCursor":
        """Restore a cursor from its checkpoint mapping."""

        raw_cell = value.get("stage_reference_cell_A")
        cell = (
            None
            if raw_cell is None
            else tuple(tuple(float(item) for item in row) for row in raw_cell)
        )
        return cls(
            stage_index=value.get("stage_index", 0),
            stage_step=value.get("stage_step", 0),
            global_step=value.get("global_step", 0),
            time_fs=value.get("time_fs", 0.0),
            stage_reference_cell_A=cell,
            reference_volume_A3=value.get("reference_volume_A3"),
        )


@dataclass(frozen=True, slots=True)
class StepContext:
    """Immutable state passed to logging, safety, and checkpoint hooks."""

    stage_index: int
    stage_name: str
    stage_kind: str
    stage_step: int
    stage_steps: int
    global_step: int
    time_fs: float
    timestep_fs: float
    progress: float
    target_volume_ratio: float | None
    final_target_volume_ratio: float | None
    target_pressure_GPa: float | None
    branch: StageBranch | None
    volume_ratio_from_engine_start: float
    reference_volume_A3: float
    stage_reference_cell_A: tuple[tuple[float, float, float], ...]
    resumed: bool
    thermostat_kind: ThermostatKind | None = None
    handoff_window_steps: int | None = None

    @property
    def exact_resume_supported(self) -> bool:
        """Whether this state can be resumed without hidden ASE state.

        Interior Nosé–Hoover-chain NVT and MTK states are excluded because ASE
        does not expose a public serializer for their extended variables.  A
        stage boundary is safe because the following stage constructs a new
        integrator.
        """

        if self.stage_step in (0, self.stage_steps):
            return True
        # Selecting a snapshot closest to a completed sampling-window mean
        # requires the whole candidate pool.  We only advertise the immutable
        # stage-boundary checkpoint, never a partial pool that would alter a
        # later handoff on resume.
        if self.handoff_window_steps is not None:
            return False
        return (
            self.stage_kind != IsotropicNPTStage.kind
            and self.thermostat_kind != "nose_hoover_chain"
        )

    def resume_cursor(self) -> RunCursor:
        """Return the cursor represented by this completed-step context."""

        if self.stage_step == self.stage_steps:
            return RunCursor(
                stage_index=self.stage_index + 1,
                stage_step=0,
                global_step=self.global_step,
                time_fs=self.time_fs,
                stage_reference_cell_A=None,
                reference_volume_A3=self.reference_volume_A3,
            )
        return RunCursor(
            stage_index=self.stage_index,
            stage_step=self.stage_step,
            global_step=self.global_step,
            time_fs=self.time_fs,
            stage_reference_cell_A=self.stage_reference_cell_A,
            reference_volume_A3=self.reference_volume_A3,
        )


class DynamicsHook(Protocol):
    """Minimal observer interface; hook implementations must not mutate cells."""

    def on_stage_start(self, atoms: Atoms, context: StepContext) -> None: ...

    def on_step(self, atoms: Atoms, context: StepContext) -> None: ...

    def on_stage_end(self, atoms: Atoms, context: StepContext) -> None: ...


class BaseDynamicsHook:
    """No-op base class for hooks interested in only some events."""

    def on_stage_start(self, atoms: Atoms, context: StepContext) -> None:
        del atoms, context

    def on_step(self, atoms: Atoms, context: StepContext) -> None:
        del atoms, context

    def on_stage_end(self, atoms: Atoms, context: StepContext) -> None:
        del atoms, context

    def on_stage_handoff(
        self,
        atoms: Atoms,
        context: StepContext,
        handoff: Mapping[str, Any],
    ) -> None:
        del atoms, context, handoff


HookCallback: TypeAlias = Callable[[Atoms, StepContext], None]


@dataclass(slots=True)
class CallbackHook(BaseDynamicsHook):
    """Adapt up to three ordinary callables to :class:`DynamicsHook`."""

    stage_start: HookCallback | None = None
    step: HookCallback | None = None
    stage_end: HookCallback | None = None

    def on_stage_start(self, atoms: Atoms, context: StepContext) -> None:
        if self.stage_start is not None:
            self.stage_start(atoms, context)

    def on_step(self, atoms: Atoms, context: StepContext) -> None:
        if self.step is not None:
            self.step(atoms, context)

    def on_stage_end(self, atoms: Atoms, context: StepContext) -> None:
        if self.stage_end is not None:
            self.stage_end(atoms, context)


class NonExactResumeError(RuntimeError):
    """Raised when a cursor would conceal loss of extended integrator state."""


@dataclass(slots=True)
class _HandoffSnapshot:
    """One mechanically self-consistent state from a stage sampling tail."""

    stage_step: int
    global_step: int
    volume_A3: float
    positions: np.ndarray
    cell: np.ndarray
    momenta: np.ndarray

    @classmethod
    def capture(
        cls, atoms: Atoms, *, stage_step: int, global_step: int
    ) -> "_HandoffSnapshot":
        return cls(
            stage_step=int(stage_step),
            global_step=int(global_step),
            volume_A3=float(atoms.get_volume()),
            positions=np.asarray(atoms.get_positions(), dtype=float).copy(),
            cell=np.asarray(atoms.get_cell(), dtype=float).copy(),
            momenta=np.asarray(atoms.get_momenta(), dtype=float).copy(),
        )

    def restore(self, atoms: Atoms) -> None:
        atoms.set_cell(self.cell, scale_atoms=False)
        atoms.set_positions(self.positions)
        atoms.set_momenta(self.momenta)


class ProtocolEngine:
    """Execute an ordered sequence of validated MD stages on one Atoms object."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        timestep_fs: Real,
        rng: np.random.Generator | None = None,
        hooks: Sequence[DynamicsHook] = (),
        reference_volume_A3: Real | None = None,
    ) -> None:
        self.atoms = atoms
        self.timestep_fs = _positive(timestep_fs, "timestep_fs")
        self.rng = np.random.default_rng() if rng is None else rng
        self.hooks = tuple(hooks)

        current_volume = float(atoms.get_volume())
        if reference_volume_A3 is None:
            self.reference_volume_A3 = _positive(
                current_volume, "initial atoms volume"
            )
        else:
            self.reference_volume_A3 = _positive(
                reference_volume_A3, "reference_volume_A3"
            )
        self.cursor = RunCursor(reference_volume_A3=self.reference_volume_A3)

    def _context(
        self,
        *,
        stage: StageSpec,
        stage_index: int,
        stage_step: int,
        global_step: int,
        time_fs: float,
        stage_reference_cell_A: tuple[tuple[float, float, float], ...],
        resumed: bool,
    ) -> StepContext:
        target_ratio: float | None = None
        final_ratio: float | None = None
        target_pressure: float | None = None
        branch: StageBranch | None = None
        thermostat_kind: ThermostatKind | None = None
        if isinstance(stage, FixedCellNVTStage):
            branch = stage.branch
            thermostat_kind = stage.thermostat_kind
        elif isinstance(stage, VolumeRampStage):
            target_ratio = volume_ratio_at_step(
                stage.target_volume_ratio,
                stage_step,
                stage.steps,
                stage.schedule,
            )
            final_ratio = float(stage.target_volume_ratio)
            branch = stage.role
            thermostat_kind = "bussi"
        elif isinstance(stage, IsotropicNPTStage):
            target_pressure = float(stage.pressure_GPa)
            branch = stage.branch
            thermostat_kind = "nose_hoover_chain"

        return StepContext(
            stage_index=stage_index,
            stage_name=stage.name,
            stage_kind=stage.kind,
            stage_step=stage_step,
            stage_steps=stage.steps,
            global_step=global_step,
            time_fs=float(time_fs),
            timestep_fs=self.timestep_fs,
            progress=stage_step / stage.steps,
            target_volume_ratio=target_ratio,
            final_target_volume_ratio=final_ratio,
            target_pressure_GPa=target_pressure,
            branch=branch,
            volume_ratio_from_engine_start=(
                float(self.atoms.get_volume()) / self.reference_volume_A3
            ),
            reference_volume_A3=self.reference_volume_A3,
            stage_reference_cell_A=stage_reference_cell_A,
            resumed=resumed,
            thermostat_kind=thermostat_kind,
            handoff_window_steps=getattr(stage, "handoff_window_steps", None),
        )

    def _emit(self, event: str, context: StepContext) -> None:
        for hook in self.hooks:
            callback = getattr(hook, event)
            callback(self.atoms, context)

    def _emit_handoff(
        self, context: StepContext, handoff: Mapping[str, Any]
    ) -> None:
        for hook in self.hooks:
            callback = getattr(hook, "on_stage_handoff", None)
            if callback is not None:
                callback(self.atoms, context, handoff)

    def _build_dynamics(
        self,
        stage: StageSpec,
        *,
        completed_steps: int,
        stage_reference_cell_A: tuple[tuple[float, float, float], ...],
    ) -> MolecularDynamics:
        common = {"timestep_fs": self.timestep_fs, "logfile": None}
        if isinstance(stage, FixedCellNVTStage):
            if (
                stage.thermostat_kind == "nose_hoover_chain"
                and completed_steps != 0
            ):
                raise NonExactResumeError(
                    "cannot resume inside a Nosé–Hoover-chain NVT stage "
                    "without its extended thermostat state; restart this stage "
                    "from its stage-boundary checkpoint"
                )
            return build_fixed_cell_nvt(
                self.atoms,
                **common,
                thermostat_kind=stage.thermostat_kind,
                temperature_K=stage.temperature_K,
                thermostat_tau_fs=stage.thermostat_tau_fs,
                rng=self.rng,
                thermostat_chain_length=stage.thermostat_chain_length,
                thermostat_substeps=stage.thermostat_substeps,
            )
        if isinstance(stage, NVEStage):
            return build_velocity_verlet(self.atoms, **common)
        if isinstance(stage, VolumeRampStage):
            return build_isotropic_volume_ramp(
                self.atoms,
                **common,
                temperature_K=stage.temperature_K,
                thermostat_tau_fs=stage.thermostat_tau_fs,
                target_volume_ratio=stage.target_volume_ratio,
                total_steps=stage.steps,
                schedule=stage.schedule,
                rng=self.rng,
                reference_cell=stage_reference_cell_A,
                completed_steps=completed_steps,
            )
        if isinstance(stage, IsotropicNPTStage):
            if completed_steps != 0:
                raise NonExactResumeError(
                    "cannot resume inside an isotropic MTK plateau without its "
                    "extended thermostat/barostat state; restart this plateau from "
                    "its stage-boundary checkpoint"
                )
            # Reaching this branch for every IsotropicNPTStage is intentional:
            # different fixed pressures are separate MTK instances, with no
            # mutation of ASE's private target-pressure attribute.
            return build_isotropic_mtk_npt(
                self.atoms,
                **common,
                temperature_K=stage.temperature_K,
                pressure_GPa=stage.pressure_GPa,
                thermostat_tau_fs=stage.thermostat_tau_fs,
                barostat_tau_fs=stage.barostat_tau_fs,
                thermostat_chain_length=stage.thermostat_chain_length,
                barostat_chain_length=stage.barostat_chain_length,
                thermostat_substeps=stage.thermostat_substeps,
                barostat_substeps=stage.barostat_substeps,
            )
        raise TypeError(f"unsupported stage specification {type(stage).__name__}")

    def run(
        self,
        stages: Sequence[StageSpec],
        *,
        cursor: RunCursor | None = None,
    ) -> RunCursor:
        """Run all remaining stages and return the next-stage cursor.

        Positions, momenta, and cells flow directly between stages.  In
        particular, this method never initializes or reinitializes velocities.
        """

        stage_list = tuple(stages)
        names = [stage.name for stage in stage_list]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")

        active = (
            RunCursor(reference_volume_A3=self.reference_volume_A3)
            if cursor is None
            else cursor
        )
        if active.reference_volume_A3 is not None and not np.isclose(
            active.reference_volume_A3,
            self.reference_volume_A3,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "cursor reference_volume_A3 does not match the engine reference "
                "volume; construct a resumed engine with the original CIF volume"
            )
        if active.stage_index > len(stage_list):
            raise ValueError("cursor stage_index lies beyond the protocol")

        expected_global_step = sum(
            stage.steps for stage in stage_list[: active.stage_index]
        )
        expected_global_step += int(active.stage_step)
        if active.global_step != expected_global_step:
            raise ValueError(
                "cursor global_step is inconsistent with its stage position: "
                f"{active.global_step} != {expected_global_step}"
            )

        expected_time_fs = active.global_step * self.timestep_fs
        if not np.isclose(
            active.time_fs,
            expected_time_fs,
            rtol=1.0e-12,
            atol=1.0e-9,
        ):
            raise ValueError(
                "cursor time_fs is inconsistent with global_step and timestep_fs: "
                f"{active.time_fs} != {active.global_step} * {self.timestep_fs}"
            )
        if active.stage_index == len(stage_list):
            if active.stage_step != 0:
                raise ValueError("a completed protocol cursor must have stage_step=0")
            self.cursor = active
            return active

        global_step = int(active.global_step)
        time_fs = float(active.time_fs)

        for stage_index in range(active.stage_index, len(stage_list)):
            stage = stage_list[stage_index]
            completed_steps = (
                int(active.stage_step) if stage_index == active.stage_index else 0
            )
            if not 0 <= completed_steps < stage.steps:
                raise ValueError(
                    f"cursor stage_step must lie in [0, {stage.steps - 1}] for "
                    f"stage {stage.name!r}"
                )

            resumed = completed_steps > 0
            if resumed:
                if active.stage_reference_cell_A is None:
                    raise ValueError(
                        "a mid-stage cursor requires stage_reference_cell_A"
                    )
                stage_reference_cell_A = active.stage_reference_cell_A
                if isinstance(stage, (FixedCellNVTStage, NVEStage)) and not np.allclose(
                    np.asarray(self.atoms.get_cell(), dtype=float),
                    np.asarray(stage_reference_cell_A, dtype=float),
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "current cell does not match the fixed cell saved for the "
                        f"resumed stage {stage.name!r}"
                    )
            else:
                stage_reference_cell_A = _cell_tuple(self.atoms)

            dynamics = self._build_dynamics(
                stage,
                completed_steps=completed_steps,
                stage_reference_cell_A=stage_reference_cell_A,
            )
            context = self._context(
                stage=stage,
                stage_index=stage_index,
                stage_step=completed_steps,
                global_step=global_step,
                time_fs=time_fs,
                stage_reference_cell_A=stage_reference_cell_A,
                resumed=resumed,
            )
            self.cursor = RunCursor(
                stage_index=stage_index,
                stage_step=completed_steps,
                global_step=global_step,
                time_fs=time_fs,
                stage_reference_cell_A=stage_reference_cell_A,
                reference_volume_A3=self.reference_volume_A3,
            )
            self._emit("on_stage_start", context)

            handoff_window = getattr(stage, "handoff_window_steps", None)
            handoff_interval = int(
                getattr(stage, "handoff_sample_interval_steps", 1)
            )
            handoff_start = (
                None
                if handoff_window is None
                else max(0, stage.steps - int(handoff_window))
            )
            handoff_candidates: list[_HandoffSnapshot] = []
            if handoff_start == 0:
                # A window equal to the complete stage includes its initial
                # boundary.  Capturing it here also keeps the interval rule
                # deterministic for a one-step stage.
                handoff_candidates.append(
                    _HandoffSnapshot.capture(
                        self.atoms,
                        stage_step=0,
                        global_step=global_step,
                    )
                )

            while completed_steps < stage.steps:
                dynamics.step()
                # Direct step execution keeps cell mutation out of observer
                # callbacks.  nsteps is public ASE bookkeeping only; equations
                # of motion in the selected integrators do not depend on it.
                dynamics.nsteps += 1
                completed_steps += 1
                global_step += 1
                # Recompute from the integer counter instead of accumulating
                # binary floating-point roundoff over very long trajectories.
                time_fs = global_step * self.timestep_fs

                context = self._context(
                    stage=stage,
                    stage_index=stage_index,
                    stage_step=completed_steps,
                    global_step=global_step,
                    time_fs=time_fs,
                    stage_reference_cell_A=stage_reference_cell_A,
                    resumed=resumed,
                )
                self.cursor = context.resume_cursor()
                self._emit("on_step", context)

                if (
                    handoff_start is not None
                    and completed_steps >= handoff_start
                    and (
                        completed_steps == stage.steps
                        or (completed_steps - handoff_start) % handoff_interval == 0
                    )
                ):
                    handoff_candidates.append(
                        _HandoffSnapshot.capture(
                            self.atoms,
                            stage_step=completed_steps,
                            global_step=global_step,
                        )
                    )

            self._emit("on_stage_end", context)
            if handoff_start is not None and stage_index + 1 < len(stage_list):
                if not handoff_candidates:
                    # The final step is always eligible, so this branch guards
                    # only future changes to candidate scheduling.
                    raise RuntimeError(
                        f"No sampling-window handoff candidates for stage {stage.name!r}"
                    )
                mean_volume = float(
                    np.mean([candidate.volume_A3 for candidate in handoff_candidates])
                )
                selected_snapshot = min(
                    handoff_candidates,
                    key=lambda candidate: (
                        abs(candidate.volume_A3 - mean_volume),
                        -candidate.stage_step,
                    ),
                )
                selected_snapshot.restore(self.atoms)
                handoff = {
                    "from_stage": stage.name,
                    "to_stage": stage_list[stage_index + 1].name,
                    "selection": "sampling_window_volume_nearest_mean_snapshot",
                    "sampling_window_steps": int(handoff_window),
                    "sample_interval_steps": handoff_interval,
                    "candidate_count": len(handoff_candidates),
                    "mean_volume_A3": mean_volume,
                    "selected_volume_A3": selected_snapshot.volume_A3,
                    "selected_stage_step": selected_snapshot.stage_step,
                    "selected_global_step": selected_snapshot.global_step,
                    "absolute_volume_difference_A3": abs(
                        selected_snapshot.volume_A3 - mean_volume
                    ),
                }
                self.cursor = context.resume_cursor()
                self._emit_handoff(context, handoff)

        self.cursor = RunCursor(
            stage_index=len(stage_list),
            stage_step=0,
            global_step=global_step,
            time_fs=time_fs,
            stage_reference_cell_A=None,
            reference_volume_A3=self.reference_volume_A3,
        )
        return self.cursor


__all__ = [
    "BaseDynamicsHook",
    "CallbackHook",
    "DynamicsHook",
    "FixedCellNVTStage",
    "FixedHoldRole",
    "IsotropicNPTStage",
    "NPTPlateauRole",
    "NVEStage",
    "NonExactResumeError",
    "ProtocolEngine",
    "RunCursor",
    "StageSpec",
    "StageBranch",
    "StepContext",
    "ThermostatKind",
    "VolumeRampRole",
    "VolumeRampStage",
]
