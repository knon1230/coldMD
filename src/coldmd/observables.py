"""Thermodynamic observables for cold-compression molecular dynamics.

The functions in this module deliberately contain no I/O.  An engine can
therefore sample a :class:`ThermoRecord` once and pass the same immutable
record to the CSV writer, safety monitor, and progress reporter without
re-evaluating the calculator.

ASE uses the Voigt order ``xx, yy, zz, yz, xz, xy`` and a tensile-positive
stress convention.  Pressures reported here are consequently
``-trace(stress) / 3``.  Stress values supplied by ASE are in eV/Angstrom^3;
all public stress and pressure fields below are converted to GPa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


# CODATA conversion: 1 eV / Angstrom^3 in GPa.
EV_A3_TO_GPA = 160.2176634
# 1 unified atomic mass unit / Angstrom^3 in g / cm^3.
AMU_A3_TO_G_CM3 = 1.66053906660

VOIGT_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


class ObservableError(RuntimeError):
    """Raised when an observable cannot be obtained or has an invalid shape."""


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Protocol information associated with one sampled MD state."""

    step: int
    time_fs: float
    stage: str
    stage_step: int = 0
    target_volume_ratio: float | None = None
    target_pressure_GPa: float | None = None
    volume_ratio_from_engine_start: float | None = None
    branch: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationContext":
        """Build a context from an engine state mapping.

        Extra keys are intentionally ignored so a complete protocol-state
        dictionary can be passed directly by the dynamics engine.
        """

        try:
            return cls(
                step=int(value["step"]),
                time_fs=float(value["time_fs"]),
                stage=str(value["stage"]),
                stage_step=int(value.get("stage_step", 0)),
                target_volume_ratio=_optional_float(
                    value.get("target_volume_ratio")
                ),
                target_pressure_GPa=_optional_float(
                    value.get("target_pressure_GPa")
                ),
                volume_ratio_from_engine_start=_optional_float(
                    value.get("volume_ratio_from_engine_start")
                ),
                branch=_optional_string(value.get("branch")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservableError(f"Invalid observation context: {exc}") from exc

    @classmethod
    def from_protocol_context(cls, value: Any) -> "ObservationContext":
        """Adapt a mapping or an engine ``StepContext``-like object.

        The dynamics layer calls the global step ``global_step`` and the stage
        label ``stage_name``.  Accepting both spellings keeps this I/O-facing
        module decoupled from the engine dataclass itself.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            normalized = dict(value)
            if "step" not in normalized and "global_step" in normalized:
                normalized["step"] = normalized["global_step"]
            if "stage" not in normalized and "stage_name" in normalized:
                normalized["stage"] = normalized["stage_name"]
            return cls.from_mapping(normalized)
        try:
            raw_step = (
                getattr(value, "step")
                if hasattr(value, "step")
                else getattr(value, "global_step")
            )
            raw_stage = (
                getattr(value, "stage")
                if hasattr(value, "stage")
                else getattr(value, "stage_name")
            )
            return cls(
                step=int(raw_step),
                time_fs=float(getattr(value, "time_fs")),
                stage=str(raw_stage),
                stage_step=int(getattr(value, "stage_step", 0)),
                target_volume_ratio=_optional_float(
                    getattr(value, "target_volume_ratio", None)
                ),
                target_pressure_GPa=_optional_float(
                    getattr(value, "target_pressure_GPa", None)
                ),
                volume_ratio_from_engine_start=_optional_float(
                    getattr(value, "volume_ratio_from_engine_start", None)
                ),
                branch=_optional_string(getattr(value, "branch", None)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ObservableError(f"Invalid protocol context: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ThermoRecord:
    """One canonical row of thermodynamic output.

    ``stress_*_GPa`` contains the configurational (calculator) stress.  The
    configurational pressure is derived from that tensor.  Total pressure and
    deviatoric stress additionally include the kinetic stress tensor when
    momenta are present.
    """

    step: int
    time_fs: float
    time_ps: float
    stage: str
    stage_step: int
    target_volume_ratio: float | None
    target_pressure_GPa: float | None
    temperature_K: float
    potential_energy_eV: float
    kinetic_energy_eV: float
    total_energy_eV: float
    volume_A3: float
    density_g_cm3: float
    stress_xx_GPa: float
    stress_yy_GPa: float
    stress_zz_GPa: float
    stress_yz_GPa: float
    stress_xz_GPa: float
    stress_xy_GPa: float
    configurational_pressure_GPa: float
    total_pressure_GPa: float
    deviatoric_stress_GPa: float
    max_force_eV_A: float
    rms_force_eV_A: float
    min_distance_A: float
    volume_ratio_from_engine_start: float | None = None
    branch: str | None = None

    def as_dict(self) -> dict[str, int | float | str | None]:
        """Return a field-order-preserving, JSON/CSV-friendly dictionary."""

        return asdict(self)

    @classmethod
    def fieldnames(cls) -> tuple[str, ...]:
        """Return the canonical CSV column order."""

        return tuple(field.name for field in fields(cls))

    def trajectory_metadata(self) -> dict[str, int | float | str | None]:
        """Return compact metadata suitable for ``Atoms.info``.

        Full thermodynamic data belong in ``thermo.csv``.  These fields are
        enough to align a trajectory frame with its protocol state.
        """

        return {
            "coldmd_step": self.step,
            "coldmd_time_fs": self.time_fs,
            "coldmd_stage": self.stage,
            "coldmd_stage_step": self.stage_step,
            "coldmd_target_volume_ratio": self.target_volume_ratio,
            "coldmd_target_pressure_GPa": self.target_pressure_GPa,
            "coldmd_volume_ratio_from_engine_start": (
                self.volume_ratio_from_engine_start
            ),
            "coldmd_branch": self.branch,
        }


THERMO_FIELDS = ThermoRecord.fieldnames()


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_voigt(stress: ArrayLike) -> NDArray[np.float64]:
    """Return a six-component ASE-order Voigt stress array."""

    array = np.asarray(stress, dtype=np.float64)
    if array.shape == (6,):
        return np.array(array, dtype=np.float64, copy=True)
    if array.shape == (3, 3):
        # Symmetrise defensively.  A physical stress tensor should already be
        # symmetric, but tiny numerical asymmetry is not scientifically useful.
        symmetric = 0.5 * (array + array.T)
        return np.array(
            [
                symmetric[0, 0],
                symmetric[1, 1],
                symmetric[2, 2],
                symmetric[1, 2],
                symmetric[0, 2],
                symmetric[0, 1],
            ],
            dtype=np.float64,
        )
    raise ObservableError(
        f"Stress must have shape (6,) or (3, 3), received {array.shape}."
    )


def voigt_to_tensor(stress: ArrayLike) -> NDArray[np.float64]:
    """Convert ASE-order Voigt data to a symmetric 3x3 tensor."""

    xx, yy, zz, yz, xz, xy = _as_voigt(stress)
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64
    )


def tensor_to_voigt(stress: ArrayLike) -> NDArray[np.float64]:
    """Convert a symmetric 3x3 tensor to ASE-order Voigt data."""

    return _as_voigt(stress)


def pressure_from_stress_GPa(stress_eV_A3: ArrayLike) -> float:
    """Return pressure in GPa from tensile-positive stress in eV/Angstrom^3."""

    tensor = voigt_to_tensor(stress_eV_A3)
    return float(-np.trace(tensor) * EV_A3_TO_GPA / 3.0)


def von_mises_stress_GPa(stress_eV_A3: ArrayLike) -> float:
    """Return the von-Mises magnitude of the deviatoric stress in GPa."""

    tensor = voigt_to_tensor(stress_eV_A3)
    deviator = tensor - np.eye(3) * np.trace(tensor) / 3.0
    return float(np.sqrt(1.5 * np.sum(deviator * deviator)) * EV_A3_TO_GPA)


def kinetic_stress_tensor_eV_A3(atoms: Any, volume_A3: float) -> NDArray[np.float64]:
    """Return the tensile-positive kinetic stress contribution.

    ASE momenta have units compatible with ``p^2 / mass`` in eV.  The ideal
    gas contribution to ASE stress is ``-sum(p outer p / m) / volume``.
    """

    if volume_A3 <= 0.0:
        raise ObservableError(f"Cell volume must be positive, got {volume_A3}.")
    try:
        momenta = np.asarray(atoms.get_momenta(), dtype=np.float64)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    except Exception as exc:  # calculator/Atoms implementations vary
        raise ObservableError(f"Could not obtain momenta and masses: {exc}") from exc
    if momenta.ndim != 2 or momenta.shape[1:] != (3,):
        raise ObservableError(f"Momenta must have shape (N, 3), got {momenta.shape}.")
    if masses.shape != (momenta.shape[0],):
        raise ObservableError(
            f"Masses must have shape ({momenta.shape[0]},), got {masses.shape}."
        )
    if np.any(masses <= 0.0):
        raise ObservableError("All atomic masses must be positive.")
    kinetic_virial = np.einsum("ia,ib,i->ab", momenta, momenta, 1.0 / masses)
    return np.asarray(-kinetic_virial / volume_A3, dtype=np.float64)


def minimum_distance_A(atoms: Any) -> float:
    """Return the minimum pair distance using the minimum-image convention."""

    natoms = len(atoms)
    if natoms < 2:
        raise ObservableError("At least two atoms are required for a pair distance.")
    try:
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    except Exception as exc:
        raise ObservableError(f"Could not calculate pair distances: {exc}") from exc
    if distances.shape != (natoms, natoms):
        raise ObservableError(
            f"Distance matrix must have shape ({natoms}, {natoms}), "
            f"got {distances.shape}."
        )
    return float(np.min(distances[np.triu_indices(natoms, k=1)]))


def compute_thermo_record(
    atoms: Any,
    *,
    step: int,
    time_fs: float,
    stage: str,
    stage_step: int = 0,
    target_volume_ratio: float | None = None,
    target_pressure_GPa: float | None = None,
    volume_ratio_from_engine_start: float | None = None,
    branch: str | None = None,
    forces: ArrayLike | None = None,
    potential_energy_eV: float | None = None,
    stress_eV_A3: ArrayLike | None = None,
    min_distance_A_value: float | None = None,
) -> ThermoRecord:
    """Sample all canonical observables from an ASE-compatible ``Atoms``.

    Expensive calculator results can be supplied by the engine via ``forces``,
    ``potential_energy_eV``, and ``stress_eV_A3``.  If omitted, they are read
    from ``atoms``.  No non-finite value is silently repaired; the safety
    monitor is responsible for turning such a state into an orderly abort.
    """

    try:
        volume = float(atoms.get_volume())
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        kinetic_energy = float(atoms.get_kinetic_energy())
        temperature = float(atoms.get_temperature())
    except Exception as exc:
        raise ObservableError(f"Could not sample atomic state: {exc}") from exc

    if potential_energy_eV is None:
        try:
            potential_energy = float(atoms.get_potential_energy())
        except Exception as exc:
            raise ObservableError(f"Could not calculate potential energy: {exc}") from exc
    else:
        potential_energy = float(potential_energy_eV)

    if forces is None:
        try:
            force_array = np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception as exc:
            raise ObservableError(f"Could not calculate forces: {exc}") from exc
    else:
        force_array = np.asarray(forces, dtype=np.float64)
    if force_array.shape != (len(atoms), 3):
        raise ObservableError(
            f"Forces must have shape ({len(atoms)}, 3), got {force_array.shape}."
        )

    if stress_eV_A3 is None:
        try:
            configurational_stress_voigt = _as_voigt(atoms.get_stress(voigt=True))
        except Exception as exc:
            raise ObservableError(f"Could not calculate stress: {exc}") from exc
    else:
        configurational_stress_voigt = _as_voigt(stress_eV_A3)

    configurational_stress = voigt_to_tensor(configurational_stress_voigt)
    kinetic_stress = kinetic_stress_tensor_eV_A3(atoms, volume)
    total_stress = configurational_stress + kinetic_stress

    force_magnitudes = np.linalg.norm(force_array, axis=1)
    density = float(np.sum(masses) * AMU_A3_TO_G_CM3 / volume)
    min_distance = (
        minimum_distance_A(atoms)
        if min_distance_A_value is None
        else float(min_distance_A_value)
    )
    stress_GPa = configurational_stress_voigt * EV_A3_TO_GPA

    return ThermoRecord(
        step=int(step),
        time_fs=float(time_fs),
        time_ps=float(time_fs) / 1000.0,
        stage=str(stage),
        stage_step=int(stage_step),
        target_volume_ratio=_optional_float(target_volume_ratio),
        target_pressure_GPa=_optional_float(target_pressure_GPa),
        temperature_K=temperature,
        potential_energy_eV=potential_energy,
        kinetic_energy_eV=kinetic_energy,
        total_energy_eV=potential_energy + kinetic_energy,
        volume_A3=volume,
        density_g_cm3=density,
        stress_xx_GPa=float(stress_GPa[0]),
        stress_yy_GPa=float(stress_GPa[1]),
        stress_zz_GPa=float(stress_GPa[2]),
        stress_yz_GPa=float(stress_GPa[3]),
        stress_xz_GPa=float(stress_GPa[4]),
        stress_xy_GPa=float(stress_GPa[5]),
        configurational_pressure_GPa=float(
            -np.trace(configurational_stress) * EV_A3_TO_GPA / 3.0
        ),
        total_pressure_GPa=float(-np.trace(total_stress) * EV_A3_TO_GPA / 3.0),
        deviatoric_stress_GPa=von_mises_stress_GPa(total_stress),
        max_force_eV_A=float(np.max(force_magnitudes)),
        rms_force_eV_A=float(np.sqrt(np.mean(force_magnitudes**2))),
        min_distance_A=min_distance,
        volume_ratio_from_engine_start=_optional_float(
            volume_ratio_from_engine_start
        ),
        branch=_optional_string(branch),
    )


class RecordSink(Protocol):
    """Callable accepted by :class:`ThermoObserver`."""

    def __call__(self, record: ThermoRecord) -> Any: ...


class ThermoObserver:
    """A zero-argument callback suitable for an ASE dynamics observer.

    ``context_provider`` must return either :class:`ObservationContext` or a
    mapping with at least ``step``, ``time_fs``, and ``stage``.  The observer
    returns the record as well as sending it to the sink, which makes direct
    unit testing and engine integration straightforward.
    """

    def __init__(
        self,
        atoms: Any,
        context_provider: Callable[[], ObservationContext | Mapping[str, Any]],
        sink: RecordSink,
    ) -> None:
        self.atoms = atoms
        self.context_provider = context_provider
        self.sink = sink

    def __call__(self) -> ThermoRecord:
        context = self.context_provider()
        context = ObservationContext.from_protocol_context(context)
        record = compute_thermo_record(
            self.atoms,
            step=context.step,
            time_fs=context.time_fs,
            stage=context.stage,
            stage_step=context.stage_step,
            target_volume_ratio=context.target_volume_ratio,
            target_pressure_GPa=context.target_pressure_GPa,
            volume_ratio_from_engine_start=(
                context.volume_ratio_from_engine_start
            ),
            branch=context.branch,
        )
        self.sink(record)
        return record


__all__ = [
    "AMU_A3_TO_G_CM3",
    "EV_A3_TO_GPA",
    "ObservationContext",
    "ObservableError",
    "THERMO_FIELDS",
    "ThermoObserver",
    "ThermoRecord",
    "VOIGT_COMPONENTS",
    "compute_thermo_record",
    "kinetic_stress_tensor_eV_A3",
    "minimum_distance_A",
    "pressure_from_stress_GPa",
    "tensor_to_voigt",
    "voigt_to_tensor",
    "von_mises_stress_GPa",
]
