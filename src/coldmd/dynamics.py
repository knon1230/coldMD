"""ASE dynamics constructors and the explicit isotropic volume-ramp driver."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ase import Atoms, units
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.md.verlet import VelocityVerlet

from .schedules import RampSchedule, isotropic_cell_at_step, volume_ratio_at_step


def _positive_finite(value: Real, name: str) -> float:
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


def _require_calculator(atoms: Atoms) -> None:
    if atoms.calc is None:
        raise ValueError("atoms must have an ASE calculator before dynamics are built")


def _require_thermal_momenta(atoms: Atoms) -> None:
    kinetic_energy = float(atoms.get_kinetic_energy())
    if not np.isfinite(kinetic_energy) or kinetic_energy <= 0.0:
        raise ValueError(
            "thermostatted dynamics require non-zero finite initial kinetic energy; "
            "initialize momenta once before the protocol starts"
        )


def _require_unconstrained(atoms: Atoms, dynamics_name: str) -> None:
    if len(atoms.constraints) > 0:
        raise ValueError(
            f"{dynamics_name} does not support atomic/cell constraints in coldmd"
        )


def _require_rng(rng: Any) -> None:
    if rng is None:
        raise ValueError("an explicit random-number generator is required")
    for method in ("standard_normal", "standard_gamma"):
        if not callable(getattr(rng, method, None)):
            raise TypeError(f"rng must provide a callable {method} method")


def initialize_momenta_once(
    atoms: Atoms,
    temperature_K: Real,
    rng: Any,
    *,
    remove_com: bool = True,
    exact_temperature: bool = False,
    allow_overwrite: bool = False,
    zero_energy_tolerance_eV: Real = 1.0e-12,
) -> None:
    """Draw initial thermal momenta once, with overwrite protection.

    Fresh-run orchestration may call this before constructing the first stage.
    Resume paths and stage transitions must not call it.  By default, any
    already non-zero kinetic energy is treated as evidence of existing MD state
    and raises instead of silently destroying trajectory continuity.
    """

    temperature = _positive_finite(temperature_K, "temperature_K")
    _require_rng(rng)
    for name, value in (
        ("remove_com", remove_com),
        ("exact_temperature", exact_temperature),
        ("allow_overwrite", allow_overwrite),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    tolerance = float(zero_energy_tolerance_eV)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_energy_tolerance_eV must be finite and non-negative")

    kinetic_energy = float(atoms.get_kinetic_energy())
    if not np.isfinite(kinetic_energy):
        raise ValueError("existing kinetic energy is non-finite")
    if kinetic_energy > tolerance and not allow_overwrite:
        raise ValueError(
            "refusing to overwrite non-zero momenta; velocity initialization is "
            "allowed only once for a fresh protocol"
        )

    thermalize_momenta(
        atoms,
        temperature,
        exact_temperature=bool(exact_temperature),
        rng=rng,
    )
    if remove_com:
        Stationary(atoms, preserve_temperature=True)

    initialized_energy = float(atoms.get_kinetic_energy())
    if not np.isfinite(initialized_energy) or initialized_energy <= 0.0:
        raise RuntimeError("thermal momentum initialization produced invalid kinetic energy")


def build_bussi_nvt(
    atoms: Atoms,
    *,
    timestep_fs: Real,
    temperature_K: Real,
    thermostat_tau_fs: Real,
    rng: Any,
    **kwargs: Any,
) -> Bussi:
    """Build fixed-cell canonical dynamics without changing the momenta."""

    _require_calculator(atoms)
    _require_thermal_momenta(atoms)
    _require_rng(rng)
    dt = _positive_finite(timestep_fs, "timestep_fs")
    temperature = _positive_finite(temperature_K, "temperature_K")
    tau = _positive_finite(thermostat_tau_fs, "thermostat_tau_fs")
    return Bussi(
        atoms,
        timestep=dt * units.fs,
        temperature_K=temperature,
        taut=tau * units.fs,
        rng=rng,
        **kwargs,
    )


def build_nose_hoover_chain_nvt(
    atoms: Atoms,
    *,
    timestep_fs: Real,
    temperature_K: Real,
    thermostat_tau_fs: Real,
    thermostat_chain_length: int = 3,
    thermostat_substeps: int = 1,
    **kwargs: Any,
) -> NoseHooverChainNVT:
    """Build fixed-cell Nosé–Hoover-chain canonical dynamics.

    ASE 3.29 does not support constrained Nosé–Hoover-chain NVT dynamics,
    so coldmd rejects constraints explicitly instead of allowing a trajectory
    whose equations of motion do not honor them.  This deterministic
    thermostat does not consume the protocol random-number generator.
    """

    _require_calculator(atoms)
    _require_thermal_momenta(atoms)
    _require_unconstrained(atoms, "Nosé–Hoover-chain NVT")
    dt = _positive_finite(timestep_fs, "timestep_fs")
    temperature = _positive_finite(temperature_K, "temperature_K")
    tdamp = _positive_finite(thermostat_tau_fs, "thermostat_tau_fs")
    tchain = _positive_integer(
        thermostat_chain_length, "thermostat_chain_length"
    )
    tloop = _positive_integer(thermostat_substeps, "thermostat_substeps")
    return NoseHooverChainNVT(
        atoms,
        timestep=dt * units.fs,
        temperature_K=temperature,
        tdamp=tdamp * units.fs,
        tchain=tchain,
        tloop=tloop,
        **kwargs,
    )


def build_fixed_cell_nvt(
    atoms: Atoms,
    *,
    thermostat_kind: str,
    timestep_fs: Real,
    temperature_K: Real,
    thermostat_tau_fs: Real,
    rng: Any = None,
    thermostat_chain_length: int = 3,
    thermostat_substeps: int = 1,
    **kwargs: Any,
) -> Bussi | NoseHooverChainNVT:
    """Build the configured fixed-cell canonical integrator.

    ``rng`` is forwarded only to Bussi.  Nosé–Hoover-chain NVT is
    deterministic after the initial momenta have been generated, and passing
    an RNG into ASE's constructor would be both unnecessary and unsupported.
    """

    if thermostat_kind == "bussi":
        return build_bussi_nvt(
            atoms,
            timestep_fs=timestep_fs,
            temperature_K=temperature_K,
            thermostat_tau_fs=thermostat_tau_fs,
            rng=rng,
            **kwargs,
        )
    if thermostat_kind == "nose_hoover_chain":
        return build_nose_hoover_chain_nvt(
            atoms,
            timestep_fs=timestep_fs,
            temperature_K=temperature_K,
            thermostat_tau_fs=thermostat_tau_fs,
            thermostat_chain_length=thermostat_chain_length,
            thermostat_substeps=thermostat_substeps,
            **kwargs,
        )
    raise ValueError(
        "thermostat_kind must be 'bussi' or 'nose_hoover_chain'; "
        f"got {thermostat_kind!r}"
    )


def build_velocity_verlet(
    atoms: Atoms,
    *,
    timestep_fs: Real,
    **kwargs: Any,
) -> VelocityVerlet:
    """Build fixed-cell microcanonical velocity-Verlet dynamics."""

    _require_calculator(atoms)
    dt = _positive_finite(timestep_fs, "timestep_fs")
    return VelocityVerlet(atoms, timestep=dt * units.fs, **kwargs)


def build_isotropic_mtk_npt(
    atoms: Atoms,
    *,
    timestep_fs: Real,
    temperature_K: Real,
    pressure_GPa: Real,
    thermostat_tau_fs: Real,
    barostat_tau_fs: Real,
    thermostat_chain_length: int = 3,
    barostat_chain_length: int = 3,
    thermostat_substeps: int = 1,
    barostat_substeps: int = 1,
    **kwargs: Any,
) -> IsotropicMTKNPT:
    """Build one fixed-pressure isotropic MTK plateau.

    A new instance must be constructed for each new target pressure.  The
    engine never mutates ASE's private ``_pressure_au`` member.
    """

    _require_calculator(atoms)
    _require_thermal_momenta(atoms)
    _require_unconstrained(atoms, "isotropic MTK NPT")
    if not bool(np.all(atoms.get_pbc())):
        raise ValueError("isotropic MTK NPT requires periodicity in all three axes")

    dt = _positive_finite(timestep_fs, "timestep_fs")
    temperature = _positive_finite(temperature_K, "temperature_K")
    pressure = _finite(pressure_GPa, "pressure_GPa")
    tdamp = _positive_finite(thermostat_tau_fs, "thermostat_tau_fs")
    pdamp = _positive_finite(barostat_tau_fs, "barostat_tau_fs")
    tchain = _positive_integer(thermostat_chain_length, "thermostat_chain_length")
    pchain = _positive_integer(barostat_chain_length, "barostat_chain_length")
    tloop = _positive_integer(thermostat_substeps, "thermostat_substeps")
    ploop = _positive_integer(barostat_substeps, "barostat_substeps")

    return IsotropicMTKNPT(
        atoms,
        timestep=dt * units.fs,
        temperature_K=temperature,
        pressure_au=pressure * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
        tchain=tchain,
        pchain=pchain,
        tloop=tloop,
        ploop=ploop,
        **kwargs,
    )


class IsotropicVolumeRampBussi(Bussi):
    """Thermostatted, strain-controlled MD with an explicit isotropic ramp.

    This is a non-equilibrium strain-controlled driver, not an NVT integrator:
    the cell is prescribed externally while Bussi stochastic rescaling controls
    temperature.  For every step it performs the following operations inside
    :meth:`step`, never in an ASE observer callback:

    1. evaluate an absolute target cell from the stage-start reference cell;
    2. affinely scale positions while leaving momenta unchanged;
    3. explicitly request forces for the mutated cell and positions;
    4. take one Bussi/velocity-Verlet step using those fresh forces.

    Consequently, a force object supplied by ASE's generic ``run`` machinery is
    intentionally ignored: it may have been evaluated before the cell change.
    """

    def __init__(
        self,
        atoms: Atoms,
        *,
        timestep_fs: Real,
        temperature_K: Real,
        thermostat_tau_fs: Real,
        target_volume_ratio: Real,
        total_steps: int,
        schedule: RampSchedule = "log_linear",
        rng: Any,
        reference_cell: ArrayLike | None = None,
        completed_steps: int = 0,
        cell_rtol: Real = 1.0e-10,
        cell_atol_A: Real = 1.0e-12,
        **kwargs: Any,
    ) -> None:
        _require_calculator(atoms)
        _require_thermal_momenta(atoms)
        _require_rng(rng)
        _require_unconstrained(atoms, "isotropic volume-ramp dynamics")
        if not bool(np.all(atoms.get_pbc())):
            raise ValueError(
                "isotropic volume-ramp dynamics require periodicity in all three axes"
            )

        self.total_ramp_steps = _positive_integer(total_steps, "total_steps")
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, Integral):
            raise TypeError("completed_steps must be an integer")
        self.completed_ramp_steps = int(completed_steps)
        if not 0 <= self.completed_ramp_steps <= self.total_ramp_steps:
            raise ValueError(
                "completed_steps must lie in the closed interval "
                f"[0, {self.total_ramp_steps}]"
            )

        self.target_volume_ratio = _positive_finite(
            target_volume_ratio, "target_volume_ratio"
        )
        # Validate the name before any mutation or ASE object construction.
        volume_ratio_at_step(
            self.target_volume_ratio,
            self.completed_ramp_steps,
            self.total_ramp_steps,
            schedule,
        )
        self.ramp_schedule = schedule

        if reference_cell is None:
            if self.completed_ramp_steps != 0:
                raise ValueError(
                    "reference_cell is required when resuming a volume ramp"
                )
            cell = np.asarray(atoms.get_cell(), dtype=float)
        else:
            cell = np.asarray(reference_cell, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("reference_cell must be a finite array with shape (3, 3)")
        if float(np.linalg.det(cell)) <= np.finfo(float).eps:
            raise ValueError("reference_cell must be right-handed and non-singular")
        self.reference_cell: NDArray[np.float64] = np.asarray(
            cell.copy(), dtype=np.float64
        )

        expected_cell = isotropic_cell_at_step(
            self.reference_cell,
            self.target_volume_ratio,
            self.completed_ramp_steps,
            self.total_ramp_steps,
            self.ramp_schedule,
        )
        rtol = _positive_finite(cell_rtol, "cell_rtol")
        atol = _positive_finite(cell_atol_A, "cell_atol_A")
        if not np.allclose(
            np.asarray(atoms.get_cell(), dtype=float),
            expected_cell,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "current cell does not match the scheduled cell at completed_steps; "
                "a safe ramp resume requires the original stage reference cell"
            )

        dt = _positive_finite(timestep_fs, "timestep_fs")
        temperature = _positive_finite(temperature_K, "temperature_K")
        tau = _positive_finite(thermostat_tau_fs, "thermostat_tau_fs")
        super().__init__(
            atoms,
            timestep=dt * units.fs,
            temperature_K=temperature,
            taut=tau * units.fs,
            rng=rng,
            **kwargs,
        )

    @property
    def current_volume_ratio(self) -> float:
        """Current scheduled volume relative to the stage-start volume."""

        return volume_ratio_at_step(
            self.target_volume_ratio,
            self.completed_ramp_steps,
            self.total_ramp_steps,
            self.ramp_schedule,
        )

    def target_cell_for_step(self, step: int) -> NDArray[np.float64]:
        """Return the absolute cell target without mutating the atoms."""

        return isotropic_cell_at_step(
            self.reference_cell,
            self.target_volume_ratio,
            step,
            self.total_ramp_steps,
            self.ramp_schedule,
        )

    def step(self, forces: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Advance one ramp step, discarding any potentially stale input force."""

        del forces
        if self.completed_ramp_steps >= self.total_ramp_steps:
            raise RuntimeError("the isotropic volume ramp is already complete")

        next_step = self.completed_ramp_steps + 1
        target_cell = self.target_cell_for_step(next_step)

        # ASE's set_cell(scale_atoms=True) does not intentionally transform
        # momenta, but restoring this copy makes that scientific invariant
        # explicit and protects it from future behavior changes.
        momenta = self.atoms.get_momenta().copy()
        self.atoms.set_cell(target_cell, scale_atoms=True, apply_constraint=False)
        self.atoms.set_momenta(momenta, apply_constraint=False)

        # This explicit call occurs after both cell and positions have changed.
        # ASE calculators therefore see the new system state.  Passing it into
        # Bussi/VelocityVerlet prevents reuse of a force evaluated for the old
        # cell at the beginning of generic Dynamics.run().
        fresh_forces = self.atoms.get_forces(md=True)
        final_forces = super().step(forces=fresh_forces)
        self.completed_ramp_steps = next_step
        return final_forces


def build_isotropic_volume_ramp(
    atoms: Atoms,
    **kwargs: Any,
) -> IsotropicVolumeRampBussi:
    """Construct :class:`IsotropicVolumeRampBussi` (factory-style public API)."""

    return IsotropicVolumeRampBussi(atoms, **kwargs)


__all__ = [
    "IsotropicVolumeRampBussi",
    "build_bussi_nvt",
    "build_fixed_cell_nvt",
    "build_isotropic_mtk_npt",
    "build_isotropic_volume_ramp",
    "build_nose_hoover_chain_nvt",
    "build_velocity_verlet",
    "initialize_momenta_once",
]
