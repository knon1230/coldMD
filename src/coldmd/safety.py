"""Fail-fast safety checks for molecular-dynamics states.

Safety checks report and abort; they never clip forces, repair cells, rescale
velocities, or otherwise change the trajectory.  The callback hook is intended
for writing an emergency checkpoint immediately before a controlled abort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .observables import ThermoRecord


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Configurable limits in the units used by :class:`ThermoRecord`.

    ``None`` disables only that particular engineering threshold.  Fundamental
    checks for non-finite arrays and a finite, right-handed, non-singular cell
    are always active.
    """

    min_temperature_K: float | None = None
    max_temperature_K: float | None = None
    max_abs_pressure_GPa: float | None = None
    max_force_eV_A: float | None = None
    min_distance_A: float | None = None
    min_cell_volume_A3: float | None = None
    min_volume_ratio: float | None = None
    max_volume_ratio: float | None = None
    max_abs_log_volume_strain_per_step: float | None = None
    max_cell_condition_number: float | None = None
    max_deviatoric_stress_GPa: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite or None, got {value!r}.")

        nonnegative = (
            "min_temperature_K",
            "max_temperature_K",
            "max_abs_pressure_GPa",
            "max_force_eV_A",
            "min_distance_A",
            "min_cell_volume_A3",
            "min_volume_ratio",
            "max_volume_ratio",
            "max_abs_log_volume_strain_per_step",
            "max_cell_condition_number",
            "max_deviatoric_stress_GPa",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if (
            self.min_temperature_K is not None
            and self.max_temperature_K is not None
            and self.min_temperature_K >= self.max_temperature_K
        ):
            raise ValueError("min_temperature_K must be less than max_temperature_K.")
        if (
            self.min_volume_ratio is not None
            and self.max_volume_ratio is not None
            and self.min_volume_ratio >= self.max_volume_ratio
        ):
            raise ValueError("min_volume_ratio must be less than max_volume_ratio.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | Any) -> "SafetyLimits":
        """Build limits from a mapping or a Pydantic/dataclass-like object."""

        names = {field.name for field in fields(cls)}
        if isinstance(config, Mapping):
            values = {name: config[name] for name in names if name in config}
        else:
            values = {
                name: getattr(config, name)
                for name in names
                if hasattr(config, name)
            }
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SafetyIssue:
    """One failed safety predicate."""

    code: str
    message: str
    observed: float | str | None = None
    limit: float | str | None = None
    immediate: bool = False

    def as_dict(self) -> dict[str, float | str | bool | None]:
        return asdict(self)


class SafetyViolation(RuntimeError):
    """Raised when one or more safety issues require termination."""

    def __init__(
        self,
        issues: Sequence[SafetyIssue],
        *,
        step: int | None = None,
        stage: str | None = None,
    ) -> None:
        if not issues:
            raise ValueError("SafetyViolation requires at least one issue.")
        self.issues = tuple(issues)
        self.step = step
        self.stage = stage
        location = []
        if step is not None:
            location.append(f"step={step}")
        if stage is not None:
            location.append(f"stage={stage}")
        prefix = (
            f"Safety violation ({', '.join(location)}): "
            if location
            else "Safety violation: "
        )
        super().__init__(prefix + "; ".join(issue.message for issue in self.issues))

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "stage": self.stage,
            "issues": [issue.as_dict() for issue in self.issues],
        }


ViolationCallback = Callable[[SafetyViolation, Any, ThermoRecord], Any]


def _issue(
    code: str,
    message: str,
    *,
    observed: float | str | None = None,
    limit: float | str | None = None,
    immediate: bool = False,
) -> SafetyIssue:
    return SafetyIssue(code, message, observed, limit, immediate)


def _extract_cell(atoms: Any) -> NDArray[np.float64]:
    try:
        cell = np.asarray(atoms.cell, dtype=np.float64)
    except Exception as exc:
        raise TypeError(f"Could not read atomic cell: {exc}") from exc
    if cell.shape != (3, 3):
        raise ValueError(f"Cell must have shape (3, 3), got {cell.shape}.")
    return np.array(cell, dtype=np.float64, copy=True)


def _raw_state_issues(atoms: Any, cell: NDArray[np.float64]) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    arrays: dict[str, Any] = {"cell": cell}
    for name, getter_name in (
        ("positions", "get_positions"),
        ("momenta", "get_momenta"),
    ):
        try:
            arrays[name] = getattr(atoms, getter_name)()
        except Exception as exc:
            issues.append(
                _issue(
                    f"unreadable_{name}",
                    f"Could not read {name}: {exc}",
                    observed=type(exc).__name__,
                    immediate=True,
                )
            )
    for name, value in arrays.items():
        try:
            array = np.asarray(value, dtype=np.float64)
        except Exception as exc:
            issues.append(
                _issue(
                    f"invalid_{name}",
                    f"{name} could not be converted to a numeric array: {exc}",
                    immediate=True,
                )
            )
            continue
        if not np.all(np.isfinite(array)):
            issues.append(
                _issue(
                    f"nonfinite_{name}",
                    f"{name} contains NaN or infinity.",
                    immediate=True,
                )
            )
    return issues


def _record_nonfinite_issues(record: ThermoRecord) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    for name, value in record.as_dict().items():
        if value is None or isinstance(value, str):
            continue
        if not np.isfinite(value):
            issues.append(
                _issue(
                    "nonfinite_observable",
                    f"Observable {name} is NaN or infinity.",
                    observed=name,
                    immediate=True,
                )
            )
    return issues


class SafetyMonitor:
    """Stateful, callback-friendly trajectory guard.

    ``check(atoms, record)`` is suitable as an engine hook.  Non-finite state
    and invalid cell geometry always raise immediately.  Other configured
    thresholds raise after ``maximum_consecutive_violations`` sampled states;
    the default is one.  Any tolerated issue is returned explicitly and is
    therefore never silently ignored.
    """

    def __init__(
        self,
        limits: SafetyLimits | Mapping[str, Any] | Any,
        *,
        reference_volume_A3: float | None = None,
        maximum_consecutive_violations: int = 1,
        on_violation: ViolationCallback | None = None,
    ) -> None:
        self.limits = (
            limits if isinstance(limits, SafetyLimits) else SafetyLimits.from_config(limits)
        )
        if maximum_consecutive_violations < 1:
            raise ValueError("maximum_consecutive_violations must be at least 1.")
        if reference_volume_A3 is not None and (
            not np.isfinite(reference_volume_A3) or reference_volume_A3 <= 0.0
        ):
            raise ValueError("reference_volume_A3 must be finite and positive.")
        self.reference_volume_A3 = reference_volume_A3
        self.maximum_consecutive_violations = int(maximum_consecutive_violations)
        self.on_violation = on_violation
        self._previous_cell: NDArray[np.float64] | None = None
        self._previous_step: int | None = None
        self._consecutive_violations = 0

    @property
    def consecutive_violations(self) -> int:
        return self._consecutive_violations

    def reset(
        self,
        *,
        reference_volume_A3: float | None = None,
        previous_cell: NDArray[np.float64] | None = None,
        previous_step: int | None = None,
    ) -> None:
        """Reset monitor history, for example at the start of a fresh run."""

        if reference_volume_A3 is not None and (
            not np.isfinite(reference_volume_A3) or reference_volume_A3 <= 0.0
        ):
            raise ValueError("reference_volume_A3 must be finite and positive.")
        self.reference_volume_A3 = reference_volume_A3
        self._previous_cell = (
            None if previous_cell is None else np.asarray(previous_cell, dtype=np.float64).copy()
        )
        self._previous_step = previous_step
        self._consecutive_violations = 0

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable monitor state."""

        return {
            "reference_volume_A3": self.reference_volume_A3,
            "previous_cell": (
                None if self._previous_cell is None else self._previous_cell.tolist()
            ),
            "previous_step": self._previous_step,
            "consecutive_violations": self._consecutive_violations,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore monitor history from a validated checkpoint mapping."""

        reference = state.get("reference_volume_A3")
        previous_cell_raw = state.get("previous_cell")
        previous_cell = (
            None
            if previous_cell_raw is None
            else np.asarray(previous_cell_raw, dtype=np.float64)
        )
        if previous_cell is not None and previous_cell.shape != (3, 3):
            raise ValueError("Checkpoint safety previous_cell must have shape (3, 3).")
        self.reset(
            reference_volume_A3=None if reference is None else float(reference),
            previous_cell=previous_cell,
            previous_step=(
                None if state.get("previous_step") is None else int(state["previous_step"])
            ),
        )
        count = int(state.get("consecutive_violations", 0))
        if count < 0:
            raise ValueError("Checkpoint consecutive_violations cannot be negative.")
        self._consecutive_violations = count

    def evaluate(self, atoms: Any, record: ThermoRecord) -> tuple[SafetyIssue, ...]:
        """Evaluate the current state without mutating atoms or monitor history."""

        issues = _record_nonfinite_issues(record)
        try:
            cell = _extract_cell(atoms)
        except (TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_cell",
                    str(exc),
                    observed=type(exc).__name__,
                    immediate=True,
                )
            )
            return tuple(issues)

        issues.extend(_raw_state_issues(atoms, cell))
        if np.all(np.isfinite(cell)):
            determinant = float(np.linalg.det(cell))
            if determinant <= 0.0:
                issues.append(
                    _issue(
                        "invalid_cell_determinant",
                        "Cell determinant must be positive.",
                        observed=determinant,
                        limit="> 0",
                        immediate=True,
                    )
                )
            elif (
                self.limits.min_cell_volume_A3 is not None
                and determinant < self.limits.min_cell_volume_A3
            ):
                issues.append(
                    _issue(
                        "cell_volume_below_minimum",
                        "Cell volume is below the configured minimum.",
                        observed=determinant,
                        limit=self.limits.min_cell_volume_A3,
                    )
                )
            try:
                condition = float(np.linalg.cond(cell))
            except np.linalg.LinAlgError:
                condition = float("inf")
            if not np.isfinite(condition):
                issues.append(
                    _issue(
                        "singular_cell",
                        "Cell is singular or numerically ill-conditioned.",
                        observed=condition,
                        immediate=True,
                    )
                )
            elif (
                self.limits.max_cell_condition_number is not None
                and condition > self.limits.max_cell_condition_number
            ):
                issues.append(
                    _issue(
                        "cell_condition_above_maximum",
                        "Cell condition number exceeds the configured maximum.",
                        observed=condition,
                        limit=self.limits.max_cell_condition_number,
                    )
                )

        limits = self.limits
        if (
            limits.min_temperature_K is not None
            and record.temperature_K < limits.min_temperature_K
        ):
            issues.append(
                _issue(
                    "temperature_below_minimum",
                    "Temperature is below the configured minimum.",
                    observed=record.temperature_K,
                    limit=limits.min_temperature_K,
                )
            )
        if (
            limits.max_temperature_K is not None
            and record.temperature_K > limits.max_temperature_K
        ):
            issues.append(
                _issue(
                    "temperature_above_maximum",
                    "Temperature exceeds the configured maximum.",
                    observed=record.temperature_K,
                    limit=limits.max_temperature_K,
                )
            )
        if limits.max_abs_pressure_GPa is not None:
            for name, pressure in (
                ("configurational", record.configurational_pressure_GPa),
                ("total", record.total_pressure_GPa),
            ):
                if abs(pressure) > limits.max_abs_pressure_GPa:
                    issues.append(
                        _issue(
                            f"{name}_pressure_above_maximum",
                            f"Absolute {name} pressure exceeds the configured maximum.",
                            observed=pressure,
                            limit=limits.max_abs_pressure_GPa,
                        )
                    )
        if limits.max_force_eV_A is not None and record.max_force_eV_A > limits.max_force_eV_A:
            issues.append(
                _issue(
                    "force_above_maximum",
                    "Maximum atomic force exceeds the configured maximum.",
                    observed=record.max_force_eV_A,
                    limit=limits.max_force_eV_A,
                )
            )
        if limits.min_distance_A is not None and record.min_distance_A < limits.min_distance_A:
            issues.append(
                _issue(
                    "distance_below_minimum",
                    "Minimum pair distance is below the configured minimum.",
                    observed=record.min_distance_A,
                    limit=limits.min_distance_A,
                )
            )
        if (
            limits.max_deviatoric_stress_GPa is not None
            and record.deviatoric_stress_GPa > limits.max_deviatoric_stress_GPa
        ):
            issues.append(
                _issue(
                    "deviatoric_stress_above_maximum",
                    "Deviatoric stress exceeds the configured maximum.",
                    observed=record.deviatoric_stress_GPa,
                    limit=limits.max_deviatoric_stress_GPa,
                )
            )

        volume_ratio = (
            record.volume_A3 / self.reference_volume_A3
            if self.reference_volume_A3 is not None
            else record.target_volume_ratio
        )
        if volume_ratio is not None:
            if limits.min_volume_ratio is not None and volume_ratio < limits.min_volume_ratio:
                issues.append(
                    _issue(
                        "volume_ratio_below_minimum",
                        "Volume ratio is below the configured minimum.",
                        observed=volume_ratio,
                        limit=limits.min_volume_ratio,
                    )
                )
            if limits.max_volume_ratio is not None and volume_ratio > limits.max_volume_ratio:
                issues.append(
                    _issue(
                        "volume_ratio_above_maximum",
                        "Volume ratio exceeds the configured maximum.",
                        observed=volume_ratio,
                        limit=limits.max_volume_ratio,
                    )
                )

        if (
            limits.max_abs_log_volume_strain_per_step is not None
            and self._previous_cell is not None
            and self._previous_step is not None
            and record.step > self._previous_step
            and np.all(np.isfinite(cell))
        ):
            previous_volume = float(np.linalg.det(self._previous_cell))
            current_volume = float(np.linalg.det(cell))
            if previous_volume > 0.0 and current_volume > 0.0:
                delta_steps = record.step - self._previous_step
                log_strain_per_step = abs(
                    np.log(current_volume / previous_volume) / delta_steps
                )
                if log_strain_per_step > limits.max_abs_log_volume_strain_per_step:
                    issues.append(
                        _issue(
                            "volume_strain_above_maximum",
                            "Absolute logarithmic volume strain per MD step "
                            "exceeds the configured maximum.",
                            observed=float(log_strain_per_step),
                            limit=limits.max_abs_log_volume_strain_per_step,
                        )
                    )

        return tuple(issues)

    def check(self, atoms: Any, record: ThermoRecord) -> tuple[SafetyIssue, ...]:
        """Evaluate, update history, and raise when an abort criterion is met."""

        issues = self.evaluate(atoms, record)
        immediate = any(issue.immediate for issue in issues)
        if issues:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        try:
            cell = _extract_cell(atoms)
        except (TypeError, ValueError):
            cell = None
        if cell is not None and np.all(np.isfinite(cell)):
            self._previous_cell = cell
            self._previous_step = record.step

        should_raise = immediate or (
            bool(issues)
            and self._consecutive_violations >= self.maximum_consecutive_violations
        )
        if should_raise:
            violation = SafetyViolation(issues, step=record.step, stage=record.stage)
            if self.on_violation is not None:
                try:
                    self.on_violation(violation, atoms, record)
                except Exception as callback_error:
                    violation.add_note(
                        "The emergency on_violation callback also failed: "
                        f"{callback_error!r}"
                    )
            raise violation
        return issues

    def __call__(self, atoms: Any, record: ThermoRecord) -> tuple[SafetyIssue, ...]:
        return self.check(atoms, record)


__all__ = [
    "SafetyIssue",
    "SafetyLimits",
    "SafetyMonitor",
    "SafetyViolation",
]
