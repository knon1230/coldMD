"""Pure schedules used by the cold-compression protocol engine.

The functions in this module deliberately do not depend on ASE.  In
particular, an isotropic ramp is evaluated from the *stage reference cell* at
every step.  It is therefore immune to the floating-point drift that results
from repeatedly multiplying the cell from the previous step.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


RampSchedule = Literal["log_linear", "log_smoothstep"]


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero; got {value!r}")
    return result


def _step_pair(step: int, total_steps: int) -> tuple[int, int]:
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise TypeError("step must be an integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
        raise TypeError("total_steps must be an integer")

    step_i = int(step)
    total_i = int(total_steps)
    if total_i <= 0:
        raise ValueError("total_steps must be greater than zero")
    if not 0 <= step_i <= total_i:
        raise ValueError(
            f"step must lie in the closed interval [0, {total_i}]; got {step_i}"
        )
    return step_i, total_i


def duration_ps_to_steps(duration_ps: Real, timestep_fs: Real) -> int:
    """Return the nearest integral number of MD steps for a duration.

    The realized duration can differ from the request by at most half a time
    step.  Configuration validation should report the realized duration when
    that distinction matters.
    """

    duration = _positive_finite(duration_ps, "duration_ps")
    timestep = _positive_finite(timestep_fs, "timestep_fs")
    raw_steps = 1000.0 * duration / timestep
    steps = int(np.rint(raw_steps))
    if steps < 1:
        raise ValueError(
            "duration_ps is shorter than half of one timestep and rounds to zero steps"
        )
    return steps


def progress_at_step(step: int, total_steps: int) -> float:
    """Return an inclusive progress coordinate: step 0 -> 0, final step -> 1."""

    step_i, total_i = _step_pair(step, total_steps)
    return step_i / total_i


def schedule_weight(
    step: int,
    total_steps: int,
    schedule: RampSchedule = "log_linear",
) -> float:
    """Return the interpolation weight for a supported ramp schedule.

    ``log_smoothstep`` applies :math:`3x^2-2x^3` to log-volume interpolation,
    giving zero ramp velocity at both endpoints.  ``log_linear`` gives a
    constant logarithmic volumetric strain rate.
    """

    progress = progress_at_step(step, total_steps)
    if schedule == "log_linear":
        return progress
    if schedule == "log_smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    raise ValueError(
        f"unsupported ramp schedule {schedule!r}; expected 'log_linear' or "
        "'log_smoothstep'"
    )


def volume_ratio_at_step(
    target_volume_ratio: Real,
    step: int,
    total_steps: int,
    schedule: RampSchedule = "log_linear",
) -> float:
    """Return ``V(step) / V(stage_start)`` on a log-volume path."""

    target = _positive_finite(target_volume_ratio, "target_volume_ratio")
    weight = schedule_weight(step, total_steps, schedule)
    # Written this way instead of target**weight to make the physical
    # log-volume interpolation explicit.
    return float(np.exp(weight * np.log(target)))


def isotropic_length_scale_at_step(
    target_volume_ratio: Real,
    step: int,
    total_steps: int,
    schedule: RampSchedule = "log_linear",
) -> float:
    """Return the isotropic length scale relative to the stage-start cell."""

    ratio = volume_ratio_at_step(target_volume_ratio, step, total_steps, schedule)
    return float(np.cbrt(ratio))


def isotropic_cell_at_step(
    reference_cell: ArrayLike,
    target_volume_ratio: Real,
    step: int,
    total_steps: int,
    schedule: RampSchedule = "log_linear",
) -> NDArray[np.float64]:
    """Return the absolute target cell for one step of an isotropic ramp.

    ``reference_cell`` is the cell at the beginning of the stage, not the cell
    from the previous step.  Cell angles and axial ratios are preserved.
    """

    cell = np.asarray(reference_cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"reference_cell must have shape (3, 3); got {cell.shape}")
    if not np.all(np.isfinite(cell)):
        raise ValueError("reference_cell contains non-finite values")
    if float(np.linalg.det(cell)) <= np.finfo(float).eps:
        raise ValueError("reference_cell must be right-handed and non-singular")

    scale = isotropic_length_scale_at_step(
        target_volume_ratio, step, total_steps, schedule
    )
    target_cell = np.asarray(cell * scale, dtype=np.float64)
    if (
        not np.all(np.isfinite(target_cell))
        or float(np.linalg.det(target_cell)) <= np.finfo(float).eps
    ):
        raise ValueError("scheduled target cell is non-finite or numerically singular")
    return target_cell


def linear_pressure_plateaus(
    start_pressure_GPa: Real,
    end_pressure_GPa: Real,
    count: int,
) -> tuple[float, ...]:
    """Return inclusive, linearly spaced fixed-pressure plateau targets."""

    start = float(start_pressure_GPa)
    end = float(end_pressure_GPa)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("pressure endpoints must be finite")
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError("count must be an integer")
    if int(count) < 1:
        raise ValueError("count must be greater than zero")
    if int(count) == 1:
        if start != end:
            raise ValueError(
                "one plateau cannot include two different pressure endpoints"
            )
        return (start,)
    return tuple(float(value) for value in np.linspace(start, end, int(count)))


def pressure_range_targets(
    start_pressure_GPa: Real,
    end_pressure_GPa: Real,
    pressure_step_GPa: Real,
    *,
    max_points: int = 10_000,
) -> tuple[float, ...]:
    """Return an inclusive fixed-increment pressure range.

    Decimal arithmetic is used after validating the public real-valued inputs so
    that human-authored values such as ``0.0 -> 1.0 by 0.1`` have deterministic
    endpoint and divisibility semantics.  ``pressure_step_GPa`` is a positive
    magnitude; the endpoint order determines the sign of the generated steps.
    """

    values: dict[str, float] = {}
    for name, value in (
        ("start_pressure_GPa", start_pressure_GPa),
        ("end_pressure_GPa", end_pressure_GPa),
        ("pressure_step_GPa", pressure_step_GPa),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        values[name] = result

    start = values["start_pressure_GPa"]
    end = values["end_pressure_GPa"]
    step = values["pressure_step_GPa"]
    if start < 0.0 or end < 0.0:
        raise ValueError("pressure range endpoints must be non-negative")
    if step <= 0.0:
        raise ValueError("pressure_step_GPa must be greater than zero")
    if start == end:
        raise ValueError(
            "pressure range endpoints must differ; use pressure_plateau for one target"
        )
    if isinstance(max_points, bool) or not isinstance(max_points, Integral):
        raise TypeError("max_points must be an integer")
    if int(max_points) < 2:
        raise ValueError("max_points must be at least two")

    try:
        start_decimal = Decimal(str(start))
        end_decimal = Decimal(str(end))
        step_decimal = Decimal(str(step))
        span = abs(end_decimal - start_decimal)
        quotient, remainder = divmod(span, step_decimal)
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - finite inputs
        raise ValueError("pressure range could not be represented as decimals") from exc

    if remainder != 0:
        raise ValueError(
            "the pressure range span must be exactly divisible by pressure_step_GPa"
        )
    intervals = int(quotient)
    point_count = intervals + 1
    if point_count > int(max_points):
        raise ValueError(
            f"pressure range expands to {point_count} points, above the "
            f"maximum of {int(max_points)}"
        )

    direction = Decimal(1) if end_decimal > start_decimal else Decimal(-1)
    targets = tuple(
        float(start_decimal + direction * step_decimal * index)
        for index in range(point_count)
    )
    # The exact Decimal divisibility check guarantees this assignment changes
    # no schedule semantics while preserving the caller's declared endpoint.
    return (*targets[:-1], float(end_decimal))


__all__ = [
    "RampSchedule",
    "duration_ps_to_steps",
    "isotropic_cell_at_step",
    "isotropic_length_scale_at_step",
    "linear_pressure_plateaus",
    "pressure_range_targets",
    "progress_at_step",
    "schedule_weight",
    "volume_ratio_at_step",
]
