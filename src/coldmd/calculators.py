"""ASE calculator construction and scientific pre-flight probes.

Two construction routes are deliberately supported:

* an explicit MACE adapter for a foundation-model alias or local checkpoint;
* a ``package.module:function`` factory for any ASE-compatible calculator.

Foundation aliases may trigger a network download in MACE.  Downloads are
blocked by default so that ``coldmd validate`` is side-effect free; callers must
opt in explicitly or provide a local model checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, chemical_symbols
from ase.units import GPa


CalculatorKind = Literal["factory", "mace"]


class CalculatorConfigurationError(ValueError):
    """Raised when a calculator cannot be constructed from its configuration."""


class ModelDownloadBlockedError(CalculatorConfigurationError):
    """Raised when a foundation alias would be allowed to access the network."""


class CalculatorProbeError(RuntimeError):
    """Raised when a calculator fails an energy/force/stress pre-flight probe."""


@dataclass(frozen=True, slots=True)
class CalculatorSpec:
    """Serializable specification for an ASE calculator.

    For ``kind='factory'``, set ``factory`` to
    ``'package.module:create_calculator'``.  Device and precision are injected
    only if the callable accepts ``device`` and ``dtype``/``default_dtype`` (or
    arbitrary keyword arguments).

    For ``kind='mace'``, exactly one of ``foundation_model`` and ``model_path``
    must be set.  ``allow_download`` is required for a foundation alias.
    """

    kind: CalculatorKind
    factory: str | None = None
    foundation_model: str | None = None
    model_path: Path | str | None = None
    device: str = "cpu"
    dtype: str = "float64"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    allow_download: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalculatorSpec":
        """Build a spec while accepting a few unambiguous config aliases."""

        raw = dict(value)
        kind = raw.pop("kind", raw.pop("type", raw.pop("backend", None)))
        factory = raw.pop("factory", None)
        foundation = raw.pop(
            "foundation_model", raw.pop("foundation", raw.pop("mace_model", None))
        )
        model_path = raw.pop("model_path", raw.pop("checkpoint", None))
        if kind is None:
            if factory is not None:
                kind = "factory"
            elif foundation is not None or model_path is not None:
                kind = "mace"
        kwargs = raw.pop("kwargs", {})
        allow_download = raw.pop(
            "allow_download", raw.pop("allow_model_download", False)
        )
        device = raw.pop("device", "cpu")
        dtype = raw.pop("dtype", raw.pop("default_dtype", "float64"))
        if raw:
            names = ", ".join(sorted(raw))
            raise CalculatorConfigurationError(
                f"Unknown calculator configuration field(s): {names}"
            )
        if not isinstance(kwargs, Mapping):
            raise CalculatorConfigurationError("calculator kwargs must be a mapping")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            factory=factory,
            foundation_model=foundation,
            model_path=model_path,
            device=str(device),
            dtype=str(dtype),
            kwargs=dict(kwargs),
            allow_download=bool(allow_download),
        )


@dataclass(frozen=True, slots=True)
class CalculatorProbeReport:
    """Numerical results from a successful calculator pre-flight."""

    calculator_class: str
    implemented_properties: tuple[str, ...]
    energy_ev: float
    forces_shape: tuple[int, int]
    maximum_force_ev_a: float
    stress_voigt_ev_a3: tuple[float, float, float, float, float, float]
    pressure_gpa: float
    supported_atomic_numbers: tuple[int, ...] | None
    supported_elements: tuple[str, ...] | None
    finite_difference_delta_ln_volume: float | None
    finite_difference_mean_stress_ev_a3: float | None
    finite_difference_pressure_gpa: float | None
    hydrostatic_stress_absolute_error_ev_a3: float | None
    hydrostatic_stress_relative_error: float | None


def build_calculator(
    specification: CalculatorSpec | Mapping[str, Any],
    *,
    allow_download: bool | None = None,
) -> Calculator:
    """Construct a configured ASE calculator.

    ``allow_download`` overrides the serialized setting, which lets a validation
    command enforce offline behavior even if a run configuration permits model
    acquisition in a separate preparation step.
    """

    spec = (
        specification
        if isinstance(specification, CalculatorSpec)
        else CalculatorSpec.from_mapping(specification)
    )
    kind = str(spec.kind).strip().lower()
    permitted_download = spec.allow_download if allow_download is None else allow_download
    if kind == "factory":
        if spec.foundation_model is not None or spec.model_path is not None:
            raise CalculatorConfigurationError(
                "factory calculator cannot also define a MACE model"
            )
        return create_factory_calculator(
            spec.factory,
            kwargs=spec.kwargs,
            device=spec.device,
            dtype=spec.dtype,
        )
    if kind == "mace":
        if spec.factory is not None:
            raise CalculatorConfigurationError(
                "MACE calculator cannot also define a dynamic factory"
            )
        return create_mace_calculator(
            foundation_model=spec.foundation_model,
            model_path=spec.model_path,
            device=spec.device,
            dtype=spec.dtype,
            kwargs=spec.kwargs,
            allow_download=bool(permitted_download),
        )
    raise CalculatorConfigurationError(
        f"Unknown calculator kind {spec.kind!r}; expected 'factory' or 'mace'"
    )


def create_factory_calculator(
    factory: str | None,
    *,
    kwargs: Mapping[str, Any] | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> Calculator:
    """Import and invoke a ``package.module:function`` calculator factory."""

    if not factory or not isinstance(factory, str):
        raise CalculatorConfigurationError(
            "Dynamic calculator requires factory='package.module:function'"
        )
    if factory.count(":") != 1:
        raise CalculatorConfigurationError(
            f"Invalid calculator factory {factory!r}; use package.module:function"
        )
    module_name, attribute_name = (part.strip() for part in factory.split(":", 1))
    if not module_name or not attribute_name:
        raise CalculatorConfigurationError(
            f"Invalid calculator factory {factory!r}; use package.module:function"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise CalculatorConfigurationError(
            f"Could not import calculator module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    callable_factory: Any = module
    try:
        for component in attribute_name.split("."):
            callable_factory = getattr(callable_factory, component)
    except AttributeError as exc:
        raise CalculatorConfigurationError(
            f"Module {module_name!r} has no calculator factory {attribute_name!r}"
        ) from exc
    if not callable(callable_factory):
        raise CalculatorConfigurationError(
            f"Calculator factory {factory!r} is not callable"
        )

    call_kwargs = dict(kwargs or {})
    _inject_runtime_argument(callable_factory, call_kwargs, "device", device)
    if dtype is not None and "dtype" not in call_kwargs and "default_dtype" not in call_kwargs:
        parameter = _accepted_precision_parameter(callable_factory)
        if parameter is not None:
            call_kwargs[parameter] = _normalise_dtype(dtype)
    try:
        calculator = callable_factory(**call_kwargs)
    except Exception as exc:
        raise CalculatorConfigurationError(
            f"Calculator factory {factory!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return _require_ase_calculator(calculator, origin=factory)


def create_mace_calculator(
    *,
    foundation_model: str | None = None,
    model_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "float64",
    kwargs: Mapping[str, Any] | None = None,
    allow_download: bool = False,
) -> Calculator:
    """Construct a MACE foundation or local-checkpoint ASE calculator."""

    if bool(foundation_model) == bool(model_path):
        raise CalculatorConfigurationError(
            "MACE requires exactly one of foundation_model or model_path"
        )
    normalised_device = _normalise_device(device)
    normalised_dtype = _normalise_dtype(dtype)
    options = dict(kwargs or {})
    protected = {"model", "model_paths", "device", "default_dtype"}.intersection(options)
    if protected:
        raise CalculatorConfigurationError(
            "Set MACE model/device/dtype in their explicit configuration fields, "
            f"not kwargs: {sorted(protected)}"
        )
    try:
        from mace.calculators import MACECalculator, mace_mp
    except Exception as exc:
        raise CalculatorConfigurationError(
            "MACE is not importable. Install the pinned mace-torch environment "
            f"before selecting the MACE backend ({type(exc).__name__}: {exc})."
        ) from exc

    if foundation_model:
        if not allow_download:
            raise ModelDownloadBlockedError(
                f"MACE foundation alias {foundation_model!r} may access the "
                "network. Validation does not download models by default. "
                "Prefetch it explicitly, set allow_download=true, or provide a "
                "local model_path."
            )
        try:
            calculator = mace_mp(
                model=foundation_model,
                device=normalised_device,
                default_dtype=normalised_dtype,
                **options,
            )
        except Exception as exc:
            raise CalculatorConfigurationError(
                f"Could not construct MACE foundation model {foundation_model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _require_ase_calculator(calculator, origin="mace_mp")

    checkpoint = Path(model_path).expanduser().resolve()  # type: ignore[arg-type]
    if not checkpoint.is_file():
        raise CalculatorConfigurationError(
            f"MACE model checkpoint does not exist: {checkpoint}"
        )
    try:
        calculator = MACECalculator(
            model_paths=str(checkpoint),
            device=normalised_device,
            default_dtype=normalised_dtype,
            **options,
        )
    except Exception as exc:
        raise CalculatorConfigurationError(
            f"Could not load MACE checkpoint {checkpoint}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _require_ase_calculator(calculator, origin=str(checkpoint))


def probe_calculator(
    calculator: Calculator,
    atoms: Atoms,
    *,
    required_properties: Sequence[str] = ("energy", "forces", "stress"),
    check_supported_elements: bool = True,
    check_stress_finite_difference: bool = True,
    finite_difference_delta_ln_volume: float = 1.0e-3,
    stress_relative_tolerance: float = 0.10,
    stress_absolute_tolerance_ev_a3: float = 1.0e-2,
) -> CalculatorProbeReport:
    """Evaluate E/F/stress and validate ASE's hydrostatic stress convention.

    The finite-difference test applies an affine isotropic strain at fixed
    fractional coordinates.  With ASE's tensile-positive stress convention,

    ``mean(sigma_xx, sigma_yy, sigma_zz) = (1/V) dE/d(ln V)``

    and pressure is ``-trace(stress)/3``.  Comparing these quantities catches
    missing volume factors, unit errors, and reversed pressure/stress signs
    before a barostat is allowed to control the cell.
    """

    calculator = _require_ase_calculator(calculator, origin="probe input")
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise CalculatorProbeError("Calculator probe requires a non-empty ase.Atoms")
    required = tuple(dict.fromkeys(str(item).lower() for item in required_properties))
    invalid_required = set(required).difference({"energy", "forces", "stress"})
    if invalid_required:
        raise CalculatorProbeError(
            f"Unsupported probe property name(s): {sorted(invalid_required)}"
        )
    implemented = tuple(
        sorted(str(item).lower() for item in getattr(calculator, "implemented_properties", ()))
    )
    missing = sorted(set(required).difference(implemented))
    if missing:
        raise CalculatorProbeError(
            f"{_calculator_name(calculator)} does not declare required ASE "
            f"properties {missing}; implemented_properties={list(implemented)}"
        )

    supported = discover_supported_atomic_numbers(calculator)
    if check_supported_elements and supported is not None:
        requested = {int(number) for number in atoms.numbers}
        unsupported = sorted(requested.difference(supported))
        if unsupported:
            symbols = [_symbol(number) for number in unsupported]
            raise CalculatorProbeError(
                f"Calculator model does not support input element(s) {symbols} "
                f"(Z={unsupported}); discovered supported Z={sorted(supported)}"
            )

    work = atoms.copy()
    work.calc = calculator
    energy = _get_finite_energy(work, label="unstrained structure")
    try:
        forces = np.asarray(work.get_forces(), dtype=float)
    except Exception as exc:
        raise CalculatorProbeError(
            f"Force evaluation failed for {_calculator_name(calculator)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if forces.shape != (len(work), 3):
        raise CalculatorProbeError(
            f"Calculator returned forces with shape {forces.shape}; expected "
            f"({len(work)}, 3)"
        )
    if not np.isfinite(forces).all():
        bad = np.argwhere(~np.isfinite(forces))[0]
        raise CalculatorProbeError(
            f"Calculator returned non-finite force at atom/axis "
            f"({int(bad[0])}, {int(bad[1])})"
        )
    try:
        stress = np.asarray(work.get_stress(voigt=True), dtype=float)
    except Exception as exc:
        raise CalculatorProbeError(
            f"Stress evaluation failed for {_calculator_name(calculator)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if stress.shape != (6,):
        raise CalculatorProbeError(
            f"Calculator returned Voigt stress with shape {stress.shape}; expected (6,)"
        )
    if not np.isfinite(stress).all():
        raise CalculatorProbeError("Calculator returned non-finite stress components")

    mean_stress = float(np.mean(stress[:3]))
    pressure_gpa = -mean_stress / GPa
    fd_stress: float | None = None
    fd_pressure_gpa: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    fd_delta: float | None = None

    if check_stress_finite_difference:
        _validate_stress_tolerances(
            finite_difference_delta_ln_volume,
            stress_relative_tolerance,
            stress_absolute_tolerance_ev_a3,
        )
        volume = float(work.get_volume())
        if not np.isfinite(volume) or volume <= 0.0:
            raise CalculatorProbeError(
                f"Cannot finite-difference stress with invalid volume {volume} A^3"
            )
        fd_delta = finite_difference_delta_ln_volume
        plus_energy = _energy_at_log_volume_offset(
            atoms, calculator, +fd_delta, label="expanded finite-difference structure"
        )
        minus_energy = _energy_at_log_volume_offset(
            atoms, calculator, -fd_delta, label="compressed finite-difference structure"
        )
        derivative_ev = (plus_energy - minus_energy) / (2.0 * fd_delta)
        fd_stress = derivative_ev / volume
        fd_pressure_gpa = -fd_stress / GPa
        absolute_error = abs(mean_stress - fd_stress)
        scale = max(abs(mean_stress), abs(fd_stress), np.finfo(float).tiny)
        relative_error = absolute_error / scale
        allowed_error = max(
            stress_absolute_tolerance_ev_a3,
            stress_relative_tolerance * max(abs(mean_stress), abs(fd_stress)),
        )
        significance = stress_absolute_tolerance_ev_a3
        opposite_sign = (
            abs(mean_stress) > significance
            and abs(fd_stress) > significance
            and np.signbit(mean_stress) != np.signbit(fd_stress)
        )
        if opposite_sign or absolute_error > allowed_error:
            reason = "opposite signs" if opposite_sign else "difference exceeds tolerance"
            raise CalculatorProbeError(
                "Hydrostatic stress is inconsistent with an isotropic energy "
                f"finite difference ({reason}). Calculator reports "
                f"mean(stress)={mean_stress:.8g} eV/A^3 "
                f"(P={pressure_gpa:.6g} GPa); energy derivative gives "
                f"{fd_stress:.8g} eV/A^3 (P={fd_pressure_gpa:.6g} GPa). "
                f"Absolute error={absolute_error:.4g} eV/A^3, allowed="
                f"{allowed_error:.4g}. ASE requires pressure=-trace(stress)/3."
            )

    supported_tuple = tuple(sorted(supported)) if supported is not None else None
    return CalculatorProbeReport(
        calculator_class=_calculator_name(calculator),
        implemented_properties=implemented,
        energy_ev=energy,
        forces_shape=(int(forces.shape[0]), int(forces.shape[1])),
        maximum_force_ev_a=float(np.max(np.linalg.norm(forces, axis=1))),
        stress_voigt_ev_a3=tuple(float(value) for value in stress),  # type: ignore[arg-type]
        pressure_gpa=float(pressure_gpa),
        supported_atomic_numbers=supported_tuple,
        supported_elements=(
            tuple(_symbol(number) for number in supported_tuple)
            if supported_tuple is not None
            else None
        ),
        finite_difference_delta_ln_volume=fd_delta,
        finite_difference_mean_stress_ev_a3=fd_stress,
        finite_difference_pressure_gpa=fd_pressure_gpa,
        hydrostatic_stress_absolute_error_ev_a3=absolute_error,
        hydrostatic_stress_relative_error=relative_error,
    )


def discover_supported_atomic_numbers(calculator: Calculator) -> set[int] | None:
    """Best-effort discovery of a model's element table.

    ASE does not standardize element support, so this inspects common MACE and
    generic calculator attributes.  ``None`` means "not discoverable", not
    "supports every element"; the numerical probe remains authoritative.
    """

    candidates: list[Any] = []
    for attribute in ("atomic_numbers", "supported_atomic_numbers", "elements"):
        if hasattr(calculator, attribute):
            candidates.append(getattr(calculator, attribute))
    z_table = getattr(calculator, "z_table", None)
    if z_table is not None:
        for attribute in ("zs", "atomic_numbers"):
            if hasattr(z_table, attribute):
                candidates.append(getattr(z_table, attribute))
    models: list[Any] = []
    model = getattr(calculator, "model", None)
    if model is not None:
        models.append(model)
    raw_models = getattr(calculator, "models", None)
    if raw_models is not None:
        try:
            models.extend(list(raw_models))
        except TypeError:
            models.append(raw_models)
    model_sets: list[set[int]] = []
    for item in models:
        for attribute in ("atomic_numbers", "supported_atomic_numbers", "elements"):
            if hasattr(item, attribute):
                parsed = _atomic_number_set(getattr(item, attribute))
                if parsed:
                    model_sets.append(parsed)
        item_z_table = getattr(item, "z_table", None)
        if item_z_table is not None and hasattr(item_z_table, "zs"):
            parsed = _atomic_number_set(getattr(item_z_table, "zs"))
            if parsed:
                model_sets.append(parsed)

    direct_sets = [parsed for value in candidates if (parsed := _atomic_number_set(value))]
    all_sets = direct_sets + model_sets
    if not all_sets:
        parameters = getattr(calculator, "parameters", None)
        if isinstance(parameters, Mapping):
            for key in ("atomic_numbers", "supported_atomic_numbers", "elements", "zs"):
                if key in parameters:
                    parsed = _atomic_number_set(parameters[key])
                    if parsed:
                        all_sets.append(parsed)
    if not all_sets:
        return None
    # Committee/ensemble models must all support an input element.
    result = set(all_sets[0])
    for item in all_sets[1:]:
        result.intersection_update(item)
    return result or None


def verify_model_sha256(
    model_path: str | Path,
    expected_sha256: str | None = None,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash a local model and optionally assert its configured identity."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise CalculatorConfigurationError(f"Model file does not exist: {path}")
    if chunk_size <= 0:
        raise CalculatorConfigurationError("SHA-256 chunk_size must be positive")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
    except OSError as exc:
        raise CalculatorConfigurationError(
            f"Could not read model file for SHA-256 verification: {path}: {exc}"
        ) from exc
    observed = digest.hexdigest()
    if expected_sha256 is not None:
        wanted = str(expected_sha256).strip().lower()
        if len(wanted) != 64 or any(character not in "0123456789abcdef" for character in wanted):
            raise CalculatorConfigurationError(
                "Expected model SHA-256 must contain exactly 64 hexadecimal digits"
            )
        if observed != wanted:
            raise CalculatorConfigurationError(
                f"Model SHA-256 mismatch for {path}: observed {observed}, "
                f"expected {wanted}"
            )
    return observed


def _atomic_number_set(value: Any) -> set[int] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().tolist()
        except Exception:
            return None
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, Mapping):
        value = list(value.keys())
    if isinstance(value, (str, bytes, int, np.integer)):
        value = [value]
    try:
        flat = np.asarray(value, dtype=object).ravel().tolist()
    except Exception:
        try:
            flat = list(value)
        except TypeError:
            return None
    result: set[int] = set()
    for item in flat:
        if isinstance(item, bytes):
            item = item.decode(errors="replace")
        if isinstance(item, str):
            number = atomic_numbers.get(item.strip())
            if number is None:
                return None
        else:
            try:
                number = int(item)
            except (TypeError, ValueError, OverflowError):
                return None
        if not 0 < number < len(chemical_symbols):
            return None
        result.add(number)
    return result or None


def _energy_at_log_volume_offset(
    atoms: Atoms,
    calculator: Calculator,
    delta_ln_volume: float,
    *,
    label: str,
) -> float:
    strained = atoms.copy()
    factor = float(np.exp(delta_ln_volume / 3.0))
    strained.set_cell(np.asarray(atoms.cell.array) * factor, scale_atoms=True)
    strained.calc = calculator
    return _get_finite_energy(strained, label=label)


def _get_finite_energy(atoms: Atoms, *, label: str) -> float:
    try:
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        raise CalculatorProbeError(
            f"Energy evaluation failed for {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not np.isfinite(energy):
        raise CalculatorProbeError(
            f"Calculator returned non-finite energy for {label}: {energy}"
        )
    return energy


def _normalise_device(device: str) -> str:
    value = str(device).strip().lower()
    if not value:
        raise CalculatorConfigurationError("calculator device cannot be empty")
    if value == "auto":
        try:
            import torch
        except Exception as exc:
            raise CalculatorConfigurationError(
                "device='auto' requires PyTorch so CUDA availability can be "
                f"checked ({type(exc).__name__}: {exc})"
            ) from exc
        if torch.cuda.is_available():
            return "cuda"
        # MPS remains available as an explicit, best-effort choice, but it is
        # not selected automatically because MACE's production ASE calculator
        # path is formally documented for CUDA/CPU.
        return "cpu"
    if not (value == "cpu" or value == "mps" or value.startswith("cuda")):
        raise CalculatorConfigurationError(
            f"Unsupported MACE device {device!r}; use auto, cpu, cuda[:index], or mps"
        )
    if value.startswith("cuda"):
        suffix = value.removeprefix("cuda")
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            raise CalculatorConfigurationError(
                f"Invalid CUDA device {device!r}; use cuda or cuda:<nonnegative index>"
            )
    return value


def _normalise_dtype(dtype: str) -> str:
    value = str(dtype).strip().lower()
    aliases = {
        "double": "float64",
        "torch.float64": "float64",
        "np.float64": "float64",
        "single": "float32",
        "torch.float32": "float32",
        "np.float32": "float32",
    }
    value = aliases.get(value, value)
    if value not in {"float32", "float64"}:
        raise CalculatorConfigurationError(
            f"Unsupported calculator dtype {dtype!r}; use float32 or float64"
        )
    return value


def _accepted_precision_parameter(callable_factory: Any) -> str | None:
    try:
        signature = inspect.signature(callable_factory)
    except (TypeError, ValueError):
        return None
    if "default_dtype" in signature.parameters:
        return "default_dtype"
    if "dtype" in signature.parameters:
        return "dtype"
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return "dtype"
    return None


def _inject_runtime_argument(
    callable_factory: Any,
    kwargs: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if value is None or name in kwargs:
        return
    try:
        signature = inspect.signature(callable_factory)
    except (TypeError, ValueError):
        return
    accepts = name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts:
        kwargs[name] = value


def _require_ase_calculator(calculator: Any, *, origin: str) -> Calculator:
    if not isinstance(calculator, Calculator):
        raise CalculatorConfigurationError(
            f"Calculator source {origin!r} returned {type(calculator).__name__}, "
            "not an ase.calculators.calculator.Calculator"
        )
    return calculator


def _validate_stress_tolerances(delta: float, rtol: float, atol: float) -> None:
    if not np.isfinite(delta) or not 0.0 < delta < 0.1:
        raise CalculatorProbeError(
            "finite_difference_delta_ln_volume must be finite and between 0 and 0.1"
        )
    if not np.isfinite(rtol) or rtol < 0.0:
        raise CalculatorProbeError("stress_relative_tolerance must be non-negative")
    if not np.isfinite(atol) or atol < 0.0:
        raise CalculatorProbeError(
            "stress_absolute_tolerance_ev_a3 must be non-negative"
        )


def _calculator_name(calculator: Calculator) -> str:
    cls = type(calculator)
    return f"{cls.__module__}.{cls.__qualname__}"


def _symbol(number: int) -> str:
    return chemical_symbols[number] if 0 < number < len(chemical_symbols) else f"Z={number}"


__all__ = [
    "CalculatorConfigurationError",
    "CalculatorKind",
    "CalculatorProbeError",
    "CalculatorProbeReport",
    "CalculatorSpec",
    "ModelDownloadBlockedError",
    "build_calculator",
    "create_factory_calculator",
    "create_mace_calculator",
    "discover_supported_atomic_numbers",
    "probe_calculator",
    "verify_model_sha256",
]
