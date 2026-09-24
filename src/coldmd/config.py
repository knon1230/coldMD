"""Validated, unit-explicit configuration for cold-compression MD.

Version 2 is deliberately stage-explicit.  A protocol is an ordered list of
fixed-volume holds, pressure-controlled holds, and prescribed-volume ramps;
the physical control variable belongs to each stage rather than to a global
``control_mode``.  Relative volume targets are defined with respect to the
start of the individual ramp stage.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import math
import re
from typing import Annotated, Any, Literal, Mapping, Sequence, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from .schedules import pressure_range_targets


FiniteFloat: TypeAlias = Annotated[
    float, Field(strict=True, allow_inf_nan=False)
]
PositiveFloat: TypeAlias = Annotated[
    float, Field(strict=True, gt=0.0, allow_inf_nan=False)
]
NonNegativeFloat: TypeAlias = Annotated[
    float, Field(strict=True, ge=0.0, allow_inf_nan=False)
]
CompressionRatio: TypeAlias = Annotated[
    float, Field(strict=True, gt=0.0, lt=1.0, allow_inf_nan=False)
]
DecompressionRatio: TypeAlias = Annotated[
    float, Field(strict=True, gt=1.0, allow_inf_nan=False)
]
PositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt: TypeAlias = Annotated[int, Field(strict=True, ge=0)]

RequiredProperty: TypeAlias = Literal["energy", "forces", "stress"]
ControlMode: TypeAlias = Literal[
    "volume_ramp", "pressure_plateaus", "stage_sequence"
]
VolumeSchedule: TypeAlias = Literal["log_linear", "log_smoothstep"]
PressureBranch: TypeAlias = Literal["cold_compression", "decompression"]
StageBranch: TypeAlias = Literal[
    "cold_compression",
    "decompression",
    "recovery",
    "initial_hold",
    "high_density_hold",
    "recovered_hold",
    "other",
]
ResearchBranch: TypeAlias = Literal[
    "compression", "decompression", "reference", "recovery", "other"
]
StagePurpose: TypeAlias = Literal[
    "equilibration", "sampling", "transition", "recovery"
]


class ConfigurationError(ValueError):
    """Raised for malformed or internally inconsistent configuration files."""


class StrictModel(BaseModel):
    """Base model shared by every public configuration section."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class InputConfig(StrictModel):
    """Input periodic structure and optional identity checks."""

    cif_path: Path = Field(description="Path to the periodic CIF structure.")
    cif_frame: NonNegativeInt = Field(
        default=0,
        description="Zero-based CIF frame/block selected by the structure reader.",
    )
    supercell: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default=(1, 1, 1),
        description="Integer replication along the three lattice vectors.",
    )
    expected_atom_count: PositiveInt | None = Field(
        default=None,
        description="Optional atom-count assertion after supercell expansion.",
    )
    expected_formula: StrictStr | None = Field(
        default=None,
        description="Optional chemical-formula assertion.",
    )
    reject_partial_occupancy: StrictBool = Field(
        default=True,
        description="Reject partially occupied/disordered CIF sites by default.",
    )

    @field_validator("cif_path")
    @classmethod
    def _require_cif_suffix(cls, value: Path) -> Path:
        if value.suffix.lower() != ".cif":
            raise ValueError("cif_path must end in '.cif'")
        return value

    @field_validator("expected_formula")
    @classmethod
    def _nonempty_formula(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("expected_formula cannot be empty")
        return value


class CalculatorConfig(StrictModel):
    """Generic Python-importable ASE Calculator factory."""

    kind: Literal["factory", "mace"] = "factory"
    factory: StrictStr | None = Field(
        default=None,
        description="Import target in 'package.module:callable' form.",
    )
    foundation_model: StrictStr | None = Field(
        default=None,
        description="MACE foundation-model alias, mutually exclusive with model_path.",
    )
    kwargs: dict[StrictStr, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to the factory.",
    )
    model_path: Path | None = Field(
        default=None,
        description="Optional local model checkpoint; resolved relative to the YAML file.",
    )
    model_sha256: StrictStr | None = Field(
        default=None,
        description="Optional expected lowercase/uppercase SHA-256 digest.",
    )
    device: StrictStr = Field(
        default="auto",
        description="auto, cpu, mps, cuda, or an indexed CUDA device such as cuda:0.",
    )
    dtype: Literal["float64", "float32"] = "float64"
    allow_download: StrictBool = Field(
        default=False,
        description="Explicit opt-in for a MACE foundation-model download.",
    )
    required_properties: tuple[RequiredProperty, ...] = (
        "energy",
        "forces",
        "stress",
    )

    @field_validator("factory")
    @classmethod
    def _valid_factory(cls, value: str | None) -> str | None:
        if value is None:
            return None
        pattern = r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+:[A-Za-z_]\w*$"
        if not re.fullmatch(pattern, value):
            raise ValueError(
                "factory must have the form 'package.module:callable'"
            )
        return value

    @field_validator("foundation_model")
    @classmethod
    def _nonempty_foundation_model(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("foundation_model cannot be empty")
        return value

    @field_validator("model_sha256")
    @classmethod
    def _valid_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("model_sha256 must contain exactly 64 hexadecimal digits")
        return value.lower()

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: str) -> str:
        if not re.fullmatch(r"(?:auto|cpu|mps|cuda(?::\d+)?)", value):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        return value

    @field_validator("required_properties")
    @classmethod
    def _unique_properties(
        cls, value: tuple[RequiredProperty, ...]
    ) -> tuple[RequiredProperty, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required_properties cannot contain duplicates")
        missing = {"energy", "forces", "stress"}.difference(value)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "cold-compression runs require energy, forces, and stress; "
                f"missing: {missing_text}"
            )
        return value

    @model_validator(mode="after")
    def _coherent_backend(self) -> "CalculatorConfig":
        if self.kind == "factory":
            if self.factory is None:
                raise ValueError("kind='factory' requires factory")
            if self.foundation_model is not None:
                raise ValueError(
                    "factory calculators cannot define foundation_model"
                )
            if self.model_path is not None and "model_path" in self.kwargs:
                raise ValueError(
                    "define a factory model path either as model_path or in kwargs, "
                    "not both"
                )
            if self.model_sha256 is not None and self.model_path is None:
                raise ValueError("model_sha256 requires model_path")
            if self.allow_download:
                raise ValueError("allow_download is only valid for a MACE foundation model")
            return self

        if self.factory is not None:
            raise ValueError("kind='mace' cannot also define factory")
        if bool(self.foundation_model) == bool(self.model_path):
            raise ValueError(
                "kind='mace' requires exactly one of foundation_model or model_path"
            )
        if self.foundation_model is not None and not self.allow_download:
            raise ValueError(
                "a MACE foundation_model requires allow_download=true; use a local "
                "model_path for an offline, content-addressed production run"
            )
        if self.model_path is not None and self.allow_download:
            raise ValueError("allow_download must be false when model_path is local")
        if self.model_sha256 is not None and self.model_path is None:
            raise ValueError("model_sha256 requires model_path")
        return self

    def to_spec_mapping(self) -> dict[str, Any]:
        """Return only fields understood by :class:`CalculatorSpec`."""

        kwargs = dict(self.kwargs)
        specification_model_path: Path | None = self.model_path
        if self.kind == "factory" and self.model_path is not None:
            # CalculatorSpec reserves model_path for its explicit MACE backend.
            # A generic factory receives the same explicit path as a kwarg.
            kwargs["model_path"] = str(self.model_path)
            specification_model_path = None
        return {
            "kind": self.kind,
            "factory": self.factory,
            "foundation_model": self.foundation_model,
            "model_path": specification_model_path,
            "device": self.device,
            "dtype": self.dtype,
            "kwargs": kwargs,
            "allow_download": self.allow_download,
        }


class ThermostatConfig(StrictModel):
    """Fixed-cell thermostat selection and supported volume-ramp parameters."""

    kind: Literal["bussi", "nose_hoover_chain"] = "bussi"
    tau_fs: PositiveFloat = Field(
        default=100.0,
        description="Thermostat relaxation/damping time in femtoseconds.",
    )
    chain_length: PositiveInt = Field(
        default=3,
        description="Nosé–Hoover thermostat-chain length; ignored by Bussi.",
    )
    substeps: PositiveInt = Field(
        default=1,
        description="Nosé–Hoover thermostat-chain substeps; ignored by Bussi.",
    )


class BarostatConfig(StrictModel):
    """Parameters for isotropic Martyna-Tobias-Klein NPT stages."""

    kind: Literal["isotropic_mtk"] = "isotropic_mtk"
    thermostat_tau_fs: PositiveFloat = Field(
        default=100.0,
        description="MTK thermostat damping time in femtoseconds.",
    )
    barostat_tau_fs: PositiveFloat = Field(
        default=1000.0,
        description="MTK barostat damping time in femtoseconds.",
    )
    thermostat_chain_length: PositiveInt = 3
    barostat_chain_length: PositiveInt = 3
    thermostat_substeps: PositiveInt = 1
    barostat_substeps: PositiveInt = 1

    @model_validator(mode="after")
    def _barostat_slower_than_thermostat(self) -> "BarostatConfig":
        if self.barostat_tau_fs <= self.thermostat_tau_fs:
            raise ValueError(
                "barostat_tau_fs must be greater than thermostat_tau_fs"
            )
        return self


class StageBase(StrictModel):
    """Fields present in every stage."""

    name: StrictStr
    duration_ps: PositiveFloat = Field(description="Stage duration in picoseconds.")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError(
                "stage name must be 1-64 safe characters: letters, digits, '_', '-', '.'"
            )
        return value


class SamplingStageBase(StageBase):
    """A stage with an optional tail window marked for structural analysis."""

    sampling_window_ps: PositiveFloat | None = Field(
        default=None,
        description="Optional analyzed tail of the stage, in picoseconds.",
    )

    @model_validator(mode="after")
    def _sampling_fits_stage(self) -> "SamplingStageBase":
        if (
            self.sampling_window_ps is not None
            and self.sampling_window_ps > self.duration_ps
        ):
            raise ValueError("sampling_window_ps cannot exceed duration_ps")
        return self


class InitialNVTStage(SamplingStageBase):
    kind: Literal["initial_nvt"] = "initial_nvt"
    initialize_velocities: StrictBool = Field(
        default=True,
        description="Draw initial Maxwell-Boltzmann momenta once at run start.",
    )


class NVTStage(SamplingStageBase):
    """Generic fixed-cell NVT stage for freely ordered protocols."""

    kind: Literal["nvt"] = "nvt"
    branch: StageBranch = "other"


class CompressionStage(StageBase):
    """Thermostatted isotropic cold-compression cell ramp."""

    kind: Literal["compression"] = "compression"
    target_volume_ratio: CompressionRatio = Field(
        description=(
            "Dimensionless Vend/Vstart for this stage; it must be below one."
        ),
    )
    schedule: VolumeSchedule = "log_linear"


class PeakNVTStage(SamplingStageBase):
    kind: Literal["peak_nvt"] = "peak_nvt"


class DecompressionStage(StageBase):
    """Thermostatted isotropic decompression cell ramp."""

    kind: Literal["decompression"] = "decompression"
    target_volume_ratio: DecompressionRatio = Field(
        description=(
            "Dimensionless Vend/Vstart for this stage; it must be above one."
        ),
    )
    schedule: VolumeSchedule = "log_linear"


class PressurePlateauStage(SamplingStageBase):
    """Fixed-target-pressure isotropic NPT plateau."""

    kind: Literal["pressure_plateau"] = "pressure_plateau"
    branch: PressureBranch
    target_pressure_GPa: NonNegativeFloat = Field(
        description="Hydrostatic target pressure in gigapascals."
    )


class NPTStage(SamplingStageBase):
    """Generic isotropic NPT stage for freely ordered protocols."""

    kind: Literal["npt"] = "npt"
    branch: StageBranch = "other"
    target_pressure_GPa: FiniteFloat = Field(
        description="Hydrostatic target pressure in gigapascals."
    )


class PressureRangeStage(StrictModel):
    """Input-only shorthand expanded into fixed-pressure NPT plateaus."""

    kind: Literal["pressure_range"] = "pressure_range"
    name_prefix: StrictStr = Field(
        description="Prefix for deterministic '<prefix>_<pressure>GPa' stage names."
    )
    branch: PressureBranch
    start_pressure_GPa: NonNegativeFloat
    end_pressure_GPa: NonNegativeFloat
    pressure_step_GPa: PositiveFloat = Field(
        description="Positive pressure increment magnitude in gigapascals."
    )
    duration_ps: PositiveFloat = Field(
        description="Duration assigned to every expanded plateau."
    )
    sampling_window_ps: PositiveFloat | None = Field(
        default=None,
        description="Tail sampling window assigned to every expanded plateau.",
    )

    @field_validator("name_prefix")
    @classmethod
    def _valid_name_prefix(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError(
                "name_prefix must be 1-64 safe characters: letters, digits, "
                "'_', '-', '.'"
            )
        return value

    @model_validator(mode="after")
    def _coherent_range(self) -> "PressureRangeStage":
        if (
            self.sampling_window_ps is not None
            and self.sampling_window_ps > self.duration_ps
        ):
            raise ValueError("sampling_window_ps cannot exceed duration_ps")
        if self.end_pressure_GPa == self.start_pressure_GPa:
            raise ValueError(
                "pressure_range endpoints must differ; use pressure_plateau "
                "for one target"
            )
        if (
            self.branch == "cold_compression"
            and self.end_pressure_GPa <= self.start_pressure_GPa
        ):
            raise ValueError(
                "cold_compression pressure_range requires end_pressure_GPa "
                "greater than start_pressure_GPa"
            )
        if (
            self.branch == "decompression"
            and self.end_pressure_GPa >= self.start_pressure_GPa
        ):
            raise ValueError(
                "decompression pressure_range requires end_pressure_GPa less "
                "than start_pressure_GPa"
            )
        # Validate divisibility and the expansion count before creating stage
        # mappings.  The returned values are regenerated by the normalizer.
        pressure_range_targets(
            self.start_pressure_GPa,
            self.end_pressure_GPa,
            self.pressure_step_GPa,
        )
        return self


class RecoveryNPTStage(SamplingStageBase):
    kind: Literal["recovery_npt"] = "recovery_npt"
    target_pressure_GPa: NonNegativeFloat = Field(
        default=0.0,
        description="Recovery target pressure in gigapascals.",
    )


class FinalNVTStage(SamplingStageBase):
    kind: Literal["final_nvt"] = "final_nvt"


# v2 stage vocabulary -------------------------------------------------------
#
# The legacy classes above remain readable so existing completed runs can still
# be resumed with their original installation.  New v2 input uses the three
# control-variable based classes below.  They intentionally all carry a tail
# sampling window: it is the statistically defined pool from which the next
# stage receives its physically consistent handoff snapshot.


class V2StageBase(StageBase):
    """Common metadata for every v2 stage."""

    branch: ResearchBranch = "other"
    purpose: StagePurpose = "equilibration"
    sampling_window_ps: PositiveFloat = Field(
        description=(
            "Tail window used for statistics and for selecting the state passed "
            "to the next stage."
        )
    )

    @model_validator(mode="after")
    def _sampling_fits_stage_v2(self) -> "V2StageBase":
        if self.sampling_window_ps > self.duration_ps:
            raise ValueError("sampling_window_ps cannot exceed duration_ps")
        return self


class NVTHoldStage(V2StageBase):
    """Fixed-cell NVT hold for equilibration or transport observation."""

    kind: Literal["nvt_hold"] = "nvt_hold"


class NPTHoldStage(V2StageBase):
    """Isotropic pressure-controlled NPT hold."""

    kind: Literal["npt_hold"] = "npt_hold"
    target_pressure_GPa: FiniteFloat = Field(
        description="Hydrostatic target pressure in gigapascals."
    )


class VolumeRampControlStage(V2StageBase):
    """Prescribed isotropic volume/strain path (a driven, non-equilibrium stage)."""

    kind: Literal["volume_ramp"] = "volume_ramp"
    target_volume_ratio: PositiveFloat = Field(
        description="Vend/Vstart for this individual prescribed-volume ramp."
    )
    schedule: VolumeSchedule = "log_linear"

    @model_validator(mode="after")
    def _valid_volume_ramp(self) -> "VolumeRampControlStage":
        if math.isclose(self.target_volume_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError("volume_ramp target_volume_ratio must differ from 1")
        expected = "compression" if self.target_volume_ratio < 1.0 else "decompression"
        if self.branch not in {"other", expected}:
            raise ValueError(
                "volume_ramp branch must agree with target_volume_ratio "
                f"({expected!r} for this value)"
            )
        if self.branch == "other":
            self.branch = expected
        return self


StageConfig: TypeAlias = Annotated[
    InitialNVTStage
    | NVTStage
    | CompressionStage
    | PeakNVTStage
    | DecompressionStage
    | PressurePlateauStage
    | NPTStage
    | PressureRangeStage
    | RecoveryNPTStage
    | FinalNVTStage
    | NVTHoldStage
    | NPTHoldStage
    | VolumeRampControlStage,
    Field(discriminator="kind"),
]


class ProtocolConfig(StrictModel):
    """Ordered, isothermal cold-compression/decompression protocol.

    ``control_mode`` is retained only to read resolved v1 run directories.  It
    is optional in v2 input and has no scientific meaning there: the explicit
    stage ``kind`` identifies whether the cell is fixed, pressure controlled,
    or prescribed to deform.
    """

    control_mode: ControlMode = "stage_sequence"
    temperature_K: PositiveFloat = Field(description="Target temperature in kelvin.")
    timestep_fs: PositiveFloat = Field(
        default=0.5,
        description="MD integration timestep in femtoseconds.",
    )
    random_seed: NonNegativeInt = Field(default=20260901)
    remove_center_of_mass_momentum: StrictBool = True
    initialize_velocities: StrictBool | None = Field(
        default=None,
        description=(
            "Optional run-level velocity initialization policy for stage_sequence; "
            "legacy protocols may continue to use initial_nvt.initialize_velocities."
        ),
    )
    thermostat: ThermostatConfig = Field(default_factory=ThermostatConfig)
    barostat: BarostatConfig = Field(default_factory=BarostatConfig)
    stages: list[StageConfig] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_pressure_ranges(cls, value: Any) -> Any:
        """Replace every input-only pressure range with canonical plateaus."""

        if isinstance(value, cls) or not isinstance(value, Mapping):
            return value
        stages = value.get("stages")
        if not isinstance(stages, (list, tuple)):
            return value
        normalized = dict(value)
        normalized["stages"] = expand_pressure_ranges(stages)
        return normalized

    @model_validator(mode="after")
    def _validate_protocol(self) -> "ProtocolConfig":
        if self.thermostat.tau_fs <= self.timestep_fs:
            raise ValueError("thermostat tau_fs must exceed timestep_fs")
        if self.barostat.thermostat_tau_fs <= self.timestep_fs:
            raise ValueError(
                "barostat thermostat_tau_fs must exceed timestep_fs"
            )
        if self.barostat.barostat_tau_fs <= self.timestep_fs:
            raise ValueError("barostat barostat_tau_fs must exceed timestep_fs")

        names = [stage.name for stage in self.stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "stage names must be unique; duplicates: " + ", ".join(duplicates)
            )

        for stage in self.stages:
            _duration_to_steps(stage.duration_ps, self.timestep_fs, stage.name)
            if isinstance(stage, (SamplingStageBase, V2StageBase)) and stage.sampling_window_ps:
                _duration_to_steps(
                    stage.sampling_window_ps,
                    self.timestep_fs,
                    f"{stage.name}.sampling_window_ps",
                )

        if self.thermostat.kind == "nose_hoover_chain" and any(
            isinstance(stage, (CompressionStage, DecompressionStage, VolumeRampControlStage))
            for stage in self.stages
        ):
            raise ValueError(
                "thermostat.kind='nose_hoover_chain' is not supported with "
                "compression or decompression stages; prescribed volume ramps "
                "currently require thermostat.kind='bussi'"
            )

        if self.control_mode == "volume_ramp":
            self._validate_volume_stages()
        elif self.control_mode == "pressure_plateaus":
            self._validate_pressure_stages()
        else:
            self._validate_stage_sequence()
        return self

    def _validate_volume_stages(self) -> None:
        forbidden = [
            stage.name
            for stage in self.stages
            if isinstance(stage, (PressurePlateauStage, NVTStage, NPTStage))
        ]
        if forbidden:
            raise ValueError(
                "pressure_plateau, nvt, and npt stages are incompatible with control_mode="
                f"'volume_ramp': {', '.join(forbidden)}"
            )

        counts = _stage_counts(self.stages)
        required_exact = {
            InitialNVTStage: 1,
            PeakNVTStage: 1,
            FinalNVTStage: 1,
        }
        for stage_type, expected in required_exact.items():
            actual = counts.get(stage_type, 0)
            if actual != expected:
                raise ValueError(
                    f"volume_ramp protocols require exactly {expected} "
                    f"{stage_type.__name__} stage; found {actual}"
                )
        if counts.get(CompressionStage, 0) < 1:
            raise ValueError("volume_ramp protocols require a compression stage")
        if counts.get(DecompressionStage, 0) < 1:
            raise ValueError("volume_ramp protocols require a decompression stage")
        if counts.get(RecoveryNPTStage, 0) > 1:
            raise ValueError("at most one recovery_npt stage is allowed")

        rank = {
            InitialNVTStage: 0,
            CompressionStage: 1,
            PeakNVTStage: 2,
            DecompressionStage: 3,
            RecoveryNPTStage: 4,
            FinalNVTStage: 5,
        }
        _require_nondecreasing_stage_order(self.stages, rank)

    def _validate_pressure_stages(self) -> None:
        forbidden = [
            stage.name
            for stage in self.stages
            if isinstance(
                stage,
                (
                    CompressionStage,
                    PeakNVTStage,
                    DecompressionStage,
                    NVTStage,
                    NPTStage,
                ),
            )
        ]
        if forbidden:
            raise ValueError(
                "compression, peak_nvt, decompression, nvt, and npt stages are "
                "incompatible "
                "with control_mode='pressure_plateaus': " + ", ".join(forbidden)
            )

        counts = _stage_counts(self.stages)
        for stage_type in (InitialNVTStage, FinalNVTStage):
            actual = counts.get(stage_type, 0)
            if actual != 1:
                raise ValueError(
                    "pressure_plateaus protocols require exactly one "
                    f"{stage_type.__name__} stage; found {actual}"
                )
        if counts.get(RecoveryNPTStage, 0) > 1:
            raise ValueError("at most one recovery_npt stage is allowed")

        plateaus = [
            stage
            for stage in self.stages
            if isinstance(stage, PressurePlateauStage)
        ]
        compression = [
            stage for stage in plateaus if stage.branch == "cold_compression"
        ]
        decompression = [
            stage for stage in plateaus if stage.branch == "decompression"
        ]
        if not compression or not decompression:
            raise ValueError(
                "pressure_plateaus protocols require both cold_compression and "
                "decompression pressure branches"
            )

        branch_order = [stage.branch for stage in plateaus]
        if branch_order != sorted(
            branch_order,
            key={"cold_compression": 0, "decompression": 1}.__getitem__,
        ):
            raise ValueError(
                "all cold_compression pressure plateaus must precede all "
                "decompression pressure plateaus"
            )
        compression_pressures = [stage.target_pressure_GPa for stage in compression]
        decompression_pressures = [
            stage.target_pressure_GPa for stage in decompression
        ]
        if not _monotonic(compression_pressures, increasing=True, strict=True):
            raise ValueError(
                "cold_compression target_pressure_GPa values must strictly increase"
            )
        if not _monotonic(decompression_pressures, increasing=False, strict=True):
            raise ValueError(
                "decompression target_pressure_GPa values must strictly decrease"
            )
        if decompression_pressures[0] >= compression_pressures[-1]:
            raise ValueError(
                "the first decompression pressure must be below the peak "
                "cold-compression pressure"
            )

        rank = {
            InitialNVTStage: 0,
            PressurePlateauStage: 1,
            RecoveryNPTStage: 2,
            FinalNVTStage: 3,
        }
        _require_nondecreasing_stage_order(self.stages, rank)

    def _validate_stage_sequence(self) -> None:
        """Accept individually valid stage types in exactly the written order.

        ``stage_sequence`` deliberately imposes no cardinality, branch-order, or
        monotonic-pressure rules.  Unique names, integral durations, sampling
        windows, stage-specific fields, and cross-section safety limits remain
        validated by the shared validators.
        """


class OutputConfig(StrictModel):
    """Run directory and loss-aware output cadence."""

    directory: Path = Path("coldmd-run")
    trajectory_filename: StrictStr = "trajectory.traj"
    thermo_filename: StrictStr = "thermo.csv"
    log_filename: StrictStr = "coldmd.log"
    trajectory_interval_steps: PositiveInt = 100
    thermo_interval_steps: PositiveInt = 10
    msd_interval_steps: PositiveInt = 10
    checkpoint_interval_steps: PositiveInt = 10_000
    write_forces: StrictBool = True
    write_stress: StrictBool = True
    write_momenta: StrictBool = True
    write_stage_structures: StrictBool = True

    @field_validator("trajectory_filename")
    @classmethod
    def _trajectory_name(cls, value: str) -> str:
        _require_plain_filename(value, "trajectory_filename")
        if Path(value).suffix.lower() not in {".traj", ".extxyz", ".xyz"}:
            raise ValueError(
                "trajectory_filename must use .traj, .extxyz, or .xyz"
            )
        # Storage normalizes trajectory format suffixes; do so once in the
        # resolved configuration to keep resume discovery unambiguous.
        return str(Path(value).with_suffix(Path(value).suffix.lower()))

    @field_validator("thermo_filename")
    @classmethod
    def _thermo_name(cls, value: str) -> str:
        _require_plain_filename(value, "thermo_filename")
        if Path(value).suffix.lower() != ".csv":
            raise ValueError("thermo_filename must end in .csv")
        return value

    @field_validator("log_filename")
    @classmethod
    def _log_name(cls, value: str) -> str:
        _require_plain_filename(value, "log_filename")
        if Path(value).suffix.lower() != ".log":
            raise ValueError("log_filename must end in .log")
        return value


class SafetyConfig(StrictModel):
    """Hard-stop guards; ``None`` disables only the stated optional guard."""

    min_temperature_K: NonNegativeFloat | None = 1.0
    max_temperature_K: PositiveFloat | None = 5000.0
    max_abs_pressure_GPa: PositiveFloat | None = None
    max_force_eV_A: PositiveFloat | None = None
    min_distance_A: PositiveFloat | None = 0.5
    min_cell_volume_A3: PositiveFloat | None = None
    min_volume_ratio: PositiveFloat | None = 0.2
    max_volume_ratio: PositiveFloat | None = 2.0
    max_abs_log_volume_strain_per_step: PositiveFloat | None = 1.0e-3
    max_cell_condition_number: PositiveFloat | None = 1.0e4
    max_deviatoric_stress_GPa: PositiveFloat | None = None
    abort_on_nonfinite: Literal[True] = True
    checkpoint_on_abort: StrictBool = True
    maximum_consecutive_violations: PositiveInt = 1

    @field_validator("abort_on_nonfinite", mode="before")
    @classmethod
    def _nonfinite_guard_is_mandatory(cls, value: Any) -> Any:
        if type(value) is not bool or value is not True:
            raise ValueError("abort_on_nonfinite must be the boolean true")
        return value

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "SafetyConfig":
        if (
            self.min_temperature_K is not None
            and self.max_temperature_K is not None
            and self.min_temperature_K >= self.max_temperature_K
        ):
            raise ValueError("min_temperature_K must be below max_temperature_K")
        if (
            self.min_volume_ratio is not None
            and self.max_volume_ratio is not None
            and self.min_volume_ratio >= self.max_volume_ratio
        ):
            raise ValueError("min_volume_ratio must be below max_volume_ratio")
        return self


class ColdMDConfig(StrictModel):
    """Top-level cold-compression MD configuration."""

    config_version: Literal[1, 2] = 2
    input: InputConfig
    calculator: CalculatorConfig
    protocol: ProtocolConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    _source_path: Path | None = PrivateAttr(default=None)

    @field_validator("config_version", mode="before")
    @classmethod
    def _strict_config_version(cls, value: Any) -> Any:
        if type(value) is not int or value not in {1, 2}:
            raise ValueError("config_version must be the integer 1 or 2")
        return value

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @model_validator(mode="after")
    def _cross_validate(self) -> "ColdMDConfig":
        if self.config_version == 2:
            if "control_mode" in self.protocol.model_fields_set:
                raise ValueError(
                    "config_version: 2 does not accept protocol.control_mode; "
                    "choose nvt_hold, npt_hold, or volume_ramp per stage"
                )
            legacy_stages = [
                stage.name
                for stage in self.protocol.stages
                if not isinstance(stage, V2StageBase)
            ]
            if legacy_stages:
                raise ValueError(
                    "config_version: 2 accepts only nvt_hold, npt_hold, and "
                    "volume_ramp stages; legacy stage(s): "
                    + ", ".join(legacy_stages)
                )
        if (
            self.safety.min_temperature_K is not None
            and self.protocol.temperature_K < self.safety.min_temperature_K
        ):
            raise ValueError(
                "protocol temperature_K is below safety min_temperature_K"
            )
        if (
            self.safety.max_temperature_K is not None
            and self.protocol.temperature_K > self.safety.max_temperature_K
        ):
            raise ValueError(
                "protocol temperature_K is above safety max_temperature_K"
            )

        if self.protocol.control_mode in {"volume_ramp", "stage_sequence"}:
            cumulative_ratio = 1.0
            cumulative_ratio_known = True
            for stage in self.protocol.stages:
                if (
                    self.protocol.control_mode == "stage_sequence"
                    and isinstance(
                        stage,
                        (NPTStage, PressurePlateauStage, RecoveryNPTStage, NPTHoldStage),
                    )
                ):
                    # An NPT stage changes the cell dynamically, so later
                    # relative ramps can no longer be bounded against the
                    # original input volume during static validation.
                    cumulative_ratio_known = False
                    continue
                if not isinstance(
                    stage,
                    (CompressionStage, DecompressionStage, VolumeRampControlStage),
                ):
                    continue
                if (
                    cumulative_ratio_known
                    and self.safety.min_volume_ratio is not None
                    and cumulative_ratio * stage.target_volume_ratio
                    < self.safety.min_volume_ratio
                ):
                    reached_ratio = cumulative_ratio * stage.target_volume_ratio
                    raise ValueError(
                        f"stage '{stage.name}' reaches V/Vinitial={reached_ratio:.8g}, "
                        "below safety min_volume_ratio"
                    )
                if (
                    cumulative_ratio_known
                    and self.safety.max_volume_ratio is not None
                    and cumulative_ratio * stage.target_volume_ratio
                    > self.safety.max_volume_ratio
                ):
                    reached_ratio = cumulative_ratio * stage.target_volume_ratio
                    raise ValueError(
                        f"stage '{stage.name}' reaches V/Vinitial={reached_ratio:.8g}, "
                        "above safety max_volume_ratio"
                    )
                if self.safety.max_abs_log_volume_strain_per_step is not None:
                    steps = _duration_to_steps(
                        stage.duration_ps,
                        self.protocol.timestep_fs,
                        stage.name,
                    )
                    average_strain_per_step = abs(
                        math.log(stage.target_volume_ratio) / steps
                    )
                    # The derivative of 3x^2-2x^3 peaks at 1.5.  This is a
                    # conservative discrete-step bound for log_smoothstep.
                    schedule_factor = (
                        1.5 if stage.schedule == "log_smoothstep" else 1.0
                    )
                    strain_per_step = schedule_factor * average_strain_per_step
                    if (
                        strain_per_step
                        > self.safety.max_abs_log_volume_strain_per_step
                    ):
                        raise ValueError(
                            f"stage '{stage.name}' has |dlnV|/step="
                            f"{strain_per_step:.8g}, above safety "
                            "max_abs_log_volume_strain_per_step"
                        )
                if cumulative_ratio_known:
                    cumulative_ratio *= stage.target_volume_ratio

        if self.safety.max_abs_pressure_GPa is not None:
            pressure_stages = [
                stage
                for stage in self.protocol.stages
                if isinstance(
                    stage,
                    (NPTStage, PressurePlateauStage, RecoveryNPTStage, NPTHoldStage),
                )
            ]
            for stage in pressure_stages:
                if abs(stage.target_pressure_GPa) > self.safety.max_abs_pressure_GPa:
                    raise ValueError(
                        f"stage '{stage.name}' target_pressure_GPa exceeds safety "
                        "max_abs_pressure_GPa"
                    )
        return self


# Compatibility aliases kept intentionally small for downstream modules.
SimulationConfig = ColdMDConfig
RootConfig = ColdMDConfig


def load_config(path: str | Path) -> ColdMDConfig:
    """Load and validate a YAML configuration.

    Known filesystem paths are resolved relative to the YAML file, not the
    caller's current directory.  The input/model files are not opened here;
    physical and calculator checks belong to ``coldmd.api.validate_config``.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"invalid YAML in {config_path}: {exc}"
        ) from exc

    if raw is None:
        raise ConfigurationError(f"configuration file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigurationError("the YAML document root must be a mapping")

    resolved = _resolve_known_paths(raw, config_path.parent)
    config = ColdMDConfig.model_validate(resolved)
    config._source_path = config_path
    return config


def load_config_dict(
    data: dict[str, Any], *, base_directory: str | Path | None = None
) -> ColdMDConfig:
    """Validate an in-memory mapping, optionally resolving relative paths."""

    raw = deepcopy(data)
    if base_directory is not None:
        raw = _resolve_known_paths(raw, Path(base_directory).expanduser().resolve())
    return ColdMDConfig.model_validate(raw)


def dump_config(config: ColdMDConfig, path: str | Path) -> Path:
    """Write a normalized YAML configuration without private runtime state."""

    destination = Path(path)
    payload = config.model_dump(mode="json", exclude_none=True)
    if config.config_version == 2:
        # ``control_mode`` is an internal default retained solely for v1
        # compatibility.  A resolved v2 file must preserve the public,
        # stage-explicit vocabulary and therefore omit it as well.
        payload["protocol"].pop("control_mode", None)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    # The caller owns overwrite policy; this helper performs one explicit write.
    destination.write_text(text, encoding="utf-8")
    return destination


def _duration_to_steps(duration_ps: float, timestep_fs: float, label: str) -> int:
    raw_steps = duration_ps * 1000.0 / timestep_fs
    nearest = round(raw_steps)
    tolerance = 1.0e-9 * max(1.0, abs(raw_steps))
    if nearest < 1 or abs(raw_steps - nearest) > tolerance:
        raise ValueError(
            f"{label} duration_ps must be an integer multiple of timestep_fs; "
            f"got {raw_steps:.12g} steps"
        )
    return int(nearest)


def expand_pressure_ranges(stages: Sequence[Any]) -> list[Any]:
    """Expand input range mappings while leaving every other stage untouched.

    This public normalization helper lets offline tooling consume an original
    v1 YAML override as well as the already-canonical ``resolved-config.yaml``.
    """

    expanded: list[Any] = []
    for item in stages:
        if isinstance(item, PressureRangeStage):
            pressure_range = item
        elif isinstance(item, Mapping) and item.get("kind") == "pressure_range":
            pressure_range = PressureRangeStage.model_validate(dict(item))
        else:
            expanded.append(item)
            continue
        targets = pressure_range_targets(
            pressure_range.start_pressure_GPa,
            pressure_range.end_pressure_GPa,
            pressure_range.pressure_step_GPa,
        )
        for target in targets:
            stage_name = _pressure_range_stage_name(
                pressure_range.name_prefix,
                target,
            )
            stage: dict[str, Any] = {
                "kind": "pressure_plateau",
                "name": stage_name,
                "branch": pressure_range.branch,
                "target_pressure_GPa": target,
                "duration_ps": pressure_range.duration_ps,
            }
            if pressure_range.sampling_window_ps is not None:
                stage["sampling_window_ps"] = pressure_range.sampling_window_ps
            expanded.append(stage)
    return expanded


def _pressure_range_stage_name(prefix: str, pressure_GPa: float) -> str:
    """Create a stable safe name from a user prefix and decimal pressure."""

    decimal_pressure = Decimal(str(float(pressure_GPa)))
    if decimal_pressure == 0:
        decimal_pressure = Decimal(0)
    label = format(decimal_pressure.normalize(), "f")
    name = f"{prefix}_{label}GPa"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise ValueError(
            f"pressure_range name_prefix {prefix!r} generates invalid or "
            f"overlong stage name {name!r}"
        )
    return name


def _stage_counts(stages: list[StageConfig]) -> dict[type[StageBase], int]:
    counts: dict[type[StageBase], int] = {}
    for stage in stages:
        stage_type = type(stage)
        counts[stage_type] = counts.get(stage_type, 0) + 1
    return counts


def _require_nondecreasing_stage_order(
    stages: list[StageConfig], rank: dict[type[StageBase], int]
) -> None:
    values = [rank[type(stage)] for stage in stages]
    if values != sorted(values):
        order = " -> ".join(stage.kind for stage in stages)
        raise ValueError(f"invalid stage order: {order}")


def _monotonic(
    values: list[float], *, increasing: bool, strict: bool
) -> bool:
    pairs = zip(values, values[1:])
    if increasing and strict:
        return all(left < right for left, right in pairs)
    if increasing:
        return all(left <= right for left, right in pairs)
    if strict:
        return all(left > right for left, right in pairs)
    return all(left >= right for left, right in pairs)


def _require_plain_filename(value: str, field_name: str) -> None:
    path = Path(value)
    if (
        not value
        or "/" in value
        or "\\" in value
        or path.name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{field_name} must be a plain filename, not a path")


def _resolve_known_paths(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    data = deepcopy(raw)

    def resolve(section: str, key: str) -> None:
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            return
        value = section_data.get(key)
        if value is None:
            return
        if not isinstance(value, (str, Path)):
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        section_data[key] = path.resolve()

    resolve("input", "cif_path")
    resolve("calculator", "model_path")
    if "output" not in data:
        data["output"] = {"directory": (base / "coldmd-run").resolve()}
    elif isinstance(data["output"], dict) and "directory" not in data["output"]:
        data["output"]["directory"] = (base / "coldmd-run").resolve()
    resolve("output", "directory")
    return data


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


__all__ = [
    "BarostatConfig",
    "CalculatorConfig",
    "ColdMDConfig",
    "CompressionStage",
    "ConfigurationError",
    "DecompressionStage",
    "FinalNVTStage",
    "InitialNVTStage",
    "InputConfig",
    "NPTStage",
    "NPTHoldStage",
    "NVTStage",
    "NVTHoldStage",
    "OutputConfig",
    "PeakNVTStage",
    "PressurePlateauStage",
    "PressureRangeStage",
    "ProtocolConfig",
    "RecoveryNPTStage",
    "RootConfig",
    "SafetyConfig",
    "SimulationConfig",
    "StageConfig",
    "ThermostatConfig",
    "VolumeRampControlStage",
    "dump_config",
    "expand_pressure_ranges",
    "load_config",
    "load_config_dict",
]
