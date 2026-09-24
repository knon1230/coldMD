"""Plots and reproducible report bundles for :mod:`coldmd.analysis`.

The public :func:`generate_analysis_report` function is deliberately tolerant:
one unavailable diagnostic is recorded as a warning while independent results
are still written.  Scientific criteria and all sampled stage names are embedded
in ``analysis.json`` so that PNG figures are never the sole record of a result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from .config import expand_pressure_ranges  # noqa: E402
from .analysis import (  # noqa: E402
    AnalysisError,
    CoordinationResult,
    HysteresisResult,
    MSDResult,
    OxygenSpeciationResult,
    RDFResult,
    SelfIntermediateScatteringResult,
    StageThermoSummary,
    StageThermoSummaryResult,
    StructureFactorEstimate,
    TrajectoryDataset,
    cell_relative_variation,
    coordination_distributions,
    estimate_first_minimum,
    estimate_structure_factor_peak,
    load_segmented_trajectories,
    oxygen_speciation,
    partial_rdf,
    self_intermediate_scattering,
    sampling_window_thermo_series,
    species_msd,
    stage_thermo_statistics,
    topology_resolved_rdf,
    write_json,
)


BRANCH_COLORS = {
    "compression": "#1f77b4",
    "decompression": "#d62728",
    "recovery": "#e377c2",
    "high_density_hold": "#9467bd",
    "recovered_hold": "#2ca02c",
    "initial_hold": "#7f7f7f",
    "other": "#bcbd22",
}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "unknown"


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)
    return path


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 180) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)
    return path


def plot_rdf(result: RDFResult, path: str | Path, *, title: str | None = None) -> Path:
    """Plot every partial RDF in compact small multiples."""

    output = Path(path)
    keys = [key for key in result.g if key != "ALL-ALL"]
    if "ALL-ALL" in result.g:
        keys.insert(0, "ALL-ALL")
    columns = min(3, max(1, len(keys)))
    rows = int(math.ceil(len(keys) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.0 * rows), squeeze=False)
    for axis, key in zip(axes.flat, keys, strict=False):
        axis.plot(result.r_A, result.g[key], linewidth=1.35)
        axis.set_title(key)
        axis.set_xlabel(r"$r$ ($\AA$)")
        axis.set_ylabel(r"$g(r)$")
        axis.set_xlim(0.0, result.rmax_A)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(keys) :]:
        axis.set_visible(False)
    fig.suptitle(title or "Partial radial distribution functions")
    fig.tight_layout()
    return _save_figure(fig, output)


def write_rdf_csv(result: RDFResult, path: str | Path) -> Path:
    keys = list(result.g)
    rows = [
        {"r_A": float(radius), **{f"g_{key}": float(result.g[key][index]) for key in keys}}
        for index, radius in enumerate(result.r_A)
    ]
    return _atomic_csv(Path(path), ["r_A", *(f"g_{key}" for key in keys)], rows)


def plot_coordination(
    result: CoordinationResult,
    path: str | Path,
    *,
    title: str | None = None,
    maximum_panels: int = 16,
) -> Path:
    """Plot the most populated directed coordination distributions."""

    output = Path(path)
    pairs = sorted(
        result.pairs.items(), key=lambda item: item[1].n_centers, reverse=True
    )[:maximum_panels]
    if not pairs:
        raise AnalysisError("No coordination distributions are available to plot")
    columns = min(4, len(pairs))
    rows = int(math.ceil(len(pairs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows), squeeze=False)
    for axis, (key, pair) in zip(axes.flat, pairs, strict=False):
        axis.bar(pair.coordination, pair.probability, width=0.8)
        axis.set_title(f"{key}; cutoff {pair.cutoff_A:.3g} Å")
        axis.set_xlabel("Coordination number")
        axis.set_ylabel("Probability")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.2)
    for axis in list(axes.flat)[len(pairs) :]:
        axis.set_visible(False)
    fig.suptitle(title or "Coordination-number distributions")
    fig.tight_layout()
    return _save_figure(fig, output)


def write_coordination_csv(result: CoordinationResult, path: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    for key, pair in result.pairs.items():
        for coordination, count, probability in zip(
            pair.coordination, pair.count, pair.probability, strict=True
        ):
            rows.append(
                {
                    "pair": key,
                    "center": pair.center,
                    "neighbor": pair.neighbor,
                    "cutoff_A": pair.cutoff_A,
                    "cutoff_source": pair.cutoff_source,
                    "coordination": int(coordination),
                    "count": int(count),
                    "probability": float(probability),
                    "mean": pair.mean,
                    "std": pair.std,
                    "n_centers": pair.n_centers,
                }
            )
    fields = [
        "pair",
        "center",
        "neighbor",
        "cutoff_A",
        "cutoff_source",
        "coordination",
        "count",
        "probability",
        "mean",
        "std",
        "n_centers",
    ]
    return _atomic_csv(Path(path), fields, rows)


def plot_msd(
    result: MSDResult,
    path: str | Path,
    *,
    title: str | None = None,
    loglog: bool = False,
    sampling_window_ps: float | None = None,
) -> Path:
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for element, curve in result.msd_A2.items():
        status = result.diffusion[element].status
        if loglog:
            mask = (result.lag_time_ps > 0.0) & (curve > 0.0)
            axis.loglog(result.lag_time_ps[mask], curve[mask], label=f"{element}: {status}")
        else:
            axis.plot(result.lag_time_ps, curve, label=f"{element}: {status}")
    axis.set_xlabel("Elapsed time from stage origin (ps)")
    axis.set_ylabel(r"MSD ($\AA^2$)")
    if not loglog:
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
    if sampling_window_ps is not None:
        sampling_start = float(result.lag_time_ps[-1]) - float(sampling_window_ps)
        positive_times = result.lag_time_ps[result.lag_time_ps > 0.0]
        shading_start = (
            float(positive_times[0])
            if loglog and len(positive_times)
            else 0.0
        )
        if math.isfinite(sampling_start) and sampling_start > shading_start:
            axis.axvspan(
                shading_start,
                sampling_start,
                color="0.75",
                alpha=0.35,
                linewidth=0.0,
            )
            axis.axvline(
                sampling_start,
                color="0.35",
                linestyle="--",
                linewidth=0.9,
            )
            axis.text(
                0.01,
                0.04,
                "shaded: excluded from sampling statistics",
                transform=axis.transAxes,
                fontsize="x-small",
                color="0.3",
            )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    axis.set_title(title or ("Species MSD (log–log)" if loglog else "Species mean-squared displacement"))
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def write_msd_csv(result: MSDResult, path: str | Path) -> Path:
    elements = list(result.msd_A2)
    rows = [
        {
            "elapsed_time_ps": float(time),
            **{f"msd_{element}_A2": float(result.msd_A2[element][index]) for element in elements},
        }
        for index, time in enumerate(result.lag_time_ps)
    ]
    return _atomic_csv(
        Path(path), ["elapsed_time_ps", *(f"msd_{item}_A2" for item in elements)], rows
    )


def write_sampling_thermo_csv(
    series: Mapping[str, Any], path: str | Path
) -> Path:
    """Write one full-stage series and explicitly label sampling-tail rows."""

    times = list(series.get("time_ps", []))
    temperature = list(series.get("temperature_K", []))
    pressure = list(series.get("pressure_GPa", []))
    volume = list(series.get("volume_A3", []))
    sampling_mask = list(series.get("in_sampling_window", []))
    if not (
        len(times)
        == len(temperature)
        == len(pressure)
        == len(volume)
        == len(sampling_mask)
    ):
        raise AnalysisError("Sampling thermo series columns have inconsistent lengths")
    rows = [
        {
            "stage_time_ps": float(times[index]),
            "in_sampling_window": bool(sampling_mask[index]),
            "temperature_K": temperature[index],
            "pressure_GPa": pressure[index],
            "volume_A3": volume[index],
        }
        for index in range(len(times))
    ]
    return _atomic_csv(
        Path(path),
        [
            "stage_time_ps",
            "in_sampling_window",
            "temperature_K",
            "pressure_GPa",
            "volume_A3",
        ],
        rows,
    )


def plot_sampling_thermo(
    series: Mapping[str, Any], path: str | Path, *, observable: str, title: str
) -> Path:
    """Plot a full stage and shade the interval excluded from tail statistics."""

    if observable not in {"pressure_GPa", "volume_A3"}:
        raise ValueError("observable must be 'pressure_GPa' or 'volume_A3'")
    times = np.asarray(series.get("time_ps", []), dtype=float)
    temperature = np.asarray(series.get("temperature_K", []), dtype=float)
    secondary = np.asarray(series.get(observable, []), dtype=float)
    if len(times) < 1 or not (len(times) == len(temperature) == len(secondary)):
        raise AnalysisError("Sampling thermo series has no aligned observations")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    axes[0].plot(times, temperature, color="#d62728", linewidth=1.25)
    axes[0].set_ylabel("T (K)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, secondary, color="#1f77b4", linewidth=1.25)
    axes[1].set_xlabel("Stage time (ps)")
    axes[1].set_ylabel("P (GPa)" if observable == "pressure_GPa" else r"V ($\AA^3$)")
    axes[1].grid(alpha=0.25)
    sampling_start = series.get("sampling_cutoff_ps")
    if sampling_start is not None:
        sampling_start = float(sampling_start)
        if math.isfinite(sampling_start) and sampling_start > float(times[0]):
            for axis in axes:
                axis.axvspan(
                    float(times[0]),
                    sampling_start,
                    color="0.75",
                    alpha=0.35,
                    linewidth=0.0,
                )
                axis.axvline(
                    sampling_start,
                    color="0.35",
                    linestyle="--",
                    linewidth=0.9,
                )
            axes[0].text(
                0.01,
                0.04,
                "shaded: excluded from sampling statistics",
                transform=axes[0].transAxes,
                fontsize="x-small",
                color="0.3",
            )
    if len(times) > 1:
        axes[1].set_xlim(float(times[0]), float(times[-1]))
    fig.suptitle(title)
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def diffusion_rows(stage: str, result: MSDResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for element, estimate in result.diffusion.items():
        rows.append(
            {
                "stage": stage,
                "species": element,
                "status": estimate.status,
                "D_A2_ps": estimate.coefficient_A2_ps,
                "D_cm2_s": estimate.coefficient_cm2_s,
                "standard_error_A2_ps": estimate.standard_error_A2_ps,
                "confidence95_low_A2_ps": (
                    estimate.confidence95_A2_ps[0]
                    if estimate.confidence95_A2_ps is not None
                    else None
                ),
                "confidence95_high_A2_ps": (
                    estimate.confidence95_A2_ps[1]
                    if estimate.confidence95_A2_ps is not None
                    else None
                ),
                "upper_bound_A2_ps": estimate.upper_bound_A2_ps,
                "r_squared": estimate.r_squared,
                "fit_start_ps": estimate.fit_start_ps,
                "fit_end_ps": estimate.fit_end_ps,
                "observation_time_ps": result.observation_time_ps,
                "reason": estimate.reason,
            }
        )
    return rows


def plot_self_intermediate_scattering(
    result: SelfIntermediateScatteringResult,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for element, curve in result.Fs.items():
        axis.plot(result.lag_time_ps, curve, label=element)
    axis.axhline(math.e**-1, color="0.5", linestyle="--", linewidth=1.0, label=r"$e^{-1}$")
    axis.set_xlabel("Lag time (ps)")
    axis.set_ylabel(r"$F_s(k,t)$")
    axis.set_xlim(left=0.0)
    axis.set_ylim(-0.1, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    axis.set_title(title or f"Self-intermediate scattering at k={result.k_inv_A:.3g} Å⁻¹")
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def write_fs_csv(result: SelfIntermediateScatteringResult, path: str | Path) -> Path:
    elements = list(result.Fs)
    rows = [
        {
            "lag_time_ps": float(time),
            "k_inv_A": result.k_inv_A,
            **{f"Fs_{element}": float(result.Fs[element][index]) for element in elements},
        }
        for index, time in enumerate(result.lag_time_ps)
    ]
    return _atomic_csv(
        Path(path),
        ["lag_time_ps", "k_inv_A", *(f"Fs_{item}" for item in elements)],
        rows,
    )


def plot_structure_factor(
    result: StructureFactorEstimate, path: str | Path, *, title: str | None = None
) -> Path:
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(result.q_inv_A, result.S_q, linewidth=1.3)
    axis.axvline(result.k_inv_A, color="#d62728", linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"$q$ ($\AA^{-1}$)")
    axis.set_ylabel(r"Unweighted $S(q)$")
    axis.grid(alpha=0.25)
    axis.set_title(title or f"Estimated S(q) peak: {result.k_inv_A:.3g} Å⁻¹")
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def plot_hysteresis(result: HysteresisResult, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # Connectivity always follows stored chronology. Branch labels control only
    # point colour, so arbitrary stage sequences remain one continuous path.
    axes[0].plot(
        result.volume_ratio,
        result.pressure_GPa,
        color="0.45",
        linewidth=1.0,
        zorder=1,
        label="chronological path",
    )
    axes[1].plot(
        result.density_g_cm3,
        result.pressure_GPa,
        color="0.45",
        linewidth=1.0,
        zorder=1,
        label="chronological path",
    )
    branches = list(dict.fromkeys(result.branch))
    for branch in branches:
        mask = np.asarray([item == branch for item in result.branch])
        color = BRANCH_COLORS.get(branch, "0.4")
        axes[0].scatter(
            result.volume_ratio[mask],
            result.pressure_GPa[mask],
            marker="o",
            s=8.0,
            color=color,
            label=branch,
            zorder=2,
        )
        finite_density = mask & np.isfinite(result.density_g_cm3)
        if np.any(finite_density):
            axes[1].scatter(
                result.density_g_cm3[finite_density],
                result.pressure_GPa[finite_density],
                marker="o",
                s=8.0,
                color=color,
                label=branch,
                zorder=2,
            )
    axes[0].set_xlabel(r"Volume ratio $V/V_0$")
    axes[1].set_xlabel(r"Density (g cm$^{-3}$)")
    for axis in axes:
        axis.set_ylabel("Pressure (GPa)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    fig.suptitle("Pressure–volume/density path")
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def write_hysteresis_csv(result: HysteresisResult, path: str | Path) -> Path:
    rows = [
        {
            "time_ps": float(result.time_ps[index]),
            "stage": result.stage[index],
            "branch": result.branch[index],
            "pressure_GPa": float(result.pressure_GPa[index]),
            "volume_A3": float(result.volume_A3[index]),
            "volume_ratio": float(result.volume_ratio[index]),
            "density_g_cm3": (
                float(result.density_g_cm3[index])
                if math.isfinite(float(result.density_g_cm3[index]))
                else ""
            ),
        }
        for index in range(len(result.time_ps))
    ]
    fields = [
        "time_ps",
        "stage",
        "branch",
        "pressure_GPa",
        "volume_A3",
        "volume_ratio",
        "density_g_cm3",
    ]
    return _atomic_csv(Path(path), fields, rows)


STAGE_THERMO_FIELDS = [
    "stage_order",
    "stage",
    "kind",
    "pv_branch",
    "target_pressure_GPa",
    "duration_ps",
    "sampling_window_ps",
    "sampling_start_stage_step",
    "selection_method",
    "n_samples",
    "n_pressure",
    "n_volume",
    "n_temperature",
    "n_density",
    "first_sample_time_ps",
    "last_sample_time_ps",
    "realized_sample_span_ps",
    "pressure_mean_GPa",
    "pressure_std_GPa",
    "volume_mean_A3",
    "volume_std_A3",
    "temperature_mean_K",
    "temperature_std_K",
    "density_mean_g_cm3",
    "density_std_g_cm3",
    "pressure_source",
    "available_rows",
    "dropped_invalid_rows",
    "status",
]


def write_stage_thermo_csv(
    result: StageThermoSummaryResult,
    path: str | Path,
) -> Path:
    """Write one unified P/V/T/density mean-and-SD table for every stage."""

    rows: list[dict[str, Any]] = []
    for summary in result.summaries:
        row = summary.to_dict()
        row["pv_branch"] = row.pop("branch")
        rows.append(row)
    return _atomic_csv(Path(path), STAGE_THERMO_FIELDS, rows)


def _stage_error(value: float | None) -> float:
    return 0.0 if value is None or not math.isfinite(float(value)) else float(value)


def _plot_stage_mean_sd(
    result: StageThermoSummaryResult,
    path: str | Path,
    *,
    x_mean: str,
    x_std: str,
    xlabel: str,
    title: str,
) -> Path:
    """Plot pressure against one stage-averaged observable with x/y SD bars."""

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    plotted = 0
    ordered = sorted(result.summaries, key=lambda summary: summary.stage_order)
    path_x = np.asarray(
        [
            np.nan if getattr(summary, x_mean) is None else float(getattr(summary, x_mean))
            for summary in ordered
        ],
        dtype=float,
    )
    path_y = np.asarray(
        [
            np.nan
            if summary.pressure_mean_GPa is None
            else float(summary.pressure_mean_GPa)
            for summary in ordered
        ],
        dtype=float,
    )
    if np.any(np.isfinite(path_x) & np.isfinite(path_y)):
        axis.plot(
            path_x,
            path_y,
            color="0.4",
            linewidth=1.25,
            zorder=1,
            label="stage sequence",
        )
    branch_order = list(dict.fromkeys(summary.branch for summary in ordered))
    for branch in branch_order:
        selected = [
            summary
            for summary in ordered
            if summary.branch == branch
            and getattr(summary, x_mean) is not None
            and summary.pressure_mean_GPa is not None
        ]
        if not selected:
            continue
        x = np.asarray([float(getattr(item, x_mean)) for item in selected])
        y = np.asarray([float(item.pressure_mean_GPa) for item in selected])
        xerr = np.asarray([_stage_error(getattr(item, x_std)) for item in selected])
        yerr = np.asarray([_stage_error(item.pressure_std_GPa) for item in selected])
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=5.0,
            linewidth=0.0,
            elinewidth=1.0,
            capsize=3.0,
            color=BRANCH_COLORS.get(branch, "0.4"),
            label=branch,
            zorder=2,
        )
        plotted += len(selected)
    if not plotted:
        plt.close(fig)
        raise AnalysisError("No finite stage means are available for the requested plot")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean total pressure (GPa)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    axis.set_title(title)
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def plot_pv_stage_mean_sd(
    result: StageThermoSummaryResult,
    path: str | Path,
) -> Path:
    return _plot_stage_mean_sd(
        result,
        path,
        x_mean="volume_mean_A3",
        x_std="volume_std_A3",
        xlabel=r"Mean volume ($\AA^3$)",
        title="Pressure–volume path: stage mean ± 1 sample SD",
    )


def plot_pt_stage_mean_sd(
    result: StageThermoSummaryResult,
    path: str | Path,
) -> Path:
    return _plot_stage_mean_sd(
        result,
        path,
        x_mean="temperature_mean_K",
        x_std="temperature_std_K",
        xlabel="Mean temperature (K)",
        title="Pressure–temperature path: stage mean ± 1 sample SD",
    )


def write_oxygen_speciation_csv(
    result: OxygenSpeciationResult,
    path: str | Path,
) -> Path:
    fields = [
        "step",
        "time_fs",
        "time_ps",
        "stage",
        "stage_step",
        "si_o_cutoff_A",
        "cutoff_source",
        "n_oxygen",
        "n_nbo",
        "n_bo",
        "n_other",
        "fraction_nbo",
        "fraction_bo",
        "fraction_other",
    ]
    return _atomic_csv(
        Path(path),
        fields,
        [frame.to_dict() for frame in result.frames],
    )


def _rdf_key_for_labels(result: RDFResult, left: str, right: str) -> str | None:
    for key, pair in result.pair_symbols.items():
        if pair == (left, right) or pair == (right, left):
            return key
    return None


def plot_oxygen_topology_rdf(
    result: RDFResult,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot NBO–NBO, NBO–BO, and BO–BO curves together."""

    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    colors = {"NBO-NBO": "#d62728", "NBO-BO": "#9467bd", "BO-BO": "#1f77b4"}
    plotted = 0
    for left, right in (("NBO", "NBO"), ("NBO", "BO"), ("BO", "BO")):
        key = _rdf_key_for_labels(result, left, right)
        if key is None:
            continue
        label = f"{left}-{right}"
        axis.plot(result.r_A, result.g[key], label=label, color=colors[label])
        plotted += 1
    if not plotted:
        plt.close(fig)
        raise AnalysisError("No NBO/BO oxygen-pair RDF curves are available")
    axis.set_xlabel(r"$r$ ($\AA$)")
    axis.set_ylabel(r"$g(r)$")
    axis.set_xlim(0.0, result.rmax_A)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title(title or "Oxygen-topology RDF")
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def plot_cation_oxygen_topology_rdf(
    result: RDFResult,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot each cation's NBO and BO curves on the same panel."""

    topology_labels = {"NBO", "BO", "O_other", "ALL"}
    cations: list[str] = []
    for pair in result.pair_symbols.values():
        for label in pair:
            if label not in topology_labels and label not in cations:
                cations.append(label)
    preferred = [item for item in ("Ca", "Si") if item in cations]
    cations = preferred + sorted(set(cations) - set(preferred))
    if not cations:
        raise AnalysisError("No cation–NBO/BO RDF curves are available")
    fig, axes = plt.subplots(
        1,
        len(cations),
        figsize=(5.1 * len(cations), 4.2),
        squeeze=False,
    )
    plotted = 0
    for axis, cation in zip(axes.flat, cations, strict=True):
        panel_curves = 0
        for oxygen, color in (("NBO", "#d62728"), ("BO", "#1f77b4")):
            key = _rdf_key_for_labels(result, cation, oxygen)
            if key is None:
                continue
            axis.plot(result.r_A, result.g[key], label=f"{cation}-{oxygen}", color=color)
            panel_curves += 1
            plotted += 1
        axis.set_xlabel(r"$r$ ($\AA$)")
        axis.set_ylabel(r"$g(r)$")
        axis.set_xlim(0.0, result.rmax_A)
        axis.grid(alpha=0.25)
        axis.set_title(f"{cation}–oxygen topology")
        if panel_curves:
            axis.legend()
        else:
            axis.set_visible(False)
    if not plotted:
        plt.close(fig)
        raise AnalysisError("No cation–NBO/BO RDF curves are available")
    fig.suptitle(title or "Cation–NBO/BO radial distribution functions")
    fig.tight_layout()
    return _save_figure(fig, Path(path))


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AnalysisError(f"Could not parse stored configuration {path}: {exc}") from exc
    return dict(value) if isinstance(value, Mapping) else {}


def _discover_config(
    output_dir: Path, explicit: str | Path | Mapping[str, Any] | None
) -> dict[str, Any]:
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if explicit is not None:
        return _read_mapping(Path(explicit))
    for name in (
        "resolved-config.yaml",
        "resolved_config.yaml",
        "config.resolved.yaml",
        "config.yaml",
    ):
        candidate = output_dir / name
        if candidate.exists():
            return _read_mapping(candidate)
    for name in ("run-manifest.json", "manifest.json"):
        candidate = output_dir / name
        if not candidate.exists():
            continue
        manifest = _read_mapping(candidate)
        # Some imported/legacy manifests embed a configuration directly.
        if _protocol_section(manifest):
            return manifest
        configured_path = manifest.get("config_path")
        if configured_path is None:
            continue
        referenced = Path(str(configured_path)).expanduser()
        if not referenced.is_absolute():
            referenced = output_dir / referenced
        referenced = referenced.resolve()
        # A run manifest is data, not an instruction to read arbitrary files.
        # Current ColdMD manifests always point to a sibling resolved config.
        if referenced.parent == output_dir.resolve() and referenced.exists():
            return _read_mapping(referenced)
    return {}


def _protocol_section(config: Mapping[str, Any]) -> dict[str, Any]:
    """Locate only an explicitly named protocol section.

    Recursive key search is unsafe here because calculator factory ``kwargs`` may
    legitimately contain unrelated ``timestep_fs`` or ``stages`` keys.
    """

    protocol = config.get("protocol")
    if isinstance(protocol, Mapping):
        return _expanded_protocol_section(protocol)
    for wrapper_name in ("resolved_config", "config"):
        wrapper = config.get(wrapper_name)
        if isinstance(wrapper, Mapping) and isinstance(wrapper.get("protocol"), Mapping):
            return _expanded_protocol_section(wrapper["protocol"])
    return {}


def _expanded_protocol_section(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize pressure-range shorthand in an explicit analysis override."""

    result = dict(protocol)
    stages = result.get("stages")
    if isinstance(stages, Sequence) and not isinstance(stages, (str, bytes)):
        try:
            result["stages"] = expand_pressure_ranges(stages)
        except (TypeError, ValueError) as exc:
            raise AnalysisError(
                f"Could not expand pressure_range in analysis configuration: {exc}"
            ) from exc
    return result


def _output_section(config: Mapping[str, Any]) -> dict[str, Any]:
    """Locate only an explicitly named output section."""

    output = config.get("output")
    if isinstance(output, Mapping):
        return dict(output)
    for wrapper_name in ("resolved_config", "config"):
        wrapper = config.get(wrapper_name)
        if isinstance(wrapper, Mapping) and isinstance(wrapper.get("output"), Mapping):
            return dict(wrapper["output"])
    return {}


def _plain_output_filename(value: Any, *, field: str, default: str) -> str:
    """Validate a stored output filename before resolving it below a run directory."""

    if value is None:
        return default
    filename = str(value)
    path = Path(filename)
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or path.name != filename
        or filename in {".", ".."}
    ):
        raise AnalysisError(f"output.{field} must be a plain filename, not a path")
    return filename


def _configured_sampling_windows(protocol: Mapping[str, Any]) -> dict[str, float]:
    """Return per-stage tail windows from a resolved protocol configuration."""

    stages = protocol.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return {}
    windows: dict[str, float] = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        name = stage.get("name")
        window = stage.get("sampling_window_ps")
        if name is None or window is None:
            continue
        try:
            parsed = float(window)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            windows[str(name)] = parsed
    return windows


def _configured_stage_kinds(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Return explicit configuration kinds keyed by stage name."""

    stages = protocol.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return {}
    result: dict[str, str] = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        name, kind = stage.get("name"), stage.get("kind")
        if name is not None and kind is not None:
            result[str(name)] = str(kind)
    return result


def _is_nvt_kind(kind: str | None, records: Sequence[Any]) -> bool:
    if kind in {"nvt_hold", "nvt", "initial_nvt", "peak_nvt", "final_nvt"}:
        return True
    observed = {record.stage_kind for record in records if record.stage_kind is not None}
    return observed == {"fixed_nvt"}


def _time_series_observable(kind: str | None, records: Sequence[Any]) -> str | None:
    if _is_nvt_kind(kind, records):
        return "pressure_GPa"
    if kind in {
        "npt_hold",
        "npt",
        "pressure_plateau",
        "recovery_npt",
        "volume_ramp",
        "compression",
        "decompression",
    }:
        return "volume_A3"
    observed = {record.stage_kind for record in records if record.stage_kind is not None}
    if observed == {"isotropic_npt_plateau"} or observed.intersection(
        {"cold_compression", "decompression"}
    ):
        return "volume_A3"
    return None


def _sampling_tail(
    records: Sequence[Any],
    *,
    window_ps: float | None,
    timestep_fs: float | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Select the configured tail using stored physical time or MD step metadata."""

    available = list(records)
    if not available:
        raise AnalysisError("Cannot select a sampling window from an empty stage")
    selected = available
    method = "full_stage"
    realized_span_ps: float | None = None

    if window_ps is not None:
        times_fs = np.asarray(
            [
                np.nan if record.time_fs is None else float(record.time_fs)
                for record in available
            ],
            dtype=float,
        )
        if np.all(np.isfinite(times_fs)):
            cutoff = float(times_fs[-1] - 1000.0 * window_ps)
            selected = [
                record
                for record, time_fs in zip(available, times_fs, strict=True)
                if time_fs >= cutoff - 1.0e-9
            ]
            method = "stored_time_fs_tail"
            if len(selected) > 1:
                realized_span_ps = (
                    float(selected[-1].time_fs) - float(selected[0].time_fs)
                ) / 1000.0
        elif timestep_fs is not None and all(
            record.stage_step is not None for record in available
        ):
            final_step = int(available[-1].stage_step)
            step_window = 1000.0 * window_ps / timestep_fs
            cutoff = final_step - step_window
            selected = [
                record
                for record in available
                if float(record.stage_step) >= cutoff - 1.0e-9
            ]
            method = "stored_stage_step_tail"
            if len(selected) > 1:
                realized_span_ps = (
                    (int(selected[-1].stage_step) - int(selected[0].stage_step))
                    * timestep_fs
                    / 1000.0
                )
        elif timestep_fs is not None and all(record.step is not None for record in available):
            final_step = int(available[-1].step)
            step_window = 1000.0 * window_ps / timestep_fs
            cutoff = final_step - step_window
            selected = [
                record for record in available if float(record.step) >= cutoff - 1.0e-9
            ]
            method = "stored_global_step_tail"
            if len(selected) > 1:
                realized_span_ps = (
                    (int(selected[-1].step) - int(selected[0].step))
                    * timestep_fs
                    / 1000.0
                )
        else:
            raise AnalysisError(
                "sampling_window_ps is configured, but trajectory time/step metadata "
                "are insufficient to select its tail"
            )
        if not selected:
            # This should be impossible because the final frame is inside its own tail.
            selected = [available[-1]]

    first = selected[0]
    last = selected[-1]
    selection = {
        "configured_sampling_window_ps": window_ps,
        "selection_method": method,
        "available_frames": len(available),
        "selected_frames": len(selected),
        "first_global_step": first.step,
        "last_global_step": last.step,
        "first_stage_step": first.stage_step,
        "last_stage_step": last.stage_step,
        "first_time_ps": None if first.time_fs is None else first.time_fs / 1000.0,
        "last_time_ps": None if last.time_fs is None else last.time_fs / 1000.0,
        "realized_sample_span_ps": realized_span_ps,
    }
    return selected, selection


def _group_by_stage(dataset: TrajectoryDataset) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for record in dataset.records:
        groups[record.stage].append(record)
    return dict(groups)


def _selected_structural_stages(
    groups: Mapping[str, Sequence[Any]], requested: Sequence[str] | None
) -> list[str]:
    """Choose representative structures without requiring legacy stage kinds.

    Legacy semantic names take precedence when present.  A freely named generic
    sequence falls back to its first stage, observed minimum-volume stage, and
    last stage, retaining YAML/trajectory order and removing duplicates.
    """

    if requested is not None:
        absent = [stage for stage in requested if stage not in groups]
        if absent:
            raise AnalysisError(f"Requested analysis stages are absent: {absent}")
        return list(requested)

    stages = list(groups)
    selected: list[str] = []
    initial = next((stage for stage in stages if "initial" in stage.lower()), stages[0])
    selected.append(initial)
    peak_names = [
        stage
        for stage in stages
        if any(token in stage.lower() for token in ("peak", "high", "compressed"))
    ]
    if peak_names:
        selected.append(peak_names[-1])
    else:
        selected.append(
            min(
                stages,
                key=lambda stage: float(
                    np.mean([record.atoms.get_volume() for record in groups[stage]])
                ),
            )
        )
    final_names = [
        stage
        for stage in stages
        if any(token in stage.lower() for token in ("final", "aging", "recovered"))
    ]
    selected.append(final_names[-1] if final_names else stages[-1])
    return list(dict.fromkeys(selected))


def _coordination_from_rdf(
    records: Sequence[Any], rdf: RDFResult, *, max_frames: int | None = None
) -> CoordinationResult:
    """Use only pairs with a resolved RDF minimum in automatic reports."""

    cutoffs: dict[str, float] = {}
    pairs: list[tuple[str, str]] = []
    for key, symbols in rdf.pair_symbols.items():
        if key == "ALL-ALL":
            continue
        try:
            cutoff = estimate_first_minimum(rdf.r_A, rdf.g[key])
        except AnalysisError:
            continue
        left, right = symbols
        cutoffs[f"{left}-{right}"] = cutoff
        pairs.append((left, right))
        if left != right:
            pairs.append((right, left))
    if not pairs:
        raise AnalysisError(
            "No partial RDF contains a resolved first minimum; "
            "configure coordination cutoffs manually"
        )
    return coordination_distributions(
        records,
        pairs=pairs,
        cutoffs_A=cutoffs,
        rdf=rdf,
        max_frames=max_frames,
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _markdown_report(
    *,
    analysis_dir: Path,
    output_dir: Path,
    dataset: TrajectoryDataset,
    selected_stages: Sequence[str],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage_selections: Mapping[str, Mapping[str, Any]],
    thermo_summary: StageThermoSummaryResult | None,
    artifacts: Sequence[Path],
    warnings: Sequence[str],
) -> str:
    artifact_names = {_relative(path, analysis_dir): path for path in artifacts}
    lines = [
        "# Cold-compression MD offline analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Source output: `{output_dir}`",
        "",
        f"Trajectory frames: {len(dataset.records)} across {len(dataset.files)} segment(s)",
        "",
        "## Scope and interpretation",
        "",
        "Structural disorder and transport are reported separately; loss of long-range "
        "order alone does not establish diffusive atomic transport. Classifications apply "
        "only to the sampled fixed-cell interval and use the explicit status labels "
        "`diffusive`, `nondiffusive_on_timescale`, and `undetermined`.",
        "Displayed MSD curves use one origin at the first stored stage frame; diffusion "
        "estimates use a separate multi-origin fit with block uncertainty.",
        "",
        "## Selected stages",
        "",
        "| Stage | Selected/available frames | Tail window (ps) | Realized span (ps) "
        "| Selected step range | Mean volume (Å³) | Cell variation | Analyses |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for stage in selected_stages:
        result_names = ", ".join(stage_results.get(stage, {})) or "none"
        selection = stage_selections[stage]
        window = selection.get("configured_sampling_window_ps")
        window_text = "full stage" if window is None else f"{float(window):.6g}"
        realized = selection.get("realized_sample_span_ps")
        realized_text = "—" if realized is None else f"{float(realized):.6g}"
        first_step = selection.get("first_global_step")
        last_step = selection.get("last_global_step")
        step_range = "—" if first_step is None else f"{first_step}–{last_step}"
        lines.append(
            f"| `{stage}` | {selection['selected_frames']}/{selection['available_frames']} | "
            f"{window_text} | {realized_text} | {step_range} | "
            f"{selection['mean_volume_A3']:.6g} | "
            f"{selection['cell_relative_variation']:.3g} | {result_names} |"
        )

    lines.extend(["", "## Transport estimates", ""])
    transport_rows = 0
    for stage in selected_stages:
        msd = stage_results.get(stage, {}).get("msd_full_stage")
        if not isinstance(msd, MSDResult):
            continue
        if transport_rows == 0:
            lines.extend(
                [
                    "| Stage | Species | Status | D (Å²/ps) | Uncertainty (SE) | Fit R² |",
                    "|---|---|---|---:|---:|---:|",
                ]
            )
        for element, estimate in msd.diffusion.items():
            coefficient = (
                "—" if estimate.coefficient_A2_ps is None else f"{estimate.coefficient_A2_ps:.4g}"
            )
            uncertainty = (
                "—"
                if estimate.standard_error_A2_ps is None
                else f"{estimate.standard_error_A2_ps:.3g}"
            )
            r_squared = "—" if estimate.r_squared is None else f"{estimate.r_squared:.3f}"
            lines.append(
                f"| `{stage}` | {element} | `{estimate.status}` | {coefficient} | "
                f"{uncertainty} | {r_squared} |"
            )
            transport_rows += 1
    if transport_rows == 0:
        lines.append(
            "No selected stage contained enough uniformly sampled fixed-cell frames for "
            "a transport estimate."
        )

    if thermo_summary is not None:
        lines.extend(
            [
                "",
                "## Stage pressure–volume–temperature statistics",
                "",
                "Values use each configured sampling-window tail. Uncertainty is "
                "±1 sample standard deviation (ddof=1), not a standard error.",
                "",
                "| Stage | Branch | n | P (GPa) | V (Å³) | T (K) | Density (g cm⁻³) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in thermo_summary.summaries:
            def mean_sd(mean: float | None, deviation: float | None) -> str:
                if mean is None:
                    return "—"
                if deviation is None:
                    return f"{mean:.6g} (SD unavailable)"
                return f"{mean:.6g} ± {deviation:.3g}"

            lines.append(
                f"| `{summary.stage}` | {summary.branch} | {summary.n_samples} | "
                f"{mean_sd(summary.pressure_mean_GPa, summary.pressure_std_GPa)} | "
                f"{mean_sd(summary.volume_mean_A3, summary.volume_std_A3)} | "
                f"{mean_sd(summary.temperature_mean_K, summary.temperature_std_K)} | "
                f"{mean_sd(summary.density_mean_g_cm3, summary.density_std_g_cm3)} |"
            )

    stage_rank = {_safe_name(stage): index for index, stage in enumerate(selected_stages)}

    def figure_order(name: str) -> tuple[int, int, int, str]:
        path = Path(name)
        if len(path.parts) == 1:
            global_rank = {
                "pv_hysteresis_mean_sd.png": 0,
            }.get(path.name, 10)
            return (0, global_rank, 0, name)
        file_rank = {
            "rdf.png": 0,
            "rdf_oxygen_topology_oxygen.png": 1,
            "rdf_oxygen_topology_cation.png": 2,
            "coordination.png": 3,
        }.get(path.name, 10)
        return (1, stage_rank.get(path.parts[0], len(stage_rank)), file_rank, name)

    pngs = sorted(
        (name for name in artifact_names if name.endswith(".png")),
        key=figure_order,
    )
    if pngs:
        lines.extend(["", "## Figures", ""])
        for name in pngs:
            lines.extend([f"### {Path(name).stem}", "", f"![{Path(name).stem}]({name})", ""])

    if warnings:
        lines.extend(["", "## Warnings and unavailable diagnostics", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "## Machine-readable files",
            "",
            "- `analysis.json`: complete inputs, criteria, arrays, statuses, and warnings",
            "- `thermo_stage_mean_sd.csv`: all-stage P/V/T/density means, sample SDs, and sampling provenance",
            "- Stage CSV files: elemental RDF first, then NBO/BO topology RDF and oxygen speciation, coordination, MSD, diffusion, and Fs",
            "",
        ]
    )
    return "\n".join(lines)


def generate_analysis_report(
    output_dir: str | Path,
    *,
    analysis_dir: str | Path | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    stages: Sequence[str] | None = None,
    all_stages: bool = False,
    si_o_cutoff_A: float | None = None,
    timestep_fs: float | None = None,
    frame_interval_fs: float | None = None,
    k_inv_A: float | None = None,
    rdf_bins: int = 300,
    max_structural_frames: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    """Create JSON, CSV, PNG, and Markdown analysis artifacts.

    Existing non-empty analysis directories are protected unless ``force=True``.
    RDF and coordination are produced for representative initial/high-density/final
    stages unless explicitly selected. Thermodynamic stage statistics always cover
    every configured stage. MSD and :math:`F_s(k,t)` are attempted only for
    fixed-cell stages.
    """

    if stages and all_stages:
        raise ValueError("stages and all_stages are mutually exclusive")
    if si_o_cutoff_A is not None and (
        not math.isfinite(float(si_o_cutoff_A)) or float(si_o_cutoff_A) <= 0.0
    ):
        raise ValueError("si_o_cutoff_A must be a finite positive distance")

    root = Path(output_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    destination = (
        Path(analysis_dir).resolve() if analysis_dir is not None else root / "analysis"
    )
    backup_dir: Path | None = None
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(
            f"Analysis directory is not empty: {destination}. Use force=True to replace artifacts."
        )
    if destination.exists() and any(destination.iterdir()) and force:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = destination.with_name(f"{destination.name}.backup-{timestamp}")
        suffix = 1
        while candidate.exists():
            candidate = destination.with_name(
                f"{destination.name}.backup-{timestamp}-{suffix:02d}"
            )
            suffix += 1
        destination.rename(candidate)
        backup_dir = candidate
    destination.mkdir(parents=True, exist_ok=True)

    stored_config = _discover_config(root, config)
    protocol_config = _protocol_section(stored_config)
    output_config = _output_section(stored_config)
    if timestep_fs is None:
        configured_timestep = protocol_config.get("timestep_fs")
        if configured_timestep is not None:
            timestep_fs = float(configured_timestep)
    sampling_windows = _configured_sampling_windows(protocol_config)
    stage_kinds = _configured_stage_kinds(protocol_config)

    configured_thermo_name = _plain_output_filename(
        output_config.get("thermo_filename"),
        field="thermo_filename",
        default="thermo.csv",
    )
    thermo_path = root / configured_thermo_name
    used_legacy_thermo_fallback = False
    if not thermo_path.exists() and configured_thermo_name != "thermo.csv":
        legacy_thermo = root / "thermo.csv"
        if legacy_thermo.exists():
            thermo_path = legacy_thermo
            used_legacy_thermo_fallback = True
    dataset = load_segmented_trajectories(
        root, thermo_csv=thermo_path if thermo_path.exists() else None
    )
    groups = _group_by_stage(dataset)
    selected = list(groups) if all_stages else _selected_structural_stages(groups, stages)
    warnings: list[str] = []
    if used_legacy_thermo_fallback:
        warnings.append(
            f"Configured thermodynamic file {configured_thermo_name!r} was absent; "
            "used legacy 'thermo.csv' instead."
        )
    artifacts: list[Path] = []
    results: dict[str, dict[str, Any]] = {}
    stage_selections: dict[str, dict[str, Any]] = {}
    diffusion_table: list[dict[str, Any]] = []

    for stage in selected:
        available_records = groups[stage]
        records, selection = _sampling_tail(
            available_records,
            window_ps=sampling_windows.get(stage),
            timestep_fs=timestep_fs,
        )
        selection["mean_volume_A3"] = float(
            np.mean([record.atoms.get_volume() for record in records])
        )
        selection["cell_relative_variation"] = cell_relative_variation(records)
        stage_selections[stage] = selection
        stage_key = _safe_name(stage)
        stage_dir = destination / stage_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        results[stage] = {}
        try:
            rdf = partial_rdf(
                records,
                bins=rdf_bins,
                include_total=True,
                max_frames=max_structural_frames,
            )
            results[stage]["rdf"] = rdf
            artifacts.append(write_rdf_csv(rdf, stage_dir / "rdf.csv"))
            artifacts.append(plot_rdf(rdf, stage_dir / "rdf.png", title=f"RDF — {stage}"))
        except (AnalysisError, ValueError) as exc:
            warnings.append(f"{stage}: RDF unavailable — {exc}")
            rdf = None

        if rdf is not None:
            try:
                coordination = _coordination_from_rdf(
                    records, rdf, max_frames=max_structural_frames
                )
                results[stage]["coordination"] = coordination
                artifacts.append(
                    write_coordination_csv(coordination, stage_dir / "coordination.csv")
                )
                artifacts.append(
                    plot_coordination(
                        coordination,
                        stage_dir / "coordination.png",
                        title=f"Coordination — {stage}",
                    )
                )
            except (AnalysisError, ValueError) as exc:
                warnings.append(f"{stage}: coordination unavailable — {exc}")

        # General elemental RDFs are deliberately written first. Oxygen
        # topology is a second, explicitly derived view whose NBO/BO labels are
        # recomputed for every frame rather than stored as chemical identities.
        try:
            speciation = oxygen_speciation(
                records,
                si_o_cutoff_A=si_o_cutoff_A,
                rdf_bins=rdf_bins,
                max_frames=max_structural_frames,
            )
            topology_rdf = topology_resolved_rdf(
                records,
                si_o_cutoff_A=speciation.cutoffs_A_by_stage,
                rdf_bins=rdf_bins,
                bins=rdf_bins,
                include_total=False,
                max_frames=max_structural_frames,
            )
            # Preserve the scientifically meaningful original source (automatic
            # first minimum or fixed user override) rather than the internal
            # stage-map reuse used to avoid estimating the cutoff twice.
            topology_rdf.metadata["oxygen_speciation"] = speciation.to_dict()
            results[stage]["oxygen_speciation"] = speciation
            results[stage]["oxygen_topology_rdf"] = topology_rdf
            artifacts.append(
                write_oxygen_speciation_csv(
                    speciation,
                    stage_dir / "oxygen_speciation.csv",
                )
            )
            artifacts.append(
                write_rdf_csv(
                    topology_rdf,
                    stage_dir / "rdf_oxygen_topology.csv",
                )
            )
            artifacts.append(
                plot_oxygen_topology_rdf(
                    topology_rdf,
                    stage_dir / "rdf_oxygen_topology_oxygen.png",
                    title=f"NBO/BO oxygen RDF — {stage}",
                )
            )
            artifacts.append(
                plot_cation_oxygen_topology_rdf(
                    topology_rdf,
                    stage_dir / "rdf_oxygen_topology_cation.png",
                    title=f"Cation–NBO/BO RDF — {stage}",
                )
            )
        except (AnalysisError, ValueError) as exc:
            warnings.append(f"{stage}: NBO/BO topology RDF unavailable — {exc}")

        variation = cell_relative_variation(records)
        if variation > 1.0e-5:
            warnings.append(
                f"{stage}: cell variation {variation:.3g} exceeds the fixed-cell "
                "transport tolerance; Fs was not evaluated."
            )
            continue
        try:
            structure_factor: StructureFactorEstimate | None = None
            selected_k = k_inv_A
            if selected_k is None:
                structure_factor = estimate_structure_factor_peak(
                    records, max_frames=min(max_structural_frames, 50)
                )
                selected_k = structure_factor.k_inv_A
                results[stage]["structure_factor"] = structure_factor
                artifacts.append(
                    plot_structure_factor(
                        structure_factor,
                        stage_dir / "structure_factor.png",
                        title=f"Estimated unweighted S(q) — {stage}",
                    )
                )
            fs_result = self_intermediate_scattering(
                records,
                k_inv_A=selected_k,
                frame_interval_fs=frame_interval_fs,
                timestep_fs=timestep_fs,
            )
            if structure_factor is not None:
                fs_result.k_source = structure_factor.method
            results[stage]["self_intermediate_scattering"] = fs_result
            artifacts.append(
                write_fs_csv(
                    fs_result, stage_dir / "self_intermediate_scattering.csv"
                )
            )
            artifacts.append(
                plot_self_intermediate_scattering(
                    fs_result,
                    stage_dir / "self_intermediate_scattering.png",
                    title=f"Self-intermediate scattering — {stage}",
                )
            )
        except (AnalysisError, ValueError) as exc:
            warnings.append(f"{stage}: self-intermediate scattering unavailable — {exc}")

    # MSD is intentionally a full-stage diagnostic, never a sampling-tail
    # diagnostic.  Only explicitly fixed-cell NVT stages are eligible: for an
    # NPT or prescribed-ramp cell, affine box motion is not diffusion.
    for stage, full_records in groups.items():
        if not _is_nvt_kind(stage_kinds.get(stage), full_records):
            continue
        stage_key = _safe_name(stage)
        stage_dir = destination / stage_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        results.setdefault(stage, {})
        try:
            msd = species_msd(
                full_records,
                frame_interval_fs=frame_interval_fs,
                timestep_fs=timestep_fs,
            )
            results[stage]["msd_full_stage"] = msd
            diffusion_table.extend(diffusion_rows(stage, msd))
            artifacts.append(write_msd_csv(msd, stage_dir / "msd_full_stage.csv"))
            artifacts.append(
                plot_msd(
                    msd,
                    stage_dir / "msd_full_stage.png",
                    title=f"Full-stage MSD — {stage}",
                    sampling_window_ps=sampling_windows.get(stage),
                )
            )
            artifacts.append(
                plot_msd(
                    msd,
                    stage_dir / "msd_full_stage_loglog.png",
                    title=f"Full-stage MSD (log–log) — {stage}",
                    loglog=True,
                    sampling_window_ps=sampling_windows.get(stage),
                )
            )
            if len(msd.lag_time_ps) > 1:
                frame_dt = float(np.min(np.diff(msd.lag_time_ps)))
                if frame_dt > 0.02:
                    warnings.append(
                        f"{stage}: MSD frame spacing is {frame_dt:.4g} ps; it may "
                        "be too coarse to resolve the ballistic regime. Reduce "
                        "output.msd_interval_steps for a future run."
                    )
        except (AnalysisError, ValueError) as exc:
            warnings.append(f"{stage}: full-stage NVT MSD unavailable — {exc}")

    if diffusion_table:
        diffusion_fields = [
            "stage",
            "species",
            "status",
            "D_A2_ps",
            "D_cm2_s",
            "standard_error_A2_ps",
            "confidence95_low_A2_ps",
            "confidence95_high_A2_ps",
            "upper_bound_A2_ps",
            "r_squared",
            "fit_start_ps",
            "fit_end_ps",
            "observation_time_ps",
            "reason",
        ]
        artifacts.append(
            _atomic_csv(destination / "diffusion_estimates.csv", diffusion_fields, diffusion_table)
        )

    thermo_summary: StageThermoSummaryResult | None = None
    if thermo_path.exists():
        try:
            configured_stages = protocol_config.get("stages")
            thermo_summary = stage_thermo_statistics(
                thermo_path,
                protocol_stages=(
                    configured_stages
                    if isinstance(configured_stages, Sequence)
                    and not isinstance(configured_stages, (str, bytes))
                    else None
                ),
                timestep_fs=timestep_fs,
            )
            artifacts.append(
                write_stage_thermo_csv(
                    thermo_summary,
                    destination / "thermo_stage_mean_sd.csv",
                )
            )
            artifacts.append(
                plot_pv_stage_mean_sd(
                    thermo_summary,
                    destination / "pv_hysteresis_mean_sd.png",
                )
            )
            for summary in thermo_summary.summaries:
                missing_sd = [
                    label
                    for label, mean, deviation in (
                        ("pressure", summary.pressure_mean_GPa, summary.pressure_std_GPa),
                        ("volume", summary.volume_mean_A3, summary.volume_std_A3),
                        (
                            "temperature",
                            summary.temperature_mean_K,
                            summary.temperature_std_K,
                        ),
                        (
                            "density",
                            summary.density_mean_g_cm3,
                            summary.density_std_g_cm3,
                        ),
                    )
                    if mean is not None and deviation is None
                ]
                if summary.status == "missing_stage_data":
                    warnings.append(
                        f"{summary.stage}: configured stage has no thermodynamic samples."
                    )
                elif missing_sd:
                    warnings.append(
                        f"{summary.stage}: sample SD is unavailable for "
                        + ", ".join(missing_sd)
                        + " because fewer than two finite samples were selected."
                    )

            time_series = sampling_window_thermo_series(
                thermo_path,
                protocol_stages=(
                    configured_stages
                    if isinstance(configured_stages, Sequence)
                    and not isinstance(configured_stages, (str, bytes))
                    else None
                ),
                timestep_fs=timestep_fs,
            )
            for stage, series in time_series.items():
                observable = _time_series_observable(
                    stage_kinds.get(stage) or series.get("kind"),
                    groups.get(stage, []),
                )
                if observable is None:
                    warnings.append(
                        f"{stage}: no explicit NVT/NPT/ramp kind was available; "
                        "full-stage thermo plot was skipped."
                    )
                    continue
                stage_dir = destination / _safe_name(stage)
                stage_dir.mkdir(parents=True, exist_ok=True)
                results.setdefault(stage, {})["sampling_thermo"] = series
                artifacts.append(
                    write_sampling_thermo_csv(series, stage_dir / "sampling_thermo.csv")
                )
                suffix = "T_P" if observable == "pressure_GPa" else "T_V"
                artifacts.append(
                    plot_sampling_thermo(
                        series,
                        stage_dir / f"sampling_thermo_{suffix}.png",
                        observable=observable,
                        title=(
                            f"Full-stage T and {'P' if suffix == 'T_P' else 'V'} "
                            f"(sampling tail marked) — {stage}"
                        ),
                    )
                )
        except (AnalysisError, ValueError) as exc:
            warnings.append(f"Stage P/V/T/density summary unavailable — {exc}")
    else:
        warnings.append(
            f"Thermodynamic file {configured_thermo_name!r} is absent; "
            "stage P/V/T/density means and standard deviations were not evaluated."
        )

    payload: dict[str, Any] = {
        "schema_version": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_output_dir": str(root),
        "previous_analysis_backup": str(backup_dir) if backup_dir is not None else None,
        "trajectory_files": [str(path) for path in dataset.files],
        "trajectory_frame_count": len(dataset.records),
        "thermo_file": str(thermo_path) if thermo_path.exists() else None,
        "available_stages": dataset.stages(),
        "selected_stages": selected,
        "parameters": {
            "timestep_fs": timestep_fs,
            "frame_interval_fs": frame_interval_fs,
            "msd_interval_steps": output_config.get("msd_interval_steps"),
            "k_inv_A": k_inv_A,
            "rdf_bins": rdf_bins,
            "max_structural_frames": max_structural_frames,
            "structural_stage_mode": (
                "all" if all_stages else "explicit" if stages else "representative_three"
            ),
            "si_o_cutoff_A_override": si_o_cutoff_A,
            "fixed_cell_relative_tolerance": 1.0e-5,
            "msd_origin_mode": "single origin at first stored stage frame",
            "diffusion_estimator": "separate multi-origin block estimator",
            "transport_status_vocabulary": [
                "diffusive",
                "nondiffusive_on_timescale",
                "undetermined",
            ],
        },
        "stages": results,
        "stage_sampling": stage_selections,
        "thermo_stage_summary": thermo_summary,
        "warnings": warnings,
    }
    json_path = write_json(payload, destination / "analysis.json")
    artifacts.append(json_path)

    markdown = _markdown_report(
        analysis_dir=destination,
        output_dir=root,
        dataset=dataset,
        selected_stages=selected,
        stage_results=results,
        stage_selections=stage_selections,
        thermo_summary=thermo_summary,
        artifacts=artifacts,
        warnings=warnings,
    )
    report_path = destination / "report.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(markdown + "\n", encoding="utf-8")
    temporary.replace(report_path)
    artifacts.append(report_path)

    return {
        "analysis_dir": destination,
        "previous_analysis_backup": backup_dir,
        "report": report_path,
        "json": json_path,
        "artifacts": artifacts,
        "warnings": warnings,
        "results": results,
        "thermo_stage_summary": thermo_summary,
    }


# A shorter name for use by ``coldmd.api.analyze``.
analyze_output = generate_analysis_report


__all__ = [
    "analyze_output",
    "diffusion_rows",
    "generate_analysis_report",
    "plot_cation_oxygen_topology_rdf",
    "plot_coordination",
    "plot_hysteresis",
    "plot_msd",
    "plot_oxygen_topology_rdf",
    "plot_pt_stage_mean_sd",
    "plot_pv_stage_mean_sd",
    "plot_rdf",
    "plot_self_intermediate_scattering",
    "plot_structure_factor",
    "write_coordination_csv",
    "write_fs_csv",
    "write_hysteresis_csv",
    "write_msd_csv",
    "write_oxygen_speciation_csv",
    "write_rdf_csv",
    "write_stage_thermo_csv",
]
