"""Contracts introduced by the stage-explicit ColdMD v2 workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


def v2_config() -> dict[str, object]:
    return {
        "config_version": 2,
        "input": {"cif_path": "input.cif", "expected_atom_count": 2},
        "calculator": {
            "kind": "factory",
            "factory": "package.module:create_calculator",
        },
        "protocol": {
            "temperature_K": 300.0,
            "timestep_fs": 1.0,
            "initialize_velocities": True,
            "thermostat": {"kind": "bussi", "tau_fs": 100.0},
            "stages": [
                {
                    "kind": "npt_hold",
                    "name": "equilibrate",
                    "branch": "compression",
                    "purpose": "equilibration",
                    "target_pressure_GPa": 2.0,
                    "duration_ps": 2.0,
                    "sampling_window_ps": 1.0,
                },
                {
                    "kind": "nvt_hold",
                    "name": "sample_fixed_cell",
                    "branch": "compression",
                    "purpose": "sampling",
                    "duration_ps": 2.0,
                    "sampling_window_ps": 1.0,
                },
                {
                    "kind": "volume_ramp",
                    "name": "driven_unload",
                    "branch": "decompression",
                    "purpose": "transition",
                    "target_volume_ratio": 1.05,
                    "duration_ps": 2.0,
                    "sampling_window_ps": 1.0,
                },
            ],
        },
    }


class V2ConfigurationTests(unittest.TestCase):
    def test_v2_is_stage_explicit_and_round_trips_without_control_mode(self) -> None:
        from coldmd.api import _build_stage_specs
        from coldmd.config import dump_config, load_config, load_config_dict

        config = load_config_dict(v2_config())
        stages = _build_stage_specs(config)
        self.assertEqual(
            [stage.kind for stage in stages],
            ["isotropic_npt_plateau", "fixed_nvt", "decompression"],
        )
        self.assertEqual([stage.handoff_window_steps for stage in stages], [1000] * 3)

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "resolved-v2.yaml"
            dump_config(config, path)
            self.assertNotIn("control_mode", path.read_text(encoding="utf-8"))
            reloaded = load_config(path)
        self.assertEqual(reloaded.config_version, 2)
        self.assertEqual([stage.kind for stage in reloaded.protocol.stages], [
            "npt_hold",
            "nvt_hold",
            "volume_ramp",
        ])

    def test_v2_rejects_legacy_global_mode_and_stage_kinds(self) -> None:
        from pydantic import ValidationError

        from coldmd.config import load_config_dict

        with_mode = deepcopy(v2_config())
        with_mode["protocol"]["control_mode"] = "stage_sequence"  # type: ignore[index]
        with self.assertRaisesRegex(ValidationError, "does not accept protocol.control_mode"):
            load_config_dict(with_mode)

        with_legacy_stage = deepcopy(v2_config())
        legacy_stage = with_legacy_stage["protocol"]["stages"][0]  # type: ignore[index]
        legacy_stage["kind"] = "npt"
        legacy_stage.pop("branch")
        legacy_stage.pop("purpose")
        with self.assertRaisesRegex(ValidationError, "accepts only nvt_hold"):
            load_config_dict(with_legacy_stage)


class V2AnalysisTests(unittest.TestCase):
    def test_stage_series_retains_full_duration_and_marks_sampling_tail(self) -> None:
        from coldmd.analysis import sampling_window_thermo_series

        rows = [
            {
                "step": step,
                "stage": "fixed",
                "stage_step": step,
                "time_fs": step * 500.0,
                "temperature_K": 300.0 + step,
                "total_pressure_GPa": 1.0 + step,
                "volume_A3": 100.0,
            }
            for step in range(5)
        ]
        series = sampling_window_thermo_series(
            rows,
            protocol_stages=[
                {
                    "name": "fixed",
                    "kind": "nvt_hold",
                    "sampling_window_ps": 1.0,
                    "branch": "compression",
                }
            ],
            timestep_fs=500.0,
        )["fixed"]
        self.assertEqual(series["time_ps"], [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(
            series["in_sampling_window"], [False, False, True, True, True]
        )
        self.assertEqual(series["sampling_cutoff_ps"], 1.0)
        self.assertEqual(
            series["temperature_K"], [300.0, 301.0, 302.0, 303.0, 304.0]
        )
        self.assertEqual(series["pressure_GPa"], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_published_msd_is_single_origin_and_spans_full_stage(self) -> None:
        try:
            from ase import Atoms
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE is unavailable: {exc}")

        import matplotlib.pyplot as plt

        from coldmd.analysis import FrameRecord, species_msd
        from coldmd.reporting import plot_msd

        records = []
        for index in range(12):
            atoms = Atoms(
                "Cu",
                positions=[[0.01 * index**2, 0.0, 0.0]],
                cell=np.eye(3) * 20.0,
                pbc=True,
            )
            records.append(
                FrameRecord(
                    atoms,
                    Path(f"frame-{index}.traj"),
                    0,
                    index,
                    index,
                    index * 1000.0,
                    "fixed",
                    index,
                    "fixed_nvt",
                )
            )

        result = species_msd(records, maximum_lag_fraction=0.5)

        self.assertEqual(result.origin_mode, "single origin: first stored frame of the stage")
        self.assertIn("multi-origin", result.diffusion_estimator)
        self.assertEqual(result.lag_time_ps[-1], 11.0)
        self.assertAlmostEqual(result.msd_A2["Cu"][2], 0.04**2)

        captured: list[tuple[int, list[float], list[str]]] = []

        def inspect(fig, path):  # type: ignore[no-untyped-def]
            axis = fig.axes[0]
            cutoff_lines = [
                line
                for line in axis.lines
                if line.get_linestyle() == "--"
            ]
            captured.append(
                (
                    len(axis.patches),
                    [float(line.get_xdata()[0]) for line in cutoff_lines],
                    [text.get_text() for text in axis.texts],
                )
            )
            plt.close(fig)
            return Path(path)

        with patch("coldmd.reporting._save_figure", side_effect=inspect):
            plot_msd(result, "linear.png", sampling_window_ps=3.0)
            plot_msd(result, "loglog.png", loglog=True, sampling_window_ps=3.0)

        self.assertEqual([item[0] for item in captured], [1, 1])
        self.assertEqual([item[1] for item in captured], [[8.0], [8.0]])
        self.assertTrue(
            all(
                "shaded: excluded from sampling statistics" in item[2]
                for item in captured
            )
        )


class V2HandoffTests(unittest.TestCase):
    def test_next_stage_receives_actual_snapshot_nearest_tail_mean_volume(self) -> None:
        try:
            from ase import Atoms
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE is unavailable: {exc}")

        from coldmd.engine import (
            BaseDynamicsHook,
            FixedCellNVTStage,
            ProtocolEngine,
            VolumeRampStage,
        )

        atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.0, pbc=True)
        starts: list[tuple[str, float]] = []
        handoffs: list[dict[str, object]] = []

        class Recorder(BaseDynamicsHook):
            def on_stage_start(self, observed, context):  # type: ignore[no-untyped-def]
                starts.append((context.stage_name, float(observed.get_volume())))

            def on_stage_handoff(self, observed, context, handoff):  # type: ignore[no-untyped-def]
                handoffs.append(dict(handoff))

        class FakeDynamics:
            def __init__(self, observed, stage_name):  # type: ignore[no-untyped-def]
                self.atoms = observed
                self.stage_name = stage_name
                self.nsteps = 0

            def step(self) -> None:
                if self.stage_name == "driven":
                    # The tail volumes are 125, 64, 27, and 8 A^3.  Their
                    # mean is 56 A^3, so the mechanically complete 64 A^3
                    # snapshot is the deterministic selected handoff state.
                    side = (4.0, 3.0, 2.0)[self.nsteps]
                    self.atoms.set_cell(np.eye(3) * side, scale_atoms=True)

        def fake_build(self, stage, **_):  # type: ignore[no-untyped-def]
            return FakeDynamics(self.atoms, stage.name)

        stages = [
            VolumeRampStage(
                name="driven",
                role="cold_compression",
                steps=3,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
                target_volume_ratio=0.5,
                handoff_window_steps=3,
                handoff_sample_interval_steps=1,
            ),
            FixedCellNVTStage(
                name="inherited",
                steps=1,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
            ),
        ]
        engine = ProtocolEngine(atoms, timestep_fs=1.0, hooks=(Recorder(),))
        with patch.object(ProtocolEngine, "_build_dynamics", new=fake_build):
            cursor = engine.run(stages)

        self.assertEqual(cursor.global_step, 4)
        self.assertEqual(starts[1][0], "inherited")
        self.assertAlmostEqual(starts[1][1], 64.0)
        self.assertEqual(handoffs[0]["candidate_count"], 4)
        self.assertAlmostEqual(float(handoffs[0]["selected_volume_A3"]), 64.0)
        self.assertEqual(
            handoffs[0]["selection"],
            "sampling_window_volume_nearest_mean_snapshot",
        )

    def test_handoff_checkpoint_rebases_safety_history_to_selected_cell(self) -> None:
        try:
            from ase import Atoms
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE is unavailable: {exc}")

        from coldmd.safety import SafetyLimits, SafetyMonitor
        from coldmd.storage import RunOutputHook

        atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
        context = {
            "stage_index": 0,
            "stage_name": "equilibrate",
            "stage_kind": "isotropic_npt_plateau",
            "stage_step": 4,
            "stage_steps": 4,
            "global_step": 4,
            "time_fs": 4.0,
            "stage_reference_cell_A": np.eye(3).tolist(),
        }
        handoff = {
            "from_stage": "equilibrate",
            "to_stage": "sample",
            "selected_volume_A3": 64.0,
        }
        with TemporaryDirectory() as temporary:
            hook = RunOutputHook(
                temporary,
                safety_monitor=SafetyMonitor(
                    SafetyLimits(max_abs_log_volume_strain_per_step=1.0e-3),
                    reference_volume_A3=64.0,
                ),
                rng=np.random.default_rng(1),
                config_hash="a" * 64,
                model_hash="b" * 64,
                reference_volume_A3=64.0,
                write_stage_structures=False,
            )
            try:
                with hook:
                    hook.on_stage_handoff(atoms, context, handoff)
                saved = hook.load_latest_checkpoint()
            finally:
                hook.close()

        state = saved.protocol_state["safety_monitor"]
        self.assertEqual(state["previous_step"], 4)
        np.testing.assert_allclose(state["previous_cell"], np.eye(3) * 4.0)
        self.assertTrue(
            saved.metadata["stage_handoff"]
            ["safety_history_rebased_to_selected_snapshot"]
        )


if __name__ == "__main__":
    unittest.main()
