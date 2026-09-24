import importlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


def import_analysis_with_minimal_ase():
    """Import pure analysis helpers when ASE is absent from the test host."""

    try:
        return importlib.import_module("coldmd.analysis")
    except ImportError as original:
        try:
            importlib.import_module("ase")
        except ImportError:
            pass
        else:
            raise original

    ase = types.ModuleType("ase")
    geometry = types.ModuleType("ase.geometry")
    io = types.ModuleType("ase.io")

    class Atoms:
        pass

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("minimal ASE test stub cannot perform geometry or I/O")

    ase.Atoms = Atoms
    geometry.find_mic = unavailable
    io.iread = unavailable
    sys.modules.update(
        {
            "ase": ase,
            "ase.geometry": geometry,
            "ase.io": io,
        }
    )
    sys.modules.pop("coldmd.analysis", None)
    return importlib.import_module("coldmd.analysis")


class CrossModuleContractTests(unittest.TestCase):
    def test_nvt_stage_locks_the_complete_cell_inherited_from_previous_stage(self):
        try:
            from ase import Atoms
            from coldmd.engine import (
                CallbackHook,
                FixedCellNVTStage,
                IsotropicNPTStage,
                ProtocolEngine,
            )
        except ImportError as exc:
            self.skipTest(f"ASE dynamics dependency is unavailable: {exc}")

        atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
        inherited_cell = np.asarray(
            [[9.0, 1.0, 0.0], [0.0, 8.0, 0.5], [0.0, 0.0, 7.0]],
            dtype=float,
        )
        stage_start_cells = []

        class OneStepDynamics:
            def __init__(self, action=None):
                self.action = action
                self.nsteps = 0

            def step(self):
                if self.action is not None:
                    self.action()

        stages = [
            IsotropicNPTStage(
                name="npt",
                steps=1,
                temperature_K=300.0,
                pressure_GPa=0.0,
                thermostat_tau_fs=100.0,
                barostat_tau_fs=1000.0,
                branch="other",
            ),
            FixedCellNVTStage(
                name="nvt",
                steps=1,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
                branch="cold_compression",
            ),
        ]
        hook = CallbackHook(
            stage_start=lambda _atoms, context: stage_start_cells.append(
                np.asarray(context.stage_reference_cell_A, dtype=float)
            )
        )
        engine = ProtocolEngine(atoms, timestep_fs=0.5, hooks=(hook,))

        def build_dynamics(stage, **_kwargs):
            action = (
                lambda: atoms.set_cell(inherited_cell, scale_atoms=True)
                if isinstance(stage, IsotropicNPTStage)
                else None
            )
            return OneStepDynamics(action)

        with patch.object(engine, "_build_dynamics", side_effect=build_dynamics):
            engine.run(stages)

        self.assertEqual(len(stage_start_cells), 2)
        np.testing.assert_allclose(stage_start_cells[1], inherited_cell)
        np.testing.assert_allclose(np.asarray(atoms.cell), inherited_cell)

    def test_extended_xyz_sidecar_uses_the_storage_writer_stem_convention(self):
        _load_segment_metadata = (
            import_analysis_with_minimal_ase()._load_segment_metadata
        )

        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary) / "trajectory-000000.extxyz"
            sidecar = Path(temporary) / "trajectory-000000.meta.jsonl"
            sidecar.write_text(
                json.dumps(
                    {
                        "segment": 0,
                        "frame": 0,
                        "metadata": {
                            "coldmd_step": 10,
                            "coldmd_stage": "arbitrary-stage-name",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metadata = _load_segment_metadata(segment)
            self.assertEqual(metadata[0]["coldmd_step"], 10)
            self.assertEqual(metadata[0]["coldmd_stage"], "arbitrary-stage-name")

    def test_user_facing_source_avoids_superseded_phase_process_terms(self):
        project = Path(__file__).resolve().parents[1]
        forbidden = ("m" + "elt", "liq" + "uid", "qu" + "ench")
        offenders = []
        roots = (project / "src", project / "examples", project / "docs")
        paths = [project / "README.md"]
        for root in roots:
            paths.extend(root.rglob("*"))
        for path in paths:
            if path.suffix.lower() not in {".py", ".yaml", ".yml"}:
                if path.suffix.lower() != ".md":
                    continue
            text = path.read_text(encoding="utf-8").lower()
            for term in forbidden:
                if term in text:
                    offenders.append(f"{path.relative_to(project)}: {term}")
        self.assertEqual(offenders, [])

    def test_checkpoint_cursor_rejects_lossy_numeric_coercion(self):
        try:
            from coldmd.engine import RunCursor
        except ImportError as exc:
            self.skipTest(f"ASE dynamics dependency is unavailable: {exc}")

        for invalid in (True, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    RunCursor.from_mapping({"stage_index": invalid})

    def test_cursor_round_trip_preserves_original_reference_volume(self):
        try:
            from coldmd.engine import RunCursor
        except ImportError as exc:
            self.skipTest(f"ASE dynamics dependency is unavailable: {exc}")

        original = RunCursor(
            stage_index=2,
            stage_step=11,
            global_step=211,
            time_fs=105.5,
            stage_reference_cell_A=(
                (8.0, 0.0, 0.0),
                (0.0, 8.0, 0.0),
                (0.0, 0.0, 8.0),
            ),
            reference_volume_A3=1000.0,
        )
        restored = RunCursor.from_mapping(original.as_dict())
        self.assertEqual(restored, original)

    def test_hysteresis_volume_ratio_is_global_not_stage_local(self):
        thermo_hysteresis = import_analysis_with_minimal_ase().thermo_hysteresis

        rows = [
            {
                "time_ps": 0.0,
                "stage": "s1",
                "branch": "cold_compression",
                "total_pressure_GPa": 0.0,
                "volume_A3": 100.0,
                "density_g_cm3": 2.0,
                "target_volume_ratio": 1.0,
            },
            {
                "time_ps": 1.0,
                "stage": "s1",
                "branch": "cold_compression",
                "total_pressure_GPa": 10.0,
                "volume_A3": 50.0,
                "density_g_cm3": 4.0,
                "target_volume_ratio": 0.5,
            },
            {
                "time_ps": 2.0,
                "stage": "s2",
                "branch": "decompression",
                "total_pressure_GPa": 10.0,
                "volume_A3": 50.0,
                "density_g_cm3": 4.0,
                "target_volume_ratio": 1.0,
            },
            {
                "time_ps": 3.0,
                "stage": "s2",
                "branch": "decompression",
                "total_pressure_GPa": 0.0,
                "volume_A3": 100.0,
                "density_g_cm3": 2.0,
                "target_volume_ratio": 2.0,
            },
        ]
        result = thermo_hysteresis(rows)
        self.assertEqual(result.branch, ["compression", "compression", "decompression", "decompression"])
        self.assertEqual(result.volume_ratio.tolist(), [1.0, 0.5, 0.5, 1.0])

    def test_hysteresis_prefers_persisted_engine_reference_ratio(self):
        thermo_hysteresis = import_analysis_with_minimal_ase().thermo_hysteresis

        # The first retained thermo row need not be the engine's V0 row (for
        # example, a legacy file may have been trimmed).  The explicit ratio
        # written by RunOutputHook must therefore win over volume/first-volume.
        rows = [
            {
                "time_ps": 2.0,
                "stage": "dense",
                "branch": "cold_compression",
                "total_pressure_GPa": 20.0,
                "volume_A3": 70.0,
                "volume_ratio_from_engine_start": 0.7,
                "density_g_cm3": 3.0,
            },
            {
                "time_ps": 3.0,
                "stage": "recovery",
                "branch": "recovery",
                "total_pressure_GPa": 0.0,
                "volume_A3": 90.0,
                "volume_ratio_from_engine_start": 0.9,
                "density_g_cm3": 2.4,
            },
        ]

        result = thermo_hysteresis(rows)
        self.assertEqual(result.volume_ratio.tolist(), [0.7, 0.9])
        self.assertEqual(result.branch, ["compression", "recovery"])

    def test_trajectory_discovery_excludes_stage_structure_snapshots(self):
        discover_trajectory_files = (
            import_analysis_with_minimal_ase().discover_trajectory_files
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trajectory = run_dir / "frames-000000.extxyz"
            trajectory.touch()
            stage_dir = run_dir / "stage_structures"
            stage_dir.mkdir()
            (stage_dir / "stage-000-initial-start-step-000.extxyz").touch()

            self.assertEqual(discover_trajectory_files(run_dir), [trajectory.resolve()])

    def test_namespaced_trajectory_metadata_has_provenance_precedence(self):
        load_segmented_trajectories = (
            import_analysis_with_minimal_ase().load_segmented_trajectories
        )

        class Frame:
            def __init__(self):
                self.info = {
                    "stage": "unrelated-generic-stage",
                    "time_fs": 999.0,
                    "stage_step": 999,
                    "coldmd_step": 7,
                    "coldmd_stage": "canonical-stage",
                    "coldmd_time_fs": 3.5,
                    "coldmd_stage_step": 2,
                }

            def copy(self):
                return self

        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary) / "trajectory-000000.traj"
            segment.touch()
            with patch("coldmd.analysis.iread", return_value=iter([Frame()])):
                dataset = load_segmented_trajectories([segment])

        record = dataset.records[0]
        self.assertEqual(record.stage, "canonical-stage")
        self.assertEqual(record.time_fs, 3.5)
        self.assertEqual(record.stage_step, 2)

    def test_reporting_protocol_lookup_cannot_be_shadowed_by_factory_kwargs(self):
        analysis_module = import_analysis_with_minimal_ase()
        from coldmd.reporting import (
            _configured_sampling_windows,
            _output_section,
            _plain_output_filename,
            _protocol_section,
        )

        config = {
            "calculator": {
                "kwargs": {
                    "timestep_fs": 99.0,
                    "stages": [
                        {"name": "wrong", "sampling_window_ps": 99.0}
                    ],
                }
            },
            "protocol": {
                "timestep_fs": 0.5,
                "stages": [
                    {"name": "canonical", "sampling_window_ps": 2.0}
                ],
            },
            "output": {"thermo_filename": "custom-thermo.csv"},
        }

        protocol = _protocol_section(config)
        self.assertEqual(protocol["timestep_fs"], 0.5)
        self.assertEqual(
            _configured_sampling_windows(protocol), {"canonical": 2.0}
        )
        output = _output_section(config)
        self.assertEqual(output["thermo_filename"], "custom-thermo.csv")
        self.assertEqual(
            _plain_output_filename(
                output["thermo_filename"],
                field="thermo_filename",
                default="thermo.csv",
            ),
            "custom-thermo.csv",
        )
        with self.assertRaises(analysis_module.AnalysisError):
            _plain_output_filename(
                "../escape.csv", field="thermo_filename", default="thermo.csv"
            )

    def test_sampling_window_selects_only_the_configured_stage_tail(self):
        import_analysis_with_minimal_ase()
        from coldmd.reporting import _sampling_tail

        @dataclass
        class Record:
            step: int
            time_fs: float
            stage_step: int

        records = [
            Record(step=index, time_fs=index * 500.0, stage_step=index)
            for index in range(7)
        ]
        selected, metadata = _sampling_tail(
            records,
            window_ps=1.0,
            timestep_fs=500.0,
        )

        self.assertEqual([record.step for record in selected], [4, 5, 6])
        self.assertEqual(metadata["selection_method"], "stored_time_fs_tail")
        self.assertEqual(metadata["realized_sample_span_ps"], 1.0)
        self.assertEqual(metadata["selected_frames"], 3)


if __name__ == "__main__":
    unittest.main()
