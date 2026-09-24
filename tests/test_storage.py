import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from coldmd.observables import ThermoRecord
from coldmd.safety import SafetyLimits, SafetyMonitor
from coldmd.storage import (
    CheckpointCompatibilityError,
    CheckpointManager,
    RunOutputHook,
    collect_version_metadata,
)


class FakeAtoms:
    def __init__(self):
        self.numbers = np.array([14, 8], dtype=np.int64)
        self.positions = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
        self.cell = np.diag([8.0, 9.0, 10.0])
        self.pbc = np.array([True, True, True])
        self.momenta = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        self.calc = object()

    def __len__(self):
        return len(self.numbers)

    def has(self, name):
        return name == "momenta"

    def get_atomic_numbers(self):
        return self.numbers.copy()

    def get_positions(self):
        return self.positions.copy()

    def get_cell(self):
        return self.cell.copy()

    def get_pbc(self):
        return self.pbc.copy()

    def get_momenta(self):
        return self.momenta.copy()

    def set_cell(self, value, scale_atoms=False):
        if scale_atoms:
            raise AssertionError("checkpoint restore must not affinely rescale positions")
        self.cell = np.asarray(value, dtype=float).copy()

    def set_pbc(self, value):
        self.pbc = np.asarray(value, dtype=bool).copy()

    def set_positions(self, value):
        self.positions = np.asarray(value, dtype=float).copy()

    def set_momenta(self, value):
        self.momenta = np.asarray(value, dtype=float).copy()


def thermo_record(*, step=5, stage="stage-a", temperature_K=600.0):
    return ThermoRecord(
        step=step,
        time_fs=step * 0.5,
        time_ps=step * 0.0005,
        stage=stage,
        stage_step=step,
        target_volume_ratio=None,
        target_pressure_GPa=None,
        temperature_K=temperature_K,
        potential_energy_eV=-1.0,
        kinetic_energy_eV=0.1,
        total_energy_eV=-0.9,
        volume_A3=720.0,
        density_g_cm3=1.0,
        stress_xx_GPa=0.0,
        stress_yy_GPa=0.0,
        stress_zz_GPa=0.0,
        stress_yz_GPa=0.0,
        stress_xz_GPa=0.0,
        stress_xy_GPa=0.0,
        configurational_pressure_GPa=0.0,
        total_pressure_GPa=0.0,
        deviatoric_stress_GPa=0.0,
        max_force_eV_A=1.0,
        rms_force_eV_A=0.5,
        min_distance_A=1.6,
    )


class CheckpointTests(unittest.TestCase):
    def test_version_metadata_fingerprints_source_and_dependency_closure(self):
        versions = collect_version_metadata()
        digest = versions.get("coldmd-source-sha256", "")
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))
        self.assertIn("dependency:numpy", versions)
        self.assertIn("platform", versions)

    def test_checkpoint_round_trip_restores_mechanics_protocol_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary) / "checkpoints")
            atoms = FakeAtoms()
            original_calc = atoms.calc
            rng = np.random.default_rng(12345)
            rng.random(3)
            protocol = {
                "cursor": {
                    "stage_index": 2,
                    "stage_step": 17,
                    "global_step": 117,
                    "time_fs": 58.5,
                },
                "tuple_value": (1, 2),
                "array_value": np.array([3, 4], dtype=np.int32),
            }
            manager.save(
                atoms,
                protocol,
                rng=rng,
                config_hash="a" * 64,
                model_hash="b" * 64,
                metadata={"purpose": "resume-test"},
                versions={"test-package": "1.2.3"},
            )
            expected_random = rng.random(5)

            checkpoint = manager.load(
                expected_config_hash="a" * 64,
                expected_model_hash="b" * 64,
                expected_versions={"test-package": "1.2.3"},
            )
            self.assertEqual(checkpoint.protocol_state["tuple_value"], (1, 2))
            np.testing.assert_array_equal(
                checkpoint.protocol_state["array_value"], np.array([3, 4])
            )

            atoms.positions[:] = 99.0
            atoms.cell[:] = np.eye(3)
            atoms.momenta[:] = 0.0
            checkpoint.apply_to(atoms)
            self.assertIs(atoms.calc, original_calc)
            np.testing.assert_allclose(
                atoms.positions, [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]
            )
            np.testing.assert_allclose(atoms.cell, np.diag([8.0, 9.0, 10.0]))

            restored_rng = np.random.default_rng()
            checkpoint.restore_rng(restored_rng)
            np.testing.assert_allclose(restored_rng.random(5), expected_random)

    def test_checkpoint_rejects_wrong_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(temporary)
            manager.save(
                FakeAtoms(),
                {"cursor": {}},
                rng=np.random.default_rng(1),
                config_hash="a" * 64,
                model_hash="b" * 64,
            )
            with self.assertRaises(CheckpointCompatibilityError):
                manager.load(expected_model_hash="c" * 64)

    def test_exact_version_set_rejects_saved_only_fingerprint_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(temporary)
            manager.save(
                FakeAtoms(),
                {"cursor": {}},
                rng=np.random.default_rng(1),
                config_hash="a" * 64,
                model_hash="b" * 64,
                versions={"saved-only-runtime": "1.0"},
            )
            checkpoint = manager.load()
            current = dict(checkpoint.versions)
            current.pop("saved-only-runtime")

            with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "fingerprint key set differs",
            ):
                manager.load(
                    expected_versions=current,
                    require_exact_version_set=True,
                )


class OutputHookTests(unittest.TestCase):
    def _resumed_hook(self, directory, **kwargs):
        return RunOutputHook(
            directory,
            safety_monitor=SafetyMonitor(SafetyLimits()),
            rng=np.random.default_rng(9),
            config_hash="a" * 64,
            model_hash="b" * 64,
            append=True,
            **kwargs,
        )

    def test_stage_structure_writes_extxyz_momenta_and_cif_companion(self):
        try:
            from ase import Atoms
            from ase.io import read
        except ImportError as exc:
            self.skipTest(f"ASE is unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temporary:
            atoms = Atoms(
                "Cu2",
                positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
                cell=np.eye(3) * 8.0,
                pbc=True,
            )
            expected_momenta = np.asarray(
                [[0.2, -0.1, 0.3], [-0.15, 0.05, -0.25]], dtype=float
            )
            atoms.set_momenta(expected_momenta)
            hook = RunOutputHook(
                temporary,
                safety_monitor=SafetyMonitor(SafetyLimits()),
                rng=np.random.default_rng(21),
                config_hash="a" * 64,
                model_hash="b" * 64,
            )
            try:
                pair = hook._write_stage_structure(
                    atoms,
                    {
                        "stage_index": 3,
                        "stage_name": "branch-ready",
                        "global_step": 125,
                    },
                    "handoff",
                )
            finally:
                hook.close()

            self.assertIsNotNone(pair)
            extxyz, cif = pair  # type: ignore[misc]
            self.assertTrue(extxyz.is_file())
            self.assertTrue(cif.is_file())
            self.assertEqual(extxyz.stem, cif.stem)
            restored = read(extxyz, format="extxyz")
            self.assertTrue(restored.has("momenta"))
            np.testing.assert_allclose(restored.get_momenta(), expected_momenta)
            viewed = read(cif, format="cif")
            self.assertEqual(viewed.get_chemical_symbols(), atoms.get_chemical_symbols())

    def test_resume_primes_each_durable_sink_from_its_actual_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            hook = self._resumed_hook(
                temporary,
                resume_step=100,
                last_thermo_step=98,
                last_trajectory_step=99,
            )
            try:
                self.assertEqual(hook._last_safety_step, 100)
                self.assertEqual(hook._last_checkpoint_step, 100)
                self.assertEqual(hook._last_thermo_step, 98)
                self.assertEqual(hook._last_trajectory_step, 99)
            finally:
                hook.close()

    def test_resume_does_not_assume_checkpoint_boundary_outputs_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            hook = self._resumed_hook(temporary, resume_step=100)
            try:
                self.assertEqual(hook._last_safety_step, 100)
                self.assertEqual(hook._last_checkpoint_step, 100)
                self.assertIsNone(hook._last_thermo_step)
                self.assertIsNone(hook._last_trajectory_step)
            finally:
                hook.close()

    def test_stage_boundary_does_not_double_count_same_safety_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = SafetyMonitor(
                SafetyLimits(max_temperature_K=500.0),
                maximum_consecutive_violations=2,
            )
            hook = RunOutputHook(
                temporary,
                safety_monitor=monitor,
                rng=np.random.default_rng(9),
                config_hash="a" * 64,
                model_hash="b" * 64,
            )
            try:
                atoms = FakeAtoms()
                first = thermo_record(stage="previous")
                issues = hook._check_safety(
                    atoms,
                    {"global_step": 5, "stage_name": "previous"},
                    first,
                )
                self.assertTrue(issues)
                self.assertEqual(monitor.consecutive_violations, 1)

                second = replace(first, stage="next", stage_step=0)
                duplicate = hook._check_safety(
                    atoms,
                    {"global_step": 5, "stage_name": "next"},
                    second,
                )
                self.assertEqual(duplicate, ())
                self.assertEqual(monitor.consecutive_violations, 1)
            finally:
                hook.close()

    def test_nhc_and_mtk_checkpoint_capability_is_boundary_only(self):
        cases = (
            ("fixed_nvt", "nose_hoover_chain", 1, False),
            ("fixed_nvt", "nose_hoover_chain", 0, True),
            ("fixed_nvt", "nose_hoover_chain", 3, True),
            ("fixed_nvt", "bussi", 1, True),
            ("isotropic_npt_plateau", "nose_hoover_chain", 1, False),
            ("isotropic_npt_plateau", "nose_hoover_chain", 3, True),
        )
        for stage_kind, thermostat_kind, stage_step, expected in cases:
            context = {
                "stage_kind": stage_kind,
                "thermostat_kind": thermostat_kind,
                "stage_step": stage_step,
                "stage_steps": 3,
            }
            with self.subTest(
                stage_kind=stage_kind,
                thermostat_kind=thermostat_kind,
                stage_step=stage_step,
            ):
                self.assertIs(
                    RunOutputHook._exact_resume_supported(context), expected
                )

    def test_interval_checkpoint_skips_interior_nhc_and_keeps_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            events = []
            hook = RunOutputHook(
                temporary,
                safety_monitor=SafetyMonitor(SafetyLimits()),
                rng=np.random.default_rng(9),
                config_hash="a" * 64,
                model_hash="b" * 64,
                reference_volume_A3=720.0,
                event_callback=lambda name, payload: events.append((name, payload)),
            )
            interior = {
                "stage_index": 0,
                "stage_name": "nhc",
                "stage_kind": "fixed_nvt",
                "thermostat_kind": "nose_hoover_chain",
                "stage_step": 1,
                "stage_steps": 3,
                "global_step": 1,
                "time_fs": 0.5,
                "stage_reference_cell_A": np.diag([8.0, 9.0, 10.0]).tolist(),
            }
            boundary = {
                **interior,
                "stage_step": 3,
                "global_step": 3,
                "time_fs": 1.5,
            }
            saved = SimpleNamespace(
                manifest=Path(temporary) / "checkpoint.json",
                arrays=Path(temporary) / "checkpoint.npz",
            )
            try:
                with patch.object(
                    hook.checkpoint_manager, "save", return_value=saved
                ) as save:
                    self.assertIsNone(
                        hook._checkpoint(
                            FakeAtoms(), interior, event="interval", force=True
                        )
                    )
                    save.assert_not_called()
                    self.assertIs(
                        hook._checkpoint(
                            FakeAtoms(), boundary, event="stage_end", force=True
                        ),
                        saved,
                    )
                    save.assert_called_once()
            finally:
                hook.close()

            skipped = [payload for name, payload in events if name == "checkpoint_skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertIn("extended-system state", skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()
