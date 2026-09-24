from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class ForkStateTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from ase import Atoms
            from ase.io import write
            import yaml
        except ImportError as exc:
            self.skipTest(f"fork dependencies are unavailable: {exc}")

        from coldmd.api import _config_hash
        from coldmd.config import dump_config, load_config

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_cif = self.root / "identity.cif"
        identity = Atoms(
            "Cu2",
            positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        write(self.input_cif, identity, format="cif")
        config_data = {
            "config_version": 2,
            "input": {
                "cif_path": str(self.input_cif),
                "expected_atom_count": 2,
            },
            "calculator": {
                "kind": "factory",
                "factory": "ase.calculators.emt:EMT",
            },
            "protocol": {
                "temperature_K": 300.0,
                "timestep_fs": 1.0,
                "random_seed": 210,
                "stages": [
                    {
                        "kind": "nvt_hold",
                        "name": "child",
                        "branch": "decompression",
                        "purpose": "sampling",
                        "duration_ps": 0.02,
                        "sampling_window_ps": 0.01,
                    }
                ],
            },
        }
        self.config_path = self.root / "child.yaml"
        self.config_path.write_text(
            yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
        )
        self.config = load_config(self.config_path)

        self.parent = self.root / "parent"
        self.stage_dir = self.parent / "stage_structures"
        self.stage_dir.mkdir(parents=True)
        dump_config(self.config, self.parent / "resolved-config.yaml")
        (self.parent / "run-manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "config_sha256": _config_hash(self.config),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.state = self.stage_dir / (
            "stage-007-parent-handoff-step-000000000100.extxyz"
        )
        snapshot = identity.copy()
        snapshot.set_cell(np.eye(3) * 7.5, scale_atoms=False)
        snapshot.set_positions([[0.2, 0.1, 0.0], [2.3, 2.4, 2.5]])
        snapshot.set_momenta([[0.3, -0.1, 0.2], [-0.2, 0.15, -0.1]])
        write(self.state, snapshot, format="extxyz")
        (self.parent / "coldmd.log").write_text(
            json.dumps(
                {
                    "event": "stage_handoff",
                    "stage": "parent_original_name",
                    "structure": str(self.state),
                    "selected_global_step": 80,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_complete_mechanics_and_records_parent_only_as_provenance(self) -> None:
        from coldmd.api import _load_fork_state

        fork_state = _load_fork_state(self.config, self.parent, self.state.name)

        np.testing.assert_allclose(
            fork_state.atoms.get_momenta(),
            [[0.3, -0.1, 0.2], [-0.2, 0.15, -0.1]],
        )
        np.testing.assert_allclose(
            fork_state.atoms.get_positions(),
            [[0.2, 0.1, 0.0], [2.3, 2.4, 2.5]],
        )
        self.assertAlmostEqual(fork_state.atoms.get_volume(), 7.5**3)
        provenance = fork_state.provenance
        self.assertEqual(provenance["source_boundary_global_step"], 100)
        self.assertEqual(provenance["source_selected_global_step"], 80)
        self.assertEqual(provenance["source_mechanical_state_time_ps"], 0.08)
        self.assertIn("global_step", provenance["reset_in_child"])
        self.assertIn("momenta", provenance["inheritance"])
        self.assertEqual(len(provenance["source_state_sha256"]), 64)

    def test_public_check_only_probes_child_without_creating_run_output(self) -> None:
        from coldmd.api import fork

        result = fork(
            self.config_path,
            from_run=self.parent,
            state=self.state.name,
            check_only=True,
        )

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["fork_ready"])
        self.assertEqual(result["child_origin"]["global_step"], 0)
        self.assertEqual(result["child_origin"]["time_fs"], 0.0)
        self.assertEqual(
            result["child_origin"]["volume_ratio_from_engine_start"], 1.0
        )
        self.assertEqual(result["constructed_stage_kinds"], ["fixed_nvt"])
        self.assertFalse(self.config.output.directory.exists())
        self.assertFalse((self.root / ".parent.coldmd.lock").exists())

    def test_public_fork_runs_child_from_zero_and_persists_lineage(self) -> None:
        from ase.io import read

        from coldmd.api import fork

        child = self.root / "child-run"
        result = fork(
            self.config_path,
            from_run=self.parent,
            state=self.state.name,
            output_dir=child,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_global_step"], 20)
        manifest = json.loads(
            (child / "run-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["run_mode"], "fork")
        self.assertEqual(
            manifest["fork_provenance"]["source_state_sha256"],
            result["fork_provenance"]["source_state_sha256"],
        )
        with (child / "thermo.csv").open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        self.assertEqual(int(first["step"]), 0)
        self.assertEqual(float(first["time_fs"]), 0.0)
        self.assertEqual(float(first["volume_ratio_from_engine_start"]), 1.0)
        extxyz_files = sorted((child / "stage_structures").glob("*.extxyz"))
        cif_files = sorted((child / "stage_structures").glob("*.cif"))
        self.assertEqual(len(extxyz_files), len(cif_files))
        self.assertGreaterEqual(len(extxyz_files), 2)
        initial = read(extxyz_files[0], format="extxyz")
        self.assertTrue(initial.has("momenta"))

    def test_rejects_position_only_extxyz(self) -> None:
        from ase import Atoms
        from ase.io import write

        from coldmd.api import _load_fork_state
        from coldmd.storage import CheckpointCompatibilityError

        missing = self.stage_dir / "stage-002-no-momenta-end-step-000000000010.extxyz"
        atoms = Atoms(
            "Cu2",
            positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        write(missing, atoms, format="extxyz")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "no momenta"):
            _load_fork_state(self.config, self.parent, missing.name)

    def test_accepts_explicit_zero_momenta_without_redrawing_them(self) -> None:
        from ase import Atoms
        from ase.io import write

        from coldmd.api import _load_fork_state

        zero_state = self.stage_dir / (
            "stage-003-zero-temperature-end-step-000000000020.extxyz"
        )
        atoms = Atoms(
            "Cu2",
            positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        atoms.set_momenta(np.zeros((2, 3)))
        write(zero_state, atoms, format="extxyz")

        fork_state = _load_fork_state(self.config, self.parent, zero_state.name)

        np.testing.assert_array_equal(
            fork_state.atoms.get_momenta(), np.zeros((2, 3))
        )
        self.assertEqual(fork_state.provenance["source_temperature_K"], 0.0)

    def test_rejects_cif_as_fork_source(self) -> None:
        from coldmd.api import _load_fork_state

        cif = self.stage_dir / "stage-007-parent-handoff-step-000000000100.cif"
        cif.write_bytes(self.input_cif.read_bytes())
        with self.assertRaisesRegex(ValueError, "canonical ColdMD stage snapshot"):
            _load_fork_state(self.config, self.parent, cif.name)

    def test_parent_and_child_directories_must_not_overlap(self) -> None:
        from coldmd.api import _assert_separate_fork_directories

        with self.assertRaisesRegex(ValueError, "non-nested"):
            _assert_separate_fork_directories(
                self.parent, self.parent / "children" / "branch"
            )


class ForkCLIContractTests(unittest.TestCase):
    def test_fork_command_parses_explicit_parent_state_and_check_mode(self) -> None:
        from coldmd.cli import build_parser

        arguments = build_parser().parse_args(
            [
                "fork",
                "child.yaml",
                "--from-run",
                "parent",
                "--state",
                "stage.extxyz",
                "--output-dir",
                "child-run",
                "--check-only",
            ]
        )
        self.assertEqual(arguments.command, "fork")
        self.assertEqual(arguments.from_run, Path("parent"))
        self.assertEqual(arguments.state, Path("stage.extxyz"))
        self.assertEqual(arguments.output_dir, Path("child-run"))
        self.assertTrue(arguments.check_only)


if __name__ == "__main__":
    unittest.main()
