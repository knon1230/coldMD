import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from coldmd.observables import THERMO_FIELDS
from coldmd.reconcile import _durable_move, reconcile_outputs_to_checkpoint


def _thermo_row(step: int) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in THERMO_FIELDS}
    row.update(
        {
            "step": step,
            "time_fs": float(step),
            "time_ps": float(step) / 1000.0,
            "stage": "hold",
            "stage_step": step,
            "branch": "initial_hold",
        }
    )
    return row


def _write_thermo(path: Path, steps: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(THERMO_FIELDS))
        writer.writeheader()
        for step in steps:
            writer.writerow(_thermo_row(step))


def _read_steps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [int(row["step"]) for row in csv.DictReader(stream)]


class ReconciliationTests(unittest.TestCase):
    def test_post_checkpoint_thermo_and_events_are_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_thermo(root / "thermo.csv", [0, 5, 10, 15])
            (root / "coldmd.log").write_text(
                "\n".join(
                    json.dumps({"event": "sample", "step": step})
                    for step in (0, 10, 15)
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint = SimpleNamespace(resume_step=10, checkpoint_id="checkpoint-x")

            report = reconcile_outputs_to_checkpoint(
                root,
                checkpoint,
                trajectory_filename="trajectory.traj",
                thermo_filename="thermo.csv",
                event_filename="coldmd.log",
            )

            self.assertTrue(report.changed)
            self.assertEqual(_read_steps(root / "thermo.csv"), [0, 5, 10])
            self.assertEqual(report.last_thermo_step, 10)
            self.assertEqual(report.quarantined_thermo_rows, 1)
            self.assertFalse((root / "coldmd.log").exists())
            self.assertIsNotNone(report.attempt_directory)
            attempt = report.attempt_directory
            assert attempt is not None
            self.assertEqual(
                _read_steps(attempt / "thermo.csv"), [0, 5, 10, 15]
            )
            self.assertTrue((attempt / "coldmd.log").exists())

    def test_event_only_tail_also_triggers_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_thermo(root / "thermo.csv", [0, 10])
            (root / "coldmd.log").write_text(
                json.dumps({"event": "warning", "step": 11}) + "\n",
                encoding="utf-8",
            )
            checkpoint = SimpleNamespace(resume_step=10, checkpoint_id="checkpoint-y")

            report = reconcile_outputs_to_checkpoint(
                root,
                checkpoint,
                trajectory_filename="trajectory.traj",
                thermo_filename="thermo.csv",
                event_filename="coldmd.log",
            )

            self.assertTrue(report.changed)
            self.assertEqual(_read_steps(root / "thermo.csv"), [0, 10])
            self.assertEqual(report.original_max_event_step, 11)

    def test_stage_extxyz_and_cif_tail_alone_trigger_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_thermo(root / "thermo.csv", [0, 10])
            structures = root / "stage_structures"
            structures.mkdir()
            stem = "stage-002-branch-end-step-000000000020"
            extxyz = structures / f"{stem}.extxyz"
            cif = structures / f"{stem}.cif"
            extxyz.write_text("snapshot\n", encoding="utf-8")
            cif.write_text("data_snapshot\n", encoding="utf-8")
            checkpoint = SimpleNamespace(
                resume_step=10, checkpoint_id="checkpoint-structures"
            )

            report = reconcile_outputs_to_checkpoint(
                root,
                checkpoint,
                trajectory_filename="trajectory.traj",
                thermo_filename="thermo.csv",
            )

            self.assertTrue(report.changed)
            self.assertFalse(extxyz.exists())
            self.assertFalse(cif.exists())
            assert report.attempt_directory is not None
            archived = report.attempt_directory / "stage_structures"
            self.assertTrue((archived / extxyz.name).is_file())
            self.assertTrue((archived / cif.name).is_file())

    def test_fsync_failure_after_rename_rolls_back_canonical_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_thermo(root / "thermo.csv", [0, 10, 20])
            checkpoint = SimpleNamespace(resume_step=10, checkpoint_id="checkpoint-z")

            from coldmd import reconcile as reconcile_module

            real_fsync_file = reconcile_module._fsync_file
            calls = 0

            def fail_first_file_fsync(path):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("injected fsync failure after rename")
                return real_fsync_file(path)

            with mock.patch.object(
                reconcile_module,
                "_fsync_file",
                side_effect=fail_first_file_fsync,
            ):
                with self.assertRaisesRegex(OSError, "injected fsync failure"):
                    reconcile_outputs_to_checkpoint(
                        root,
                        checkpoint,
                        trajectory_filename="trajectory.traj",
                        thermo_filename="thermo.csv",
                    )

            self.assertEqual(_read_steps(root / "thermo.csv"), [0, 10, 20])
            journal = json.loads(
                (root / "resume-reconciliation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(journal["status"], "rolled_back")

    def test_nested_archive_parent_entry_is_fsynced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.extxyz"
            source.write_text("frame\n", encoding="utf-8")
            destination = root / "attempt" / "stage_structures" / source.name

            from coldmd import reconcile as reconcile_module

            synced: list[Path] = []
            real_fsync_directory = reconcile_module._fsync_directory

            def record_directory(path):
                synced.append(Path(path))
                return real_fsync_directory(path)

            with mock.patch.object(
                reconcile_module,
                "_fsync_directory",
                side_effect=record_directory,
            ):
                _durable_move(source, destination)

            self.assertTrue(destination.is_file())
            self.assertIn(root / "attempt", synced)
            self.assertIn(root / "attempt" / "stage_structures", synced)


if __name__ == "__main__":
    unittest.main()
