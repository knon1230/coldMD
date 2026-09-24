import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from ase import Atoms
from ase.io.trajectory import Trajectory

from coldmd.reporting import (
    _discover_config,
    _protocol_section,
    _selected_structural_stages,
    generate_analysis_report,
    plot_pv_stage_mean_sd,
)


class ReportingV1IntegrationTests(unittest.TestCase):
    def test_stage_path_connects_consecutive_branches_in_protocol_order(self):
        import matplotlib.pyplot as plt

        from coldmd.analysis import stage_thermo_statistics

        protocol = [
            {"name": "load_50", "kind": "npt_hold", "branch": "compression"},
            {"name": "load_100", "kind": "npt_hold", "branch": "compression"},
            {"name": "unload_75", "kind": "npt_hold", "branch": "decompression"},
        ]
        rows = [
            {
                "stage": stage["name"],
                "stage_step": 0,
                "time_ps": float(index),
                "total_pressure_GPa": pressure,
                "volume_A3": volume,
                "temperature_K": 300.0,
                "density_g_cm3": 2.0,
            }
            for index, (stage, pressure, volume) in enumerate(
                zip(protocol, [50.0, 100.0, 75.0], [90.0, 70.0, 78.0], strict=True)
            )
        ]
        summary = stage_thermo_statistics(rows, protocol_stages=protocol)
        captured: dict[str, list[float]] = {}

        def inspect(fig, path, dpi=180):  # type: ignore[no-untyped-def]
            line = next(
                item for item in fig.axes[0].lines if item.get_label() == "stage sequence"
            )
            captured["x"] = list(line.get_xdata())
            captured["y"] = list(line.get_ydata())
            plt.close(fig)
            return Path(path)

        with patch("coldmd.reporting._save_figure", side_effect=inspect):
            plot_pv_stage_mean_sd(summary, "unused.png")

        self.assertEqual(captured["x"], [90.0, 70.0, 78.0])
        self.assertEqual(captured["y"], [50.0, 100.0, 75.0])

    def test_representative_selection_does_not_require_legacy_stage_names(self):
        def record(volume):
            return type(
                "Record",
                (),
                {"atoms": type("AtomsLike", (), {"get_volume": lambda self: volume})()},
            )()

        groups = {
            "equilibrate_at_start": [record(100.0)],
            "arbitrary_middle_hold": [record(84.0)],
            "return_to_ambient": [record(98.0)],
        }

        self.assertEqual(
            _selected_structural_stages(groups, None),
            ["equilibrate_at_start", "arbitrary_middle_hold", "return_to_ambient"],
        )

    def test_original_config_override_expands_pressure_ranges_for_statistics(self):
        protocol = _protocol_section(
            {
                "protocol": {
                    "stages": [
                        {
                            "kind": "pressure_range",
                            "name_prefix": "compression",
                            "branch": "cold_compression",
                            "start_pressure_GPa": 0.0,
                            "end_pressure_GPa": 2.0,
                            "pressure_step_GPa": 1.0,
                            "duration_ps": 2.0,
                            "sampling_window_ps": 1.0,
                        }
                    ]
                }
            }
        )
        self.assertEqual(
            [stage["name"] for stage in protocol["stages"]],
            ["compression_0GPa", "compression_1GPa", "compression_2GPa"],
        )

    def test_run_manifest_fallback_only_follows_local_resolved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            resolved = run_dir / "stored.yaml"
            resolved.write_text(
                "protocol:\n  timestep_fs: 1.0\n  stages: []\n",
                encoding="utf-8",
            )
            (run_dir / "run-manifest.json").write_text(
                '{"config_path": "stored.yaml"}\n',
                encoding="utf-8",
            )
            discovered = _discover_config(run_dir, None)
            self.assertEqual(discovered["protocol"]["timestep_fs"], 1.0)

    def test_report_writes_all_stage_mean_sd_and_topology_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            stages = [
                {
                    "kind": "npt_hold",
                    "name": "initial_300K",
                    "branch": "compression",
                    "purpose": "equilibration",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 0.4,
                    "sampling_window_ps": 0.2,
                },
                {
                    "kind": "nvt_hold",
                    "name": "recovered_glass_aging",
                    "branch": "decompression",
                    "purpose": "sampling",
                    "duration_ps": 0.4,
                    "sampling_window_ps": 0.2,
                },
            ]
            (run_dir / "resolved-config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "protocol": {"timestep_fs": 1.0, "stages": stages},
                        "output": {"thermo_filename": "thermo.csv"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            trajectory_path = run_dir / "trajectory-000000.traj"
            writer = Trajectory(str(trajectory_path), mode="w")
            rows = []
            step = 0
            try:
                for stage_index, stage in enumerate(stages):
                    frame_count = 12 if stage["kind"] == "nvt_hold" else 5
                    for stage_step in range(frame_count):
                        shift = 0.01 * stage_step
                        atoms = Atoms(
                            symbols=["Si", "Si", "O", "O", "Ca"],
                            positions=[
                                [3.0 + shift, 5.0, 5.0],
                                [7.0 + shift, 5.0, 5.0],
                                [5.0 + shift, 5.0, 5.0],  # BO: two Si within 2.1 A
                                [3.0 + shift, 7.0, 5.0],  # NBO: one Si within 2.1 A
                                [5.0 + shift, 7.0, 5.0],
                            ],
                            cell=[10.0, 10.0, 10.0],
                            pbc=True,
                        )
                        atoms.info.update(
                            {
                                "coldmd_step": step,
                                "coldmd_time_fs": float(step * 100.0),
                                "coldmd_stage": stage["name"],
                                "coldmd_stage_step": stage_step,
                            }
                        )
                        writer.write(atoms)
                        rows.append(
                            {
                                "step": step,
                                "time_fs": step * 100.0,
                                "time_ps": step * 0.1,
                                "stage": stage["name"],
                                "stage_step": stage_step,
                                "branch": (
                                    "initial_hold" if stage_index == 0 else "recovered_hold"
                                ),
                                "target_pressure_GPa": "",
                                "total_pressure_GPa": stage_index + stage_step * 0.1,
                                "volume_A3": 1000.0 + stage_index * 5.0 + stage_step,
                                "temperature_K": 300.0 + stage_step,
                                "density_g_cm3": 2.5 - stage_step * 0.001,
                            }
                        )
                        step += 1
            finally:
                writer.close()

            with (run_dir / "thermo.csv").open("w", encoding="utf-8", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                csv_writer.writerows(rows)

            result = generate_analysis_report(
                run_dir,
                si_o_cutoff_A=2.1,
                rdf_bins=40,
                max_structural_frames=5,
            )

            analysis_dir = Path(result["analysis_dir"])
            unified = analysis_dir / "thermo_stage_mean_sd.csv"
            self.assertTrue(unified.exists())
            with unified.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["stage"] for row in summary_rows],
                ["initial_300K", "recovered_glass_aging"],
            )
            self.assertEqual(
                [row["pv_branch"] for row in summary_rows],
                ["compression", "decompression"],
            )
            self.assertTrue(all(row["temperature_mean_K"] for row in summary_rows))
            self.assertTrue(all(row["density_std_g_cm3"] for row in summary_rows))
            self.assertTrue((analysis_dir / "pv_hysteresis_mean_sd.png").exists())
            self.assertFalse((analysis_dir / "pt_path_mean_sd.png").exists())
            self.assertFalse((analysis_dir / "pv_hysteresis.csv").exists())

            npt_dir = analysis_dir / "initial_300K"
            nvt_dir = analysis_dir / "recovered_glass_aging"
            self.assertTrue((npt_dir / "sampling_thermo.csv").exists())
            self.assertTrue((npt_dir / "sampling_thermo_T_V.png").exists())
            self.assertTrue((nvt_dir / "sampling_thermo.csv").exists())
            self.assertTrue((nvt_dir / "sampling_thermo_T_P.png").exists())
            self.assertTrue((nvt_dir / "msd_full_stage.csv").exists())
            self.assertTrue((nvt_dir / "msd_full_stage.png").exists())
            self.assertTrue((nvt_dir / "msd_full_stage_loglog.png").exists())

            with (nvt_dir / "sampling_thermo.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                stage_rows = list(csv.DictReader(handle))
            self.assertEqual(len(stage_rows), 12)
            self.assertEqual(stage_rows[0]["stage_time_ps"], "0.0")
            self.assertEqual(stage_rows[0]["in_sampling_window"], "False")
            self.assertEqual(stage_rows[-1]["in_sampling_window"], "True")
            with (nvt_dir / "msd_full_stage.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                msd_reader = csv.DictReader(handle)
                self.assertEqual(msd_reader.fieldnames[0], "elapsed_time_ps")

            for stage in ("initial_300K", "recovered_glass_aging"):
                stage_dir = analysis_dir / stage
                self.assertTrue((stage_dir / "rdf.csv").exists())
                self.assertTrue((stage_dir / "oxygen_speciation.csv").exists())
                self.assertTrue((stage_dir / "rdf_oxygen_topology.csv").exists())
                self.assertTrue(
                    (stage_dir / "rdf_oxygen_topology_oxygen.png").exists()
                )
                self.assertTrue(
                    (stage_dir / "rdf_oxygen_topology_cation.png").exists()
                )

            markdown = (analysis_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("`recovered_glass_aging` | Ca", markdown)
            self.assertLess(
                markdown.index("initial_300K/rdf.png"),
                markdown.index("initial_300K/rdf_oxygen_topology_oxygen.png"),
            )

            self.assertEqual(result["thermo_stage_summary"].summaries[0].n_samples, 3)

    def test_all_stages_and_explicit_selection_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            generate_analysis_report(
                ".",
                stages=["initial"],
                all_stages=True,
            )


if __name__ == "__main__":
    unittest.main()
