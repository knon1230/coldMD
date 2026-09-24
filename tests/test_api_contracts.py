import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_config import valid_volume_config


class APIContractTests(unittest.TestCase):
    @staticmethod
    def _imports():
        try:
            from coldmd.api import _build_stage_specs
            from coldmd.config import load_config_dict
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")
        return _build_stage_specs, load_config_dict

    def test_volume_stage_mapping_preserves_roles_and_integral_steps(self):
        _build_stage_specs, load_config_dict = self._imports()
        config = load_config_dict(valid_volume_config())
        stages = _build_stage_specs(config)

        self.assertEqual([stage.steps for stage in stages], [2000] * 5)
        self.assertEqual(stages[1].role, "cold_compression")
        self.assertEqual(stages[3].role, "decompression")
        self.assertAlmostEqual(stages[1].target_volume_ratio, 0.8)
        self.assertAlmostEqual(stages[3].target_volume_ratio, 1.25)
        for stage in (stages[0], stages[2], stages[4]):
            self.assertEqual(stage.thermostat_kind, "bussi")
            self.assertEqual(stage.thermostat_chain_length, 3)
            self.assertEqual(stage.thermostat_substeps, 1)

    def test_pressure_stage_mapping_preserves_explicit_branch(self):
        _build_stage_specs, load_config_dict = self._imports()
        data = valid_volume_config()
        data["protocol"] = {
            "control_mode": "pressure_plateaus",
            "temperature_K": 300.0,
            "timestep_fs": 0.5,
            "stages": [
                {"kind": "initial_nvt", "name": "s0", "duration_ps": 1.0},
                {
                    "kind": "pressure_plateau",
                    "name": "s1",
                    "branch": "cold_compression",
                    "target_pressure_GPa": 10.0,
                    "duration_ps": 1.0,
                },
                {
                    "kind": "pressure_plateau",
                    "name": "s2",
                    "branch": "decompression",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 1.0,
                },
                {
                    "kind": "recovery_npt",
                    "name": "s3",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 1.0,
                },
                {"kind": "final_nvt", "name": "s4", "duration_ps": 1.0},
            ],
        }
        stages = _build_stage_specs(load_config_dict(data))

        self.assertEqual(stages[1].branch, "cold_compression")
        self.assertEqual(stages[2].branch, "decompression")
        self.assertEqual(stages[3].branch, "recovery")

    def test_free_stage_sequence_maps_generic_nvt_npt_and_ramps_in_order(self):
        _build_stage_specs, load_config_dict = self._imports()
        try:
            from coldmd.engine import (
                FixedCellNVTStage,
                IsotropicNPTStage,
                VolumeRampStage,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        data = valid_volume_config()
        data["protocol"] = {
            "control_mode": "stage_sequence",
            "temperature_K": 300.0,
            "timestep_fs": 0.5,
            "stages": [
                {
                    "kind": "npt",
                    "name": "equilibrate_at_zero",
                    "branch": "other",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 1.0,
                },
                {
                    "kind": "nvt",
                    "name": "hold_zero_cell",
                    "branch": "cold_compression",
                    "duration_ps": 0.5,
                },
                {
                    "kind": "compression",
                    "name": "smooth_down",
                    "target_volume_ratio": 0.96,
                    "duration_ps": 1.0,
                    "schedule": "log_smoothstep",
                },
                {
                    "kind": "npt",
                    "name": "equilibrate_at_peak",
                    "branch": "decompression",
                    "target_pressure_GPa": 2.5,
                    "duration_ps": 1.0,
                },
                {
                    "kind": "decompression",
                    "name": "smooth_up",
                    "target_volume_ratio": 1.0416666666666667,
                    "duration_ps": 1.0,
                    "schedule": "log_smoothstep",
                },
                {
                    "kind": "nvt",
                    "name": "final_hold",
                    "branch": "recovery",
                    "duration_ps": 0.5,
                },
            ],
        }

        stages = _build_stage_specs(load_config_dict(data))

        self.assertEqual(
            [type(stage) for stage in stages],
            [
                IsotropicNPTStage,
                FixedCellNVTStage,
                VolumeRampStage,
                IsotropicNPTStage,
                VolumeRampStage,
                FixedCellNVTStage,
            ],
        )
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "equilibrate_at_zero",
                "hold_zero_cell",
                "smooth_down",
                "equilibrate_at_peak",
                "smooth_up",
                "final_hold",
            ],
        )
        self.assertEqual(stages[0].branch, "other")
        self.assertEqual(stages[1].branch, "cold_compression")
        self.assertEqual(stages[3].branch, "decompression")
        self.assertEqual(stages[5].branch, "recovery")
        self.assertEqual([stage.steps for stage in stages], [2000, 1000, 2000, 2000, 2000, 1000])

    def test_every_fixed_nvt_stage_receives_selected_nose_hoover_controls(self):
        _build_stage_specs, load_config_dict = self._imports()
        try:
            from coldmd.engine import FixedCellNVTStage, IsotropicNPTStage
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        data = valid_volume_config()
        data["protocol"] = {
            "control_mode": "stage_sequence",
            "temperature_K": 300.0,
            "timestep_fs": 0.5,
            "thermostat": {
                "kind": "nose_hoover_chain",
                "tau_fs": 75.0,
                "chain_length": 5,
                "substeps": 2,
            },
            "barostat": {
                "kind": "isotropic_mtk",
                "thermostat_tau_fs": 125.0,
                "barostat_tau_fs": 1250.0,
                "thermostat_chain_length": 4,
                "barostat_chain_length": 6,
                "thermostat_substeps": 3,
                "barostat_substeps": 2,
            },
            "stages": [
                {"kind": "initial_nvt", "name": "initial", "duration_ps": 0.5},
                {
                    "kind": "nvt",
                    "name": "generic",
                    "branch": "cold_compression",
                    "duration_ps": 0.5,
                },
                {"kind": "peak_nvt", "name": "peak", "duration_ps": 0.5},
                {
                    "kind": "npt",
                    "name": "pressure",
                    "target_pressure_GPa": 2.0,
                    "duration_ps": 0.5,
                },
                {"kind": "final_nvt", "name": "final", "duration_ps": 0.5},
            ],
        }

        stages = _build_stage_specs(load_config_dict(data))
        fixed = [stage for stage in stages if isinstance(stage, FixedCellNVTStage)]
        self.assertEqual([stage.name for stage in fixed], ["initial", "generic", "peak", "final"])
        self.assertTrue(
            all(stage.thermostat_kind == "nose_hoover_chain" for stage in fixed)
        )
        self.assertTrue(all(stage.thermostat_tau_fs == 75.0 for stage in fixed))
        self.assertTrue(all(stage.thermostat_chain_length == 5 for stage in fixed))
        self.assertTrue(all(stage.thermostat_substeps == 2 for stage in fixed))

        npt = next(stage for stage in stages if isinstance(stage, IsotropicNPTStage))
        self.assertEqual(npt.thermostat_tau_fs, 125.0)
        self.assertEqual(npt.thermostat_chain_length, 4)
        self.assertEqual(npt.thermostat_substeps, 3)

    def test_dryrun_cli_is_a_fixed_cell_nvt_smoke_command(self):
        from coldmd.cli import build_parser

        arguments = build_parser().parse_args(
            ["dryrun", "input.yaml", "--nvt-steps", "37", "--json"]
        )
        self.assertEqual(arguments.nvt_steps, 37)
        self.assertTrue(arguments.json_output)
        self.assertEqual(arguments.handler.__name__, "_handle_dryrun")

    def test_fresh_velocity_initialization_policy_preserves_legacy_semantics(self):
        try:
            from coldmd.api import _initialize_velocities_for_fresh_run
            from coldmd.config import load_config_dict
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        legacy_data = valid_volume_config()
        legacy = load_config_dict(legacy_data)
        self.assertTrue(_initialize_velocities_for_fresh_run(legacy))

        legacy_data["protocol"]["stages"][0]["initialize_velocities"] = False
        legacy_disabled = load_config_dict(legacy_data)
        self.assertFalse(_initialize_velocities_for_fresh_run(legacy_disabled))

        sequence_data = valid_volume_config()
        sequence_data["protocol"] = {
            "control_mode": "stage_sequence",
            "temperature_K": 300.0,
            "timestep_fs": 0.5,
            "stages": [
                {
                    "kind": "npt",
                    "name": "first_npt",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 1.0,
                }
            ],
        }
        sequence_default = load_config_dict(sequence_data)
        self.assertTrue(_initialize_velocities_for_fresh_run(sequence_default))

        sequence_data["protocol"]["initialize_velocities"] = False
        sequence_disabled = load_config_dict(sequence_data)
        self.assertFalse(_initialize_velocities_for_fresh_run(sequence_disabled))

    def test_strict_resume_is_limited_to_local_mace_checkpoint(self):
        try:
            from coldmd.api import _strict_resume_available
            from coldmd.config import load_config_dict
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        data = valid_volume_config()
        data["calculator"] = {
            "kind": "factory",
            "factory": "package.module:create_calculator",
            "model_path": "model.pt",
        }
        generic = load_config_dict(data)
        self.assertFalse(
            _strict_resume_available(generic, "checkpoint_content_sha256")
        )

        data["calculator"] = {
            "kind": "mace",
            "model_path": "model.pt",
        }
        mace = load_config_dict(data)
        self.assertTrue(
            _strict_resume_available(mace, "checkpoint_content_sha256")
        )
        self.assertFalse(_strict_resume_available(mace, "calculator_spec_sha256"))

    def test_foundation_model_warning_is_machine_readable(self):
        try:
            from coldmd.api import (
                FOUNDATION_RESUME_WARNING_CODE,
                configuration_warnings,
            )
            from coldmd.config import load_config_dict
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        data = valid_volume_config()
        data["calculator"] = {
            "kind": "mace",
            "foundation_model": "medium-mpa-0",
            "allow_download": True,
            "device": "cpu",
            "dtype": "float64",
        }
        warnings = configuration_warnings(load_config_dict(data))

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], FOUNDATION_RESUME_WARNING_CODE)
        self.assertIn("strict resume is unavailable", warnings[0]["message"])

        data["calculator"] = {
            "kind": "mace",
            "model_path": "model.pt",
            "device": "cpu",
            "dtype": "float64",
        }
        self.assertEqual(configuration_warnings(load_config_dict(data)), [])

    def test_analyze_forwards_v1_stage_and_cutoff_options(self):
        try:
            from coldmd.api import analyze
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE workflow dependency is unavailable: {exc}")

        generated = {
            "analysis_dir": "analysis",
            "report": "analysis/report.md",
            "json": "analysis/analysis.json",
            "artifacts": ["analysis/analysis.json"],
            "warnings": [],
        }
        with patch(
            "coldmd.reporting.generate_analysis_report",
            return_value=generated,
        ) as report:
            result = analyze(
                "run",
                stages=["initial", "peak"],
                si_o_cutoff_A=2.15,
                force=True,
            )

        self.assertEqual(result["artifact_count"], 1)
        report.assert_called_once_with(
            "run",
            config=None,
            stages=["initial", "peak"],
            all_stages=False,
            si_o_cutoff_A=2.15,
            force=True,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            analyze("run", stages=["initial"], all_stages=True)


if __name__ == "__main__":
    unittest.main()
