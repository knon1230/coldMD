import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError
import yaml

from coldmd.config import (
    ColdMDConfig,
    NPTStage,
    NVTStage,
    PressurePlateauStage,
    dump_config,
    load_config,
    load_config_dict,
)


def valid_volume_config():
    return {
        "config_version": 1,
        "input": {
            "cif_path": "input.cif",
            "expected_atom_count": 400,
        },
        "calculator": {
            "factory": "coldmd.calculators:create_mace_calculator",
            "kwargs": {"foundation_model": "medium-mpa-0", "allow_download": True},
            "device": "cpu",
            "dtype": "float64",
        },
        "protocol": {
            "control_mode": "volume_ramp",
            "temperature_K": 300.0,
            "timestep_fs": 0.5,
            "stages": [
                {
                    "kind": "initial_nvt",
                    "name": "initial",
                    "duration_ps": 1.0,
                },
                {
                    "kind": "compression",
                    "name": "compress",
                    "duration_ps": 1.0,
                    "target_volume_ratio": 0.8,
                },
                {
                    "kind": "peak_nvt",
                    "name": "peak",
                    "duration_ps": 1.0,
                },
                {
                    "kind": "decompression",
                    "name": "decompress",
                    "duration_ps": 1.0,
                    "target_volume_ratio": 1.25,
                },
                {
                    "kind": "final_nvt",
                    "name": "final",
                    "duration_ps": 1.0,
                },
            ],
        },
    }


def valid_pressure_range_config():
    data = valid_volume_config()
    data["protocol"] = {
        "control_mode": "pressure_plateaus",
        "temperature_K": 300.0,
        "timestep_fs": 0.5,
        "stages": [
            {"kind": "initial_nvt", "name": "initial", "duration_ps": 1.0},
            {
                "kind": "pressure_range",
                "name_prefix": "compression",
                "branch": "cold_compression",
                "start_pressure_GPa": 0.0,
                "end_pressure_GPa": 2.0,
                "pressure_step_GPa": 1.0,
                "duration_ps": 1.0,
                "sampling_window_ps": 0.5,
            },
            {
                "kind": "pressure_range",
                "name_prefix": "decompression",
                "branch": "decompression",
                "start_pressure_GPa": 1.0,
                "end_pressure_GPa": 0.0,
                "pressure_step_GPa": 1.0,
                "duration_ps": 1.0,
                "sampling_window_ps": 0.5,
            },
            {"kind": "final_nvt", "name": "final", "duration_ps": 1.0},
        ],
    }
    return data


def valid_stage_sequence_config():
    data = valid_volume_config()
    data["protocol"] = {
        "control_mode": "stage_sequence",
        "temperature_K": 300.0,
        "timestep_fs": 0.5,
        "initialize_velocities": True,
        "stages": [
            {
                "kind": "npt",
                "name": "equilibrate_zero",
                "branch": "initial_hold",
                "target_pressure_GPa": 0.0,
                "duration_ps": 1.0,
                "sampling_window_ps": 0.5,
            },
            {
                "kind": "nvt",
                "name": "hold_zero",
                "branch": "initial_hold",
                "duration_ps": 0.5,
                "sampling_window_ps": 0.25,
            },
            {
                "kind": "compression",
                "name": "smooth_compression",
                "duration_ps": 1.0,
                "target_volume_ratio": 0.96,
                "schedule": "log_smoothstep",
            },
            {
                "kind": "nvt",
                "name": "compressed_hold",
                "branch": "high_density_hold",
                "duration_ps": 0.5,
            },
            {
                "kind": "npt",
                "name": "equilibrate_2p5",
                "branch": "cold_compression",
                "target_pressure_GPa": 2.5,
                "duration_ps": 1.0,
            },
            {
                "kind": "decompression",
                "name": "smooth_decompression",
                "duration_ps": 1.0,
                "target_volume_ratio": 1.0416666666666667,
            },
        ],
    }
    return data


class ConfigTests(unittest.TestCase):
    def test_thermostat_defaults_preserve_bussi_configuration(self):
        config = load_config_dict(valid_volume_config())

        self.assertEqual(config.protocol.thermostat.kind, "bussi")
        self.assertEqual(config.protocol.thermostat.tau_fs, 100.0)
        self.assertEqual(config.protocol.thermostat.chain_length, 3)
        self.assertEqual(config.protocol.thermostat.substeps, 1)

    def test_nose_hoover_chain_is_accepted_for_nvt_npt_stage_sequence(self):
        data = valid_stage_sequence_config()
        data["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain",
            "tau_fs": 80.0,
            "chain_length": 5,
            "substeps": 2,
        }
        data["protocol"]["stages"] = data["protocol"]["stages"][:2]

        config = load_config_dict(data)

        self.assertEqual(config.protocol.thermostat.kind, "nose_hoover_chain")
        self.assertEqual(config.protocol.thermostat.tau_fs, 80.0)
        self.assertEqual(config.protocol.thermostat.chain_length, 5)
        self.assertEqual(config.protocol.thermostat.substeps, 2)
        self.assertEqual(
            [stage.kind for stage in config.protocol.stages], ["npt", "nvt"]
        )

    def test_nose_hoover_chain_is_accepted_for_pressure_plateaus(self):
        data = valid_pressure_range_config()
        data["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain",
            "tau_fs": 100.0,
            "chain_length": 3,
            "substeps": 2,
        }

        config = load_config_dict(data)

        self.assertEqual(config.protocol.thermostat.kind, "nose_hoover_chain")
        self.assertEqual(config.protocol.thermostat.substeps, 2)

    def test_nose_hoover_chain_configuration_round_trips_through_yaml(self):
        data = valid_stage_sequence_config()
        data["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain",
            "tau_fs": 90.0,
            "chain_length": 4,
            "substeps": 3,
        }
        data["protocol"]["stages"] = data["protocol"]["stages"][:2]

        with TemporaryDirectory() as temporary:
            path = dump_config(
                load_config_dict(data, base_directory=temporary),
                Path(temporary) / "resolved-config.yaml",
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            restored = load_config(path)

        self.assertEqual(
            payload["protocol"]["thermostat"],
            {
                "kind": "nose_hoover_chain",
                "tau_fs": 90.0,
                "chain_length": 4,
                "substeps": 3,
            },
        )
        self.assertEqual(restored.protocol.thermostat.kind, "nose_hoover_chain")
        self.assertEqual(restored.protocol.thermostat.chain_length, 4)
        self.assertEqual(restored.protocol.thermostat.substeps, 3)

    def test_nose_hoover_chain_rejects_every_prescribed_volume_ramp(self):
        legacy_volume = valid_volume_config()
        legacy_volume["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain"
        }

        compression_only = valid_stage_sequence_config()
        compression_only["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain"
        }
        compression_only["protocol"]["stages"] = [
            compression_only["protocol"]["stages"][2]
        ]

        decompression_only = valid_stage_sequence_config()
        decompression_only["protocol"]["thermostat"] = {
            "kind": "nose_hoover_chain"
        }
        decompression_only["protocol"]["stages"] = [
            decompression_only["protocol"]["stages"][5]
        ]

        for data in (legacy_volume, compression_only, decompression_only):
            with self.subTest(
                control_mode=data["protocol"]["control_mode"],
                stages=[stage["kind"] for stage in data["protocol"]["stages"]],
            ):
                with self.assertRaisesRegex(
                    ValidationError, "prescribed volume ramps.*require.*bussi"
                ):
                    load_config_dict(data)

    def test_thermostat_kind_and_chain_controls_are_strictly_validated(self):
        cases = (
            {"kind": "langevin"},
            {"kind": "nose_hoover_chain", "chain_length": 0},
            {"kind": "nose_hoover_chain", "chain_length": 1.5},
            {"kind": "nose_hoover_chain", "substeps": 0},
            {"kind": "nose_hoover_chain", "substeps": True},
        )
        for thermostat in cases:
            data = valid_pressure_range_config()
            data["protocol"]["thermostat"] = thermostat
            with self.subTest(thermostat=thermostat):
                with self.assertRaises(ValidationError):
                    load_config_dict(data)

    def test_valid_volume_protocol(self):
        config = load_config_dict(valid_volume_config())
        self.assertEqual(config.protocol.control_mode, "volume_ramp")
        self.assertEqual(len(config.protocol.stages), 5)

    def test_unknown_key_is_rejected(self):
        data = valid_volume_config()
        data["protocol"]["mystery"] = 1
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_stage_order_is_rejected(self):
        data = valid_volume_config()
        stages = data["protocol"]["stages"]
        stages[1], stages[2] = stages[2], stages[1]
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_nonintegral_number_of_steps_is_rejected(self):
        data = valid_volume_config()
        data["protocol"]["stages"][0]["duration_ps"] = 1.00025
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_shipped_examples_validate_and_resolve_paths(self):
        project = Path(__file__).resolve().parents[1]
        volume = load_config(project / "examples" / "volume_ramp.yaml")
        pressure = load_config(project / "examples" / "pressure_plateaus.yaml")
        sequence = load_config(project / "examples" / "stage_sequence.yaml")

        self.assertEqual(volume.protocol.control_mode, "volume_ramp")
        self.assertEqual(pressure.protocol.control_mode, "pressure_plateaus")
        self.assertEqual(sequence.protocol.control_mode, "stage_sequence")
        self.assertEqual(len(sequence.protocol.stages), 18)
        self.assertTrue(volume.input.cif_path.is_absolute())
        self.assertEqual(
            volume.input.cif_path,
            (project / "examples" / "structure.cif").resolve(),
        )
        self.assertTrue(volume.output.directory.is_absolute())

    def test_factory_model_path_is_forwarded_as_factory_kwarg(self):
        data = valid_volume_config()
        data["calculator"] = {
            "kind": "factory",
            "factory": "example_package.adapter:create_calculator",
            "model_path": "potential.model",
            "model_sha256": "a" * 64,
            "device": "cpu",
            "dtype": "float64",
        }
        config = load_config_dict(data, base_directory="/tmp/coldmd-config-test")
        specification = config.calculator.to_spec_mapping()

        self.assertIsNone(specification["model_path"])
        self.assertEqual(
            specification["kwargs"]["model_path"],
            str((Path("/tmp/coldmd-config-test") / "potential.model").resolve()),
        )

    def test_pressure_branch_names_need_not_encode_the_branch(self):
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
                    "target_pressure_GPa": 5.0,
                    "duration_ps": 1.0,
                },
                {
                    "kind": "pressure_plateau",
                    "name": "s2",
                    "branch": "decompression",
                    "target_pressure_GPa": 0.0,
                    "duration_ps": 1.0,
                },
                {"kind": "final_nvt", "name": "s3", "duration_ps": 1.0},
            ],
        }
        config = load_config_dict(data)

        plateaus = config.protocol.stages[1:3]
        self.assertEqual([stage.name for stage in plateaus], ["s1", "s2"])
        self.assertEqual(
            [stage.branch for stage in plateaus],
            ["cold_compression", "decompression"],
        )

    def test_pressure_ranges_expand_to_canonical_plateau_stages(self):
        config = load_config_dict(valid_pressure_range_config())
        plateaus = [
            stage
            for stage in config.protocol.stages
            if isinstance(stage, PressurePlateauStage)
        ]

        self.assertEqual(
            [stage.name for stage in plateaus],
            [
                "compression_0GPa",
                "compression_1GPa",
                "compression_2GPa",
                "decompression_1GPa",
                "decompression_0GPa",
            ],
        )
        self.assertEqual(
            [stage.target_pressure_GPa for stage in plateaus],
            [0.0, 1.0, 2.0, 1.0, 0.0],
        )
        self.assertTrue(all(stage.duration_ps == 1.0 for stage in plateaus))
        self.assertTrue(all(stage.sampling_window_ps == 0.5 for stage in plateaus))

    def test_json_schema_advertises_pressure_range_input(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            schema = json.dumps(ColdMDConfig.model_json_schema(), sort_keys=True)
        self.assertIn('"pressure_range"', schema)
        self.assertIn("PressureRangeStage", schema)

    def test_decimal_pressure_range_and_explicit_plateau_can_be_mixed(self):
        data = valid_pressure_range_config()
        stages = data["protocol"]["stages"]
        stages[1]["end_pressure_GPa"] = 0.2
        stages[1]["pressure_step_GPa"] = 0.1
        stages[2]["start_pressure_GPa"] = 0.1
        stages[2]["pressure_step_GPa"] = 0.1
        stages.insert(
            2,
            {
                "kind": "pressure_plateau",
                "name": "peak_0.3GPa",
                "branch": "cold_compression",
                "target_pressure_GPa": 0.3,
                "duration_ps": 2.0,
                "sampling_window_ps": 1.0,
            },
        )

        config = load_config_dict(data)
        plateaus = [
            stage
            for stage in config.protocol.stages
            if isinstance(stage, PressurePlateauStage)
        ]
        self.assertEqual(
            [stage.target_pressure_GPa for stage in plateaus],
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.0],
        )

    def test_pressure_range_rejects_invalid_direction_and_spacing(self):
        cases = []

        wrong_compression = valid_pressure_range_config()
        wrong_compression["protocol"]["stages"][1].update(
            start_pressure_GPa=2.0,
            end_pressure_GPa=0.0,
        )
        cases.append(wrong_compression)

        wrong_decompression = valid_pressure_range_config()
        wrong_decompression["protocol"]["stages"][2].update(
            start_pressure_GPa=0.0,
            end_pressure_GPa=1.0,
        )
        cases.append(wrong_decompression)

        indivisible = valid_pressure_range_config()
        indivisible["protocol"]["stages"][1]["pressure_step_GPa"] = 0.3
        cases.append(indivisible)

        equal_endpoints = valid_pressure_range_config()
        equal_endpoints["protocol"]["stages"][1]["end_pressure_GPa"] = 0.0
        cases.append(equal_endpoints)

        for data in cases:
            with self.subTest(data=data["protocol"]["stages"]):
                with self.assertRaises(ValidationError):
                    load_config_dict(data)

    def test_pressure_range_rejects_bad_window_names_count_and_safety(self):
        bad_window = valid_pressure_range_config()
        bad_window["protocol"]["stages"][1]["sampling_window_ps"] = 2.0

        overlong_name = valid_pressure_range_config()
        overlong_name["protocol"]["stages"][1]["name_prefix"] = "x" * 64

        too_many = valid_pressure_range_config()
        too_many["protocol"]["stages"][1].update(
            end_pressure_GPa=10.0,
            pressure_step_GPa=0.0001,
        )

        unsafe = valid_pressure_range_config()
        unsafe["safety"] = {"max_abs_pressure_GPa": 1.0}

        for data in (bad_window, overlong_name, too_many, unsafe):
            with self.subTest(data=data["protocol"]["stages"][1]):
                with self.assertRaises(ValidationError):
                    load_config_dict(data)

    def test_generated_pressure_stage_names_must_be_unique(self):
        data = valid_pressure_range_config()
        data["protocol"]["stages"].insert(
            1,
            {
                "kind": "pressure_plateau",
                "name": "compression_0GPa",
                "branch": "cold_compression",
                "target_pressure_GPa": 0.0,
                "duration_ps": 1.0,
            },
        )
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_dumped_pressure_range_config_is_explicit_and_round_trips(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config_dict(
                valid_pressure_range_config(),
                base_directory=root,
            )
            path = dump_config(config, root / "resolved-config.yaml")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            kinds = [stage["kind"] for stage in payload["protocol"]["stages"]]
            self.assertNotIn("pressure_range", kinds)
            self.assertEqual(kinds.count("pressure_plateau"), 5)
            self.assertEqual(payload["config_version"], 1)

            reloaded = load_config(path)
            self.assertEqual(
                reloaded.model_dump(mode="json"),
                config.model_dump(mode="json"),
            )

    def test_stage_sequence_accepts_generic_stages_in_written_order(self):
        config = load_config_dict(valid_stage_sequence_config())

        self.assertEqual(config.protocol.control_mode, "stage_sequence")
        self.assertTrue(config.protocol.initialize_velocities)
        self.assertEqual(
            [stage.kind for stage in config.protocol.stages],
            ["npt", "nvt", "compression", "nvt", "npt", "decompression"],
        )
        self.assertIsInstance(config.protocol.stages[0], NPTStage)
        self.assertIsInstance(config.protocol.stages[1], NVTStage)
        self.assertEqual(config.protocol.stages[3].branch, "high_density_hold")

    def test_stage_sequence_accepts_legacy_stages_without_order_or_count_rules(self):
        data = valid_stage_sequence_config()
        data["protocol"]["stages"] = [
            {"kind": "final_nvt", "name": "legacy_final_a", "duration_ps": 1.0},
            {
                "kind": "pressure_plateau",
                "name": "legacy_plateau",
                "branch": "decompression",
                "target_pressure_GPa": 0.0,
                "duration_ps": 1.0,
            },
            {"kind": "peak_nvt", "name": "legacy_peak", "duration_ps": 1.0},
            {"kind": "initial_nvt", "name": "legacy_initial", "duration_ps": 1.0},
            {"kind": "final_nvt", "name": "legacy_final_b", "duration_ps": 1.0},
            {
                "kind": "recovery_npt",
                "name": "legacy_recovery",
                "target_pressure_GPa": 0.0,
                "duration_ps": 1.0,
            },
        ]

        config = load_config_dict(data)
        self.assertEqual(
            [stage.name for stage in config.protocol.stages],
            [
                "legacy_final_a",
                "legacy_plateau",
                "legacy_peak",
                "legacy_initial",
                "legacy_final_b",
                "legacy_recovery",
            ],
        )

    def test_stage_sequence_generic_branches_default_and_validate(self):
        data = valid_stage_sequence_config()
        del data["protocol"]["stages"][0]["branch"]
        del data["protocol"]["stages"][1]["branch"]
        config = load_config_dict(data)
        self.assertEqual(config.protocol.stages[0].branch, "other")
        self.assertEqual(config.protocol.stages[1].branch, "other")

        data["protocol"]["stages"][0]["branch"] = "not_a_branch"
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_stage_sequence_keeps_shared_stage_validation(self):
        duplicate = valid_stage_sequence_config()
        duplicate["protocol"]["stages"][1]["name"] = "equilibrate_zero"

        bad_duration = valid_stage_sequence_config()
        bad_duration["protocol"]["stages"][0]["duration_ps"] = 1.00025

        bad_window = valid_stage_sequence_config()
        bad_window["protocol"]["stages"][1]["sampling_window_ps"] = 1.0

        for data in (duplicate, bad_duration, bad_window):
            with self.subTest(stages=data["protocol"]["stages"]):
                with self.assertRaises(ValidationError):
                    load_config_dict(data)

    def test_stage_sequence_pressure_safety_includes_generic_npt(self):
        data = valid_stage_sequence_config()
        data["protocol"]["stages"][0]["target_pressure_GPa"] = -2.0
        data["safety"] = {"max_abs_pressure_GPa": 1.0}

        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_stage_sequence_volume_safety_handles_unknown_post_npt_volume(self):
        data = valid_stage_sequence_config()
        data["protocol"]["stages"] = [
            {
                "kind": "npt",
                "name": "dynamic_cell",
                "target_pressure_GPa": 1.0,
                "duration_ps": 1.0,
            },
            {
                "kind": "compression",
                "name": "relative_ramp",
                "duration_ps": 1.0,
                "target_volume_ratio": 0.8,
            },
        ]
        data["safety"] = {"min_volume_ratio": 0.9}

        # The NPT stage makes V/Vinitial unknowable at validation time, so the
        # cumulative bound cannot be applied to the following relative ramp.
        load_config_dict(data)

        data["safety"]["max_abs_log_volume_strain_per_step"] = 1.0e-5
        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_stage_sequence_volume_bound_is_used_before_first_npt(self):
        data = valid_stage_sequence_config()
        data["protocol"]["stages"] = [
            {
                "kind": "compression",
                "name": "known_relative_ramp",
                "duration_ps": 1.0,
                "target_volume_ratio": 0.8,
            },
            {
                "kind": "npt",
                "name": "dynamic_cell",
                "target_pressure_GPa": 1.0,
                "duration_ps": 1.0,
            },
        ]
        data["safety"] = {"min_volume_ratio": 0.9}

        with self.assertRaises(ValidationError):
            load_config_dict(data)

    def test_json_schema_advertises_stage_sequence_and_generic_stages(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            schema = json.dumps(ColdMDConfig.model_json_schema(), sort_keys=True)
        self.assertIn('"stage_sequence"', schema)
        self.assertIn("NVTStage", schema)
        self.assertIn("NPTStage", schema)

    def test_legacy_modes_keep_their_existing_contract(self):
        volume = valid_volume_config()
        volume["protocol"]["initialize_velocities"] = False
        config = load_config_dict(volume)
        self.assertFalse(config.protocol.initialize_velocities)

        volume["protocol"]["stages"][2] = {
            "kind": "nvt",
            "name": "generic_nvt",
            "duration_ps": 1.0,
        }
        with self.assertRaises(ValidationError):
            load_config_dict(volume)


if __name__ == "__main__":
    unittest.main()
