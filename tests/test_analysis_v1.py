"""Focused coverage for stage summaries and instantaneous oxygen topology."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    from ase import Atoms
except ImportError:  # pragma: no cover - package smoke environments lack ASE
    Atoms = None

if Atoms is None:  # Avoid importing coldmd.analysis, which correctly requires ASE.
    raise unittest.SkipTest("ASE is required for analysis-v1 tests")

from coldmd.analysis import (
    AnalysisError,
    FrameRecord,
    oxygen_speciation,
    stage_thermo_statistics,
    topology_resolved_rdf,
)


class StageThermoStatisticsTests(unittest.TestCase):
    def test_config_order_tail_sample_std_pressure_policy_and_boundary_duplicate(self):
        rows = [
            {
                "step": 0,
                "stage_step": 0,
                "time_ps": 0.0,
                "stage": "initial",
                "total_pressure_GPa": 0.0,
                "configurational_pressure_GPa": 99.0,
                "volume_A3": 100.0,
                "density_g_cm3": 2.0,
                "temperature_K": 300.0,
            },
            {
                "step": 1,
                "stage_step": 1,
                "time_ps": 1.0,
                "stage": "initial",
                "total_pressure_GPa": 2.0,
                "configurational_pressure_GPa": 99.0,
                "volume_A3": 98.0,
                "density_g_cm3": 2.1,
                "temperature_K": 301.0,
            },
            {
                "step": 2,
                "stage_step": 2,
                "time_ps": 2.0,
                "stage": "initial",
                "total_pressure_GPa": 4.0,
                "configurational_pressure_GPa": 99.0,
                "volume_A3": 96.0,
                "density_g_cm3": 2.2,
                "temperature_K": 302.0,
            },
            # The next stage can start at the endpoint.  It is a distinct
            # stage-local observation and must not be erased by de-duplication.
            {
                "step": 2,
                "stage_step": 0,
                "time_ps": 2.0,
                "stage": "plateau",
                "total_pressure_GPa": 900.0,
                "volume_A3": 1.0,
                "density_g_cm3": 99.0,
                "temperature_K": 999.0,
            },
            {
                "step": 3,
                "stage_step": 1,
                "time_ps": 3.0,
                "stage": "plateau",
                "configurational_pressure_GPa": 9.0,
                "volume_A3": 94.0,
                "density_g_cm3": 2.3,
                "temperature_K": 303.0,
            },
            {
                "step": 4,
                "stage_step": 2,
                "time_ps": 4.0,
                "stage": "plateau",
                "configurational_pressure_GPa": 11.0,
                "volume_A3": 92.0,
                "density_g_cm3": 2.4,
                "temperature_K": 304.0,
            },
            # A resumed stream can repeat a stage-local step.  The latest row
            # replaces the first one, while the distinct boundary record above
            # remains in the plateau stage.
            {
                "step": 4,
                "stage_step": 2,
                "time_ps": 4.0,
                "stage": "plateau",
                "configurational_pressure_GPa": 13.0,
                "volume_A3": 91.0,
                "density_g_cm3": 2.5,
                "temperature_K": 305.0,
            },
        ]
        protocol = [
            {
                "name": "initial",
                "kind": "initial_nvt",
                "duration_ps": 2.0,
                "sampling_window_ps": 1.0,
            },
            {
                "name": "plateau",
                "kind": "pressure_plateau",
                "branch": "cold_compression",
                "target_pressure_GPa": 1.0,
                "duration_ps": 2.0,
                "sampling_window_ps": 1.0,
            },
            {
                "name": "configured_but_missing",
                "kind": "pressure_plateau",
                "duration_ps": 2.0,
                "sampling_window_ps": 1.0,
            },
        ]

        result = stage_thermo_statistics(rows, protocol_stages=protocol)
        initial, plateau, missing = result.summaries

        self.assertEqual([item.stage for item in result.summaries], [
            "initial", "plateau", "configured_but_missing"
        ])
        self.assertEqual(result.dropped_duplicate_rows, 1)
        self.assertEqual(initial.selection_method, "stored_time_ps_tail")
        self.assertEqual(initial.n_samples, 2)
        self.assertAlmostEqual(initial.pressure_mean_GPa, 3.0)
        self.assertAlmostEqual(initial.pressure_std_GPa, np.sqrt(2.0))
        self.assertEqual(initial.pressure_source, "total_pressure_GPa")
        self.assertEqual(plateau.available_rows, 3)
        self.assertAlmostEqual(plateau.pressure_mean_GPa, 11.0)
        self.assertAlmostEqual(plateau.pressure_std_GPa, np.sqrt(8.0))
        self.assertEqual(
            plateau.pressure_source,
            "configurational_pressure_GPa (legacy fallback)",
        )
        self.assertEqual(missing.status, "missing_stage_data")
        self.assertIsNone(missing.pressure_std_GPa)

    def test_one_observation_has_explicitly_missing_sample_standard_deviation(self):
        result = stage_thermo_statistics(
            [
                {
                    "step": 1,
                    "stage_step": 1,
                    "time_ps": 0.1,
                    "stage": "one",
                    "total_pressure_GPa": 2.0,
                    "volume_A3": 100.0,
                    "density_g_cm3": 2.0,
                    "temperature_K": 300.0,
                }
            ],
            protocol_stages=[{"name": "one", "kind": "initial_nvt"}],
        )
        summary = result.summaries[0]
        self.assertEqual(summary.n_samples, 1)
        self.assertIsNone(summary.pressure_std_GPa)
        self.assertIsNone(summary.volume_std_A3)
        self.assertIsNone(summary.temperature_std_K)

    def test_generic_sequence_branches_use_explicit_aliases_and_neutral_fallback(self):
        protocol = [
            {
                "name": "load_npt",
                "kind": "npt",
                "branch": "cold_compression",
                "target_pressure_GPa": 2.5,
                "duration_ps": 1.0,
            },
            {
                "name": "load_nvt",
                "kind": "nvt",
                "branch": "loading",
                "duration_ps": 1.0,
            },
            {
                "name": "neutral_npt",
                "kind": "npt",
                "target_pressure_GPa": 0.0,
                "duration_ps": 1.0,
            },
            {
                "name": "neutral_nvt",
                "kind": "nvt",
                "duration_ps": 1.0,
            },
        ]
        rows = [
            {
                "stage": stage["name"],
                "stage_step": 0,
                "time_ps": float(index),
                "total_pressure_GPa": float(index),
                "volume_A3": 100.0 - index,
                "density_g_cm3": 2.0,
                "temperature_K": 300.0,
            }
            for index, stage in enumerate(protocol)
        ]
        result = stage_thermo_statistics(rows, protocol_stages=protocol)

        self.assertEqual(
            [summary.branch for summary in result.summaries],
            ["compression", "compression", "other", "other"],
        )
        self.assertEqual(result.summaries[0].target_pressure_GPa, 2.5)
        self.assertIsNone(result.summaries[1].target_pressure_GPa)


@unittest.skipIf(Atoms is None, "ASE is required for geometry analysis tests")
class OxygenTopologyTests(unittest.TestCase):
    @staticmethod
    def _frames() -> list[FrameRecord]:
        # O at x=9.2 is bonded across the periodic x boundary to Si at x=0.2.
        # Frame 2 moves Si(2), converting the BO and O-other populations so the
        # topology pair populations genuinely change between frames.
        symbols = ["Ca", "Si", "Si", "Si", "O", "O", "O"]
        first = Atoms(
            symbols=symbols,
            positions=[
                [2.0, 2.0, 2.0],  # Ca
                [0.2, 0.0, 0.0],  # Si: NBO O across boundary
                [4.0, 5.0, 5.0],  # Si: BO O
                [6.0, 5.0, 5.0],  # Si: BO O
                [9.2, 0.0, 0.0],  # NBO
                [5.0, 5.0, 5.0],  # BO
                [8.0, 8.0, 8.0],  # O_other
            ],
            cell=np.eye(3) * 10.0,
            pbc=True,
        )
        second = first.copy()
        second.positions[3] = [8.5, 8.0, 8.0]
        return [
            FrameRecord(first, Path("frame-0.extxyz"), 0, 0, 0, 0.0, "stage-a", 0),
            FrameRecord(second, Path("frame-1.extxyz"), 0, 1, 1, 1.0, "stage-a", 1),
        ]

    def test_framewise_classification_uses_triclinic_mic_and_tracks_fractions(self):
        result = oxygen_speciation(self._frames(), si_o_cutoff_A=1.5)
        first, second = result.frames
        self.assertEqual((first.n_nbo, first.n_bo, first.n_other), (1, 1, 1))
        self.assertEqual((second.n_nbo, second.n_bo, second.n_other), (3, 0, 0))
        self.assertAlmostEqual(first.fraction_nbo, 1.0 / 3.0)
        self.assertAlmostEqual(second.fraction_nbo, 1.0)
        self.assertEqual(result.cutoffs_A_by_stage, {"stage-a": 1.5})
        self.assertEqual(
            result.cutoff_source_by_stage["stage-a"], "fixed si_o_cutoff_A override"
        )
        self.assertIn("triclinic MIC", result.metadata["classification_mode"])

    def test_topology_rdf_uses_dynamic_per_frame_ideal_pair_count(self):
        result = topology_resolved_rdf(
            self._frames(),
            pairs=[("Si", "NBO")],
            si_o_cutoff_A=1.5,
            bins=20,
            rmax_A=4.0,
            include_total=False,
        )
        shell_volume = (4.0 * np.pi / 3.0) * (
            result.bin_edges_A[1:] ** 3 - result.bin_edges_A[:-1] ** 3
        )
        # Frame 1 has 3 Si * 1 NBO possibilities; frame 2 has 3 Si * 3 NBO.
        np.testing.assert_allclose(
            result.ideal_counts["Si-NBO"], (3.0 + 9.0) * shell_volume / 1000.0
        )
        speciation = result.metadata["oxygen_speciation"]
        self.assertEqual(speciation["frames"][0]["n_bo"], 1)
        self.assertEqual(speciation["frames"][1]["n_bo"], 0)

    def test_auto_cutoff_is_resolved_per_stage_and_fixed_cutoff_is_checked(self):
        frames = self._frames()
        second_stage = frames[1]
        frames = [frames[0], FrameRecord(
            second_stage.atoms,
            second_stage.segment,
            second_stage.segment_index,
            second_stage.frame_index,
            second_stage.step,
            second_stage.time_fs,
            "stage-b",
            second_stage.stage_step,
        )]
        fake_rdf = SimpleNamespace(
            r_A=np.linspace(0.0, 4.0, 30),
            g={"O-Si": np.ones(30)},
        )
        with patch("coldmd.analysis.partial_rdf", return_value=fake_rdf), patch(
            "coldmd.analysis.estimate_first_minimum", side_effect=[1.4, 1.6]
        ) as minimum:
            result = oxygen_speciation(frames)
        self.assertEqual(minimum.call_count, 2)
        self.assertEqual(result.cutoffs_A_by_stage, {"stage-a": 1.4, "stage-b": 1.6})
        self.assertTrue(all(
            source == "first minimum of stage-specific Si-O RDF"
            for source in result.cutoff_source_by_stage.values()
        ))
        with self.assertRaises(AnalysisError):
            oxygen_speciation(self._frames(), si_o_cutoff_A=5.1)


if __name__ == "__main__":
    unittest.main()
