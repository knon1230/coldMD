import unittest
from unittest.mock import patch

import numpy as np


class ProtocolEngineThermostatTests(unittest.TestCase):
    @staticmethod
    def _imports():
        try:
            from ase import Atoms
            from coldmd.engine import (
                FixedCellNVTStage,
                IsotropicNPTStage,
                NonExactResumeError,
                ProtocolEngine,
                RunCursor,
                StepContext,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE dynamics dependency is unavailable: {exc}")
        return (
            Atoms,
            FixedCellNVTStage,
            IsotropicNPTStage,
            NonExactResumeError,
            ProtocolEngine,
            RunCursor,
            StepContext,
        )

    @staticmethod
    def _atoms(Atoms):
        atoms = Atoms(
            "He",
            positions=[[1.0, 1.0, 1.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
        atoms.set_momenta([[0.1, -0.2, 0.3]])
        return atoms

    def test_bussi_mtk_and_nhc_dispatch_in_order_and_inherit_npt_cell(self):
        (
            Atoms,
            FixedCellNVTStage,
            IsotropicNPTStage,
            _,
            ProtocolEngine,
            _,
            _,
        ) = self._imports()
        atoms = self._atoms(Atoms)
        inherited_cell = np.asarray(
            [[9.0, 0.4, 0.0], [0.0, 8.5, 0.3], [0.0, 0.0, 8.0]],
            dtype=float,
        )
        initial_momenta = atoms.get_momenta().copy()
        calls = []

        class OneStepDynamics:
            def __init__(self, action=None):
                self.action = action
                self.nsteps = 0

            def step(self):
                if self.action is not None:
                    self.action()

        def fixed_builder(received_atoms, **kwargs):
            calls.append(
                (
                    "fixed",
                    kwargs["thermostat_kind"],
                    np.asarray(received_atoms.get_cell(), dtype=float).copy(),
                    kwargs,
                )
            )
            return OneStepDynamics()

        def npt_builder(received_atoms, **kwargs):
            calls.append(
                (
                    "npt",
                    "nose_hoover_chain",
                    np.asarray(received_atoms.get_cell(), dtype=float).copy(),
                    kwargs,
                )
            )
            return OneStepDynamics(
                lambda: received_atoms.set_cell(inherited_cell, scale_atoms=True)
            )

        stages = [
            FixedCellNVTStage(
                name="bussi",
                steps=1,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
                thermostat_kind="bussi",
            ),
            IsotropicNPTStage(
                name="mtk",
                steps=1,
                temperature_K=300.0,
                pressure_GPa=2.0,
                thermostat_tau_fs=100.0,
                barostat_tau_fs=1000.0,
                thermostat_chain_length=4,
                barostat_chain_length=5,
                thermostat_substeps=2,
                barostat_substeps=3,
            ),
            FixedCellNVTStage(
                name="nhc",
                steps=1,
                temperature_K=300.0,
                thermostat_tau_fs=80.0,
                thermostat_kind="nose_hoover_chain",
                thermostat_chain_length=6,
                thermostat_substeps=4,
            ),
        ]
        engine = ProtocolEngine(atoms, timestep_fs=0.5)

        with (
            patch("coldmd.engine.build_fixed_cell_nvt", side_effect=fixed_builder),
            patch("coldmd.engine.build_isotropic_mtk_npt", side_effect=npt_builder),
        ):
            cursor = engine.run(stages)

        self.assertEqual(
            [(kind, thermostat) for kind, thermostat, _, _ in calls],
            [
                ("fixed", "bussi"),
                ("npt", "nose_hoover_chain"),
                ("fixed", "nose_hoover_chain"),
            ],
        )
        np.testing.assert_allclose(calls[2][2], inherited_cell)
        np.testing.assert_allclose(np.asarray(atoms.get_cell()), inherited_cell)
        np.testing.assert_allclose(atoms.get_momenta(), initial_momenta)
        self.assertEqual(calls[2][3]["thermostat_chain_length"], 6)
        self.assertEqual(calls[2][3]["thermostat_substeps"], 4)
        self.assertEqual(calls[1][3]["thermostat_chain_length"], 4)
        self.assertEqual(calls[1][3]["barostat_chain_length"], 5)
        self.assertEqual(cursor.stage_index, 3)
        self.assertEqual(cursor.global_step, 3)

    def test_nhc_nvt_interior_cursor_is_rejected_but_bussi_is_resumable(self):
        (
            Atoms,
            FixedCellNVTStage,
            _,
            NonExactResumeError,
            ProtocolEngine,
            RunCursor,
            _,
        ) = self._imports()

        class OneStepDynamics:
            nsteps = 0

            def step(self):
                return None

        cell = tuple(tuple(float(value) for value in row) for row in np.eye(3) * 10.0)
        cursor = RunCursor(
            stage_index=0,
            stage_step=1,
            global_step=1,
            time_fs=0.5,
            stage_reference_cell_A=cell,
            reference_volume_A3=1000.0,
        )

        bussi_atoms = self._atoms(Atoms)
        bussi = FixedCellNVTStage(
            name="bussi",
            steps=3,
            temperature_K=300.0,
            thermostat_tau_fs=100.0,
            thermostat_kind="bussi",
        )
        with patch(
            "coldmd.engine.build_fixed_cell_nvt", return_value=OneStepDynamics()
        ) as build:
            result = ProtocolEngine(bussi_atoms, timestep_fs=0.5).run(
                [bussi], cursor=cursor
            )
        self.assertEqual(result.global_step, 3)
        self.assertEqual(build.call_args.kwargs["thermostat_kind"], "bussi")

        nhc_atoms = self._atoms(Atoms)
        nhc = FixedCellNVTStage(
            name="nhc",
            steps=3,
            temperature_K=300.0,
            thermostat_tau_fs=100.0,
            thermostat_kind="nose_hoover_chain",
        )
        with patch("coldmd.engine.build_fixed_cell_nvt") as build:
            with self.assertRaisesRegex(
                NonExactResumeError, "Nosé–Hoover-chain NVT.*stage-boundary"
            ):
                ProtocolEngine(nhc_atoms, timestep_fs=0.5).run(
                    [nhc], cursor=cursor
                )
        build.assert_not_called()

    def test_mtk_npt_interior_cursor_remains_rejected(self):
        (
            Atoms,
            _,
            IsotropicNPTStage,
            NonExactResumeError,
            ProtocolEngine,
            RunCursor,
            _,
        ) = self._imports()
        atoms = self._atoms(Atoms)
        stage = IsotropicNPTStage(
            name="npt",
            steps=3,
            temperature_K=300.0,
            pressure_GPa=0.0,
            thermostat_tau_fs=100.0,
            barostat_tau_fs=1000.0,
        )
        cell = tuple(tuple(float(value) for value in row) for row in np.eye(3) * 10.0)
        cursor = RunCursor(
            stage_index=0,
            stage_step=1,
            global_step=1,
            time_fs=0.5,
            stage_reference_cell_A=cell,
            reference_volume_A3=1000.0,
        )

        with self.assertRaisesRegex(
            NonExactResumeError, "isotropic MTK.*stage-boundary"
        ):
            ProtocolEngine(atoms, timestep_fs=0.5).run([stage], cursor=cursor)

    def test_exact_resume_support_is_boundary_only_for_extended_systems(self):
        *_, StepContext = self._imports()
        base = {
            "stage_index": 0,
            "stage_name": "stage",
            "stage_kind": "fixed_nvt",
            "stage_steps": 5,
            "global_step": 2,
            "time_fs": 1.0,
            "timestep_fs": 0.5,
            "progress": 0.4,
            "target_volume_ratio": None,
            "final_target_volume_ratio": None,
            "target_pressure_GPa": None,
            "branch": "other",
            "volume_ratio_from_engine_start": 1.0,
            "reference_volume_A3": 1000.0,
            "stage_reference_cell_A": tuple(
                tuple(float(value) for value in row) for row in np.eye(3) * 10.0
            ),
            "resumed": False,
        }

        for stage_step, expected in ((0, True), (2, False), (5, True)):
            context = StepContext(
                **base,
                stage_step=stage_step,
                thermostat_kind="nose_hoover_chain",
            )
            with self.subTest(stage_step=stage_step):
                self.assertIs(context.exact_resume_supported, expected)

        bussi = StepContext(
            **base, stage_step=2, thermostat_kind="bussi"
        )
        self.assertTrue(bussi.exact_resume_supported)

        mtk = StepContext(
            **{**base, "stage_kind": "isotropic_npt_plateau"},
            stage_step=2,
            thermostat_kind="nose_hoover_chain",
        )
        self.assertFalse(mtk.exact_resume_supported)


if __name__ == "__main__":
    unittest.main()
