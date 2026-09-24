import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import numpy as np


class FixedCellThermostatDispatchTests(unittest.TestCase):
    @staticmethod
    def _builder():
        try:
            from coldmd.dynamics import build_fixed_cell_nvt
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE dynamics dependency is unavailable: {exc}")
        return build_fixed_cell_nvt

    def test_dispatches_bussi_and_forwards_rng(self):
        build_fixed_cell_nvt = self._builder()
        atoms = SimpleNamespace()
        with patch(
            "coldmd.dynamics.build_bussi_nvt", return_value=sentinel.bussi
        ) as build:
            result = build_fixed_cell_nvt(
                atoms,
                thermostat_kind="bussi",
                timestep_fs=0.5,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
                rng=sentinel.rng,
                thermostat_chain_length=7,
                thermostat_substeps=4,
                logfile=None,
            )

        self.assertIs(result, sentinel.bussi)
        build.assert_called_once_with(
            atoms,
            timestep_fs=0.5,
            temperature_K=300.0,
            thermostat_tau_fs=100.0,
            rng=sentinel.rng,
            logfile=None,
        )

    def test_dispatches_nose_hoover_without_forwarding_rng(self):
        build_fixed_cell_nvt = self._builder()
        atoms = SimpleNamespace()
        with patch(
            "coldmd.dynamics.build_nose_hoover_chain_nvt",
            return_value=sentinel.nhc,
        ) as build:
            result = build_fixed_cell_nvt(
                atoms,
                thermostat_kind="nose_hoover_chain",
                timestep_fs=0.5,
                temperature_K=300.0,
                thermostat_tau_fs=80.0,
                rng=sentinel.must_not_be_forwarded,
                thermostat_chain_length=5,
                thermostat_substeps=2,
                logfile=None,
            )

        self.assertIs(result, sentinel.nhc)
        build.assert_called_once_with(
            atoms,
            timestep_fs=0.5,
            temperature_K=300.0,
            thermostat_tau_fs=80.0,
            thermostat_chain_length=5,
            thermostat_substeps=2,
            logfile=None,
        )

    def test_rejects_unknown_thermostat_kind(self):
        build_fixed_cell_nvt = self._builder()
        with self.assertRaisesRegex(ValueError, "bussi.*nose_hoover_chain"):
            build_fixed_cell_nvt(
                SimpleNamespace(),
                thermostat_kind="unknown",
                timestep_fs=0.5,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
            )


class NoseHooverChainDynamicsTests(unittest.TestCase):
    @staticmethod
    def _thermal_atoms():
        try:
            from ase import Atoms
            from ase.calculators.lj import LennardJones
            from coldmd.dynamics import initialize_momenta_once
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE dynamics dependency is unavailable: {exc}")

        atoms = Atoms(
            "Ar4",
            positions=[
                [1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0],
                [1.0, 1.0, 4.0],
            ],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        )
        atoms.calc = LennardJones(rc=3.5)
        initialize_momenta_once(
            atoms,
            300.0,
            np.random.default_rng(20260905),
            exact_temperature=True,
        )
        return atoms

    def test_real_ase_nhc_short_run_is_finite_and_keeps_full_cell_fixed(self):
        try:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
            from coldmd.dynamics import build_nose_hoover_chain_nvt
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE NHC dependency is unavailable: {exc}")

        atoms = self._thermal_atoms()
        initial_cell = np.asarray(atoms.get_cell(), dtype=float).copy()
        dynamics = build_nose_hoover_chain_nvt(
            atoms,
            timestep_fs=0.1,
            temperature_K=300.0,
            thermostat_tau_fs=25.0,
            thermostat_chain_length=3,
            thermostat_substeps=2,
            logfile=None,
        )

        self.assertIsInstance(dynamics, NoseHooverChainNVT)
        dynamics.run(5)

        np.testing.assert_array_equal(np.asarray(atoms.get_cell()), initial_cell)
        self.assertTrue(np.all(np.isfinite(atoms.get_positions())))
        self.assertTrue(np.all(np.isfinite(atoms.get_momenta())))
        self.assertTrue(np.isfinite(atoms.get_potential_energy()))
        self.assertGreater(atoms.get_kinetic_energy(), 0.0)

    def test_builder_rejects_constraints_before_constructing_ase_nhc(self):
        try:
            from ase.constraints import FixAtoms
            from coldmd.dynamics import build_nose_hoover_chain_nvt
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE NHC dependency is unavailable: {exc}")

        atoms = self._thermal_atoms()
        atoms.set_constraint(FixAtoms(indices=[0]))
        with self.assertRaisesRegex(ValueError, "does not support.*constraints"):
            build_nose_hoover_chain_nvt(
                atoms,
                timestep_fs=0.5,
                temperature_K=300.0,
                thermostat_tau_fs=100.0,
            )

    def test_builder_strictly_validates_chain_length_and_substeps(self):
        try:
            from coldmd.dynamics import build_nose_hoover_chain_nvt
        except ImportError as exc:
            raise unittest.SkipTest(f"ASE NHC dependency is unavailable: {exc}")

        cases = (
            {"thermostat_chain_length": 0},
            {"thermostat_chain_length": 1.5},
            {"thermostat_chain_length": True},
            {"thermostat_substeps": 0},
            {"thermostat_substeps": 1.5},
            {"thermostat_substeps": True},
        )
        for overrides in cases:
            atoms = self._thermal_atoms()
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    build_nose_hoover_chain_nvt(
                        atoms,
                        timestep_fs=0.5,
                        temperature_K=300.0,
                        thermostat_tau_fs=100.0,
                        **overrides,
                    )


if __name__ == "__main__":
    unittest.main()
