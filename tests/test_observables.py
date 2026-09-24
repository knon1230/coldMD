import unittest

import numpy as np

from coldmd.observables import (
    AMU_A3_TO_G_CM3,
    EV_A3_TO_GPA,
    compute_thermo_record,
    kinetic_stress_tensor_eV_A3,
    pressure_from_stress_GPa,
    tensor_to_voigt,
    voigt_to_tensor,
    von_mises_stress_GPa,
)


class FakeAtoms:
    def __init__(self):
        self._momenta = np.zeros((2, 3))
        self._masses = np.array([10.0, 20.0])

    def __len__(self):
        return 2

    def get_momenta(self):
        return self._momenta.copy()

    def get_masses(self):
        return self._masses.copy()

    def get_all_distances(self, mic=True):
        return np.array([[0.0, 2.0], [2.0, 0.0]])

    def get_volume(self):
        return 100.0

    def get_kinetic_energy(self):
        return 0.3

    def get_temperature(self):
        return 300.0


class ObservableTests(unittest.TestCase):
    def test_pressure_sign(self):
        stress = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        self.assertGreater(pressure_from_stress_GPa(stress), 0.0)

    def test_record_uses_consistent_supplied_results(self):
        atoms = FakeAtoms()
        record = compute_thermo_record(
            atoms,
            step=5,
            time_fs=2.5,
            stage="test",
            forces=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            potential_energy_eV=-2.0,
            stress_eV_A3=np.zeros(6),
        )
        self.assertEqual(record.step, 5)
        self.assertAlmostEqual(record.total_energy_eV, -1.7)
        self.assertAlmostEqual(record.density_g_cm3, 30.0 * AMU_A3_TO_G_CM3 / 100.0)
        self.assertEqual(record.min_distance_A, 2.0)

    def test_one_ev_per_a3_compression_has_positive_gpa_pressure(self):
        stress = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            pressure_from_stress_GPa(stress), EV_A3_TO_GPA, places=10
        )
        self.assertAlmostEqual(von_mises_stress_GPa(stress), 0.0, places=10)

    def test_voigt_order_round_trip_matches_ase_convention(self):
        voigt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tensor = voigt_to_tensor(voigt)
        np.testing.assert_array_equal(
            tensor,
            np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]),
        )
        np.testing.assert_array_equal(tensor_to_voigt(tensor), voigt)

    def test_pure_shear_von_mises_uses_the_full_symmetric_tensor(self):
        pure_xy_shear = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(
            von_mises_stress_GPa(pure_xy_shear),
            np.sqrt(3.0) * EV_A3_TO_GPA,
            places=10,
        )

    def test_kinetic_stress_contributes_positive_total_pressure(self):
        atoms = FakeAtoms()
        atoms._momenta = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        kinetic_stress = kinetic_stress_tensor_eV_A3(atoms, 100.0)
        np.testing.assert_allclose(
            kinetic_stress,
            np.diag([-2.0**2 / 10.0 / 100.0, -3.0**2 / 20.0 / 100.0, 0.0]),
        )
        expected_pressure = -np.trace(kinetic_stress) * EV_A3_TO_GPA / 3.0

        record = compute_thermo_record(
            atoms,
            step=1,
            time_fs=0.5,
            stage="hold",
            forces=np.zeros((2, 3)),
            potential_energy_eV=0.0,
            stress_eV_A3=np.zeros(6),
        )
        self.assertAlmostEqual(record.configurational_pressure_GPa, 0.0)
        self.assertAlmostEqual(record.total_pressure_GPa, expected_pressure)
        self.assertGreater(record.total_pressure_GPa, 0.0)


if __name__ == "__main__":
    unittest.main()
