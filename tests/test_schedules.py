import unittest

import numpy as np

from coldmd.schedules import (
    isotropic_cell_at_step,
    linear_pressure_plateaus,
    pressure_range_targets,
    volume_ratio_at_step,
)


class ScheduleTests(unittest.TestCase):
    def test_log_ramp_hits_exact_endpoints(self):
        self.assertEqual(volume_ratio_at_step(0.7, 0, 100), 1.0)
        self.assertAlmostEqual(volume_ratio_at_step(0.7, 100, 100), 0.7)

    def test_absolute_cell_has_target_volume(self):
        reference = np.diag([10.0, 11.0, 12.0])
        target = isotropic_cell_at_step(reference, 0.64, 100, 100)
        self.assertAlmostEqual(
            np.linalg.det(target) / np.linalg.det(reference), 0.64, places=13
        )

    def test_smoothstep_has_symmetric_log_path(self):
        first = np.log(volume_ratio_at_step(0.6, 25, 100, "log_smoothstep"))
        last = np.log(volume_ratio_at_step(0.6, 75, 100, "log_smoothstep"))
        self.assertAlmostEqual(first + last, np.log(0.6), places=13)

    def test_pressure_plateaus_are_inclusive(self):
        self.assertEqual(linear_pressure_plateaus(0.0, 20.0, 5), (0.0, 5.0, 10.0, 15.0, 20.0))

    def test_fixed_increment_pressure_ranges_are_decimal_safe_and_inclusive(self):
        self.assertEqual(
            pressure_range_targets(0.0, 0.3, 0.1),
            (0.0, 0.1, 0.2, 0.3),
        )
        self.assertEqual(
            pressure_range_targets(0.3, 0.0, 0.1),
            (0.3, 0.2, 0.1, 0.0),
        )

    def test_fixed_increment_pressure_range_rejects_irregular_endpoint(self):
        with self.assertRaisesRegex(ValueError, "exactly divisible"):
            pressure_range_targets(0.0, 1.0, 0.3)

    def test_fixed_increment_pressure_range_validates_step_and_count(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            pressure_range_targets(0.0, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "endpoints must differ"):
            pressure_range_targets(1.0, 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "above the maximum"):
            pressure_range_targets(0.0, 1.0, 0.1, max_points=10)


if __name__ == "__main__":
    unittest.main()
