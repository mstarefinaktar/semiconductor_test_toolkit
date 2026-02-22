"""Unit tests for Shmoo Plot Engine"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shmoo_plot import ShmooEngine, ShmooConfig


class TestShmooEngine(unittest.TestCase):

    def setUp(self):
        self.config = ShmooConfig(
            x_start=100, x_stop=500, x_step=10,
            y_start=0.8, y_stop=1.2, y_step=0.01
        )
        self.shmoo = ShmooEngine(self.config)

    def test_grid_dimensions(self):
        """Grid should match config parameters"""
        self.assertEqual(len(self.shmoo.x_values), 41)  # (500-100)/10 + 1
        self.assertGreater(len(self.shmoo.y_values), 0)

    def test_initial_data_zeros(self):
        """Initial shmoo data should be all zeros"""
        self.assertTrue(np.all(self.shmoo.shmoo_data == 0))

    def test_generate_shmoo(self):
        """Generated shmoo should have both pass and fail"""
        self.shmoo.generate_realistic_shmoo()

        has_pass = np.any(self.shmoo.shmoo_data == 1)
        has_fail = np.any(self.shmoo.shmoo_data == 0)

        self.assertTrue(has_pass)
        self.assertTrue(has_fail)

    def test_shmoo_values_binary(self):
        """Shmoo data should be only 0 or 1"""
        self.shmoo.generate_realistic_shmoo()

        unique_vals = np.unique(self.shmoo.shmoo_data)
        for val in unique_vals:
            self.assertIn(val, [0, 1])

    def test_high_voltage_more_pass(self):
        """Higher voltage rows should have more passes"""
        self.shmoo.generate_realistic_shmoo(noise=0.01)

        # Top row (highest voltage)
        top_passes = np.sum(self.shmoo.shmoo_data[-1, :])
        # Bottom row (lowest voltage)
        bottom_passes = np.sum(self.shmoo.shmoo_data[0, :])

        self.assertGreaterEqual(top_passes, bottom_passes)

    def test_operating_window(self):
        """Should find valid operating window"""
        self.shmoo.generate_realistic_shmoo()
        window = self.shmoo.find_operating_window()

        self.assertNotIn('error', window)
        self.assertGreater(window['voltage_range'], 0)

    def test_reproducibility(self):
        """Same seed should give same results"""
        shmoo1 = ShmooEngine(self.config)
        shmoo1.generate_realistic_shmoo(seed=42)

        shmoo2 = ShmooEngine(self.config)
        shmoo2.generate_realistic_shmoo(seed=42)

        np.testing.assert_array_equal(shmoo1.shmoo_data, shmoo2.shmoo_data)

    def test_different_seeds(self):
        """Different seeds should give different results"""
        shmoo1 = ShmooEngine(self.config)
        shmoo1.generate_realistic_shmoo(seed=42)

        shmoo2 = ShmooEngine(self.config)
        shmoo2.generate_realistic_shmoo(seed=99)

        self.assertFalse(np.array_equal(shmoo1.shmoo_data, shmoo2.shmoo_data))


if __name__ == '__main__':
    unittest.main()
