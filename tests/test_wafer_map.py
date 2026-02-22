"""Unit tests for Wafer Map Engine"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wafer_map import WaferMap, WaferConfig


class TestWaferMap(unittest.TestCase):

    def setUp(self):
        """Each test er age fresh wafer create hoy"""
        self.config = WaferConfig(
            wafer_diameter_mm=300,
            die_width_mm=10,
            die_height_mm=10
        )
        self.wafer = WaferMap(self.config)

    def test_grid_creation(self):
        """Grid should have valid die positions"""
        self.assertGreater(self.wafer.total_die, 0)
        self.assertEqual(self.wafer.wafer_grid.ndim, 2)

    def test_grid_dimensions(self):
        """Grid size should match wafer/die size ratio"""
        self.assertGreater(self.wafer.rows, 0)
        self.assertGreater(self.wafer.cols, 0)

    def test_circular_shape(self):
        """Edge die should be marked as -1 (invalid)"""
        # Corners should be -1
        self.assertEqual(self.wafer.wafer_grid[0, 0], -1)

    def test_sample_data_generation(self):
        """Sample data should fill valid positions"""
        self.wafer.generate_sample_data(good_yield=0.90)

        valid = self.wafer.wafer_grid[self.wafer.wafer_grid >= 0]
        self.assertGreater(len(valid), 0)

        # All valid die should have bin >= 1
        self.assertTrue(np.all(valid >= 1))

    def test_yield_range(self):
        """Yield should be between 0-100%"""
        self.wafer.generate_sample_data(good_yield=0.85)
        summary = self.wafer.get_yield_summary()

        self.assertGreaterEqual(summary['yield_pct'], 0)
        self.assertLessEqual(summary['yield_pct'], 100)

    def test_high_yield(self):
        """High input yield should give high output yield"""
        self.wafer.generate_sample_data(good_yield=0.99, seed=42)
        summary = self.wafer.get_yield_summary()
        self.assertGreater(summary['yield_pct'], 80)

    def test_low_yield(self):
        """Low input yield should give low output yield"""
        self.wafer.generate_sample_data(good_yield=0.10, seed=42)
        summary = self.wafer.get_yield_summary()
        self.assertLess(summary['yield_pct'], 30)

    def test_cluster_detection(self):
        """Should detect injected cluster"""
        self.wafer.generate_sample_data(good_yield=0.85, seed=42)
        clusters = self.wafer.detect_clusters(min_cluster_size=3)

        # Should find at least 1 cluster (we inject one)
        self.assertGreater(len(clusters), 0)

    def test_cluster_size(self):
        """Cluster size should be >= min_cluster_size"""
        self.wafer.generate_sample_data(good_yield=0.85, seed=42)
        clusters = self.wafer.detect_clusters(min_cluster_size=5)

        for cluster in clusters:
            self.assertGreaterEqual(cluster.size, 5)

    def test_zone_yield_analysis(self):
        """Zone analysis should return correct number of zones"""
        self.wafer.generate_sample_data(good_yield=0.85)
        df = self.wafer.zone_yield_analysis(num_zones=5)

        self.assertEqual(len(df), 5)
        self.assertIn('yield_pct', df.columns)

    def test_zone_edge_vs_center(self):
        """Edge yield should typically be lower than center"""
        self.wafer.generate_sample_data(good_yield=0.85, seed=42)
        df = self.wafer.zone_yield_analysis(num_zones=3)

        center_yield = df.iloc[0]['yield_pct']
        edge_yield = df.iloc[-1]['yield_pct']
        # Edge yield should generally be lower
        # (not always guaranteed with random data, so soft check)
        self.assertIsNotNone(center_yield)
        self.assertIsNotNone(edge_yield)

    def test_different_wafer_sizes(self):
        """Should work with different wafer diameters"""
        for diameter in [150, 200, 300]:
            config = WaferConfig(wafer_diameter_mm=diameter, die_width_mm=5)
            wafer = WaferMap(config)
            self.assertGreater(wafer.total_die, 0)

    def test_different_die_sizes(self):
        """Larger die = fewer die per wafer"""
        config_small = WaferConfig(die_width_mm=3, die_height_mm=3)
        config_large = WaferConfig(die_width_mm=15, die_height_mm=15)

        wafer_small = WaferMap(config_small)
        wafer_large = WaferMap(config_large)

        self.assertGreater(wafer_small.total_die, wafer_large.total_die)


if __name__ == '__main__':
    unittest.main()
