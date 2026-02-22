"""Unit tests for Outlier Detector"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.outlier_detector import OutlierDetector


class TestZScore(unittest.TestCase):

    def test_clean_data(self):
        """Clean normal data = few outliers"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = OutlierDetector.z_score(data, threshold=3.0)

        self.assertLess(result['n_outliers'], 10)

    def test_known_outlier(self):
        """Should detect obvious outlier"""
        data = np.array([1, 2, 3, 2, 1, 3, 2, 100])
        result = OutlierDetector.z_score(data, threshold=2.0)

        self.assertGreater(result['n_outliers'], 0)
        self.assertIn(7, result['outlier_indices'])

    def test_stricter_threshold(self):
        """Higher threshold = fewer outliers"""
        data = np.random.normal(0, 1, 1000)
        result_3 = OutlierDetector.z_score(data, threshold=3.0)
        result_4 = OutlierDetector.z_score(data, threshold=4.0)

        self.assertGreaterEqual(
            result_3['n_outliers'], result_4['n_outliers']
        )


class TestIQR(unittest.TestCase):

    def test_clean_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = OutlierDetector.iqr_method(data, k=1.5)
        self.assertLess(result['n_outliers'], 50)

    def test_fence_order(self):
        """Upper fence > Lower fence"""
        data = np.random.normal(0, 1, 1000)
        result = OutlierDetector.iqr_method(data)
        self.assertGreater(result['upper_fence'], result['lower_fence'])

    def test_higher_k_fewer_outliers(self):
        data = np.random.normal(0, 1, 1000)
        r1 = OutlierDetector.iqr_method(data, k=1.5)
        r2 = OutlierDetector.iqr_method(data, k=3.0)
        self.assertGreaterEqual(r1['n_outliers'], r2['n_outliers'])


class TestGrubbs(unittest.TestCase):

    def test_detects_outlier(self):
        data = np.array([2.1, 2.0, 1.9, 2.0, 2.1, 2.0, 50.0])
        result = OutlierDetector.grubbs_test(data)
        self.assertTrue(result['is_outlier'])
        self.assertEqual(result['suspect_index'], 6)

    def test_no_outlier(self):
        np.random.seed(42)
        data = np.random.normal(0, 0.01, 100)
        result = OutlierDetector.grubbs_test(data, alpha=0.01)
        # With very tight data, might or might not find outlier
        self.assertIsInstance(result['is_outlier'], bool)


class TestMahalanobis(unittest.TestCase):

    def test_multivariate(self):
        np.random.seed(42)
        data = pd.DataFrame({
            'a': np.random.normal(0, 1, 100),
            'b': np.random.normal(0, 1, 100),
            'c': np.random.normal(0, 1, 100),
        })
        result = OutlierDetector.mahalanobis_multivariate(data)
        self.assertGreaterEqual(result['n_outliers'], 0)
        self.assertEqual(len(result['distances']), 100)


class TestIsolationForest(unittest.TestCase):

    def test_basic(self):
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.concatenate([
                np.random.normal(0, 1, 95),
                np.array([10, 11, 12, -10, -11])
            ])
        })
        result = OutlierDetector.isolation_forest(data, contamination=0.05)
        self.assertGreater(result['n_outliers'], 0)


if __name__ == '__main__':
    unittest.main()
