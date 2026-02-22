"""Unit tests for yield analyzer"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yield_analyzer import YieldModels, ProcessCapability, PATScreening


class TestYieldModels(unittest.TestCase):

    def test_poisson_zero_defects(self):
        self.assertAlmostEqual(YieldModels.poisson(0, 1.0), 1.0)

    def test_poisson_high_defects(self):
        result = YieldModels.poisson(10, 1.0)
        self.assertLess(result, 0.01)

    def test_murphy(self):
        result = YieldModels.murphy(0, 1.0)
        self.assertAlmostEqual(result, 1.0)

    def test_negative_binomial(self):
        result = YieldModels.negative_binomial(0, 1.0, alpha=2.0)
        self.assertAlmostEqual(result, 1.0)


class TestProcessCapability(unittest.TestCase):

    def test_centered_process(self):
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=0.1, size=1000)
        result = ProcessCapability.calculate(data, lsl=4.5, usl=5.5)

        self.assertGreater(result['cpk'], 1.0)
        self.assertGreater(result['cp'], 1.0)
        self.assertLess(result['ppm_total'], 100)

    def test_off_center_process(self):
        np.random.seed(42)
        data = np.random.normal(loc=5.3, scale=0.1, size=1000)
        result = ProcessCapability.calculate(data, lsl=4.5, usl=5.5)

        self.assertLess(result['cpk'], result['cp'])

    def test_zero_std(self):
        data = np.array([5.0, 5.0, 5.0])
        result = ProcessCapability.calculate(data, lsl=4.5, usl=5.5)
        self.assertIn('error', result)


class TestPATScreening(unittest.TestCase):

    def test_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(2.5, 0.1, 1000)
        result = PATScreening.dynamic_pat(data)
        self.assertLess(result['far_outliers'], 5)

    def test_with_outliers(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(2.5, 0.1, 990),
            np.array([10.0, -5.0, 15.0])
        ])
        result = PATScreening.dynamic_pat(data)
        self.assertGreater(result['far_outliers'], 0)


if __name__ == '__main__':
    unittest.main()
