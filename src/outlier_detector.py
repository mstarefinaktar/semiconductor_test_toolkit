"""
Outlier Detection Engine
=========================

Multi-method outlier detection for semiconductor test data.
Methods: Z-score, IQR, Grubbs, Mahalanobis, Isolation Forest

Author: Mst Arefin Aktar
Date: 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
import matplotlib.pyplot as plt


class OutlierDetector:
    """Multi-method outlier detection for semiconductor test data"""

    @staticmethod
    def z_score(data: np.ndarray, threshold: float = 3.0) -> dict:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold

        return {
            'method': 'Z-Score',
            'threshold': threshold,
            'outlier_mask': outlier_mask,
            'n_outliers': int(np.sum(outlier_mask)),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
        }

    @staticmethod
    def iqr_method(data: np.ndarray, k: float = 1.5) -> dict:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        outlier_mask = (data < lower) | (data > upper)

        return {
            'method': f'IQR (k={k})',
            'lower_fence': lower, 'upper_fence': upper,
            'outlier_mask': outlier_mask,
            'n_outliers': int(np.sum(outlier_mask)),
        }

    @staticmethod
    def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> dict:
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        abs_dev = np.abs(data - mean)
        max_idx = np.argmax(abs_dev)
        g_stat = abs_dev[max_idx] / std

        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))

        return {
            'method': 'Grubbs Test',
            'suspect_index': int(max_idx),
            'suspect_value': float(data[max_idx]),
            'g_statistic': round(g_stat, 4),
            'g_critical': round(g_crit, 4),
            'is_outlier': bool(g_stat > g_crit),
        }

    @staticmethod
    def mahalanobis_multivariate(data: pd.DataFrame,
                                  threshold_percentile: float = 97.5) -> dict:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        cov_matrix = np.cov(scaled.T)
        cov_inv = np.linalg.pinv(cov_matrix)
        mean = np.mean(scaled, axis=0)

        distances = np.array([mahalanobis(scaled[i], mean, cov_inv) for i in range(len(scaled))])

        dof = data.shape[1]
        threshold = np.sqrt(stats.chi2.ppf(threshold_percentile / 100, dof))
        outlier_mask = distances > threshold

        return {
            'method': 'Mahalanobis Distance',
            'distances': distances,
            'threshold': threshold,
            'outlier_mask': outlier_mask,
            'n_outliers': int(np.sum(outlier_mask)),
        }

    @staticmethod
    def isolation_forest(data: pd.DataFrame, contamination: float = 0.01) -> dict:
        clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        predictions = clf.fit_predict(data)
        scores = clf.decision_function(data)
        outlier_mask = predictions == -1

        return {
            'method': 'Isolation Forest',
            'outlier_mask': outlier_mask,
            'n_outliers': int(np.sum(outlier_mask)),
            'anomaly_scores': scores,
        }

    @staticmethod
    def compare_methods(data: np.ndarray, test_name: str = "Test",
                        save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        methods = [
            ('Z-Score (3σ)', OutlierDetector.z_score(data, 3.0)),
            ('Z-Score (4σ)', OutlierDetector.z_score(data, 4.0)),
            ('IQR (1.5x)', OutlierDetector.iqr_method(data, 1.5)),
            ('IQR (3.0x)', OutlierDetector.iqr_method(data, 3.0)),
        ]

        for ax, (name, result) in zip(axes.flat, methods):
            mask = result['outlier_mask']
            colors = ['red' if m else 'green' for m in mask]
            ax.scatter(range(len(data)), data, c=colors, s=10, alpha=0.5)
            ax.set_title(f'{name}: {result["n_outliers"]} outliers', fontweight='bold')

        plt.suptitle(f'Outlier Detection: {test_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
