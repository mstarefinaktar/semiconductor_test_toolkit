"""
Yield Analytics Engine
=======================

Industry-standard yield analysis:
- Poisson, Murphy, Seeds, Negative Binomial yield models
- Cp/Cpk Process Capability
- PAT (Part Average Testing) Outlier Screening

Author: Mst Arefin Aktar
Date: 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict
import matplotlib.pyplot as plt


class YieldModels:
    """Semiconductor Yield Prediction Models"""

    @staticmethod
    def poisson(D: float, A: float) -> float:
        """Y = e^(-D*A)"""
        return np.exp(-D * A)

    @staticmethod
    def murphy(D: float, A: float) -> float:
        """Y = ((1-e^(-DA))/(DA))^2"""
        DA = D * A
        if DA == 0:
            return 1.0
        return ((1 - np.exp(-DA)) / DA) ** 2

    @staticmethod
    def seeds(D: float, A: float) -> float:
        """Y = 1/(1+DA)"""
        return 1 / (1 + D * A)

    @staticmethod
    def negative_binomial(D: float, A: float, alpha: float = 2.0) -> float:
        """Y = (1+DA/α)^(-α) -- Industry Standard"""
        return (1 + D * A / alpha) ** (-alpha)

    @staticmethod
    def compare_models(D_range: np.ndarray, die_area: float,
                       save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 7))

        models = {
            'Poisson': lambda d: YieldModels.poisson(d, die_area),
            'Murphy': lambda d: YieldModels.murphy(d, die_area),
            'Seeds': lambda d: YieldModels.seeds(d, die_area),
            'Neg.Binom (α=1)': lambda d: YieldModels.negative_binomial(d, die_area, 1),
            'Neg.Binom (α=3)': lambda d: YieldModels.negative_binomial(d, die_area, 3),
        }

        for name, model in models.items():
            yields = [model(d) * 100 for d in D_range]
            ax.plot(D_range, yields, label=name, linewidth=2)

        ax.set_xlabel('Defect Density (defects/cm²)', fontsize=12)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_title(f'Yield Model Comparison (Die Area = {die_area} cm²)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


class ProcessCapability:
    """
    Cp/Cpk Process Capability Analysis

    Cpk Interpretation:
    < 1.0  = Not capable
    1.0    = Barely capable
    1.33   = Capable (minimum for production)
    1.67   = Good
    > 2.0  = Excellent
    """

    @staticmethod
    def calculate(data: np.ndarray, lsl: float, usl: float) -> dict:
        mean = np.mean(data)
        sigma = np.std(data, ddof=1)

        if sigma == 0:
            return {'error': 'Zero standard deviation'}

        cp = (usl - lsl) / (6 * sigma)
        cpu = (usl - mean) / (3 * sigma)
        cpl = (mean - lsl) / (3 * sigma)
        cpk = min(cpu, cpl)
        sigma_level = 3 * cpk

        ppm_upper = stats.norm.sf((usl - mean) / sigma) * 1e6
        ppm_lower = stats.norm.cdf((lsl - mean) / sigma) * 1e6
        ppm_total = ppm_upper + ppm_lower
        expected_yield = (1 - ppm_total / 1e6) * 100

        return {
            'mean': round(mean, 6), 'std_dev': round(sigma, 6),
            'lsl': lsl, 'usl': usl,
            'cp': round(cp, 3), 'cpk': round(cpk, 3),
            'cpu': round(cpu, 3), 'cpl': round(cpl, 3),
            'sigma_level': round(sigma_level, 2),
            'ppm_total': round(ppm_total, 1),
            'expected_yield_pct': round(expected_yield, 4),
            'n_samples': len(data),
            'min': round(np.min(data), 6), 'max': round(np.max(data), 6),
        }

    @staticmethod
    def plot(data: np.ndarray, lsl: float, usl: float,
             test_name: str = "Test", save_path: Optional[str] = None):
        result = ProcessCapability.calculate(data, lsl, usl)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram
        ax1 = axes[0]
        ax1.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

        x = np.linspace(np.min(data) - 3 * result['std_dev'], np.max(data) + 3 * result['std_dev'], 200)
        pdf = stats.norm.pdf(x, result['mean'], result['std_dev'])
        ax1.plot(x, pdf, 'r-', linewidth=2, label='Normal Fit')
        ax1.axvline(lsl, color='red', linestyle='--', linewidth=2, label=f'LSL={lsl}')
        ax1.axvline(usl, color='red', linestyle='--', linewidth=2, label=f'USL={usl}')
        ax1.axvline(result['mean'], color='green', linestyle='-', linewidth=2, label=f'Mean={result["mean"]:.4f}')

        ax1.set_title(f'{test_name} Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)

        stats_text = f"Cp  = {result['cp']:.3f}\nCpk = {result['cpk']:.3f}\nPPM = {result['ppm_total']:.0f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Box plot
        ax2 = axes[1]
        ax2.boxplot(data, vert=True, widths=0.5)
        ax2.axhline(lsl, color='red', linestyle='--', linewidth=2, label='LSL')
        ax2.axhline(usl, color='red', linestyle='--', linewidth=2, label='USL')
        ax2.set_title(f'{test_name} Box Plot', fontsize=12, fontweight='bold')
        ax2.legend()

        plt.suptitle(f'Process Capability: {test_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return result


class PATScreening:
    """
    Part Average Testing (PAT) - Outlier Screening
    AEC-Q001 standard for automotive reliability
    """

    @staticmethod
    def dynamic_pat(data: np.ndarray, near_sigma: float = 4.0,
                    far_sigma: float = 6.0) -> dict:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        clean_mask = (data >= q1 - 3 * iqr) & (data <= q3 + 3 * iqr)
        clean_data = data[clean_mask]

        mean = np.mean(clean_data)
        sigma = np.std(clean_data, ddof=1)

        near_lo, near_hi = mean - near_sigma * sigma, mean + near_sigma * sigma
        far_lo, far_hi = mean - far_sigma * sigma, mean + far_sigma * sigma

        classifications = []
        for val in data:
            if val < far_lo or val > far_hi:
                classifications.append('FAR_OUTLIER')
            elif val < near_lo or val > near_hi:
                classifications.append('NEAR_OUTLIER')
            else:
                classifications.append('PASS')

        return {
            'mean': round(mean, 6), 'sigma': round(sigma, 6),
            'near_lo': round(near_lo, 6), 'near_hi': round(near_hi, 6),
            'far_lo': round(far_lo, 6), 'far_hi': round(far_hi, 6),
            'near_outliers': sum(1 for c in classifications if c == 'NEAR_OUTLIER'),
            'far_outliers': sum(1 for c in classifications if c == 'FAR_OUTLIER'),
            'total_parts': len(data),
            'classifications': classifications,
        }

    @staticmethod
    def plot_pat(data: np.ndarray, pat_result: dict,
                 test_name: str = "Test", save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['red' if c == 'FAR_OUTLIER' else 'orange' if c == 'NEAR_OUTLIER' else 'green'
                  for c in pat_result['classifications']]

        ax.scatter(range(len(data)), data, c=colors, s=10, alpha=0.6)
        ax.axhline(pat_result['near_hi'], color='orange', linestyle='--', label='Near limit')
        ax.axhline(pat_result['near_lo'], color='orange', linestyle='--')
        ax.axhline(pat_result['far_hi'], color='red', linestyle='--', label='Far limit')
        ax.axhline(pat_result['far_lo'], color='red', linestyle='--')
        ax.axhline(pat_result['mean'], color='blue', linestyle='-', label='Mean')

        ax.set_title(f'PAT Screening: {test_name} | Near: {pat_result["near_outliers"]} | Far: {pat_result["far_outliers"]}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
