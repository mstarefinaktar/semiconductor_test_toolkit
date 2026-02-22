"""
Test Time Optimization Engine
===============================

Tools for reducing semiconductor test time:
- Correlation analysis (redundant test removal)
- Fail-rate ordering
- Multi-site efficiency analysis

Author: Mst Arefin Aktar
Date: 2026
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import matplotlib.pyplot as plt


class TestTimeOptimizer:
    """
    Test time = MONEY in semiconductor testing.
    Every millisecond saved = millions of dollars/year.
    """

    def __init__(self, test_data: pd.DataFrame):
        self.data = test_data
        self.original_test_time = self._calc_total_time()

    def _calc_total_time(self) -> float:
        return self.data.groupby('test_name')['test_time_ms'].first().sum()

    def correlation_analysis(self, threshold: float = 0.95) -> pd.DataFrame:
        pivot = self.data.pivot_table(index='part_id', columns='test_name',
                                       values='result', aggfunc='first')
        corr_matrix = pivot.corr()

        pairs = []
        tests = corr_matrix.columns.tolist()

        for i in range(len(tests)):
            for j in range(i + 1, len(tests)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr >= threshold:
                    time_i = self.data[self.data['test_name'] == tests[i]]['test_time_ms'].iloc[0]
                    time_j = self.data[self.data['test_name'] == tests[j]]['test_time_ms'].iloc[0]

                    pairs.append({
                        'test_1': tests[i], 'test_2': tests[j],
                        'correlation': round(corr, 4),
                        'removable': tests[i] if time_i > time_j else tests[j],
                        'time_saved_ms': max(time_i, time_j)
                    })

        result = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
        if len(result) > 0:
            print(f"Correlated pairs: {len(result)}, Potential savings: {result['time_saved_ms'].sum():.1f} ms")
        return result

    def fail_rate_ordering(self) -> pd.DataFrame:
        test_stats = []
        for test_name in self.data['test_name'].unique():
            td = self.data[self.data['test_name'] == test_name]
            total = len(td)
            fails = (td['pass_fail'] == 'FAIL').sum()
            fail_rate = fails / total if total > 0 else 0
            test_time = td['test_time_ms'].iloc[0]
            efficiency = fail_rate / test_time if test_time > 0 else 0

            test_stats.append({
                'test_name': test_name, 'fail_rate': round(fail_rate, 6),
                'fail_count': fails, 'test_time_ms': test_time,
                'efficiency_score': round(efficiency, 8),
            })

        return pd.DataFrame(test_stats).sort_values('efficiency_score', ascending=False)

    def multi_site_efficiency(self, max_sites: int = 16) -> Dict:
        single_time = self.original_test_time
        overhead_per_site = 0.5

        results = {}
        for n in range(1, max_sites + 1):
            overhead = overhead_per_site * (n - 1)
            ms_time = single_time + overhead
            throughput = n / ms_time
            single_tp = 1 / single_time
            speedup = throughput / single_tp
            efficiency = (throughput / (n * single_tp)) * 100

            results[n] = {
                'sites': n, 'test_time_ms': round(ms_time, 2),
                'speedup': round(speedup, 2), 'efficiency_pct': round(efficiency, 1),
            }

        return results

    def generate_report(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Test time distribution
        ax1 = axes[0, 0]
        times = self.data.groupby('test_name')['test_time_ms'].first()
        times.sort_values(ascending=False).head(20).plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_title('Top 20 Tests by Time', fontweight='bold')

        # Fail rate
        ax2 = axes[0, 1]
        rates = self.data.groupby('test_name')['pass_fail'].apply(lambda x: (x == 'FAIL').mean())
        rates.sort_values(ascending=False).head(20).plot(kind='barh', ax=ax2, color='coral')
        ax2.set_title('Top 20 Tests by Fail Rate', fontweight='bold')

        # Cumulative time
        ax3 = axes[1, 0]
        cum_pct = times.sort_values(ascending=False).cumsum() / times.sum() * 100
        ax3.plot(range(len(cum_pct)), cum_pct.values, 'b-', linewidth=2)
        ax3.axhline(80, color='red', linestyle='--', label='80%')
        ax3.set_title('Cumulative Test Time', fontweight='bold')
        ax3.legend()

        # Multi-site
        ax4 = axes[1, 1]
        ms = self.multi_site_efficiency(16)
        sites = [r['sites'] for r in ms.values()]
        speedups = [r['speedup'] for r in ms.values()]
        ax4.plot(sites, speedups, 'bo-', label='Actual')
        ax4.plot(range(1, 17), range(1, 17), 'r--', label='Ideal')
        ax4.set_title('Multi-Site Scaling', fontweight='bold')
        ax4.legend()

        plt.suptitle('Test Time Optimization Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
