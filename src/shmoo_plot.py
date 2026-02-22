"""
Shmoo Plot Engine
==================

2D pass/fail visualization for voltage/frequency characterization.

Author: Mst Arefin Aktar
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ShmooConfig:
    x_param: str = "Frequency (MHz)"
    y_param: str = "Voltage (V)"
    x_start: float = 100
    x_stop: float = 500
    x_step: float = 10
    y_start: float = 0.8
    y_stop: float = 1.2
    y_step: float = 0.01


class ShmooEngine:
    """
    Professional Shmoo Plot Generator

    Usage:
        config = ShmooConfig(x_start=100, x_stop=500, y_start=0.75, y_stop=1.25)
        shmoo = ShmooEngine(config)
        shmoo.generate_realistic_shmoo()
        shmoo.plot(title="VDD vs Frequency")
    """

    def __init__(self, config: ShmooConfig):
        self.config = config
        self.x_values = np.arange(config.x_start, config.x_stop + config.x_step, config.x_step)
        self.y_values = np.arange(config.y_start, config.y_stop + config.y_step, config.y_step)
        self.shmoo_data = np.zeros((len(self.y_values), len(self.x_values)))

    def generate_realistic_shmoo(self, vdd_nominal: float = 1.0,
                                  freq_nominal: float = 300,
                                  noise: float = 0.05, seed: int = 42):
        np.random.seed(seed)
        vth = 0.3
        alpha = 1.5

        for i, v in enumerate(self.y_values):
            for j, f in enumerate(self.x_values):
                if v > vth:
                    f_max = freq_nominal * ((v - vth) / (vdd_nominal - vth)) ** alpha
                else:
                    f_max = 0

                f_max_noisy = f_max + np.random.normal(0, noise * freq_nominal)
                v_min = vth + (vdd_nominal - vth) * (f / freq_nominal) ** (1 / alpha)
                v_min_noisy = v_min + np.random.normal(0, noise * 0.1)

                self.shmoo_data[i, j] = 1 if (f <= f_max_noisy and v >= v_min_noisy) else 0

        return self.shmoo_data

    def plot(self, title: str = "Shmoo Plot", save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(14, 8))

        cmap = ListedColormap(['#FF4444', '#44FF44'])
        im = ax.imshow(self.shmoo_data, cmap=cmap, aspect='auto', origin='lower',
                       extent=[self.x_values[0], self.x_values[-1],
                               self.y_values[0], self.y_values[-1]])

        ax.set_xlabel(self.config.x_param, fontsize=12)
        ax.set_ylabel(self.config.y_param, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['FAIL', 'PASS'])

        pass_rate = np.sum(self.shmoo_data) / self.shmoo_data.size * 100
        ax.text(0.02, 0.02, f'Pass Rate: {pass_rate:.1f}%',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        return fig

    def find_operating_window(self) -> dict:
        max_freq = {}
        for i, v in enumerate(self.y_values):
            passes = np.where(self.shmoo_data[i, :] == 1)[0]
            if len(passes) > 0:
                max_freq[v] = self.x_values[passes[-1]]

        if not max_freq:
            return {'error': 'No passing region'}

        voltages = sorted(max_freq.keys())
        return {
            'min_voltage': round(min(voltages), 3),
            'max_voltage': round(max(voltages), 3),
            'voltage_range': round(max(voltages) - min(voltages), 3),
        }
