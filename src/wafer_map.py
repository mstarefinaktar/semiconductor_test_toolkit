"""
Wafer Map Engine
=================

Professional wafer map visualization and analysis tool.
Supports bin maps, parametric heatmaps, and cluster detection.

Author: Mst Arefin Aktar
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy import ndimage
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class WaferConfig:
    wafer_diameter_mm: float = 300.0
    die_width_mm: float = 5.0
    die_height_mm: float = 5.0
    edge_exclusion_mm: float = 3.0


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    x_coords: List[int]
    y_coords: List[int]
    centroid: Tuple[float, float]


class WaferMap:
    """
    Professional Wafer Map Engine

    Features:
    - Bin-based wafer map (pass/fail coloring)
    - Parametric heatmap
    - Cluster detection (neighboring bad die)
    - Zone yield analysis (center vs edge)

    Usage:
        config = WaferConfig(wafer_diameter_mm=300, die_width_mm=8)
        wafer = WaferMap(config)
        wafer.generate_sample_data(good_yield=0.85)
        wafer.plot_bin_map(title="Wafer 01")
    """

    BIN_COLORS = {
        1: '#00FF00', 2: '#FF0000', 3: '#0000FF',
        4: '#FFFF00', 5: '#FF00FF', 6: '#00FFFF',
        7: '#FFA500', 8: '#800080', 0: '#808080',
    }

    def __init__(self, config: Optional[WaferConfig] = None):
        self.config = config or WaferConfig()
        self._setup_grid()

    def _setup_grid(self):
        radius = self.config.wafer_diameter_mm / 2 - self.config.edge_exclusion_mm
        self.cols = int(2 * radius / self.config.die_width_mm)
        self.rows = int(2 * radius / self.config.die_height_mm)
        self.wafer_grid = np.full((self.rows, self.cols), -1, dtype=int)

        cx, cy = self.cols / 2, self.rows / 2
        for r in range(self.rows):
            for c in range(self.cols):
                dx = (c - cx + 0.5) * self.config.die_width_mm
                dy = (r - cy + 0.5) * self.config.die_height_mm
                if np.sqrt(dx ** 2 + dy ** 2) <= radius:
                    self.wafer_grid[r, c] = 0

        self.total_die = np.sum(self.wafer_grid >= 0)
        print(f"Wafer grid: {self.rows}x{self.cols}, Total die: {self.total_die}")

    def generate_sample_data(self, good_yield: float = 0.85, seed: int = 42):
        np.random.seed(seed)
        cx, cy = self.cols / 2, self.rows / 2

        for r in range(self.rows):
            for c in range(self.cols):
                if self.wafer_grid[r, c] < 0:
                    continue

                dist = np.sqrt((c - cx) ** 2 + (r - cy) ** 2)
                max_dist = np.sqrt(cx ** 2 + cy ** 2)
                edge_factor = 1.0 - 0.3 * (dist / max_dist) ** 2

                if np.random.random() < good_yield * edge_factor:
                    self.wafer_grid[r, c] = 1
                else:
                    self.wafer_grid[r, c] = np.random.choice(
                        [2, 3, 4, 5, 6, 7],
                        p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
                    )

        # Add cluster defect
        cluster_x = int(cx + np.random.randint(-10, 10))
        cluster_y = int(cy + np.random.randint(-10, 10))
        for r in range(max(0, cluster_y - 4), min(self.rows, cluster_y + 4)):
            for c in range(max(0, cluster_x - 4), min(self.cols, cluster_x + 4)):
                if self.wafer_grid[r, c] >= 0:
                    if np.sqrt((c - cluster_x) ** 2 + (r - cluster_y) ** 2) <= 4:
                        self.wafer_grid[r, c] = 2

        return self.wafer_grid

    def plot_bin_map(self, title: str = "Wafer Bin Map",
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 12)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        wafer_circle = plt.Circle(
            (self.cols / 2, self.rows / 2), self.cols / 2 + 1,
            fill=False, edgecolor='black', linewidth=2
        )
        ax.add_patch(wafer_circle)

        die_patches = []
        die_colors = []

        for r in range(self.rows):
            for c in range(self.cols):
                b = self.wafer_grid[r, c]
                if b < 0:
                    continue
                rect = patches.Rectangle((c, self.rows - r - 1), 1, 1)
                die_patches.append(rect)
                die_colors.append(self.BIN_COLORS.get(b, '#FFFFFF'))

        collection = PatchCollection(die_patches, facecolors=die_colors,
                                     edgecolors='black', linewidths=0.5)
        ax.add_collection(collection)

        ax.set_xlim(-2, self.cols + 2)
        ax.set_ylim(-2, self.rows + 2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')

        # Legend
        legend_elements = []
        for b in sorted(self.BIN_COLORS.keys()):
            count = np.sum(self.wafer_grid == b)
            if count > 0:
                legend_elements.append(
                    patches.Patch(facecolor=self.BIN_COLORS[b], label=f'Bin {b}: {count}')
                )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Yield
        valid = self.wafer_grid[self.wafer_grid >= 0]
        good = np.sum(valid == 1)
        total = len(valid)
        yield_pct = good / total * 100 if total > 0 else 0

        ax.text(0.02, 0.98, f'Total: {total}\nGood: {good}\nYield: {yield_pct:.1f}%',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_parametric_heatmap(self, x_coords, y_coords, values,
                                test_name: str = "Test",
                                save_path: Optional[str] = None):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        heatmap = np.full((self.rows, self.cols), np.nan)
        for x, y, v in zip(x_coords, y_coords, values):
            if 0 <= y < self.rows and 0 <= x < self.cols:
                heatmap[y, x] = v

        masked = np.ma.masked_where(self.wafer_grid < 0, heatmap)
        im = ax.imshow(masked, cmap='RdYlGn_r', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'{test_name} Value')
        ax.set_title(f'Parametric Wafer Map: {test_name}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        return fig

    def detect_clusters(self, fail_bins: List[int] = None,
                        min_cluster_size: int = 3) -> List[ClusterInfo]:
        if fail_bins is None:
            fail_bins = [2, 3, 4, 5, 6, 7]

        fail_map = np.zeros_like(self.wafer_grid)
        for b in fail_bins:
            fail_map[self.wafer_grid == b] = 1

        labeled, num_features = ndimage.label(fail_map)

        clusters = []
        for cid in range(1, num_features + 1):
            coords = np.where(labeled == cid)
            size = len(coords[0])
            if size >= min_cluster_size:
                clusters.append(ClusterInfo(
                    cluster_id=cid, size=size,
                    y_coords=coords[0].tolist(),
                    x_coords=coords[1].tolist(),
                    centroid=(np.mean(coords[1]), np.mean(coords[0]))
                ))

        clusters.sort(key=lambda c: c.size, reverse=True)
        print(f"Clusters found (>={min_cluster_size} die): {len(clusters)}")
        for c in clusters[:5]:
            print(f"  Cluster {c.cluster_id}: {c.size} die at ({c.centroid[0]:.1f}, {c.centroid[1]:.1f})")

        return clusters

    def zone_yield_analysis(self, num_zones: int = 5) -> pd.DataFrame:
        cx, cy = self.cols / 2, self.rows / 2
        max_radius = min(cx, cy)

        zone_data = []
        for zone in range(num_zones):
            r_inner = zone * max_radius / num_zones
            r_outer = (zone + 1) * max_radius / num_zones
            total = good = 0

            for r in range(self.rows):
                for c in range(self.cols):
                    if self.wafer_grid[r, c] < 0:
                        continue
                    dist = np.sqrt((c - cx) ** 2 + (r - cy) ** 2)
                    if r_inner <= dist < r_outer:
                        total += 1
                        if self.wafer_grid[r, c] == 1:
                            good += 1

            yield_pct = good / total * 100 if total > 0 else 0
            zone_data.append({
                'zone': zone + 1,
                'type': 'Center' if zone == 0 else ('Edge' if zone == num_zones - 1 else 'Middle'),
                'total_die': total,
                'good_die': good,
                'yield_pct': round(yield_pct, 2)
            })

        df = pd.DataFrame(zone_data)
        print("\nZone Yield Analysis:")
        print(df.to_string(index=False))
        return df

    def get_yield_summary(self) -> Dict:
        valid = self.wafer_grid[self.wafer_grid >= 0]
        total = len(valid)
        good = np.sum(valid == 1)
        return {
            'total_die': int(total),
            'good_die': int(good),
            'yield_pct': round(good / total * 100, 2) if total > 0 else 0,
        }
