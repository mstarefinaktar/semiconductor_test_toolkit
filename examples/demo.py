"""
Demo Script - Semiconductor Test Toolkit
==========================================

Run this to see all modules in action with synthetic data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

print("=" * 60)
print("  SEMICONDUCTOR TEST TOOLKIT - DEMO")
print("=" * 60)

# ==============================
# 1. Wafer Map
# ==============================
print("\n WAFER MAP ENGINE")
print("-" * 40)

from src.wafer_map import WaferMap, WaferConfig

config = WaferConfig(wafer_diameter_mm=300, die_width_mm=8, die_height_mm=8)
wafer = WaferMap(config)
wafer.generate_sample_data(good_yield=0.82, seed=42)
wafer.plot_bin_map(title="DEMO LOT: ABC123 | Wafer: 01")

summary = wafer.get_yield_summary()
print(f"Yield: {summary['yield_pct']}%")

clusters = wafer.detect_clusters(min_cluster_size=3)
zone_df = wafer.zone_yield_analysis(num_zones=5)


# ==============================
# 2. Yield Analysis
# ==============================
print("\n YIELD ANALYSIS")
print("-" * 40)

from src.yield_analyzer import YieldModels, ProcessCapability, PATScreening

# Yield Models
D = np.linspace(0, 5, 100)
YieldModels.compare_models(D, die_area=1.0)

# Cp/Cpk
np.random.seed(42)
test_data = np.random.normal(loc=3.30, scale=0.02, size=1000)
result = ProcessCapability.plot(test_data, lsl=3.20, usl=3.40, test_name="VDD Supply Current")
print(f"Cp={result['cp']}, Cpk={result['cpk']}, PPM={result['ppm_total']}")

# PAT Screening
data_with_outliers = np.concatenate([
    np.random.normal(2.5, 0.1, 980),
    np.random.normal(3.5, 0.2, 15),
    np.array([5.0, 0.1, 5.5, -0.5, 6.0])
])
pat = PATScreening.dynamic_pat(data_with_outliers)
PATScreening.plot_pat(data_with_outliers, pat, test_name="IOH Current")
print(f"Near outliers: {pat['near_outliers']}, Far outliers: {pat['far_outliers']}")


# ==============================
# 3. Shmoo Plot
# ==============================
print("\n SHMOO PLOT ENGINE")
print("-" * 40)

from src.shmoo_plot import ShmooEngine, ShmooConfig

shmoo_config = ShmooConfig(
    x_start=100, x_stop=500, x_step=5,
    y_start=0.75, y_stop=1.25, y_step=0.01
)
shmoo = ShmooEngine(shmoo_config)
shmoo.generate_realistic_shmoo(vdd_nominal=1.0, freq_nominal=300, noise=0.03)
shmoo.plot(title="Device Shmoo: VDD vs Frequency")

window = shmoo.find_operating_window()
print(f"Operating Window: {window}")


# ==============================
# 4. Outlier Detection
# ==============================
print("\n  OUTLIER DETECTION")
print("-" * 40)

from src.outlier_detector import OutlierDetector

np.random.seed(42)
normal_data = np.random.normal(2.5, 0.1, 1000)
outliers = np.array([4.0, 0.5, 4.5, 0.2, 5.0])
test_values = np.concatenate([normal_data, outliers])
np.random.shuffle(test_values)

OutlierDetector.compare_methods(test_values, "IDDQ Current")
grubbs = OutlierDetector.grubbs_test(test_values)
print(f"Grubbs Test: {grubbs}")


# ==============================
# 5. STDF Parser (requires .stdf file)
# ==============================
print("\n STDF PARSER")
print("-" * 40)
print("To use STDF parser:")
print("  from src.stdf_parser import STDFV4Parser")
print("  parser = STDFV4Parser('path/to/file.stdf')")
print("  parser.parse()")
print("  df = parser.get_ptr_dataframe(site=0)")

print("\n" + "=" * 60)
print("  DEMO COMPLETE ")
print("=" * 60)
