# semiconductor_test_toolkit
  Professional Python toolkit for IC test engineering -  STDF parser, wafer map, yield analysis, shmoo plots 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ATE](https://img.shields.io/badge/ATE-V93000%20|%20J750-orange.svg)

## Overview

A comprehensive toolkit for parsing, analyzing, and visualizing semiconductor test data from ATE platforms like **Advantest V93000** and **Teradyne UltraFLEX**.

##  Features

| Module | Description | Status |
|--------|-------------|--------|
| **STDF Parser** | Binary STDF V4 file parser | Complete |
| **Wafer Map** | Die-level visualization & cluster detection | Complete |
| **Yield Analyzer** | Cp/Cpk, yield models, PAT screening | Complete |
| **Shmoo Engine** | V/F characterization plots |  Complete |
| **Outlier Detection** | Z-score, Mahalanobis, Isolation Forest | Complete |
| **Test Time Optimizer** | Correlation, ordering, multi-site | Complete |



```bash
git clone https://github.com/mstarefinaktar/semiconductor_test_toolkit.git
cd semiconductor-test-toolkit
pip install -r requirements.txt

#Parse STDF File

from src.stdf_parser import STDFV4Parser

parser = STDFV4Parser("path/to/file.stdf")
parser.parse()
parser.print_summary()

df = parser.get_ptr_dataframe(site=0)
df.to_csv("output.csv")


#Generate Wafer Map

from src.wafer_map import WaferMap, WaferConfig

config = WaferConfig(wafer_diameter_mm=300, die_width_mm=8)
wafer = WaferMap(config)
wafer.generate_sample_data(good_yield=0.85)
wafer.plot_bin_map(title="Lot: ABC123 | Wafer: 01")


#Yield Analysis (Cp/Cpk)

from src.yield_analyzer import ProcessCapability
import numpy as np

data = np.random.normal(loc=3.30, scale=0.02, size=1000)
result = ProcessCapability.calculate(data, lsl=3.20, usl=3.40)
print(f"Cpk: {result['cpk']}")
ProcessCapability.plot(data, lsl=3.20, usl=3.40, test_name="VDD Current")


#Shmoo Plot

from src.shmoo_plot import ShmooEngine, ShmooConfig

config = ShmooConfig(x_start=100, x_stop=500, y_start=0.75, y_stop=1.25)
shmoo = ShmooEngine(config)
shmoo.generate_realistic_shmoo(vdd_nominal=1.0, freq_nominal=300)
shmoo.plot(title="VDD vs Frequency Shmoo")


#Sample Outputs
Wafer Bin Map

      ╭──────────╮
    ╭─┤ ■■■■■■■■ ├─╮      ■ = PASS (Green)
   │  │ ■■□■■■■■ │  │     □ = FAIL (Red)
   │  │ ■■■■■■■■ │  │
    ╰─┤ ■■■■■□■■ ├─╯
      ╰──────────╯

Shmoo Plot

Voltage ▲
        │  ░░░░████████
        │  ░░░█████████     █ = PASS
        │  ░░██████████     ░ = FAIL
        │  ░░░█████████
        └──────────────► Frequency

# Architecture

STDF Binary File
      │
      ▼
┌─────────────┐    ┌──────────────┐
│ STDF Parser │───▶│  DataFrame   │
└─────────────┘    └──────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌───────────┐   ┌──────────┐
    │Wafer Map │   │  Yield    │   │  Shmoo   │
    │  Engine  │   │ Analyzer  │   │  Engine  │
    └──────────┘   └───────────┘   └──────────┘
          │               │               │
          ▼               ▼               ▼
    ┌──────────────────────────────────────────┐
    │         Outlier Detection Engine          │
    └──────────────────────────────────────────┘

# Tech Stack
Python 3.8+
pandas - Data manipulation
numpy - Numerical computing
matplotlib - Visualization
scipy - Statistical analysis
scikit-learn - ML-based outlier detection

## Project Structure


semiconductor_test_toolkit/
├── src/
│   ├── stdf_parser.py
│   ├── wafer_map.py
│   ├── yield_analyzer.py
│   ├── shmoo_plot.py
│   ├── outlier_detector.py
│   └── test_time_optimizer.py
├── tests/
│   ├── test_outlier_detector.py
│   ├── test_shmoo_plot.py
│   ├── test_stdf_parser.py
│   ├── test_wafer_map.py
│   └── test_yield_analyzer.py
├── examples/
│   └── demo.py
├── data/sample/README.md
├── requirements.txt
├── setup.py
└── README.md

---
# Disclaimer
This toolkit is for educational and personal use. All sample data is synthetic.






