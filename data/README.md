# Data Directory Guide

This folder contains local data files for the **Aircraft Predictive Maintenance System**.

## 1. Provided Datasets
- `data/synthetic_engine_fleet.csv`: Pre-generated 50-engine simulated degradation dataset created via `generate_synthetic_data.py`. All records have `is_synthetic = True` to distinguish offline simulation from flight benchmarks.
- `CMAPSSData/`: Official NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan run-to-failure benchmarks:
  - `FD001`: Sea level, HPC degradation (100 train / 100 test engines)
  - `FD002`: 6 flight conditions, HPC degradation (260 train / 259 test engines)
  - `FD003`: Sea level, HPC + Fan degradation (100 train / 100 test engines)
  - `FD004`: 6 flight conditions, HPC + Fan degradation (248 train / 249 test engines)

## 2. Custom User Upload Formats
The application and CLI pipeline support two custom CSV formats:

### Format A: Standard C-MAPSS 26-Column Layout (Space or Comma-separated)
```
unit_nr,time_cycles,setting_1,setting_2,setting_3,s_1,s_2,...,s_21
1,1,-0.0007,-0.0004,100.0,518.67,641.82,...,23.4190
```

### Format B: Custom Sensor Field Format
```
unit_nr,time_cycles,temperature,pressure,vibration,fuel_flow,RUL
1,1,642.5,553.2,0.22,521.8,125
1,2,642.8,552.9,0.24,521.6,124
```
*(If `RUL` is omitted from run-to-failure records, the pipeline automatically derives $RUL = \max(cycles) - cycle$)*.
