# ✈️ Aircraft Predictive Maintenance System
> **Turbofan Engine Degradation Modeling, Causal Telemetry Analytics & Remaining Useful Life (RUL) Estimation**  
> *Developed for an Airbus Data Analyst / Aerospace Machine Learning Internship Portfolio*

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![Tests](https://img.shields.io/badge/Pytest-16%20Passed-brightgreen?logo=pytest)
![Domain](https://img.shields.io/badge/Aerospace-Airbus%20Skywise%20Aligned-00205B)

---

## 1. Executive Summary & Aerospace Context

In commercial aviation, unexpected powerplant or component failures trigger **Aircraft On Ground (AOG)** events—costing airlines tens of thousands of dollars per hour in delay penalties, passenger re-accommodation, and unscheduled maintenance. Platforms like **Airbus Skywise** address this challenge by aggregating real-time flight sensor telemetry to anticipate component degradation before failure occurs.

This project delivers an end-to-end, industrial-grade **Aircraft Predictive Maintenance System**. It processes multivariate turbofan sensor telemetry to forecast **Remaining Useful Life (RUL)** and categorize dispatch maintenance urgency into actionable risk tiers:
- 🔴 **CRITICAL** ($\le 30$ cycles): Immediate AOG prevention. Ground engine for borescope inspection.
- 🟠 **HIGH** (31–75 cycles): Priority hangar check; pre-order Line Replaceable Units (LRUs) for next turnaround.
- 🟡 **MEDIUM** (76–125 cycles): Elevated wear trend; increase telemetry polling frequency.
- 🟢 **LOW** (> 125 cycles): Nominal operational status within standard envelope.

---

## 2. Technical Architecture & Anti-Leakage Protocol

```
                                      AEROSPACE PREDICTIVE MAINTENANCE PIPELINE
                                      
  ┌────────────────────────────────┐       ┌────────────────────────────────┐       ┌────────────────────────────────┐
  │      Telemetry Ingestion       │       │    Data Quality & Auditing     │       │   Causal Feature Engineering   │
  │ • NASA C-MAPSS (FD001-FD004)   │──────>│ • Zero-variance filter         │──────>│ • Backward rolling mean (5,15) │
  │ • User-Uploaded CSV / TSV      │       │ • Missingness & deduplication  │       │ • Wear volatility (rolling std)│
  │ • Synthetic Demonstration Fleet│       │ • Sensor range validation      │       │ • Lag rate-of-change (1, 5)    │
  └────────────────────────────────┘       └────────────────────────────────┘       └────────────────────────────────┘
                                                                                                     │
                                                                                                     ▼
  ┌────────────────────────────────┐       ┌────────────────────────────────┐       ┌────────────────────────────────┐
  │  Airbus Operational Dashboard  │       │   Held-Out Benchmark & Score   │       │   Leakage-Aware GroupKFold     │
  │ • Real-time RUL inference      │<──────│ • MAE, RMSE, R² metrics        │<──────│ • Grouped strictly by unit_nr  │
  │ • Fleet health risk matrix     │       │ • NASA Asymmetric Loss S       │       │ • Dummy & Ridge Baselines      │
  │ • CSV dispatch action export   │       │ • Parity & residual diagnosis  │       │ • Random Forest & Boosted Trees│
  └────────────────────────────────┘       └────────────────────────────────┘       └────────────────────────────────┘
```

### Core Methodological Highlights

1. **Leakage-Aware Splitting (GroupKFold):**
   Standard random row-wise splits allow cycles from the same engine unit to exist simultaneously in training and validation sets, causing catastrophic data leakage (the model simply interpolates unit-specific noise). This pipeline enforces **GroupKFold cross-validation partitioned strictly by `unit_nr`**: an engine's entire run-to-failure trajectory belongs 100% to either training or validation.

2. **Causal Temporal Feature Engineering:**
   Engines degrade over operational cycles. All engineered features (rolling means, rolling standard deviations, lag differences) look strictly backward in time ($t, t-1, \dots, t-w$). Future cycles and target-derived variables ($T_{max}$) are completely isolated from feature matrices.

3. **Piecewise-Linear Target Construction ($RUL_{max} = 125$):**
   In turbofan mechanics, brand-new engines exhibit an incubation plateau where wear is undetectable. Following NASA PHM08 benchmark best practices, RUL is capped at $RUL_{max} = 125$ cycles:
   $$RUL(u, t) = \min(125, T_{max}(u) - t)$$

4. **The Official NASA C-MAPSS Asymmetric Scoring Function:**
   In aviation, an **early prediction** ($\hat{y} < y$) leads to conservative inspection (economic cost, but safe). A **late prediction** ($\hat{y} > y$) overestimates remaining life and risks in-flight shutdown or uncontained failure (severe safety hazard). The NASA loss function mathematically penalizes late errors exponentially harder than early errors:
   $$d_i = \hat{y}_i - y_i$$
   $$s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \quad \text{(Early / Safe)} \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \quad \text{(Late / Unsafe)} \end{cases}, \quad S = \sum_{i=1}^N s_i$$

---

## 3. Repository Structure

```
project_a_predictive_maintenance/
├── app.py                      # Production Streamlit telemetry & dispatch dashboard
├── train.py                    # CLI pipeline for leakage-aware model training & CV
├── evaluate.py                 # CLI tool for held-out test evaluation & residual plots
├── generate_synthetic_data.py  # Realistic multi-engine synthetic dataset generator
├── requirements.txt            # Pinned dependencies
├── README.md                   # System documentation & Airbus portfolio guide
├── .gitignore                  # Git exclusions for environments, caches, and models
├── CMAPSSData/                 # NASA C-MAPSS benchmark dataset files
│   ├── train_FD001.txt ...     # Run-to-failure training trajectories
│   ├── test_FD001.txt ...      # Truncated operational test trajectories
│   ├── RUL_FD001.txt ...       # Ground truth test RUL vectors
│   └── readme.txt              # Dataset physics documentation
├── data/                       # Local data storage
│   ├── README.md               # Data schema guide
│   └── synthetic_engine_fleet.csv # Pre-generated 50-engine demo dataset
├── models/                     # Serialized pipelines & evaluation reports
│   ├── best_model.joblib       # Production fitted predictor artifact
│   ├── training_summary.json   # Out-of-fold cross-validation metrics
│   ├── evaluation_parity_plot.png # Parity diagnostic plot
│   └── evaluation_residual_plot.png # Residual distribution plot
├── src/                        # Modular core source package
│   ├── __init__.py
│   ├── config.py               # Sensor metadata, risk thresholds, and constants
│   ├── data_loader.py          # NASA C-MAPSS parser & custom CSV auto-detection
│   ├── preprocessor.py         # Quality auditing, cleaning, and leakage-safe scaling
│   ├── feature_engineering.py  # Causal rolling windows & lag rate-of-change
│   ├── models.py               # TurbofanPredictor, baselines, and GroupKFold engine
│   ├── metrics.py              # MAE, RMSE, R2, NASA score, and risk tiering
│   ├── synthetic_generator.py  # Turbofan degradation physics simulation
│   └── utils.py                # Artifact persistence, JSON reporting, and plotting
└── tests/                      # Comprehensive pytest test suite (16 tests)
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessor.py
    ├── test_feature_engineering.py
    ├── test_models.py
    ├── test_metrics.py
    └── test_synthetic_generator.py
```

---

## 4. Installation & Local Setup

### Prerequisites
- Python 3.11+
- Git

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd project_a_predictive_maintenance

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Commands

### 1. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*Access the application in your browser at `http://localhost:8501`.*

### 2. Train Models with GroupKFold Cross-Validation (CLI)
```bash
# Train on NASA C-MAPSS FD001 (default)
python train.py --dataset FD001 --rul-clip 125 --cv-splits 5

# Train on synthetic demo fleet
python train.py --dataset synthetic --rul-clip 125

# Train on custom CSV
python train.py --dataset path/to/telemetry.csv
```

### 3. Evaluate Model on Held-Out Test Fleet (CLI)
```bash
python evaluate.py --dataset FD001 --model best_model
```
*Outputs official test metrics (MAE, RMSE, R², NASA Score) and exports `models/fleet_test_predictions.csv`.*

### 4. Generate New Synthetic Demonstration Data (CLI)
```bash
python generate_synthetic_data.py --units 50 --min-cycles 140 --max-cycles 320 --rul-clip 125
```

### 5. Run the Automated Pytest Test Suite
```bash
python -m pytest tests/ -v
```

---

## 6. Model Benchmark & Empirical Results

Evaluated on **NASA C-MAPSS FD001** (100 training engines, 100 test engines, HPC degradation):

| Model Architecture | 5-Fold Group CV MAE | 5-Fold Group CV RMSE | 5-Fold Group CV R² | NASA C-MAPSS Score | Held-Out Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Dummy Baseline** | 36.97 cycles | 41.67 cycles | -0.000 | 1,271,029.3 | 41.52 cycles |
| **Linear Baseline (Ridge)** | 14.55 cycles | 18.25 cycles | 0.808 | 24,953.0 | 21.40 cycles |
| **Random Forest Regressor** | 11.18 cycles | 16.65 cycles | 0.840 | 25,370.2 | 18.45 cycles |
| **Gradient Boosted Trees (Tuned)** | **11.13 cycles** | **16.45 cycles** | **0.844** | **25,752.0** | **18.20 cycles** |

*Notice the substantial improvement from the simple baseline (RMSE 41.67) to tuned gradient-boosted trees (RMSE 16.45).*

---

## 7. Supported CSV Input Formats

### Format 1: NASA C-MAPSS Standard 26-Column Format
Space or comma-delimited without header:
- Column 1: `unit_nr` (Engine ID)
- Column 2: `time_cycles` (Operational cycle index)
- Columns 3–5: Operational settings (`setting_1`, `setting_2`, `setting_3`)
- Columns 6–26: Sensor measurements (`s_1` through `s_21`)

### Format 2: Generic Predictive Maintenance CSV
```csv
unit_nr,time_cycles,temperature,pressure,vibration,fuel_flow,RUL
1,1,642.5,553.2,0.22,521.8,125
1,2,642.8,552.9,0.24,521.6,124
```
*The data loader automatically detects headers, normalizes column aliases (e.g., `engine_id`, `cycles`), and computes run-to-failure RUL if omitted.*

---

## 8. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `streamlit: command not found` | Virtual environment not active or path not updated | Use `python -m streamlit run app.py` |
| `pytest: term not recognized` | Windows execution path resolution | Run using `python -m pytest tests/ -v` |
| `FileNotFoundError: train_FD001.txt` | Dataset directory missing or relative path mismatch | Ensure `CMAPSSData/` exists in the project root directory |
| PowerShell script execution policy error | Restricted PowerShell execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Port 8501 already in use | Previous Streamlit instance still running | Run `streamlit run app.py --server.port 8502` |

---

## 9. Airbus Data Analyst Resume Bullets

Copy and adapt these bullet points for your resume or portfolio:

- **Engineered an End-to-End Turbofan Predictive Maintenance System** on NASA C-MAPSS multi-sensor datasets, implementing piecewise-linear RUL targets ($RUL_{max} = 125$) and causal rolling statistics to forecast AOG risk.
- **Architected Leakage-Free Validation via GroupKFold** partitioned strictly by engine unit ID, preventing trajectory leakage and benchmarking simple linear baselines against tuned gradient-boosted models to achieve an $R^2$ of 0.84 and RMSE of 16.4 cycles.
- **Formulated Aerospace Safety Evaluation** incorporating the official NASA asymmetric loss function to penalize dangerous late predictions over conservative early alerts, directly aligning ML metrics with airline dispatch reliability.
- **Built an Interactive Streamlit & Plotly Telemetry Hub** providing real-time degradation curve tracking, dynamic sensor simulators, automated maintenance risk tiering (Critical/High/Medium/Low), and exportable CSV dispatch schedules.
- **Authored Comprehensive Pytest Quality Assurance Suite** containing 16 unit tests spanning data validation, causal feature isolation, metric accuracy, and model persistence.
