"""
Data Loader Module for Aircraft Predictive Maintenance.
Supports NASA C-MAPSS benchmark datasets (FD001-FD004), custom user-uploaded CSVs,
and synthetic demonstration datasets.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import io
import pandas as pd
import numpy as np

from src.config import (
    ALL_RAW_COLS,
    INDEX_COLS,
    SETTING_COLS,
    SENSOR_COLS,
    CMAPSS_DIR,
    DEFAULT_RUL_CLIP,
    TARGET_COL,
)


def load_cmapss_dataset(
    dataset_id: str = "FD001",
    rul_clip: Optional[int] = DEFAULT_RUL_CLIP,
    cmapss_dir: Path = CMAPSS_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load NASA C-MAPSS turbofan engine degradation dataset (FD001, FD002, FD003, or FD004).

    Target Construction:
    - Train Set: Run-to-failure trajectories. For each engine unit, the maximum observed
      cycle represents total life T_max. True RUL at cycle t is (T_max - t).
      If rul_clip is specified, values are capped at rul_clip (piecewise linear degradation).
    - Test Set: Trajectories truncated at an unknown operational point prior to failure.
      The RUL_FD00x.txt file provides the true remaining useful life at the final cycle.

    Args:
        dataset_id: Sub-dataset identifier ('FD001', 'FD002', 'FD003', or 'FD004').
        rul_clip: Upper threshold for piecewise-linear RUL clipping (typically 125).
        cmapss_dir: Path to directory containing raw C-MAPSS text files.

    Returns:
        train_df: Full degradation trajectories with computed 'RUL'.
        test_df: Truncated test trajectories with true 'RUL' across all cycles.
        rul_vector: Series containing true RUL at the final observed cycle per test unit.
        metadata: Dictionary with dataset descriptive properties and descriptions.
    """
    dataset_id = dataset_id.upper()
    train_path = cmapss_dir / f"train_{dataset_id}.txt"
    test_path = cmapss_dir / f"test_{dataset_id}.txt"
    rul_path = cmapss_dir / f"RUL_{dataset_id}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Ground truth RUL file not found: {rul_path}")

    # Load raw space-delimited text files
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_RAW_COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_RAW_COLS)
    rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_rul"])["true_rul"]
    rul_series.index = np.arange(1, len(rul_series) + 1)
    rul_series.index.name = "unit_nr"

    # Compute training RUL target
    # RUL(u, t) = max_cycle(u) - t
    max_cycles_train = train_df.groupby("unit_nr")["time_cycles"].transform("max")
    linear_train_rul = max_cycles_train - train_df["time_cycles"]
    if rul_clip is not None and rul_clip > 0:
        train_df[TARGET_COL] = np.minimum(linear_train_rul, rul_clip).astype(float)
    else:
        train_df[TARGET_COL] = linear_train_rul.astype(float)

    # Compute test RUL across all recorded test cycles
    # RUL_test(u, t) = RUL_final(u) + (max_test_cycle(u) - t)
    max_cycles_test = test_df.groupby("unit_nr")["time_cycles"].transform("max")
    # Map final RUL to each test row based on unit_nr
    final_rul_mapped = test_df["unit_nr"].map(rul_series)
    linear_test_rul = final_rul_mapped + (max_cycles_test - test_df["time_cycles"])
    if rul_clip is not None and rul_clip > 0:
        test_df[TARGET_COL] = np.minimum(linear_test_rul, rul_clip).astype(float)
    else:
        test_df[TARGET_COL] = linear_test_rul.astype(float)

    train_df["is_synthetic"] = False
    test_df["is_synthetic"] = False

    dataset_descriptions = {
        "FD001": {"conditions": 1, "fault_modes": "HPC Degradation", "desc": "Sea Level, High Pressure Compressor wear"},
        "FD002": {"conditions": 6, "fault_modes": "HPC Degradation", "desc": "6 Operating conditions, HPC wear"},
        "FD003": {"conditions": 1, "fault_modes": "HPC + Fan Degradation", "desc": "Sea Level, HPC and Fan wear"},
        "FD004": {"conditions": 6, "fault_modes": "HPC + Fan Degradation", "desc": "6 Operating conditions, HPC and Fan wear"}
    }

    metadata = {
        "dataset_id": dataset_id,
        "is_synthetic": False,
        "rul_clip": rul_clip,
        "num_train_units": train_df["unit_nr"].nunique(),
        "num_test_units": test_df["unit_nr"].nunique(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "conditions": dataset_descriptions.get(dataset_id, {}).get("conditions", 1),
        "fault_modes": dataset_descriptions.get(dataset_id, {}).get("fault_modes", "Standard degradation"),
        "description": dataset_descriptions.get(dataset_id, {}).get("desc", "NASA C-MAPSS dataset")
    }

    return train_df, test_df, rul_series, metadata


def load_user_csv(
    file_source: Union[str, Path, io.BytesIO],
    rul_clip: Optional[int] = DEFAULT_RUL_CLIP
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse a user-supplied CSV/TSV dataset.
    Automatically standardizes column identifiers, detects unit IDs, cycle counts,
    sensor features, and target values.

    Args:
        file_source: Path or file-like object (e.g. Streamlit UploadedFile).
        rul_clip: Optional clipping threshold for RUL.

    Returns:
        df: Cleaned dataframe with standardized naming.
        metadata: Detection metadata report.
    """
    # Attempt to read CSV with pandas sniffing
    try:
        if isinstance(file_source, (str, Path)):
            df = pd.read_csv(file_source, sep=None, engine="python")
        else:
            # File-like object
            df = pd.read_csv(file_source, sep=None, engine="python")
    except Exception:
        # Fallback to comma or whitespace
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        df = pd.read_csv(file_source, sep=r"\s+", header=None)

    # Check if raw 26-column C-MAPSS format without headers
    if df.shape[1] == 26 and (isinstance(df.columns[0], int) or df.columns[0] == "0"):
        df.columns = ALL_RAW_COLS

    # Standardize column naming aliases
    rename_map = {}
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if norm in ["unit", "unit_id", "engine", "engine_id", "id", "unit_number"]:
            rename_map[col] = "unit_nr"
        elif norm in ["cycle", "cycles", "time", "time_cycle", "time_cycles", "operating_hours"]:
            rename_map[col] = "time_cycles"
        elif norm in ["rul", "remaining_useful_life", "remaining_life", "target"]:
            rename_map[col] = TARGET_COL

    df = df.rename(columns=rename_map)

    # Check if 'is_synthetic' column is present
    is_synthetic = False
    if "is_synthetic" in df.columns:
        is_synthetic = bool(df["is_synthetic"].iloc[0])
    elif "synthetic" in str(file_source).lower():
        is_synthetic = True

    # If RUL is not present but unit_nr and time_cycles are present, calculate run-to-failure RUL
    calculated_target = False
    if TARGET_COL not in df.columns and "unit_nr" in df.columns and "time_cycles" in df.columns:
        max_c = df.groupby("unit_nr")["time_cycles"].transform("max")
        linear_rul = max_c - df["time_cycles"]
        if rul_clip is not None and rul_clip > 0:
            df[TARGET_COL] = np.minimum(linear_rul, rul_clip).astype(float)
        else:
            df[TARGET_COL] = linear_rul.astype(float)
        calculated_target = True

    # Detect sensor columns (all numeric non-index non-target columns)
    non_sensor = ["unit_nr", "time_cycles", TARGET_COL, "is_synthetic"]
    detected_sensors = [c for c in df.columns if c not in non_sensor and pd.api.types.is_numeric_dtype(df[c])]

    metadata = {
        "is_synthetic": is_synthetic,
        "num_rows": len(df),
        "num_cols": df.shape[1],
        "has_unit_nr": "unit_nr" in df.columns,
        "has_time_cycles": "time_cycles" in df.columns,
        "has_target": TARGET_COL in df.columns,
        "calculated_target": calculated_target,
        "unique_units": df["unit_nr"].nunique() if "unit_nr" in df.columns else 1,
        "detected_sensors": detected_sensors,
        "columns": list(df.columns)
    }

    return df, metadata
