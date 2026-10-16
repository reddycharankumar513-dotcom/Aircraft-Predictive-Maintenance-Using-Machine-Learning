"""
Data Preprocessing, Cleaning, and Validation Module.
Ensures rigorous data quality checks, handles missing values, identifies invalid records,
drops uninformative zero-variance sensors, and provides leakage-safe feature scaling.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.config import TARGET_COL, INDEX_COLS, SETTING_COLS, SENSOR_COLS


def audit_data_quality(df: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks on the raw telemetry dataset.

    Checks:
    - Missing cells per column and total
    - Duplicate rows
    - Data types
    - Negative or zero cycles
    - Negative RUL values
    - Zero-variance / constant sensor columns

    Returns:
        Dictionary of data quality metrics and diagnostic tables.
    """
    total_rows = len(df)
    missing_series = df.isna().sum()
    missing_cols = missing_series[missing_series > 0].to_dict()
    total_missing = int(missing_series.sum())
    duplicates = int(df.duplicated().sum())

    # Check for invalid values
    invalid_cycles = 0
    if "time_cycles" in df.columns:
        invalid_cycles = int((df["time_cycles"] <= 0).sum())

    invalid_rul = 0
    if TARGET_COL in df.columns:
        invalid_rul = int((df[TARGET_COL] < 0).sum())

    # Check for constant / zero-variance columns among numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1 or df[col].std() == 0]

    # Sensor specific statistics
    target_sensors = sensor_cols if sensor_cols else [c for c in numeric_cols if c not in INDEX_COLS and c != TARGET_COL]
    sensor_stats = {}
    if target_sensors:
        sensor_stats = df[target_sensors].describe().T[["mean", "std", "min", "max"]].to_dict(orient="index")

    return {
        "total_rows": total_rows,
        "total_cols": df.shape[1],
        "total_missing_cells": total_missing,
        "missing_by_column": missing_cols,
        "duplicate_rows": duplicates,
        "invalid_cycles_count": invalid_cycles,
        "invalid_rul_count": invalid_rul,
        "constant_columns": constant_cols,
        "numeric_column_count": len(numeric_cols),
        "sensor_stats": sensor_stats,
        "is_clean": (total_missing == 0 and duplicates == 0 and invalid_cycles == 0 and invalid_rul == 0)
    }


def clean_telemetry_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_missing: bool = True,
    drop_constants: bool = True,
    threshold_variance: float = 1e-6
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset by removing duplicates, imputing missing values, and dropping flatline sensors.

    Args:
        df: Input raw dataframe.
        drop_duplicates: Whether to remove duplicate rows.
        fill_missing: Whether to impute missing values using median.
        drop_constants: Whether to drop columns with near-zero variance.
        threshold_variance: Variance cutoff below which a sensor is deemed non-informative.

    Returns:
        Cleaned dataframe and a log of changes made.
    """
    cleaned = df.copy()
    changes_log = {}

    # 1. Deduplication
    initial_len = len(cleaned)
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
        dropped_dups = initial_len - len(cleaned)
        changes_log["dropped_duplicates"] = dropped_dups

    # 2. Convert numeric columns
    for col in cleaned.columns:
        if col not in ["is_synthetic"]:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # 3. Filter invalid rows (non-positive cycle numbers if present)
    if "time_cycles" in cleaned.columns:
        valid_cycle_mask = cleaned["time_cycles"] > 0
        dropped_invalid = int((~valid_cycle_mask).sum())
        cleaned = cleaned[valid_cycle_mask]
        changes_log["dropped_invalid_cycles"] = dropped_invalid

    # 4. Missing value imputation
    if fill_missing and cleaned.isna().sum().sum() > 0:
        # Groupwise forward-fill if unit_nr is present, followed by median fill
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        if "unit_nr" in cleaned.columns:
            cleaned[numeric_cols] = cleaned.groupby("unit_nr")[numeric_cols].transform(lambda grp: grp.ffill().bfill())
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
        changes_log["imputed_missing_cells"] = True

    # 5. Drop flatline / constant columns
    dropped_cols = []
    if drop_constants:
        numeric_features = [c for c in cleaned.columns if c not in INDEX_COLS and c != TARGET_COL and c != "is_synthetic"]
        for c in numeric_features:
            col_std = cleaned[c].std()
            if pd.isna(col_std) or col_std <= threshold_variance:
                dropped_cols.append(c)
        if dropped_cols:
            cleaned = cleaned.drop(columns=dropped_cols)
        changes_log["dropped_constant_columns"] = dropped_cols

    return cleaned, changes_log


class TelemetryScaler(BaseEstimator, TransformerMixin):
    """
    Leakage-safe scaler specifically designed for multi-unit aerospace sensor streams.
    Learns normal operating distributions strictly from the training dataset.
    """

    def __init__(self, method: str = "standard"):
        """
        Args:
            method: 'standard' (StandardScaler) or 'robust' (RobustScaler for heavy outliers).
        """
        self.method = method
        self.scaler = RobustScaler() if method == "robust" else StandardScaler()
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler on training features only."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            self.scaler.fit(X.values)
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters, maintaining dataframe column names."""
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            vals = self.scaler.transform(X.values)
            return pd.DataFrame(vals, columns=cols, index=X.index)
        else:
            vals = self.scaler.transform(X)
            cols = self.feature_names_ if self.feature_names_ else None
            return pd.DataFrame(vals, columns=cols)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
