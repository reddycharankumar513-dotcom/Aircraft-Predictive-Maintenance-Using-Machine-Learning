"""
Feature Engineering Module for Aircraft Engine Predictive Maintenance.
Constructs causal temporal features (rolling statistics, lag differences, degradation trends)
with strict prevention of target leakage across time and engine units.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import INDEX_COLS, TARGET_COL, PRIMARY_SENSORS


def build_temporal_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    window_sizes: Tuple[int, ...] = (5, 15),
    include_rolling_std: bool = True,
    include_lags: bool = True
) -> pd.DataFrame:
    """
    Generate causal, backward-looking time-series features per engine unit.

    Safety & Anti-Leakage Protocol:
    1. Grouped by 'unit_nr': Rolling calculations never cross different engine boundaries.
    2. Strictly causal: Only past and current cycle values (t, t-1, ...) are used.
    3. min_periods=1: No forward-filling or lookahead is performed.
    4. Excludes target 'RUL' from all feature calculations.

    Args:
        df: Input dataframe containing 'unit_nr', 'time_cycles', and sensor columns.
        sensor_cols: List of sensors to engineer features from. If None, auto-detected.
        window_sizes: Tuple of rolling window lengths in operational cycles.
        include_rolling_std: Whether to compute rolling standard deviation (volatility).
        include_lags: Whether to compute lag differences (rate of change).

    Returns:
        pd.DataFrame containing original columns plus new engineered features.
    """
    data = df.sort_values(["unit_nr", "time_cycles"]).copy()

    # Identify candidate sensor columns
    if sensor_cols is None:
        exclude_cols = set(INDEX_COLS + [TARGET_COL, "is_synthetic"])
        sensor_cols = [c for c in data.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])]

    # Grouped rolling features
    new_features = {}
    grouped = data.groupby("unit_nr")

    for col in sensor_cols:
        col_series = grouped[col]
        for w in window_sizes:
            # Rolling mean (smoothed degradation baseline)
            mean_feat_name = f"{col}_roll_mean_{w}"
            new_features[mean_feat_name] = col_series.rolling(window=w, min_periods=1).mean().values

            # Rolling standard deviation (wear-induced sensor jitter)
            if include_rolling_std:
                std_feat_name = f"{col}_roll_std_{w}"
                # fillna(0) for window length 1
                new_features[std_feat_name] = col_series.rolling(window=w, min_periods=1).std().fillna(0.0).values

        if include_lags:
            for lag in [1, 5]:
                lag_feat_name = f"{col}_diff_{lag}"
                new_features[lag_feat_name] = col_series.diff(periods=lag).fillna(0.0).values

    # Concatenate new features in a single block for memory efficiency
    feat_df = pd.DataFrame(new_features, index=data.index)
    enhanced_df = pd.concat([data, feat_df], axis=1)

    return enhanced_df


def extract_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Extract feature matrix X, target y (RUL), and grouping vector (unit_nr).
    Ensures strict separation of features, target, and grouping metadata.

    Args:
        df: Input dataframe.
        feature_cols: Explicit list of feature columns to use. If None, extracts all
                      numeric non-target and non-unit columns.

    Returns:
        X: Feature matrix.
        y: Target series (RUL).
        groups: Engine unit series for GroupKFold validation.
        feature_names: List of column names used in X.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    y = df[TARGET_COL].copy()
    groups = df["unit_nr"].copy() if "unit_nr" in df.columns else pd.Series(np.ones(len(df)), index=df.index)

    if feature_cols is not None:
        valid_cols = [c for c in feature_cols if c in df.columns and c not in [TARGET_COL, "unit_nr", "is_synthetic"]]
    else:
        exclude = set(["unit_nr", TARGET_COL, "is_synthetic"])
        valid_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[valid_cols].copy()
    return X, y, groups, valid_cols
