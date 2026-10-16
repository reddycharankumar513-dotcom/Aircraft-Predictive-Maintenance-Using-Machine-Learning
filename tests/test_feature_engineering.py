"""
Unit tests for causal temporal feature engineering and anti-leakage isolation.
"""

import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import build_temporal_features, extract_features_and_target


def test_build_temporal_features_group_isolation():
    """
    Ensure feature calculation for unit 1 is completely independent of unit 2.
    """
    unit1 = pd.DataFrame({
        "unit_nr": [1, 1, 1],
        "time_cycles": [1, 2, 3],
        "s_2": [10.0, 20.0, 30.0],
        "RUL": [50.0, 49.0, 48.0]
    })

    unit2 = pd.DataFrame({
        "unit_nr": [2, 2, 2],
        "time_cycles": [1, 2, 3],
        "s_2": [1000.0, 2000.0, 3000.0], # Very large values
        "RUL": [80.0, 79.0, 78.0]
    })

    combined = pd.concat([unit1, unit2], ignore_index=True)

    feat_single = build_temporal_features(unit1, sensor_cols=["s_2"], window_sizes=(2,))
    feat_combined = build_temporal_features(combined, sensor_cols=["s_2"], window_sizes=(2,))

    # The rolling mean of unit 1 in combined must exactly match single unit
    u1_combined = feat_combined[feat_combined["unit_nr"] == 1].reset_index(drop=True)
    np.testing.assert_allclose(feat_single["s_2_roll_mean_2"].values, u1_combined["s_2_roll_mean_2"].values)

    # In unit 2 at cycle 1, the rolling mean of s_2 must be 1000.0, NOT affected by unit 1's 30.0!
    u2_combined = feat_combined[feat_combined["unit_nr"] == 2].reset_index(drop=True)
    assert u2_combined.loc[0, "s_2_roll_mean_2"] == pytest.approx(1000.0)


def test_extract_features_and_target():
    df = pd.DataFrame({
        "unit_nr": [1, 1, 2, 2],
        "time_cycles": [1, 2, 1, 2],
        "s_2": [10.0, 20.0, 30.0, 40.0],
        "RUL": [100.0, 99.0, 100.0, 99.0]
    })

    X, y, groups, feat_cols = extract_features_and_target(df)

    assert "RUL" not in X.columns
    assert "unit_nr" not in X.columns
    assert "time_cycles" in X.columns
    assert "s_2" in X.columns
    assert len(y) == 4
    assert len(groups) == 4
