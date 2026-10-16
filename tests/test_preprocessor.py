"""
Unit tests for data quality audit, cleaning, and scaling.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessor import audit_data_quality, clean_telemetry_data, TelemetryScaler


def test_audit_data_quality():
    df = pd.DataFrame({
        "unit_nr": [1, 1, 2, 2],
        "time_cycles": [1, 2, -1, 2], # 1 invalid negative cycle
        "s_1": [500.0, 500.0, 500.0, 500.0], # constant column
        "s_2": [600.0, np.nan, 610.0, 620.0], # 1 missing
        "RUL": [100.0, 99.0, 50.0, 49.0]
    })

    audit = audit_data_quality(df)
    assert audit["total_missing_cells"] == 1
    assert audit["invalid_cycles_count"] == 1
    assert "s_1" in audit["constant_columns"]
    assert audit["is_clean"] is False


def test_clean_telemetry_data():
    df = pd.DataFrame({
        "unit_nr": [1, 1, 1, 1],
        "time_cycles": [1, 1, 2, 3], # row 1 is duplicate
        "s_1": [10.0, 10.0, 10.0, 10.0], # constant
        "s_2": [100.0, 100.0, np.nan, 104.0],
        "RUL": [50.0, 50.0, 49.0, 48.0]
    })

    cleaned, log = clean_telemetry_data(df, drop_duplicates=True, drop_constants=True)
    assert log["dropped_duplicates"] == 1
    assert "s_1" in log["dropped_constant_columns"]
    assert "s_1" not in cleaned.columns
    assert cleaned["s_2"].isna().sum() == 0


def test_telemetry_scaler():
    df = pd.DataFrame({
        "feat_a": [10.0, 20.0, 30.0],
        "feat_b": [100.0, 200.0, 300.0]
    })

    scaler = TelemetryScaler(method="standard")
    scaled = scaler.fit_transform(df)

    assert scaled.shape == df.shape
    assert list(scaled.columns) == list(df.columns)
    assert scaled["feat_a"].mean() == pytest.approx(0.0, abs=1e-6)
    assert scaled["feat_a"].std() == pytest.approx(1.0, abs=0.25)
