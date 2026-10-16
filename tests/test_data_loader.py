"""
Unit tests for C-MAPSS and custom CSV data loaders.
"""

from pathlib import Path
import io
import pandas as pd
import pytest
from src.data_loader import load_cmapss_dataset, load_user_csv
from src.config import CMAPSS_DIR, TARGET_COL


def test_load_cmapss_dataset_fd001():
    train_df, test_df, rul_series, meta = load_cmapss_dataset("FD001", rul_clip=125, cmapss_dir=CMAPSS_DIR)

    assert meta["dataset_id"] == "FD001"
    assert meta["num_train_units"] == 100
    assert meta["num_test_units"] == 100
    assert len(train_df) > 20000
    assert len(test_df) > 13000
    assert TARGET_COL in train_df.columns
    assert TARGET_COL in test_df.columns
    assert len(rul_series) == 100
    assert (train_df[TARGET_COL] <= 125).all()
    assert (train_df[TARGET_COL] >= 0).all()


def test_load_user_csv():
    csv_text = """unit_id,cycle,sensor_temp,sensor_vib,rul
1,1,640.2,0.22,120
1,2,641.5,0.24,119
2,1,639.8,0.21,145
2,2,640.1,0.22,144
"""
    buffer = io.StringIO(csv_text)
    df, meta = load_user_csv(buffer)

    assert meta["has_unit_nr"] == True
    assert meta["has_time_cycles"] == True
    assert meta["has_target"] == True
    assert meta["unique_units"] == 2
    assert "s_temp" in meta["detected_sensors"] or "sensor_temp" in meta["detected_sensors"]
