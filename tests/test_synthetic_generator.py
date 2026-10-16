"""
Unit tests for synthetic dataset generator.
"""

import pytest
from src.synthetic_generator import generate_synthetic_turbofan_fleet, split_synthetic_train_test
from src.config import ALL_RAW_COLS, TARGET_COL


def test_generate_synthetic_turbofan_fleet():
    num_units = 10
    df = generate_synthetic_turbofan_fleet(num_units=num_units, min_cycles=50, max_cycles=100, rul_clip=125, random_seed=42)

    assert df["unit_nr"].nunique() == num_units
    assert df["is_synthetic"].all() == True
    for col in ALL_RAW_COLS + [TARGET_COL]:
        assert col in df.columns

    # Verify RUL decreases with cycle within each engine
    for unit_id in range(1, num_units + 1):
        u_df = df[df["unit_nr"] == unit_id].sort_values("time_cycles")
        assert (u_df[TARGET_COL].diff().dropna() <= 0).all()


def test_split_synthetic_train_test():
    df = generate_synthetic_turbofan_fleet(num_units=20, min_cycles=60, max_cycles=120, random_seed=42)
    train_df, test_df, rul_series = split_synthetic_train_test(df, test_ratio=0.25, random_seed=42)

    train_units = set(train_df["unit_nr"].unique())
    test_units = set(test_df["unit_nr"].unique())

    # Guarantee zero leakage between train and test units
    assert len(train_units.intersection(test_units)) == 0
    assert len(test_units) == 5
    assert len(rul_series) == 5
    assert (rul_series >= 0).all()
