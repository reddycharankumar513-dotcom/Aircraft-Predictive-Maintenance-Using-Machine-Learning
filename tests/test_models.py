"""
Unit tests for model predictors, GroupKFold validation, and persistence.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.models import TurbofanPredictor, run_group_cross_validation
from src.utils import save_model_artifact, load_model_artifact


def make_dummy_data():
    np.random.seed(42)
    rows = []
    for u in range(1, 7):
        for c in range(1, 15):
            deg = c / 15.0
            s1 = 500.0 + deg * 20.0 + np.random.normal(0, 0.5)
            s2 = 200.0 - deg * 10.0 + np.random.normal(0, 0.5)
            rul = max(0.0, 15 - c)
            rows.append([u, c, s1, s2, rul])
    return pd.DataFrame(rows, columns=["unit_nr", "time_cycles", "s_1", "s_2", "RUL"])


def test_baseline_and_rf_models():
    df = make_dummy_data()
    X = df[["time_cycles", "s_1", "s_2"]]
    y = df["RUL"]

    # Mean Baseline
    mean_model = TurbofanPredictor(model_type="mean_baseline")
    mean_model.fit(X, y)
    preds_mean = mean_model.predict(X)
    assert len(preds_mean) == len(X)
    assert (preds_mean >= 0).all()

    # Ridge Baseline
    ridge_model = TurbofanPredictor(model_type="ridge")
    ridge_model.fit(X, y)
    preds_ridge = ridge_model.predict(X)
    assert len(preds_ridge) == len(X)
    assert (preds_ridge >= 0).all()

    # Random Forest
    rf_model = TurbofanPredictor(model_type="random_forest", model_params={"n_estimators": 10})
    rf_model.fit(X, y)
    preds_rf = rf_model.predict(X)
    assert len(preds_rf) == len(X)
    assert (preds_rf >= 0).all()

    # Check feature importances
    fi = rf_model.get_feature_importances()
    assert fi is not None
    assert len(fi) == 3


def test_group_cross_validation():
    df = make_dummy_data()
    X = df[["time_cycles", "s_1", "s_2"]]
    y = df["RUL"]
    groups = df["unit_nr"]

    cv_res = run_group_cross_validation(
        model_type="ridge",
        X=X,
        y=y,
        groups=groups,
        n_splits=3,
        scale_features=True
    )

    assert "mean_mae" in cv_res
    assert "mean_rmse" in cv_res
    assert "mean_r2" in cv_res
    assert len(cv_res["fold_details"]) == 3
    # Check that in each fold, train units and val units are completely disjoint
    for f in cv_res["fold_details"]:
        assert f["train_units"] + f["val_units"] == 6


def test_model_persistence(tmp_path):
    df = make_dummy_data()
    X = df[["time_cycles", "s_1", "s_2"]]
    y = df["RUL"]

    model = TurbofanPredictor(model_type="ridge")
    model.fit(X, y)

    saved_path = save_model_artifact(model, "test_ridge", metadata={"test": True}, output_dir=tmp_path)
    loaded_model = load_model_artifact("test_ridge", model_dir=tmp_path)

    orig_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)

    np.testing.assert_allclose(orig_preds, loaded_preds)
