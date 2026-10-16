"""
Unit tests for aerospace evaluation metrics and risk categorization.
"""

import numpy as np
import pytest
from src.metrics import calculate_cmapss_score, evaluate_regression_predictions, categorize_maintenance_risk


def test_cmapss_score_zero_error():
    y_true = np.array([50.0, 100.0, 20.0])
    y_pred = np.array([50.0, 100.0, 20.0])
    score = calculate_cmapss_score(y_true, y_pred)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_cmapss_score_asymmetry():
    """
    Verify that late prediction (+error, dangerous) receives a higher penalty
    than early prediction (-error, conservative) of the exact same absolute magnitude.
    """
    y_true = np.array([100.0])
    
    # Early prediction by 10 cycles (d = -10)
    score_early = calculate_cmapss_score(y_true, np.array([90.0]))
    # Late prediction by 10 cycles (d = +10)
    score_late = calculate_cmapss_score(y_true, np.array([110.0]))

    # Early: exp(10/13) - 1 ≈ 1.158
    # Late: exp(10/10) - 1 ≈ 1.718
    assert score_late > score_early
    assert score_early == pytest.approx(np.exp(10.0 / 13.0) - 1.0, rel=1e-4)
    assert score_late == pytest.approx(np.exp(10.0 / 10.0) - 1.0, rel=1e-4)


def test_evaluate_regression_predictions():
    y_true = np.array([100.0, 50.0, 20.0])
    y_pred = np.array([105.0, 45.0, 20.0])  # errors: +5, -5, 0

    metrics = evaluate_regression_predictions(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(10.0 / 3.0, abs=1e-3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(50.0 / 3.0), abs=1e-3)
    assert metrics["sample_count"] == 3
    assert metrics["late_prediction_pct"] == pytest.approx(100.0 / 3.0, abs=1e-2)
    assert metrics["early_prediction_pct"] == pytest.approx(100.0 / 3.0, abs=1e-2)


def test_categorize_maintenance_risk():
    crit = categorize_maintenance_risk(15.0)
    assert crit["tier"] == "CRITICAL"
    assert "Immediate" in crit["action"]

    high = categorize_maintenance_risk(45.0)
    assert high["tier"] == "HIGH"

    med = categorize_maintenance_risk(90.0)
    assert med["tier"] == "MEDIUM"

    low = categorize_maintenance_risk(150.0)
    assert low["tier"] == "LOW"
