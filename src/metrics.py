"""
Evaluation Metrics and Aerospace Maintenance Risk Assessment Module.
Includes standard regression metrics (MAE, RMSE, R2) and the official NASA C-MAPSS
asymmetric scoring function, which penalizes dangerous late predictions over conservative early ones.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RISK_THRESHOLDS, RISK_COLORS


def calculate_cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the official NASA C-MAPSS PHM08 asymmetric penalty score.

    Mathematical Formulation:
        d = y_pred - y_true
        s_i = exp(-d / 13) - 1  if d < 0  (Early prediction: Conservative, safe)
        s_i = exp(d / 10) - 1   if d >= 0 (Late prediction: Dangerous, potential AOG)
        Total Score S = sum(s_i)

    A lower score is superior. Late predictions are penalized significantly more heavily
    because overestimating engine life risks catastrophic in-flight failure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true

    scores = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0
    )
    return float(np.sum(scores))


def evaluate_regression_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Compute comprehensive aerospace prognostic metrics for RUL estimation.

    Returns:
        Dictionary containing MAE, RMSE, R2, NASA Score, Early/Late error breakdown.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Standard metrics
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    
    # NASA Asymmetric Score
    cmapss_score = calculate_cmapss_score(y_true_arr, y_pred_arr)

    # Aerospace risk diagnostics: Late vs Early predictions
    errors = y_pred_arr - y_true_arr
    late_mask = errors > 0
    early_mask = errors < 0
    exact_mask = errors == 0

    total_count = len(errors)
    late_pct = float(np.sum(late_mask) / total_count * 100) if total_count > 0 else 0.0
    early_pct = float(np.sum(early_mask) / total_count * 100) if total_count > 0 else 0.0

    mean_late_error = float(np.mean(errors[late_mask])) if np.any(late_mask) else 0.0
    mean_early_error = float(np.mean(np.abs(errors[early_mask]))) if np.any(early_mask) else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cmapss_score": cmapss_score,
        "sample_count": total_count,
        "late_prediction_pct": late_pct,
        "early_prediction_pct": early_pct,
        "mean_late_error": mean_late_error,
        "mean_early_error": mean_early_error
    }


def categorize_maintenance_risk(rul: float) -> Dict[str, Any]:
    """
    Categorize remaining useful life into an actionable Airbus fleet maintenance tier.

    Tiers:
    - CRITICAL (<= 30 cycles): Urgent inspection; ground engine for borescope/module replacement.
    - HIGH (31-75 cycles): Plan line maintenance; pre-position spare parts at next hub.
    - MEDIUM (76-125 cycles): Trending wear; increase telemetry polling frequency.
    - LOW (> 125 cycles): Healthy engine; standard scheduled turnaround.
    """
    rul_val = float(rul)
    if rul_val <= RISK_THRESHOLDS["CRITICAL"]:
        tier = "CRITICAL"
        desc = "Urgent: Immediate AOG avoidance action. Borescope inspection required."
        action = "Ground Engine / Schedule Immediate Replacement"
    elif rul_val <= RISK_THRESHOLDS["HIGH"]:
        tier = "HIGH"
        desc = "Elevated Wear: Pre-order Line Replaceable Units (LRUs) for next hub turnaround."
        action = "Schedule Maintenance Turnaround (< 75 cycles)"
    elif rul_val <= RISK_THRESHOLDS["MEDIUM"]:
        tier = "MEDIUM"
        desc = "Moderate Wear: Increase condition monitoring telemetry analysis."
        action = "Monitor Trend & Flag for Routine C-Check"
    else:
        tier = "LOW"
        desc = "Nominal: Engine health parameters within standard operating envelope."
        action = "Standard Flight Operations"

    return {
        "tier": tier,
        "color": RISK_COLORS[tier],
        "description": desc,
        "action": action,
        "rul_val": rul_val
    }
