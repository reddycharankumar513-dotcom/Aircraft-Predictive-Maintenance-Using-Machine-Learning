"""
Utility Functions for Persistence, Reporting, and Visualization.
Supports saving/loading model pipelines, exporting JSON metrics,
and generating publication-quality aerospace diagnostic figures.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MODELS_DIR


def save_model_artifact(
    model_obj: Any,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Path = MODELS_DIR
) -> Path:
    """
    Serialize trained model pipeline and accompanying metadata to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{filename}.joblib"
    joblib.dump(model_obj, model_path)

    if metadata is not None:
        meta_path = output_dir / f"{filename}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return model_path


def load_model_artifact(
    filename: str,
    model_dir: Path = MODELS_DIR
) -> Any:
    """
    Load a serialized model pipeline from disk.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / f"{filename}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path)


def save_evaluation_report(
    metrics: Dict[str, Any],
    filename: str,
    output_dir: Path = MODELS_DIR
) -> Path:
    """
    Save evaluation metrics dictionary as formatted JSON report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{filename}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return report_path


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted RUL",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate diagnostic Actual vs Predicted scatter plot with perfect-prediction 45-degree parity line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, color="#1E3A8A", edgecolors="none", s=25, label="Operational Engines")
    
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=1.8, label="Ideal Parity (y=x)")
    
    ax.set_xlabel("Actual RUL (Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted RUL (Cycles)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Residuals Distribution",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot histogram and KDE of prediction residuals (y_pred - y_true).
    Highlights asymmetric aerospace risk: errors > 0 are late predictions (dangerous).
    """
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(residuals, bins=35, color="#2563EB", alpha=0.65, edgecolor="white", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvspan(0, max(1.0, float(np.max(residuals))), color="#DC2626", alpha=0.12, label="Late Prediction (Dangerous)")
    ax.axvspan(min(-1.0, float(np.min(residuals))), 0, color="#16A34A", alpha=0.12, label="Early Prediction (Conservative)")

    ax.set_xlabel("Residual: Predicted - Actual (Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
