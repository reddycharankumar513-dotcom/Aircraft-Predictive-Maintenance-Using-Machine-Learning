"""
Evaluation and Benchmark CLI Script.
Evaluates saved predictive models against held-out test datasets (NASA C-MAPSS or custom),
computes aerospace metrics (MAE, RMSE, R2, NASA C-MAPSS Score), generates residual diagnostics,
and outputs unit-by-unit fleet maintenance predictions.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import MODELS_DIR, DEFAULT_RUL_CLIP
from src.data_loader import load_cmapss_dataset, load_user_csv
from src.preprocessor import clean_telemetry_data
from src.feature_engineering import build_temporal_features
from src.metrics import evaluate_regression_predictions, categorize_maintenance_risk
from src.utils import load_model_artifact, plot_actual_vs_predicted, plot_residual_distribution, save_evaluation_report


def evaluate_test_set(
    dataset_name: str = "FD001",
    model_name: str = "best_model",
    rul_clip: int = DEFAULT_RUL_CLIP,
    output_dir: Path = MODELS_DIR
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AIRBUS PREDICTIVE MAINTENANCE SYSTEM - MODEL EVALUATION")
    print("=" * 78)

    # 1. Load Model Artifact
    print(f"[1/4] Loading trained model artifact: '{model_name}'...")
    try:
        model = load_model_artifact(model_name, model_dir=output_dir)
    except Exception as e:
        print(f"Error: Could not load model '{model_name}'. Please run train.py first. ({e})")
        return

    # 2. Load Test Data
    print(f"[2/4] Loading test dataset: '{dataset_name}'...")
    if dataset_name.upper() in ["FD001", "FD002", "FD003", "FD004"]:
        _, test_df, rul_series, meta = load_cmapss_dataset(dataset_name, rul_clip=rul_clip)
    else:
        test_path = Path(dataset_name)
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {test_path}")
        test_df, meta = load_user_csv(test_path, rul_clip=rul_clip)
        rul_series = None

    # Preprocess & engineer features
    cleaned_test, _ = clean_telemetry_data(test_df, drop_constants=False)
    enhanced_test = build_temporal_features(cleaned_test, window_sizes=(5, 15))

    # 3. Predict on Test Engines
    print(f"[3/4] Generating RUL predictions for {test_df['unit_nr'].nunique()} test engines...")

    # For NASA C-MAPSS benchmark, evaluate on the LAST observed cycle for each test unit
    last_cycle_indices = enhanced_test.groupby("unit_nr")["time_cycles"].idxmax()
    last_cycle_df = enhanced_test.loc[last_cycle_indices].copy()

    # Align features expected by the trained model
    X_test_last = last_cycle_df[model.feature_names_]
    preds_last = model.predict(X_test_last)
    last_cycle_df["pred_rul"] = preds_last

    # Ground truth
    if rul_series is not None:
        # Match unit_nr
        y_true_last = last_cycle_df["unit_nr"].map(rul_series).values
    elif "RUL" in last_cycle_df.columns:
        y_true_last = last_cycle_df["RUL"].values
    else:
        y_true_last = None

    # 4. Compute Metrics
    print("\n[4/4] Performance Assessment Report:")
    if y_true_last is not None:
        metrics = evaluate_regression_predictions(y_true_last, preds_last)
        print("-" * 78)
        print("HELD-OUT TEST SET BENCHMARK METRICS (FINAL OPERATIONAL CYCLE)")
        print("-" * 78)
        print(f"Mean Absolute Error (MAE)        : {metrics['mae']:>8.2f} cycles")
        print(f"Root Mean Squared Error (RMSE)   : {metrics['rmse']:>8.2f} cycles")
        print(f"Coefficient of Determination (R²): {metrics['r2']:>8.3f}")
        print(f"NASA C-MAPSS Asymmetric Score    : {metrics['cmapss_score']:>8.1f}")
        print(f"Late Predictions (Unsafe %):     : {metrics['late_prediction_pct']:>8.1f} %")
        print(f"Early Predictions (Safe %):      : {metrics['early_prediction_pct']:>8.1f} %")
        print("-" * 78)

        # Generate diagnostic plots
        fig_parity = plot_actual_vs_predicted(
            y_true_last, preds_last,
            title=f"Actual vs Predicted RUL ({dataset_name} Test Units)",
            save_path=output_dir / "evaluation_parity_plot.png"
        )
        fig_residuals = plot_residual_distribution(
            y_true_last, preds_last,
            title=f"Error Residuals ({dataset_name} Test Units)",
            save_path=output_dir / "evaluation_residual_plot.png"
        )
        print(f"Saved diagnostic plots to: {output_dir}")

        save_evaluation_report(metrics, "test_evaluation_report", output_dir=output_dir)

    # 5. Build Fleet Maintenance Action Table
    action_records = []
    for _, row in last_cycle_df.iterrows():
        unit = int(row["unit_nr"])
        cycle = int(row["time_cycles"])
        pred = float(row["pred_rul"])
        act = float(y_true_last[len(action_records)]) if y_true_last is not None else None
        risk = categorize_maintenance_risk(pred)
        
        record = {
            "unit_nr": unit,
            "last_cycle": cycle,
            "predicted_rul": round(pred, 1),
            "actual_rul": round(act, 1) if act is not None else "N/A",
            "error": round(pred - act, 1) if act is not None else "N/A",
            "risk_tier": risk["tier"],
            "maintenance_action": risk["action"]
        }
        action_records.append(record)

    fleet_report_df = pd.DataFrame(action_records).sort_values("predicted_rul")
    report_csv = output_dir / "fleet_test_predictions.csv"
    fleet_report_df.to_csv(report_csv, index=False)
    print(f"\nFleet Maintenance Action Plan exported to: {report_csv}")
    print("\nSample Priority Engine Queue (Earliest Failure Predicted):")
    print(fleet_report_df.head(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained turbofan model on test dataset.")
    parser.add_argument("--dataset", type=str, default="FD001", help="Test dataset ('FD001'-'FD004' or custom CSV path)")
    parser.add_argument("--model", type=str, default="best_model", help="Saved model file base name (default: best_model)")
    parser.add_argument("--rul-clip", type=int, default=DEFAULT_RUL_CLIP, help="Piecewise RUL clipping upper bound")

    args = parser.parse_args()
    evaluate_test_set(
        dataset_name=args.dataset,
        model_name=args.model,
        rul_clip=args.rul_clip
    )


if __name__ == "__main__":
    main()
