"""
Model Training and Cross-Validation CLI Pipeline.
Executes leakage-aware GroupKFold validation, benchmarks simple baselines against tuned models,
evaluates using MAE, RMSE, R2, and the NASA C-MAPSS asymmetric score, and saves model artifacts.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import MODELS_DIR, DEFAULT_RUL_CLIP, PRIMARY_SENSORS
from src.data_loader import load_cmapss_dataset, load_user_csv
from src.synthetic_generator import generate_synthetic_turbofan_fleet
from src.preprocessor import clean_telemetry_data, audit_data_quality
from src.feature_engineering import build_temporal_features, extract_features_and_target
from src.models import TurbofanPredictor, run_group_cross_validation
from src.metrics import evaluate_regression_predictions
from src.utils import save_model_artifact, save_evaluation_report


def train_and_evaluate(
    dataset_name: str = "FD001",
    rul_clip: int = DEFAULT_RUL_CLIP,
    n_splits: int = 5,
    save_models: bool = True
):
    print("=" * 78)
    print("AIRBUS PREDICTIVE MAINTENANCE SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 78)
    print(f"Target Dataset: {dataset_name} | RUL Clip Threshold: {rul_clip} | Group CV Folds: {n_splits}")

    # 1. Load Data
    test_df = None
    rul_series = None
    dataset_meta = {}

    if dataset_name.upper() in ["FD001", "FD002", "FD003", "FD004"]:
        train_df, test_df, rul_series, dataset_meta = load_cmapss_dataset(dataset_name, rul_clip=rul_clip)
        source_label = f"NASA C-MAPSS {dataset_name.upper()}"
    elif dataset_name.lower() == "synthetic":
        train_df = generate_synthetic_turbofan_fleet(num_units=60, rul_clip=rul_clip)
        source_label = "Synthetic Turbofan Fleet (Demonstration Data)"
        dataset_meta = {"is_synthetic": True, "dataset_id": "SYNTHETIC"}
    else:
        # Custom file path
        csv_path = Path(dataset_name)
        if not csv_path.exists():
            raise FileNotFoundError(f"Custom dataset file not found: {csv_path}")
        train_df, dataset_meta = load_user_csv(csv_path, rul_clip=rul_clip)
        source_label = f"Custom Dataset ({csv_path.name})"

    print(f"\n[1/5] Loaded data from: {source_label}")
    print(f"      Total records: {len(train_df):,} | Unique Engine Units: {train_df['unit_nr'].nunique()}")

    # 2. Quality Audit & Clean
    audit = audit_data_quality(train_df)
    print(f"\n[2/5] Data Quality Audit:")
    print(f"      Missing values: {audit['total_missing_cells']} | Duplicates: {audit['duplicate_rows']}")
    print(f"      Constant/Zero-variance features identified: {audit['constant_columns']}")

    cleaned_train, clean_log = clean_telemetry_data(train_df, drop_constants=True)
    print(f"      Active features remaining after zero-variance filter: {cleaned_train.shape[1]}")

    # 3. Feature Engineering
    print(f"\n[3/5] Engineering Causal Temporal Features (Grouped by Unit, No Leakage)...")
    enhanced_train = build_temporal_features(cleaned_train, window_sizes=(5, 15))
    X, y, groups, feature_cols = extract_features_and_target(enhanced_train)
    print(f"      Constructed {len(feature_cols)} total predictive features.")

    # 4. Group Cross-Validation Comparison
    print(f"\n[4/5] Executing {n_splits}-Fold GroupKFold Cross-Validation (Zero Unit Overlap):")
    model_candidates = [
        ("mean_baseline", "Mean Dummy Baseline", {}),
        ("ridge", "Linear Baseline (Ridge)", {"alpha": 10.0}),
        ("random_forest", "Random Forest Regressor (Tuned)", {"n_estimators": 100, "max_depth": 14}),
        ("gradient_boosting", "Gradient Boosted Trees (Tuned)", {"max_iter": 120, "learning_rate": 0.08})
    ]

    cv_leaderboard = []
    models_trained = {}

    for m_type, m_label, m_params in model_candidates:
        print(f"      Evaluating {m_label}...")
        cv_res = run_group_cross_validation(
            model_type=m_type,
            X=X,
            y=y,
            groups=groups,
            n_splits=n_splits,
            model_params=m_params,
            scale_features=True
        )
        cv_leaderboard.append({
            "model_type": m_type,
            "label": m_label,
            "cv_mae": cv_res["mean_mae"],
            "cv_rmse": cv_res["mean_rmse"],
            "cv_r2": cv_res["mean_r2"],
            "cv_cmapss_score": cv_res["mean_cmapss_score"]
        })

    # Summary Table
    lb_df = pd.DataFrame(cv_leaderboard).sort_values("cv_rmse")
    print("\n" + "-" * 78)
    print("CROSS-VALIDATION BENCHMARK SUMMARY (OUT-OF-FOLD UNIT PERFORMANCE)")
    print("-" * 78)
    for _, row in lb_df.iterrows():
        print(f"{row['label']:<34} | MAE: {row['cv_mae']:>6.2f} | RMSE: {row['cv_rmse']:>6.2f} | R²: {row['cv_r2']:>6.3f} | Score: {row['cv_cmapss_score']:>9.1f}")
    print("-" * 78)

    # 5. Fit Final Production Model on Full Training Dataset
    best_candidate = lb_df.iloc[0]
    best_type = best_candidate["model_type"]
    print(f"\n[5/5] Finalizing Best Model: {best_candidate['label']} ({best_type})")
    
    # Retrieve best parameters
    best_params = next(p for t, l, p in model_candidates if t == best_type)
    final_model = TurbofanPredictor(model_type=best_type, model_params=best_params, scale_features=True)
    final_model.fit(X, y)

    # Feature Importances
    fi = final_model.get_feature_importances()
    if fi is not None:
        print("\nTop 8 Most Predictive Degradation Features:")
        for feat, val in fi.head(8).items():
            print(f"  - {feat:<30} : {val:.4f}")

    # Save Artifacts
    if save_models:
        meta_to_save = {
            "dataset_name": dataset_name,
            "source_label": source_label,
            "is_synthetic": bool(dataset_meta.get("is_synthetic", False)),
            "rul_clip": rul_clip,
            "model_type": best_type,
            "feature_cols": feature_cols,
            "cv_benchmark": cv_leaderboard,
            "train_samples": len(X),
            "train_units": int(groups.nunique())
        }
        model_file = save_model_artifact(final_model, "best_model", metadata=meta_to_save)
        report_file = save_evaluation_report(meta_to_save, "training_summary")
        print(f"\nTrained model artifacts successfully saved:")
        print(f"  - Model binary: {model_file}")
        print(f"  - Summary report: {report_file}")

    print("\nTraining workflow completed successfully.")
    return final_model, lb_df


def main():
    parser = argparse.ArgumentParser(description="Train turbofan predictive maintenance model.")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset name ('FD001'-'FD004', 'synthetic', or CSV path)")
    parser.add_argument("--rul-clip", type=int, default=DEFAULT_RUL_CLIP, help="Piecewise RUL clipping upper bound (default: 125)")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of GroupKFold splits (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Do not save model artifacts to disk")

    args = parser.parse_args()
    train_and_evaluate(
        dataset_name=args.dataset,
        rul_clip=args.rul_clip,
        n_splits=args.cv_splits,
        save_models=not args.no_save
    )


if __name__ == "__main__":
    main()
