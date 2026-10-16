"""
Model Architecture and Leakage-Aware Training Module.
Provides simple regression baselines and tuned non-linear models (Random Forest, Gradient Boosting),
with cross-validation strictly partitioned by engine unit IDs (GroupKFold) to prevent trajectory leakage.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import evaluate_regression_predictions


class TurbofanPredictor:
    """
    Unified predictor wrapper for turbofan RUL estimation.
    Encapsulates preprocessing, feature scaling, model fitting, and evaluation.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True
    ):
        """
        Args:
            model_type: One of 'mean_baseline', 'linear_baseline', 'ridge',
                        'random_forest', or 'gradient_boosting'.
            model_params: Dict of model hyperparameter overrides.
            scale_features: Whether to apply StandardScaler to features.
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.scale_features = scale_features
        self.estimator: Optional[BaseEstimator] = None
        self.scaler: Optional[StandardScaler] = StandardScaler() if scale_features else None
        self.feature_names_: List[str] = []
        self.cv_results_: Dict[str, Any] = {}
        self.is_fitted: bool = False

        self._build_estimator()

    def _build_estimator(self):
        """Construct the underlying scikit-learn estimator."""
        params = self.model_params.copy()

        if self.model_type == "mean_baseline":
            self.estimator = DummyRegressor(strategy="mean")
        elif self.model_type == "linear_baseline":
            self.estimator = LinearRegression()
        elif self.model_type == "ridge":
            alpha = params.pop("alpha", 1.0)
            self.estimator = Ridge(alpha=alpha, random_state=42)
        elif self.model_type == "random_forest":
            n_estimators = params.pop("n_estimators", 100)
            max_depth = params.pop("max_depth", 15)
            min_samples_split = params.pop("min_samples_split", 5)
            min_samples_leaf = params.pop("min_samples_leaf", 2)
            self.estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            max_iter = params.pop("max_iter", 150)
            learning_rate = params.pop("learning_rate", 0.08)
            max_depth = params.pop("max_depth", 8)
            min_samples_leaf = params.pop("min_samples_leaf", 20)
            self.estimator = HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model pipeline on training features.

        Args:
            X: Feature matrix.
            y: Target RUL series.
        """
        self.feature_names_ = list(X.columns)
        X_vals = X.values

        if self.scaler is not None:
            X_vals = self.scaler.fit_transform(X_vals)

        self.estimator.fit(X_vals, y.values)
        self.is_fitted = True

        # Cache feature importances
        if hasattr(self.estimator, "feature_importances_"):
            self.cached_importances_ = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            self.cached_importances_ = np.abs(self.estimator.coef_)
        elif self.model_type == "gradient_boosting":
            # Fast permutation importance on subsample
            try:
                from sklearn.inspection import permutation_importance
                sub_n = min(400, len(X_vals))
                sub_idx = np.random.RandomState(42).choice(len(X_vals), sub_n, replace=False)
                pi = permutation_importance(
                    self.estimator, X_vals[sub_idx], y.values[sub_idx],
                    n_repeats=2, random_state=42
                )
                self.cached_importances_ = np.maximum(0.0, pi.importances_mean)
            except Exception:
                self.cached_importances_ = None
        else:
            self.cached_importances_ = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate remaining useful life predictions. Enforces non-negativity.

        Args:
            X: Feature matrix.

        Returns:
            Non-negative RUL prediction array.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Ensure correct feature alignment
        if isinstance(X, pd.DataFrame):
            # Check for missing expected columns
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f"Features missing from input: {missing}")
            X_aligned = X[self.feature_names_].values
        else:
            X_aligned = np.asarray(X)

        if self.scaler is not None:
            X_aligned = self.scaler.transform(X_aligned)

        raw_preds = self.estimator.predict(X_aligned)
        # RUL physically cannot be negative
        return np.maximum(0.0, raw_preds)

    def get_feature_importances(self) -> Optional[pd.Series]:
        """Extract feature importances or linear coefficients if available."""
        if not self.is_fitted:
            return None

        if hasattr(self, "cached_importances_") and self.cached_importances_ is not None:
            return pd.Series(self.cached_importances_, index=self.feature_names_).sort_values(ascending=False)
        elif hasattr(self.estimator, "feature_importances_"):
            return pd.Series(self.estimator.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
        elif hasattr(self.estimator, "coef_"):
            return pd.Series(np.abs(self.estimator.coef_), index=self.feature_names_).sort_values(ascending=False)
        return None


def run_group_cross_validation(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
    scale_features: bool = True
) -> Dict[str, Any]:
    """
    Perform GroupKFold cross-validation grouped strictly by engine unit IDs.
    Guarantees zero engine leakage between train and validation folds.

    Args:
        model_type: Algorithm name.
        X: Feature matrix.
        y: Target RUL series.
        groups: Engine unit series (unit_nr).
        n_splits: Number of group splits (folds).
        model_params: Optional hyperparameters.
        scale_features: Whether to scale features.

    Returns:
        Aggregated validation metrics and fold details.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    val_preds_all = np.zeros(len(y))
    val_indices_all = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = TurbofanPredictor(model_type=model_type, model_params=model_params, scale_features=scale_features)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        val_preds_all[val_idx] = preds
        val_indices_all.extend(val_idx)

        metrics = evaluate_regression_predictions(y_va.values, preds)
        metrics["fold"] = fold_idx + 1
        metrics["train_units"] = int(groups.iloc[train_idx].nunique())
        metrics["val_units"] = int(groups.iloc[val_idx].nunique())
        fold_metrics.append(metrics)

    # Compute overall metrics across all out-of-fold predictions
    overall_metrics = evaluate_regression_predictions(y.values[val_indices_all], val_preds_all[val_indices_all])

    # Compute fold averages and standard deviations
    mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
    std_mae = float(np.std([m["mae"] for m in fold_metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    std_rmse = float(np.std([m["rmse"] for m in fold_metrics]))
    mean_r2 = float(np.mean([m["r2"] for m in fold_metrics]))
    std_r2 = float(np.std([m["r2"] for m in fold_metrics]))
    mean_score = float(np.mean([m["cmapss_score"] for m in fold_metrics]))

    return {
        "model_type": model_type,
        "n_splits": n_splits,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_cmapss_score": mean_score,
        "overall_metrics": overall_metrics,
        "fold_details": fold_metrics
    }
