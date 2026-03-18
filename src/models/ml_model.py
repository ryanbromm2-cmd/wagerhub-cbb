"""
Machine learning models for CBB game total prediction.
XGBoost + LightGBM + Ridge regression ensemble.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xgboost as xgb
    import lightgbm as lgb

from src.features.feature_engineering import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model storage path
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MISSING_SENTINEL = -1.0


# ── Feature list ──────────────────────────────────────────────────────────────

# All features used by the ML model (aligned with FeatureEngineer output)
ML_FEATURE_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c != "data_completeness"]


class MLModelTrainer:
    """
    Trains XGBoost, LightGBM, and Ridge models to predict game totals.

    Usage::

        trainer = MLModelTrainer(config)
        trainer.train(features_df, targets)
        trainer.save_models("models/")
        metrics = trainer.evaluate(X_test, y_test)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})

        xgb_params = model_cfg.get("xgboost", {})
        lgb_params = model_cfg.get("lightgbm", {})
        ridge_params = model_cfg.get("ridge", {})
        ens_params = model_cfg.get("ensemble", {})

        self.xgb_weight: float = float(ens_params.get("xgboost_weight", 0.40))
        self.lgb_weight: float = float(ens_params.get("lightgbm_weight", 0.40))
        self.ridge_weight: float = float(ens_params.get("ridge_weight", 0.20))

        # XGBoost
        self._xgb_model = xgb.XGBRegressor(
            n_estimators=int(xgb_params.get("n_estimators", 400)),
            max_depth=int(xgb_params.get("max_depth", 5)),
            learning_rate=float(xgb_params.get("learning_rate", 0.04)),
            subsample=float(xgb_params.get("subsample", 0.8)),
            colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
            reg_alpha=float(xgb_params.get("reg_alpha", 0.1)),
            reg_lambda=float(xgb_params.get("reg_lambda", 1.0)),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            objective="reg:squarederror",
        )

        # LightGBM
        self._lgb_model = lgb.LGBMRegressor(
            n_estimators=int(lgb_params.get("n_estimators", 400)),
            max_depth=int(lgb_params.get("max_depth", 6)),
            learning_rate=float(lgb_params.get("learning_rate", 0.04)),
            num_leaves=int(lgb_params.get("num_leaves", 40)),
            subsample=float(lgb_params.get("subsample", 0.8)),
            colsample_bytree=float(lgb_params.get("colsample_bytree", 0.8)),
            reg_alpha=float(lgb_params.get("reg_alpha", 0.1)),
            reg_lambda=float(lgb_params.get("reg_lambda", 1.0)),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        # Ridge in a Pipeline (imputation + scaling + ridge)
        self._ridge_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(ridge_params.get("alpha", 5.0)))),
            ]
        )

        # Stored imputer for XGB/LGB (handle missing values)
        self._imputer = SimpleImputer(strategy="median")

        self._trained = False
        self._feature_importances: Optional[pd.DataFrame] = None
        self._feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        time_col: str = "date",
    ) -> None:
        """
        Train all three models on the provided feature matrix.

        Args:
            features_df: DataFrame where each row is a game and columns
                         are feature values.  Should contain ML_FEATURE_COLUMNS.
            target:      Series of actual game totals (same index as features_df).
            time_col:    Name of the date/time column (for sorting).
        """
        logger.info(f"Training ML models on {len(features_df)} samples...")

        # Align feature columns
        X, self._feature_names = self._align_features(features_df)
        y = target.values

        if len(X) < 50:
            logger.warning("Fewer than 50 training samples — model may be unreliable.")

        # Impute missing values for tree models
        X_imputed = self._imputer.fit_transform(X)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self._xgb_model.fit(
            X_imputed, y,
            eval_set=[(X_imputed, y)],
            verbose=False,
        )

        # Train LightGBM
        logger.info("Training LightGBM...")
        self._lgb_model.fit(X_imputed, y)

        # Train Ridge (pipeline handles imputation/scaling)
        logger.info("Training Ridge...")
        self._ridge_pipeline.fit(X, y)

        self._trained = True
        self._build_feature_importances()
        logger.info("All ML models trained successfully.")

    def _align_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """
        Align DataFrame columns to expected feature list.
        Adds zero-filled columns for any missing features.
        Returns (ndarray, feature_names).
        """
        expected = ML_FEATURE_COLUMNS
        df_copy = df.copy()

        # Add missing columns as NaN
        for col in expected:
            if col not in df_copy.columns:
                df_copy[col] = np.nan

        # Replace sentinel values with NaN for proper imputation
        df_copy = df_copy[expected].replace(MISSING_SENTINEL, np.nan)

        return df_copy.values.astype(float), expected

    # ── Cross-validation ──────────────────────────────────────────────────

    def time_aware_cross_validate(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        time_col: str = "date",
        n_splits: int = 5,
    ) -> dict:
        """
        Time-series aware cross-validation using rolling window splits.
        Ensures train always precedes test in time.

        Returns dict with mean/std of MAE, RMSE, R2 across folds.
        """
        logger.info(f"Running {n_splits}-fold time-series cross-validation...")

        # Sort by time
        if time_col in features_df.columns:
            idx = features_df[time_col].argsort()
            features_df = features_df.iloc[idx]
            target = target.iloc[idx]

        X, feature_names = self._align_features(features_df)
        y = target.values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results: dict[str, list] = {"mae": [], "rmse": [], "r2": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(X_train) < 30:
                logger.debug(f"Fold {fold}: too few train samples, skipping.")
                continue

            # Fit a fresh XGB on this fold (representative of the ensemble)
            imp = SimpleImputer(strategy="median")
            X_train_imp = imp.fit_transform(X_train)
            X_test_imp = imp.transform(X_test)

            fold_xgb = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            fold_xgb.fit(X_train_imp, y_train)
            preds = fold_xgb.predict(X_test_imp)

            mae = mean_absolute_error(y_test, preds)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            cv_results["mae"].append(mae)
            cv_results["rmse"].append(rmse)
            cv_results["r2"].append(r2)
            logger.debug(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

        summary = {}
        for metric, vals in cv_results.items():
            if vals:
                summary[f"cv_{metric}_mean"] = round(np.mean(vals), 4)
                summary[f"cv_{metric}_std"] = round(np.std(vals), 4)
            else:
                summary[f"cv_{metric}_mean"] = None
                summary[f"cv_{metric}_std"] = None

        logger.info(
            f"CV results: MAE={summary.get('cv_mae_mean'):.2f} "
            f"± {summary.get('cv_mae_std'):.2f}"
        )
        return summary

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluate the trained ensemble on test data.

        Returns dict with MAE, RMSE, R2, and calibration by total bucket.
        """
        if not self._trained:
            raise RuntimeError("Models have not been trained. Call train() first.")

        X_arr, _ = self._align_features(X_test)
        y_arr = y_test.values

        predictor = MLModelPredictor(self)
        preds = []
        for i in range(len(X_arr)):
            row_dict = {self._feature_names[j]: X_arr[i, j] for j in range(len(self._feature_names))}
            result = predictor.predict(row_dict)
            preds.append(result.get("ml_ensemble_total", 0))

        preds_arr = np.array(preds)
        mae = mean_absolute_error(y_arr, preds_arr)
        rmse = math.sqrt(mean_squared_error(y_arr, preds_arr))
        r2 = r2_score(y_arr, preds_arr)

        # Calibration by total bucket
        calibration = {}
        buckets = [(0, 130), (130, 145), (145, 160), (160, 300)]
        for lo, hi in buckets:
            mask = (y_arr >= lo) & (y_arr < hi)
            if mask.sum() > 0:
                bucket_mae = mean_absolute_error(y_arr[mask], preds_arr[mask])
                calibration[f"{lo}-{hi}"] = round(bucket_mae, 3)

        result = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "n_samples": len(y_arr),
            "calibration_by_bucket": calibration,
        }
        logger.info(f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
        return result

    # ── Feature importance ────────────────────────────────────────────────

    def _build_feature_importances(self) -> None:
        if not self._trained or not self._feature_names:
            return

        xgb_imp = self._xgb_model.feature_importances_
        lgb_imp = self._lgb_model.feature_importances_

        # Normalize
        xgb_imp_norm = xgb_imp / (xgb_imp.sum() + 1e-9)
        lgb_imp_norm = lgb_imp / (lgb_imp.sum() + 1e-9)

        # Weighted average
        avg_imp = (
            self.xgb_weight * xgb_imp_norm
            + self.lgb_weight * lgb_imp_norm
        ) / (self.xgb_weight + self.lgb_weight)

        self._feature_importances = (
            pd.DataFrame(
                {"feature": self._feature_names, "importance": avg_imp}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return DataFrame of feature importances sorted descending."""
        if self._feature_importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        return self._feature_importances.copy()

    # ── Serialization ─────────────────────────────────────────────────────

    def save_models(self, path: str = None) -> str:
        """
        Save all trained models to disk using joblib.

        Args:
            path: Directory to save models (default: models/).

        Returns:
            Path to saved model bundle.
        """
        if not self._trained:
            raise RuntimeError("Cannot save untrained models.")

        save_dir = Path(path) if path else _MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = save_dir / "cbb_totals_models.joblib"

        bundle = {
            "xgb": self._xgb_model,
            "lgb": self._lgb_model,
            "ridge_pipeline": self._ridge_pipeline,
            "imputer": self._imputer,
            "feature_names": self._feature_names,
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
            "ridge_weight": self.ridge_weight,
            "feature_importances": self._feature_importances,
        }
        joblib.dump(bundle, str(bundle_path), compress=3)
        logger.info(f"Models saved to {bundle_path}")
        return str(bundle_path)

    def load_models(self, path: str = None) -> None:
        """
        Load previously saved models from disk.

        Args:
            path: Path to the saved model bundle (.joblib file) or directory.
        """
        if path:
            p = Path(path)
            if p.is_dir():
                p = p / "cbb_totals_models.joblib"
        else:
            p = _MODEL_DIR / "cbb_totals_models.joblib"

        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        bundle = joblib.load(str(p))
        self._xgb_model = bundle["xgb"]
        self._lgb_model = bundle["lgb"]
        self._ridge_pipeline = bundle["ridge_pipeline"]
        self._imputer = bundle["imputer"]
        self._feature_names = bundle["feature_names"]
        self.xgb_weight = bundle.get("xgb_weight", 0.40)
        self.lgb_weight = bundle.get("lgb_weight", 0.40)
        self.ridge_weight = bundle.get("ridge_weight", 0.20)
        self._feature_importances = bundle.get("feature_importances")
        self._trained = True
        logger.info(f"Models loaded from {p}")


class MLModelPredictor:
    """
    Generates predictions from trained ML models.

    Can be used standalone (with a loaded trainer) or via the ensemble.
    """

    def __init__(self, trainer: MLModelTrainer):
        self._trainer = trainer

    @classmethod
    def from_disk(cls, path: str = None, config: dict = None) -> "MLModelPredictor":
        """Load a predictor from saved model files."""
        trainer = MLModelTrainer(config or {})
        trainer.load_models(path)
        return cls(trainer)

    def predict(self, features: dict) -> dict:
        """
        Generate predictions from all three models plus ensemble.

        Args:
            features: Feature dict (from FeatureEngineer).

        Returns:
            Dict with:
            - xgb_total (float)
            - lgb_total (float)
            - ridge_total (float)
            - ml_ensemble_total (float)
            - model_agreement_score (float 0–1)
        """
        if not self._trainer._trained:
            raise RuntimeError("Models are not trained. Call train() or load_models() first.")

        # Build feature vector
        feature_names = self._trainer._feature_names or ML_FEATURE_COLUMNS
        X_row = []
        for fname in feature_names:
            v = features.get(fname, np.nan)
            if v is None or (isinstance(v, float) and (math.isnan(v) or v == MISSING_SENTINEL)):
                v = np.nan
            try:
                X_row.append(float(v))
            except (TypeError, ValueError):
                X_row.append(np.nan)

        X_arr = np.array(X_row).reshape(1, -1)

        # Impute for tree models
        X_imputed = self._trainer._imputer.transform(X_arr)

        xgb_pred = float(self._trainer._xgb_model.predict(X_imputed)[0])
        lgb_pred = float(self._trainer._lgb_model.predict(X_imputed)[0])
        ridge_pred = float(self._trainer._ridge_pipeline.predict(X_arr)[0])

        # Weighted ensemble
        w_xgb = self._trainer.xgb_weight
        w_lgb = self._trainer.lgb_weight
        w_ridge = self._trainer.ridge_weight
        total_w = w_xgb + w_lgb + w_ridge

        ensemble = (
            w_xgb * xgb_pred
            + w_lgb * lgb_pred
            + w_ridge * ridge_pred
        ) / total_w

        # Model agreement: 1 - (std / mean) normalized
        preds = [xgb_pred, lgb_pred, ridge_pred]
        pred_mean = np.mean(preds)
        pred_std = np.std(preds)
        if pred_mean > 0:
            cv = pred_std / pred_mean     # Coefficient of variation
            agreement = max(0.0, 1.0 - cv * 10)  # Scale: CV of 0.1 → agreement 0
        else:
            agreement = 0.5

        result = {
            "xgb_total": round(xgb_pred, 2),
            "lgb_total": round(lgb_pred, 2),
            "ridge_total": round(ridge_pred, 2),
            "ml_ensemble_total": round(ensemble, 2),
            "model_agreement_score": round(float(np.clip(agreement, 0, 1)), 3),
        }

        logger.debug(
            f"ML predict: xgb={xgb_pred:.1f}, lgb={lgb_pred:.1f}, "
            f"ridge={ridge_pred:.1f}, ensemble={ensemble:.1f}, "
            f"agreement={agreement:.2f}"
        )
        return result

    def is_trained(self) -> bool:
        return self._trainer._trained

    def get_feature_importance(self) -> pd.DataFrame:
        return self._trainer.get_feature_importance()
