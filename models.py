"""
Model builders and ensemble weight optimization.
Provides factories for various machine learning models with predefined hyperparameters.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

from config import RANDOM_STATE


def make_hgbt(**kw):
    """HistGradientBoosting regressor with sensible defaults."""
    p = dict(loss="squared_error", max_iter=2000, max_depth=5,
             min_samples_leaf=20, l2_regularization=2.0, learning_rate=0.03,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
             tol=1e-4, random_state=RANDOM_STATE)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def make_hgbt_clf(**kw):
    """HistGradientBoosting classifier (for working shift detection)."""
    p = dict(max_iter=1000, max_depth=4, min_samples_leaf=20,
             l2_regularization=1.0, learning_rate=0.03,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
             tol=1e-4, random_state=RANDOM_STATE)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def make_xgb(**kw):
    """XGBoost regressor with defaults."""
    p = dict(n_estimators=2000, learning_rate=0.02, max_depth=5,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
             reg_alpha=0.1, reg_lambda=1.5, early_stopping_rounds=100,
             random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    p.update(kw)
    return xgb.XGBRegressor(**p)


def make_lgb(**kw):
    """LightGBM regressor with defaults."""
    p = dict(n_estimators=3000, learning_rate=0.02, num_leaves=63,
             max_depth=6, subsample=0.8, colsample_bytree=0.8,
             min_child_samples=20, reg_alpha=0.1, reg_lambda=1.5,
             random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    p.update(kw)
    return lgb.LGBMRegressor(**p)


def make_cat(cat_feature_indices=None, **kw):
    """CatBoost regressor with categorical feature support."""
    p = dict(iterations=3000, learning_rate=0.02, depth=6,
             l2_leaf_reg=3, early_stopping_rounds=100,
             random_seed=RANDOM_STATE, verbose=0, thread_count=-1)
    if cat_feature_indices:
        p["cat_features"] = cat_feature_indices
    p.update(kw)
    return CatBoostRegressor(**p)


def make_ridge_cv():
    """Ridge regression with cross-validated alpha selection."""
    return StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0], cv=5)


def make_rf(**kw):
    """Random Forest regressor (optional, slower)."""
    p = dict(n_estimators=300, max_depth=12, min_samples_leaf=10,
             n_jobs=-1, random_state=RANDOM_STATE)
    p.update(kw)
    return RandomForestRegressor(**p)


def optimize_ensemble_weights(oof_preds, y_true):
    """
    Use SLSQP optimization to find ensemble weights that minimize RMSE.
    Ensures weights sum to 1 and each weight is between 0 and 1.

    Args:
        oof_preds: Dict mapping model names to out-of-fold predictions
        y_true: Ground truth target values

    Returns:
        Dict mapping model names to optimal weights
    """
    model_names = list(oof_preds.keys())
    P = np.column_stack([oof_preds[m] for m in model_names])

    def neg_rmse(w):
        return np.sqrt(mean_squared_error(y_true, P @ w))

    n = len(model_names)
    result = minimize(
        neg_rmse, np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = result.x
    best_rmse = np.sqrt(mean_squared_error(y_true, P @ weights))
    equal_rmse = np.sqrt(mean_squared_error(y_true, P.mean(axis=1)))

    print("\n  [SLSQP] Optimized ensemble weights:")
    for name, w in zip(model_names, weights):
        print(f"    {name:12s}: {w:.4f}")
    print(f"  Equal-weight RMSE: {equal_rmse:.2f} L  →  SLSQP RMSE: {best_rmse:.2f} L")

    return dict(zip(model_names, weights))
