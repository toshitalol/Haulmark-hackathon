"""
Ensemble prediction functionality.
Generates predictions using trained two-stage ensemble model.
"""

import numpy as np
import pandas as pd

from config import USE_RF, CAT_FEATURE_NAMES


def predict_ensemble(bundle, X, feat_names_override=None):
    """
    Generate predictions using the two-stage ensemble.
    Averages predictions across CV folds for each model type,
    blends using optimized weights.

    Args:
        bundle: Dict from train_two_stage() with trained models
        X: Feature matrix to predict on (n_samples, n_features)
        feat_names_override: Optional feature names override

    Returns:
        Array of predictions
    """
    clf_models = bundle["clf_models"]
    reg_models = bundle["reg_models"]
    weights = bundle["weights"]
    scalers = bundle["ridge_scalers"]
    feat_names = feat_names_override or bundle["feat_names"]

    # Stage 1: Probability of working shift
    prob_working = np.mean([m.predict_proba(X)[:, 1] for m in clf_models], axis=0)

    # Stage 2: Fuel predictions from each model
    reg_preds = {}
    reg_preds["hgbt"] = np.mean([m.predict(X) for m in reg_models["hgbt"]], axis=0)

    if reg_models["ridge"]:
        reg_preds["ridge"] = np.mean(
            [m.predict(sc.transform(X)) for sc, m in zip(scalers, reg_models["ridge"])], axis=0)

    reg_preds["xgb"] = np.mean([m.predict(X) for m in reg_models["xgb"]], axis=0)
    reg_preds["lgb"] = np.mean([m.predict(X) for m in reg_models["lgb"]], axis=0)

    if USE_RF and reg_models.get("rf"):
        reg_preds["rf"] = np.mean([m.predict(X) for m in reg_models["rf"]], axis=0)

    # CatBoost needs DataFrame with categorical columns
    X_cat = pd.DataFrame(X, columns=feat_names)
    for f in CAT_FEATURE_NAMES:
        if f in X_cat.columns:
            X_cat[f] = X_cat[f].astype(int).astype(str)
    reg_preds["cat"] = np.mean([m.predict(X_cat) for m in reg_models["cat"]], axis=0)

    # Blend regressor predictions using optimized weights
    reg_blend = sum(weights.get(k, 0.0) * v for k, v in reg_preds.items())

    # Combine: only predict fuel if shift is working
    return np.clip(prob_working * reg_blend, 0, None)
