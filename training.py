"""
Two-stage training approach: classifier + regressor ensemble.
Handles working-shift classification and fuel amount prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from config import RANDOM_STATE, N_CV_SPLITS, ZERO_FUEL_THRESH, WEIGHT_HOLDOUT_FRAC, USE_RF, CAT_FEATURE_NAMES
from models import make_hgbt_clf, make_hgbt, make_xgb, make_lgb, make_cat, make_ridge_cv, make_rf, optimize_ensemble_weights


def train_two_stage(X, y, feat_names, date_groups, cat_feat_idx):
    """
    Two-stage training approach:
     Stage 1: Train classifier to predict working vs. non-working shifts
        (working = fuel consumption > ZERO_FUEL_THRESH)
     Stage 2: Train ensemble of regressors on working shifts only
     Final prediction = P(working) * fuel_prediction

    This helps separate zero-fuel shifts (errors/missing data) from operational shifts.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Fuel consumption target (n_samples,)
        feat_names: List of feature names
        date_groups: Date group for each sample (for GroupKFold)
        cat_feat_idx: Indices of categorical features

    Returns:
        Dict with trained models, weights, and CV metrics
    """
    kf = GroupKFold(n_splits=N_CV_SPLITS)
    is_working = (y > ZERO_FUEL_THRESH).astype(int)
    print(f"  Working shifts: {is_working.sum():,} / {len(y):,} ({is_working.mean()*100:.1f}%)")

    # ─── STAGE 1: Classify working shifts ───────────────────────────────────
    print("\n  [Stage 1] Training working-shift classifier …")
    oof_prob = np.zeros(len(y))
    clf_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, is_working, date_groups)):
        m_c = make_hgbt_clf()
        m_c.fit(X[tr_idx], is_working[tr_idx])
        oof_prob[val_idx] = m_c.predict_proba(X[val_idx])[:, 1]
        clf_models.append(m_c)
        print(f"    Clf fold {fold+1} stopped at {m_c.n_iter_} iterations")

    clf_acc = ((oof_prob > 0.5).astype(int) == is_working).mean()
    print(f"  Classifier OOF accuracy: {clf_acc*100:.1f}%")

    # ─── STAGE 2: Regress fuel on working shifts only ───────────────────────
    print("\n  [Stage 2] Training fuel regressor on working shifts only …")
    work_idx = np.where(is_working == 1)[0]
    X_work = X[work_idx]
    y_work = y[work_idx]
    g_work = date_groups[work_idx]

    # Out-of-fold predictions from each model
    oof_reg_preds = {
        "hgbt": np.zeros(len(y_work)),
        "ridge": np.zeros(len(y_work)),
        "xgb": np.zeros(len(y_work)),
        "lgb": np.zeros(len(y_work)),
        "cat": np.zeros(len(y_work))
    }
    if USE_RF:
        oof_reg_preds["rf"] = np.zeros(len(y_work))

    reg_models = {k: [] for k in oof_reg_preds}
    ridge_scalers = []

    # Train all models in cross-validation loop
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_work, y_work, g_work)):
        X_tr, X_val = X_work[tr_idx], X_work[val_idx]
        y_tr, y_val = y_work[tr_idx], y_work[val_idx]

        # HistGradientBoosting
        m_h = make_hgbt()
        m_h.fit(X_tr, y_tr)
        oof_reg_preds["hgbt"][val_idx] = m_h.predict(X_val)
        reg_models["hgbt"].append(m_h)
        print(f"    Fold {fold+1} HGBT stopped at {m_h.n_iter_} iters")

        # Ridge Regression (with scaling)
        scaler, m_r = make_ridge_cv()
        m_r.fit(scaler.fit_transform(X_tr), y_tr)
        oof_reg_preds["ridge"][val_idx] = m_r.predict(scaler.transform(X_val))
        reg_models["ridge"].append(m_r)
        ridge_scalers.append(scaler)

        # XGBoost
        m_x = make_xgb()
        m_x.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_reg_preds["xgb"][val_idx] = m_x.predict(X_val)
        reg_models["xgb"].append(m_x)
        print(f"    Fold {fold+1} XGB stopped at {m_x.best_iteration} iters")

        # LightGBM
        m_l = make_lgb()
        m_l.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof_reg_preds["lgb"][val_idx] = m_l.predict(X_val)
        reg_models["lgb"].append(m_l)
        print(f"    Fold {fold+1} LGB stopped at {m_l.best_iteration_} iters")

        # CatBoost (requires DataFrame with categorical types)
        X_tr_cat = pd.DataFrame(X_tr, columns=feat_names)
        X_val_cat = pd.DataFrame(X_val, columns=feat_names)
        for idx in cat_feat_idx:
            col_name = feat_names[idx]
            X_tr_cat[col_name] = X_tr_cat[col_name].astype(int).astype(str)
            X_val_cat[col_name] = X_val_cat[col_name].astype(int).astype(str)
        m_c2 = make_cat(cat_feature_indices=cat_feat_idx)
        m_c2.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=100)
        oof_reg_preds["cat"][val_idx] = m_c2.predict(X_val_cat)
        reg_models["cat"].append(m_c2)
        print(f"    Fold {fold+1} CAT stopped at {m_c2.best_iteration_} iters")

        # Optional: Random Forest
        if USE_RF:
            m_rf = make_rf()
            m_rf.fit(X_tr, y_tr)
            oof_reg_preds["rf"][val_idx] = m_rf.predict(X_val)
            reg_models["rf"].append(m_rf)

        # Report fold performance
        fold_blend = np.mean([oof_reg_preds[k][val_idx] for k in oof_reg_preds], axis=0)
        rmse = np.sqrt(mean_squared_error(y_val, fold_blend))
        print(f"    ── Fold {fold+1} blended RMSE = {rmse:.2f} L\n")

    # ─── Optimize ensemble weights using holdout set ───────────────────────
    work_dates = date_groups[work_idx]
    unique_dates = np.unique(work_dates)
    n_holdout = max(1, int(len(unique_dates) * WEIGHT_HOLDOUT_FRAC))
    holdout_dates = set(unique_dates[-n_holdout:])  # Most recent dates
    holdout_mask = np.array([d in holdout_dates for d in work_dates])
    fit_mask = ~holdout_mask

    print(f"\n  [L3] Fitting SLSQP on {fit_mask.sum():,} rows, validating on {holdout_mask.sum():,} holdout rows …")
    weights = optimize_ensemble_weights(
        {k: v[fit_mask] for k, v in oof_reg_preds.items()}, y_work[fit_mask])

    # Evaluate on holdout (truly unseen dates)
    P_holdout = np.column_stack([oof_reg_preds[k][holdout_mask] for k in oof_reg_preds])
    holdout_rmse = np.sqrt(mean_squared_error(
        y_work[holdout_mask], P_holdout @ np.array([weights[k] for k in oof_reg_preds])))
    print(f"  [L3] Holdout RMSE (unseen): {holdout_rmse:.2f} L")

    # Overall OOF performance on working shifts
    P_all = np.column_stack([oof_reg_preds[k] for k in oof_reg_preds])
    w_arr = np.array([weights[k] for k in oof_reg_preds])
    reg_oof_rmse = np.sqrt(mean_squared_error(y_work, P_all @ w_arr))
    print(f"  Overall regressor OOF RMSE (working shifts) = {reg_oof_rmse:.2f} L")

    # Overall two-stage OOF (including zero-fuel shifts)
    full_oof = np.zeros(len(y))
    full_oof[work_idx] = oof_prob[work_idx] * (P_all @ w_arr)
    two_stage_rmse = np.sqrt(mean_squared_error(y, full_oof))
    print(f"  Two-stage OOF RMSE (all shifts incl. zeros) = {two_stage_rmse:.2f} L")

    # Show top feature importances from XGBoost fold 1
    if reg_models["xgb"]:
        fi = pd.Series(reg_models["xgb"][0].feature_importances_, index=feat_names)
        print("\n  Top 15 feature importances (XGBoost fold 1):")
        print(fi.sort_values(ascending=False).head(15).to_string())

    return {
        "clf_models": clf_models,
        "reg_models": reg_models,
        "ridge_scalers": ridge_scalers,
        "weights": weights,
        "oof_rmse": two_stage_rmse,
        "reg_oof_rmse": reg_oof_rmse,
        "holdout_rmse": holdout_rmse,
        "feat_names": feat_names,
    }
