"""
Secondary output generation.
Builds route benchmarks, efficiency rankings, cycle segmentation, and daily consistency reports.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import RANDOM_STATE, N_ROUTE_CLUSTERS


def compute_route_clusters(combined_geo: pd.DataFrame, n_clusters: int = N_ROUTE_CLUSTERS) -> pd.DataFrame:
    """
    Cluster shifts into route types based on their spatial footprint.
    Uses (centroid, start, end) coordinates to capture location and direction.
    Returns route_id for each shift.
    """
    coord_cols = ["shift_centroid_x", "shift_centroid_y",
                  "shift_start_x", "shift_start_y",
                  "shift_end_x", "shift_end_y"]

    valid = combined_geo.dropna(subset=coord_cols).copy()
    if len(valid) < n_clusters:
        print(f"    WARNING: only {len(valid)} shifts with GPS, reducing clusters to {len(valid)//2}")
        n_clusters = max(2, len(valid) // 2)

    print(f"    Clustering {len(valid):,} shifts into {n_clusters} route clusters …")

    # Normalize each coordinate to [0,1]
    X_clust = valid[coord_cols].values.copy().astype(float)
    col_range = X_clust.max(axis=0) - X_clust.min(axis=0)
    col_range[col_range == 0] = 1.0
    X_clust = (X_clust - X_clust.min(axis=0)) / col_range

    # Cluster
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    valid["route_id"] = km.fit_predict(X_clust)

    # Assign route_id=-1 to shifts without GPS data
    out = combined_geo[["vehicle", "adj_date", "shift_key"]].copy()
    out = out.merge(valid[["vehicle", "adj_date", "shift_key", "route_id"]],
                    on=["vehicle", "adj_date", "shift_key"], how="left")
    out["route_id"] = out["route_id"].fillna(-1).astype(int)
    print(f"    Route cluster distribution:\n{out['route_id'].value_counts().sort_index().to_string()}")
    return out


def build_route_benchmark(
    train_data: pd.DataFrame,
    bundle: dict,
    ALL_FEATURES: list,
    global_feat_meds: pd.Series,
) -> tuple:
    """
    Build a vehicle-agnostic fuel benchmark for each route.
    Helps evaluate how efficiently each vehicle performs on a given route.

    Process:
      1. Zero out vehicle identity and history features
      2. Predict fuel using the ensemble (as if "unknown vehicle")
      3. Average predictions by route_id
      4. Compare vs actual mean fuel
    """
    from prediction import predict_ensemble

    print("\n  [SO1] Computing route-level fuel benchmarks …")

    train_rt = train_data.copy()
    train_rt["route_id"] = train_rt["route_id"].fillna(-1).astype(int)

    # Create anonymous feature matrix (remove vehicle-specific info)
    X_anon = train_data[ALL_FEATURES].fillna(global_feat_meds).values.astype(np.float64)
    feat_idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    # Zero out vehicle identity
    if "vehicle_enc" in feat_idx:
        X_anon[:, feat_idx["vehicle_enc"]] = -1

    # Zero out vehicle history
    for hf in ["veh_mean_fuel", "veh_lag1_fuel", "veh_lag2_fuel",
               "veh_roll3_fuel", "veh_shift_mean_fuel"]:
        if hf in feat_idx:
            X_anon[:, feat_idx[hf]] = global_feat_meds.get(hf, 0.0)
    if "veh_std_fuel" in feat_idx:
        X_anon[:, feat_idx["veh_std_fuel"]] = global_feat_meds.get("veh_std_fuel", 0.0)

    # Predict using anonymous features
    route_pred_fuel = predict_ensemble(bundle, X_anon)
    train_rt["route_pred_fuel"] = route_pred_fuel

    # Aggregate to route level
    route_bench = (
        train_rt.groupby("route_id")
        .agg(
            n_shifts=("vehicle", "count"),
            n_unique_vehicles=("vehicle", "nunique"),
            mean_dist_km=("total_dist_km", "mean"),
            mean_uphill_m=("uphill_m", "mean"),
            mean_downhill_m=("downhill_m", "mean"),
            mean_laden_dist_km=("laden_dist_km", "mean"),
            mean_terrain_intensity=("terrain_intensity", "mean"),
            mean_pct_on_road=("pct_on_road", "mean"),
            route_benchmark_fuel_L=("route_pred_fuel", "mean"),
            actual_mean_fuel_L=("actual_fuel", "mean"),
        )
        .reset_index()
    )

    # Calculate accuracy
    route_bench["benchmark_accuracy_pct"] = (
        100 * (1 - abs(route_bench["route_benchmark_fuel_L"] - route_bench["actual_mean_fuel_L"])
               / (route_bench["actual_mean_fuel_L"] + 1e-6))
    ).clip(0, 100)

    print(f"    Route benchmark built for {len(route_bench)} routes.")
    print(route_bench[["route_id", "n_shifts", "mean_dist_km",
                        "route_benchmark_fuel_L", "actual_mean_fuel_L",
                        "benchmark_accuracy_pct"]].to_string(index=False))

    return route_bench, train_rt


def compute_dumper_efficiency(
    predictions_df: pd.DataFrame,
    route_bench: pd.DataFrame,
    label: str = "train",
) -> pd.DataFrame:
    """
    Calculate efficiency of each vehicle by comparing actual/predicted fuel
    against the route benchmark.

    efficiency_ratio = predicted_fuel / route_benchmark_fuel
    Ratio < 1 = better than benchmark, > 1 = worse
    """
    print(f"\n  [SO2] Computing dumper efficiency component ({label}) …")

    eff = predictions_df.copy()
    eff = eff.merge(
        route_bench[["route_id", "route_benchmark_fuel_L"]],
        on="route_id", how="left"
    )

    # If no benchmark available, use global mean
    eff["route_benchmark_fuel_L"] = eff["route_benchmark_fuel_L"].fillna(
        eff["predicted_fuel"].mean()
    )

    # Calculate efficiency
    eff["efficiency_ratio"] = (
        eff["predicted_fuel"] / (eff["route_benchmark_fuel_L"] + 1e-6)
    ).clip(0, 5)  # Cap at 5x

    # Per-vehicle mean efficiency
    veh_eff = (
        eff.groupby("vehicle")["efficiency_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"efficiency_ratio": "veh_mean_efficiency"})
    )

    # Rank vehicles
    veh_eff["veh_efficiency_rank"] = veh_eff["veh_mean_efficiency"].rank(method="min").astype(int)
    eff = eff.merge(veh_eff, on="vehicle", how="left")

    # Actual efficiency if training
    if label == "train" and "actual_fuel" in eff.columns:
        eff["actual_efficiency_ratio"] = (
            eff["actual_fuel"] / (eff["route_benchmark_fuel_L"] + 1e-6)
        ).clip(0, 5)

    # Output columns
    out_cols = ["vehicle", "adj_date", "shift_key", "route_id",
                "predicted_fuel", "route_benchmark_fuel_L", "efficiency_ratio",
                "veh_mean_efficiency", "veh_efficiency_rank"]
    if "actual_fuel" in eff.columns:
        out_cols += ["actual_fuel", "actual_efficiency_ratio"]

    print(f"    Top 10 most efficient vehicles:")
    top10 = veh_eff.sort_values("veh_mean_efficiency").head(10)
    print(top10.to_string(index=False))

    return eff[out_cols]


def build_cycle_segmentation_output(
    train_feats: pd.DataFrame,
    test_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Output shift-level details about haul cycle identification.
    Useful for understanding segmentation quality.
    """
    print("\n  [SO3] Building cycle segmentation output …")

    cols = ["vehicle", "adj_date", "shift_key", "cycle_method",
            "trip_count", "mean_cycle_dist_km", "mean_cycle_dur_h",
            "laden_dist_km", "unladen_dist_km", "laden_climb_m",
            "n_dump_events", "n_stops"]

    def safe_select(df, label):
        """Select available columns and add split label."""
        present = [c for c in cols if c in df.columns]
        out = df[present].copy()
        out["split"] = label
        return out

    # Combine train and test
    combined = pd.concat([
        safe_select(train_feats, "train"),
        safe_select(test_feats, "test"),
    ], ignore_index=True)

    print(f"    Cycle segmentation rows: {len(combined):,}")
    if "cycle_method" in combined.columns:
        method_counts = combined["cycle_method"].value_counts()
        print(f"    Method breakdown: {method_counts.to_dict()}")

    return combined


def build_daily_consistency(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_preds: np.ndarray,
    test_preds: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate shift-level predictions to daily level.
    Allows checking consistency across a day and comparing daily totals.
    """
    print("\n  [SO4] Building daily fuel consistency output …")

    # Train aggregation
    train_daily = train_data[["vehicle", "adj_date", "shift_key", "actual_fuel"]].copy()
    train_daily["predicted_fuel"] = train_preds

    train_pivot = (
        train_daily
        .pivot_table(index=["vehicle", "adj_date"], columns="shift_key",
                     values="predicted_fuel", aggfunc="sum")
        .reset_index()
    )
    train_pivot.columns.name = None

    # Ensure all shifts present
    for sh in ["A", "B", "C"]:
        if sh not in train_pivot.columns:
            train_pivot[sh] = 0.0

    train_pivot = train_pivot.rename(columns={
        "A": "predicted_shift_A", "B": "predicted_shift_B", "C": "predicted_shift_C"
    })

    train_pivot["predicted_daily_total"] = (
        train_pivot["predicted_shift_A"].fillna(0) +
        train_pivot["predicted_shift_B"].fillna(0) +
        train_pivot["predicted_shift_C"].fillna(0)
    )

    # Actual daily totals
    actual_daily = (
        train_daily.groupby(["vehicle", "adj_date"])["actual_fuel"].sum()
        .reset_index().rename(columns={"actual_fuel": "actual_daily_total"})
    )
    train_out = train_pivot.merge(actual_daily, on=["vehicle", "adj_date"], how="left")

    train_out["daily_residual"] = train_out["predicted_daily_total"] - train_out["actual_daily_total"]
    train_out["abs_pct_error"] = (
        100 * abs(train_out["daily_residual"]) / (train_out["actual_daily_total"] + 1e-6)
    ).clip(0, 500)
    train_out["split"] = "train"

    mae = train_out["daily_residual"].abs().mean()
    mape = train_out["abs_pct_error"].mean()
    print(f"    Train daily aggregation — MAE: {mae:.1f} L  |  MAPE: {mape:.1f}%")

    # Test (no actuals)
    test_daily = test_data[["vehicle", "adj_date", "shift_key"]].copy()
    test_daily["predicted_fuel"] = test_preds

    test_pivot = (
        test_daily
        .pivot_table(index=["vehicle", "adj_date"], columns="shift_key",
                     values="predicted_fuel", aggfunc="sum")
        .reset_index()
    )
    test_pivot.columns.name = None

    for sh in ["A", "B", "C"]:
        if sh not in test_pivot.columns:
            test_pivot[sh] = 0.0

    test_pivot = test_pivot.rename(columns={
        "A": "predicted_shift_A", "B": "predicted_shift_B", "C": "predicted_shift_C"
    })
    test_pivot["predicted_daily_total"] = (
        test_pivot["predicted_shift_A"].fillna(0) +
        test_pivot["predicted_shift_B"].fillna(0) +
        test_pivot["predicted_shift_C"].fillna(0)
    )

    test_pivot["actual_daily_total"] = np.nan
    test_pivot["daily_residual"] = np.nan
    test_pivot["abs_pct_error"] = np.nan
    test_pivot["split"] = "test"

    # Combine
    combined = pd.concat([train_out, test_pivot], ignore_index=True)
    print(f"    Daily consistency rows: {len(combined):,} ({len(train_out)} train + {len(test_pivot)} test)")

    return combined
