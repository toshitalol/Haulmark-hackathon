"""
Telemetry feature engineering.
Extracts shift-level features from raw GPS and sensor data.
Handles time gaps, speeds, accelerations, cycle identification, etc.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer

from config import (
    SPEED_MOVING, MAX_GAP_TIME, MAX_GAP_ACCEL, DUMP_V_THRESH,
    MINE_DIST_THRESH, DEFAULT_DIST_THRESH,
)
from data_loading import norm_veh, safe_col, compute_shift_and_date

transformer = Transformer.from_crs("epsg:4326", "epsg:32645", always_xy=True)


def _process_single_geo_shift(name, group, mine_geoms, compute_geo_features_for_group):
    """
    Process geofencing for a single vehicle shift (parallelizable).
    Returns one row of geo features.
    """
    from geometry import compute_geo_features_for_group, ON_ROAD_THRESH_M

    veh, date, shift = name
    mine_enc = int(group["mine_enc"].iloc[0]) if "mine_enc" in group.columns else -1
    geoms = mine_geoms.get(mine_enc)

    nan_row = {
        "vehicle": veh, "adj_date": date, "shift_key": shift,
        "n_geo_dump_entries": np.nan, "time_in_dump_s": np.nan,
        "mean_road_offset_m": np.nan, "pct_on_road": np.nan,
        "in_ml_boundary_pct": np.nan, "mean_bench_dist_m": np.nan,
        "shift_centroid_x": np.nan, "shift_centroid_y": np.nan,
        "shift_start_x": np.nan, "shift_start_y": np.nan,
        "shift_end_x": np.nan, "shift_end_y": np.nan,
    }

    if geoms is None:
        return nan_row

    return {
        "vehicle": veh, "adj_date": date, "shift_key": shift,
        **compute_geo_features_for_group(
            group["x_utm"].values, group["y_utm"].values,
            group["dt_time"].values, geoms,
        )
    }


def compute_geo_shift_features(df, mine_geoms):
    """
    Compute geofencing features for all shifts in dataset.
    Parallelized across CPU cores.
    """
    from geometry import compute_geo_features_for_group

    if "x_utm" not in df.columns:
        return pd.DataFrame()

    print("    Computing geofencing features across all CPU cores …")

    # Process each shift in parallel
    records = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_process_single_geo_shift)(name, group, mine_geoms, compute_geo_features_for_group)
        for name, group in df.groupby(["vehicle", "adj_date", "shift_key"], sort=False)
    )

    geo_df = pd.DataFrame(records)
    print(f"    Geo feature rows: {len(geo_df):,}")
    return geo_df


def get_advanced_features(df, label="", fleet_df=None):
    """
    Extract comprehensive shift-level features from raw telemetry data.
    Handles time gaps, calculates speeds, accelerations, cycle identification,
    laden/unladen segments, and aggregates to shift level.

    Returns:
        agg_df: One row per shift with all engineered features
        df_slim: Minimal telemetry data needed for geofencing
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), None

    print(f"  Processing {label} telemetry ({len(df):,} rows) …")
    df = df.copy()
    df["vehicle"] = norm_veh(df["vehicle"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)

    # Time deltas (seconds between successive points)
    raw_dt = df.groupby("vehicle")["ts"].diff().dt.total_seconds().fillna(30)
    df["dt_time"] = raw_dt.clip(0, MAX_GAP_TIME)  # Cap large gaps
    df["dt_accel"] = raw_dt.clip(1, MAX_GAP_ACCEL)  # Separate cap

    # Calculate shift and date from timestamps
    calc_date, calc_shift = compute_shift_and_date(df["ts"])
    df["adj_date_calc"] = calc_date
    df["shift_calc"] = calc_shift

    # Use provided date/shift if available
    if "date_dpr" in df.columns and "shift_dpr" in df.columns:
        df["adj_date"] = df["date_dpr"].astype(str).replace("nan", np.nan).fillna(df["adj_date_calc"])
        df["shift_key"] = df["shift_dpr"].astype(str).replace("nan", np.nan).fillna(df["shift_calc"])
    else:
        df["adj_date"] = df["adj_date_calc"]
        df["shift_key"] = df["shift_calc"]

    # Remove rows with invalid date/shift
    df = df[df["adj_date"].notna() & df["shift_key"].notna()].copy()

    # Convert GPS to UTM coordinates
    if "longitude" in df.columns and "latitude" in df.columns:
        df["x_utm"], df["y_utm"] = transformer.transform(
            df["longitude"].values, df["latitude"].values)

    # Altitude processing
    df["smooth_alt"] = df.groupby("vehicle")["altitude"].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).mean())
    df["delta_alt"] = df.groupby("vehicle")["smooth_alt"].diff().fillna(0)

    # Speed and movement
    df["speed"] = df["speed"].fillna(0).clip(lower=0)
    df["ignition"] = safe_col(df, "ignition", 1).fillna(1)

    # Acceleration and jerk
    df["acceleration"] = (
        df.groupby("vehicle")["speed"]
        .transform(lambda x: x.diff().fillna(0)) / df["dt_accel"]
    )
    df["jerk"] = (
        df.groupby("vehicle")["acceleration"]
        .transform(lambda x: x.diff().fillna(0)) / df["dt_accel"]
    )

    # Operating time
    df["idle_sec"] = ((df["speed"] < SPEED_MOVING) & (df["ignition"] == 1)).astype(float) * df["dt_time"]
    df["run_sec"] = (df["ignition"] == 1).astype(float) * df["dt_time"]
    df["dist_m"] = safe_col(df, "disthav", 0).fillna(0).clip(lower=0)

    # Dump signal detection
    has_analog = "analog_input_1" in df.columns
    df["is_dump_sig"] = (
        (df["analog_input_1"].fillna(0) > DUMP_V_THRESH).astype(int)
        if has_analog else 0
    )

    # Add mine info from fleet database
    if fleet_df is not None:
        mine_lut = fleet_df[["vehicle", "mine_enc"]].drop_duplicates("vehicle")
        df = df.merge(mine_lut, on="vehicle", how="left")
        df["mine_enc"] = df["mine_enc"].fillna(-1).astype(int)
    else:
        df["mine_enc"] = -1

    # Cycle (haul trip) identification
    method_map = {}

    def map_cycles(group):
        """Identify haul cycles using dump signal or distance heuristic."""
        is_dump = group["is_dump_sig"].values
        veh = group.name

        if has_analog and is_dump.sum() > 10:
            method_map[veh] = "signal"
            dump_events = (is_dump[1:] == 1) & (is_dump[:-1] == 0)
            return pd.Series(
                np.cumsum(np.concatenate(([0], dump_events))).astype(np.int32),
                index=group.index)

        # Distance heuristic
        method_map[veh] = "heuristic"
        n = len(group)
        cycles = np.zeros(n, dtype=np.int32)
        cycle, dist_since_dump = 0, 0.0

        dist_thresh = MINE_DIST_THRESH.get(int(group["mine_enc"].iloc[0]), DEFAULT_DIST_THRESH)
        is_stop = (group["speed"].values < SPEED_MOVING).astype(int)
        dists = group["dist_m"].values

        for i in range(1, n):
            dist_since_dump += dists[i]
            if is_stop[i] == 1 and is_stop[i - 1] == 0 and dist_since_dump > dist_thresh:
                cycle += 1
                dist_since_dump = 0.0
            cycles[i] = cycle

        return pd.Series(cycles, index=group.index)

    print(f"    Mapping haul cycles …")
    df["cycle_id"] = df.groupby("vehicle", group_keys=False).apply(map_cycles)
    df["cycle_method"] = df["vehicle"].map(method_map).fillna("heuristic")

    # Laden/unladen classification
    df["cum_dist"] = df.groupby(["vehicle", "cycle_id"])["dist_m"].cumsum()
    cycle_max = df.groupby(["vehicle", "cycle_id"])["dist_m"].transform("sum")
    df["is_laden"] = (df["cum_dist"] > cycle_max * 0.20).astype(int)

    # Segment distances and climbs
    df["laden_dist"] = df["dist_m"] * df["is_laden"]
    df["unladen_dist"] = df["dist_m"] * (1 - df["is_laden"])
    df["laden_climb"] = df["delta_alt"].clip(lower=0) * df["is_laden"]
    df["unladen_climb"] = df["delta_alt"].clip(lower=0) * (1 - df["is_laden"])
    df["moving_speed"] = df["speed"].where(df["speed"] >= SPEED_MOVING)

    # Aggregate to shift level
    grp = ["vehicle", "adj_date", "shift_key"]

    # Cycle-level statistics
    cycle_dist_per_shift = (
        df.groupby(grp + ["cycle_id"])["dist_m"].sum().reset_index()
        .groupby(grp)["dist_m"].mean().reset_index()
        .rename(columns={"dist_m": "mean_cycle_dist_km"})
    )
    cycle_dist_per_shift["mean_cycle_dist_km"] /= 1000.0

    cycle_dur_per_shift = (
        df.groupby(grp + ["cycle_id"])["ts"]
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
        .reset_index().rename(columns={"ts": "cycle_dur_h"})
        .groupby(grp)["cycle_dur_h"].mean().reset_index()
        .rename(columns={"cycle_dur_h": "mean_cycle_dur_h"})
    )

    print(f"    Aggregating shift features …")
    agg_df = df.groupby(grp, sort=False).agg(
        total_dist_km=("dist_m", lambda x: x.sum() / 1000),
        n_stops=("speed", lambda x: ((x.shift(1) >= SPEED_MOVING) & (x < SPEED_MOVING)).sum()),
        idle_h=("idle_sec", lambda x: x.sum() / 3600),
        run_h=("run_sec", lambda x: x.sum() / 3600),
        trip_count=("cycle_id", "nunique"),
        shift_duration_h=("ts", lambda x: (x.max() - x.min()).total_seconds() / 3600),
        accel_variance=("acceleration", lambda x: x[x > 0].var() if (x > 0).any() else 0.0),
        hard_braking=("acceleration", lambda x: (x < -2.0).sum()),
        n_dump_events=("is_dump_sig", lambda x: (x.diff() == 1).sum()),
        mean_alt=("smooth_alt", "mean"),
        std_alt=("smooth_alt", "std"),
        net_lift=("smooth_alt", lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) >= 2 else 0.0),
        uphill_m=("delta_alt", lambda x: x[x > 0].sum()),
        downhill_m=("delta_alt", lambda x: (-x[x < 0]).sum()),
        n_points=("ts", "count"),
        laden_dist_km=("laden_dist", lambda x: x.sum() / 1000),
        laden_climb_m=("laden_climb", "sum"),
        unladen_dist_km=("unladen_dist", lambda x: x.sum() / 1000),
        unladen_climb_m=("unladen_climb", "sum"),
        mean_speed=("moving_speed", "mean"),
        speed_std=("moving_speed", "std"),
        max_speed=("moving_speed", "max"),
        speed_p75=("moving_speed", lambda x: x.quantile(0.75) if x.notna().any() else 0.0),
        mean_jerk=("jerk", lambda x: np.abs(x).mean()),
        max_jerk=("jerk", lambda x: np.abs(x).max()),
        cycle_method=("cycle_method", lambda x: x.mode()[0] if len(x) > 0 else "heuristic"),
    ).reset_index()

    # Attach cycle-level features
    agg_df = agg_df.merge(cycle_dist_per_shift, on=grp, how="left")
    agg_df = agg_df.merge(cycle_dur_per_shift, on=grp, how="left")

    # Fill missing values
    fill_zero = [
        "laden_dist_km", "laden_climb_m", "unladen_dist_km", "unladen_climb_m",
        "mean_speed", "speed_std", "max_speed", "speed_p75",
        "accel_variance", "hard_braking", "n_dump_events",
        "uphill_m", "downhill_m", "std_alt",
        "mean_jerk", "max_jerk", "mean_cycle_dist_km", "mean_cycle_dur_h",
    ]
    for c in fill_zero:
        if c in agg_df.columns:
            agg_df[c] = agg_df[c].fillna(0)

    # Derived features
    agg_df["laden_ratio"] = agg_df["laden_dist_km"] / (agg_df["total_dist_km"] + 1e-6)
    agg_df["laden_climb_rate"] = agg_df["laden_climb_m"] / (agg_df["laden_dist_km"] * 1000 + 1e-6)
    agg_df["idle_ratio"] = agg_df["idle_h"] / (agg_df["run_h"] + 1e-6)
    agg_df["laden_climb_work"] = agg_df["laden_dist_km"] * agg_df["laden_climb_rate"]
    agg_df["alt_variability"] = agg_df["std_alt"] / (agg_df["mean_alt"].abs() + 1e-6)
    agg_df["terrain_intensity"] = (agg_df["uphill_m"] + agg_df["downhill_m"]) / (agg_df["total_dist_km"] * 1000 + 1e-6)
    agg_df["work_proxy"] = agg_df["laden_climb_m"] * agg_df["laden_dist_km"]
    agg_df["dist_per_runhrs"] = agg_df["total_dist_km"] / (agg_df["run_h"] + 1e-6)
    agg_df["idle_per_dist"] = agg_df["idle_h"] / (agg_df["total_dist_km"] + 1e-6)

    print(f"    Feature rows produced: {len(agg_df):,}")

    # Prepare slim dataframe for geofencing
    keep_cols = ["vehicle", "adj_date", "shift_key", "mine_enc", "dt_time"]
    if "x_utm" in df.columns:
        keep_cols += ["x_utm", "y_utm"]

    return agg_df, df[keep_cols].copy()
