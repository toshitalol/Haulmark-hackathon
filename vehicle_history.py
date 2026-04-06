"""
Vehicle history feature engineering.
Creates time-aware historical features for each vehicle to avoid data leakage.
"""

import numpy as np
import pandas as pd


def add_time_aware_vehicle_history(train_df, global_mean, global_std):
    """
    Create time-aware historical features for each vehicle.
    Uses expanding windows to avoid data leakage:
     - veh_mean_fuel: mean fuel from all previous shifts
     - veh_lag1/lag2: fuel from 1st/2nd most recent shift
     - veh_roll3_fuel: rolling average of last 3 shifts
     - veh_shift_mean_fuel: mean fuel for this vehicle in this shift (A/B/C)
    """
    df = train_df.copy()
    df = df.sort_values(["vehicle", "adj_date", "shift_key"]).reset_index(drop=True)

    # Expanding mean/std: use only data before current row
    df["veh_mean_fuel"] = (
        df.groupby("vehicle")["actual_fuel"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(global_mean)
    )
    df["veh_std_fuel"] = (
        df.groupby("vehicle")["actual_fuel"]
        .transform(lambda x: x.shift(1).expanding().std()).fillna(global_std)
    )

    # Lag features
    df["veh_lag1_fuel"] = df.groupby("vehicle")["actual_fuel"].shift(1).fillna(global_mean)
    df["veh_lag2_fuel"] = df.groupby("vehicle")["actual_fuel"].shift(2).fillna(global_mean)

    # Rolling average of previous 3 shifts
    df["veh_roll3_fuel"] = (
        df.groupby("vehicle")["actual_fuel"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(global_mean)
    )

    # Mean fuel for this vehicle in this shift type
    df["veh_shift_mean_fuel"] = (
        df.groupby(["vehicle", "shift_key"])["actual_fuel"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(df["veh_mean_fuel"])
    )

    return df


def apply_vehicle_history_test(test_df, train_df, global_mean, global_std):
    """
    Apply vehicle history features to test set using training data statistics.
    For unseen vehicles, falls back to global mean/std.
    """
    # Extract final statistics from training data for each vehicle
    veh_stats = (
        train_df.groupby("vehicle")["actual_fuel"]
        .agg(veh_mean_fuel="mean", veh_std_fuel="std", veh_lag1_fuel="last")
        .reset_index()
    )

    # Get penultimate value for lag2
    penult = (
        train_df.groupby("vehicle")["actual_fuel"]
        .apply(lambda x: x.iloc[-2] if len(x) >= 2 else np.nan)
        .reset_index().rename(columns={"actual_fuel": "veh_lag2_fuel"})
    )

    # Get rolling 3-shift average from training tail
    roll3 = (
        train_df.groupby("vehicle")["actual_fuel"]
        .apply(lambda x: x.tail(3).mean())
        .reset_index().rename(columns={"actual_fuel": "veh_roll3_fuel"})
    )

    # Mean per shift type
    veh_shift_mean = (
        train_df.groupby(["vehicle", "shift_key"])["actual_fuel"]
        .mean().reset_index().rename(columns={"actual_fuel": "veh_shift_mean_fuel"})
    )

    # Join all onto test set
    out = test_df.copy()
    for tbl, key in [(veh_stats, "vehicle"), (penult, "vehicle"),
                     (roll3, "vehicle"), (veh_shift_mean, ["vehicle", "shift_key"])]:
        out = out.merge(tbl, on=key, how="left")

    # Fill missing (unseen vehicles) with global statistics
    defaults = {
        "veh_mean_fuel": global_mean, "veh_lag1_fuel": global_mean,
        "veh_lag2_fuel": global_mean, "veh_roll3_fuel": global_mean,
        "veh_shift_mean_fuel": global_mean, "veh_std_fuel": global_std,
    }
    for col, val in defaults.items():
        out[col] = out[col].fillna(val)

    return out
