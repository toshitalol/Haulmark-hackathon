"""
Shared utility functions for data preparation.
Handles encoding, attaching metadata, filling values, etc.
"""

import numpy as np
import pandas as pd


def encode_df(df, vehicle_map, shift_map):
    """
    Encode categorical features (vehicle, shift) to integers.
    Extract date components (day of week, month, etc).
    """
    df = df.copy()
    df["vehicle_enc"] = df["vehicle"].map(vehicle_map).fillna(-1).astype(int)
    df["shift_enc"] = df["shift_key"].map(shift_map).fillna(-1).astype(int)
    df["date_dt"] = pd.to_datetime(df["adj_date"], errors="coerce")
    df["dayofweek"] = df["date_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df["date_dt"].dt.month
    df["dayofmonth"] = df["date_dt"].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)  # Sat/Sun
    return df


def add_elapsed_days(df, start_date_str):
    """Add feature: days elapsed since training start."""
    start = pd.to_datetime(start_date_str)
    df["elapsed_days"] = (
        pd.to_datetime(df["adj_date"], errors="coerce") - start
    ).dt.days.fillna(0).clip(lower=0)
    return df


def attach_fleet(df, fleet_df):
    """Join vehicle metadata from fleet database."""
    return df.merge(
        fleet_df[["vehicle", "mine_enc", "dump_switch_enc", "tankcap"]],
        on="vehicle", how="left")


def attach_rfid(df, rfid):
    """Join RFID refueling data (aggregated at shift level)."""
    if rfid.empty:
        df["refuel_litres"] = 0.0
        df["refuel_count"] = 0.0
        return df

    df = df.merge(rfid, on=["vehicle", "adj_date", "shift_key"], how="left")
    df["refuel_litres"] = df["refuel_litres"].fillna(0)
    df["refuel_count"] = df["refuel_count"].fillna(0)
    return df


def fill_metadata(df):
    """Fill missing metadata with sensible defaults."""
    if "mine_enc" in df.columns:
        df["mine_enc"] = df["mine_enc"].fillna(-1).astype(int)
    else:
        df["mine_enc"] = pd.Series(-1, index=df.index).astype(int)

    if "dump_switch_enc" in df.columns:
        df["dump_switch_enc"] = df["dump_switch_enc"].fillna(0).astype(int)
    else:
        df["dump_switch_enc"] = pd.Series(0, index=df.index).astype(int)

    if "tankcap" in df.columns:
        df["tankcap"] = df["tankcap"].fillna(1311.0)
    else:
        df["tankcap"] = pd.Series(1311.0, index=df.index)

    return df


def audit_runhrs_leakage(train_df, runhrs_leak_corr_cap=0.95):
    """
    Check if runhrs feature is too correlated with target (data leakage).
    Returns True if feature should be kept, False if dropped.
    """
    if "runhrs" not in train_df.columns:
        return False

    mask = train_df["runhrs"].notna() & train_df["actual_fuel"].notna()
    if mask.sum() < 50:
        return False

    # Calculate correlation
    corr = abs(np.corrcoef(
        train_df.loc[mask, "runhrs"].values,
        train_df.loc[mask, "actual_fuel"].values,
    )[0, 1])

    print(f"  [B1] runhrs ↔ actual_fuel correlation = {corr:.4f}")

    if corr > runhrs_leak_corr_cap:
        print(f"  [B1] WARNING: corr > {runhrs_leak_corr_cap} — runhrs_feat DROPPED")
        return False

    print(f"  [B1] Correlation acceptable — keeping runhrs_feat")
    return True
