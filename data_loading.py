"""
Data loading and basic utilities.
Handles loading parquet files, CSVs, and RFID refueling data.
"""

from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

from config import RAW, RFID_FILE


def norm_veh(s):
    """Normalize vehicle names: lowercase, remove special characters."""
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def load_parquets(file_list, label):
    """
    Load and concatenate multiple parquet files.
    Skips missing files with warning.
    Normalizes column names to lowercase.
    """
    seen: Set[str] = set()
    dfs = []

    for name in file_list:
        if name in seen:
            continue

        seen.add(name)
        p = RAW / name

        if p.exists():
            print(f"    Loading {name} …")
            df = pd.read_parquet(p)
            df.columns = [str(c).lower().strip() for c in df.columns]
            dfs.append(df)
        else:
            print(f"    WARNING: {name} not found")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def safe_col(df, col, default=0):
    """
    Safely get a column if it exists, otherwise return Series of defaults.
    Prevents KeyError when column is missing.
    """
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def compute_shift_and_date(ts_series):
    """
    Determine shift (A/B/C) and adjusted date from timestamps.

    Shifts:
     - A = 6-14h
     - B = 14-22h
     - C = 22-6h (midnight shift)

    Adjusted date = calendar date for A/B, next day for C.
    """
    # Convert to Asia/Kolkata timezone
    if ts_series.dt.tz is None:
        ts = ts_series.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts_series.dt.tz_convert("Asia/Kolkata")

    hour = ts.dt.hour
    date = ts.dt.date

    # Assign shift based on hour
    shift = np.select(
        [hour >= 22, hour < 6, (hour >= 6) & (hour < 14)],
        ["C", "C", "A"],
        default="B",
    )

    # For night shift (C), advance date to next calendar day
    next_date = (ts + pd.Timedelta(days=1)).dt.date
    dpr_date = np.where(hour >= 22, next_date.astype(str), date.astype(str))

    return (
        pd.Series(dpr_date, index=ts_series.index, name="adj_date"),
        pd.Series(shift, index=ts_series.index, name="shift_key"),
    )


def load_rfid_refuels():
    """
    Load RFID refueling transactions and aggregate to shift level.
    Returns one row per vehicle-date-shift with total liters and count.
    """
    p = RAW / RFID_FILE
    if not p.exists():
        return pd.DataFrame()

    print(f"  Loading {RFID_FILE} …")
    rf = pd.read_parquet(p)
    rf.columns = [str(c).lower().strip() for c in rf.columns]

    # Filter to dumper vehicles only
    rf = rf[rf["vehicle"].str.startswith("Dump", na=False)].copy()
    rf["vehicle"] = norm_veh(rf["vehicle"])
    rf["litres"] = pd.to_numeric(rf["litres"], errors="coerce").fillna(0)

    # Aggregate by vehicle, date, shift
    agg = (
        rf.groupby(["vehicle", "date_dpr", "shift_dpr"])
        .agg(refuel_litres=("litres", "sum"), refuel_count=("litres", "count"))
        .reset_index()
        .rename(columns={"date_dpr": "adj_date", "shift_dpr": "shift_key"})
        .drop_duplicates(subset=["vehicle", "adj_date", "shift_key"])
    )
    print(f"    Refuel rows aggregated: {len(agg):,}")
    return agg
