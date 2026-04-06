import struct
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.spatial import KDTree
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
# Convert GPS coordinates from lat/lon (EPSG:4326) to UTM (EPSG:32645)
# converting distances in meters from degree coordinates
from pyproj import Transformer
transformer = Transformer.from_crs("epsg:4326", "epsg:32645", always_xy=True)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

# Point to the root data directory where all files are stored
RAW = Path(r"C:\Users\dogra\Desktop\haulmark hackathon\data")

# GPKG files contain mine geometries (dump sites, haul roads, boundaries, benches)
GPKG_FILES = {
    0: RAW / "mine_001_anonymized.gpkg",
    1: RAW / "mine_002_anonymized.gpkg",
}
# Training telemetry: GPS, speed, acceleration, altitude data for model building
TRAIN_TELEMETRY = [
    "telemetry_2026-01-01_2026-01-10.parquet",
    "telemetry_2026-01-11_2026-01-20.parquet",
    "telemetry_2026-02-01_2026-02-10.parquet",
    "telemetry_2026-02-11_2026-02-20.parquet",
    "telemetry_2026-03-01_2026-03-11.parquet",
]
# Test telemetry: same format, used for predictions on new data
TEST_TELEMETRY = [
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
]
# Summary files: aggregated shift-level fuel consumption and operational details
SUMMARY_FILES = [
    "smry_jan_train_ordered.csv",
    "smry_feb_train_ordered.csv",
    "smry_mar_train_ordered.csv",
]

# RFID refueling transactions: when, where, and how much fuel was added
RFID_FILE = "rfid_refuels_2026-01-01_2026-03-31.parquet"
#defining the hyperparameters and the thresholds for feature engineering and model training
RANDOM_STATE         = 42

# Dump signal: analog voltage threshold above which we consider the dumper active
DUMP_V_THRESH        = 2.5

# Time gap handling: telemetry can have gaps; cap them to avoid unrealistic jumps
MAX_GAP_TIME         = 300  # seconds (used in dt_time calculation)
MAX_GAP_ACCEL        = 120  # seconds (used in acceleration calculations)

# Speed threshold: what counts as "moving"
SPEED_MOVING         = 1.0  # km/h

# Distance from haul road center that still counts as "on road"
ON_ROAD_THRESH_M     = 50.0  # meters

# Distance thresholds for cycle segmentation (how far to travel = one haul cycle)
# Different for different mines due to different layout sizes
MINE_DIST_THRESH     = {0: 1200, 1: 600}  # meters
DEFAULT_DIST_THRESH  = 800  # fallback if mine unknown

# Tank capacity safety margin: don't predict above this % of tank capacity
TANK_CAP_MARGIN      = 1.05

# Cross-validation folds: split training data this many ways
N_CV_SPLITS          = 5

# Fuel consumption threshold: below this = likely not a working shift (zero fuel)
ZERO_FUEL_THRESH     = 1.0  # liters

# Runhrs leakage check: if correlation > this, drop the runhrs feature
RUNHRS_LEAK_CORR_CAP = 0.95

# Holdout set: reserve this fraction of unique dates for final validation
WEIGHT_HOLDOUT_FRAC  = 0.10

# Whether to include RandomForest in the ensemble (slower, optional)
USE_RF               = False

# Route clustering: group similar shift patterns into K route clusters
N_ROUTE_CLUSTERS     = 20

#gpkg geometry parsing where we are extracting the geometries from the GPKG files and converting them into numpy arrays
def _parse_gpkg_wkb_linestring_2d(blob: bytes) -> np.ndarray:

   # Handles linestrings, polygons, and multigeometries.
    try:
        # GPKG uses WKB format with envelope info at the start
        flags         = blob[3]
        env_indicator = (flags >> 1) & 0x07
        env_size      = [0, 32, 48, 48, 64][min(env_indicator, 4)]
        wkb           = blob[8 + env_size:]  # Skip GPKG envelope, get WKB part
        
        # Determine byte order (endianness) from first byte
        byte_order    = wkb[0]
        bo            = "<" if byte_order == 1 else ">"
        
        # Read geometry type and check if 2D or 3D
        wkb_type      = struct.unpack_from(bo + "I", wkb, 1)[0]
        wkb_type_2d   = wkb_type % 1000  # Strip Z/M info
        offset        = 5
        
        # Extract coordinates based on geometry type
        if wkb_type_2d == 2:  # Linestring
            n_pts  = struct.unpack_from(bo + "I", wkb, offset)[0]
            offset += 4
            coords = np.frombuffer(wkb[offset: offset + n_pts * 16], dtype=np.float64)
            if byte_order == 0:
                coords = coords.byteswap()  # Convert from big-endian if needed
            return coords.reshape(n_pts, 2)
            
        elif wkb_type_2d == 3:  # Polygon (we use outer ring only)
            _      = struct.unpack_from(bo + "I", wkb, offset)[0]  # num_rings
            offset += 4
            n_pts  = struct.unpack_from(bo + "I", wkb, offset)[0]  # points in first ring
            offset += 4
            coords = np.frombuffer(wkb[offset: offset + n_pts * 16], dtype=np.float64)
            if byte_order == 0:
                coords = coords.byteswap()
            return coords.reshape(n_pts, 2)
            
        elif wkb_type_2d in (1002, 1003):  # Multilinestring / Multipolygon
            n_pts = struct.unpack_from(bo + "I", wkb, offset)[0]
            offset += 4
            pts   = []
            for _ in range(n_pts):
                x, y = struct.unpack_from(bo + "dd", wkb, offset)
                pts.append((x, y))
                offset += 24
            return np.array(pts, dtype=np.float64)

    except Exception:
        pass
    
    # Return empty array if parsing fails
    return np.empty((0, 2), dtype=np.float64)


def load_gpkg_geometries(gpkg_path: Path) -> Dict:
    """
    Load all mine geometry layers from GPKG file.
    Returns dict with geometry lists for dump sites, haul roads, boundaries, benches,
    plus a KDTree for fast nearest-bench lookup.
    """
    result = {
        "ob_dump": [],      # Overburden dump sites
        "haul_road": [],    # Main haul roads
        "ml_boundary": [],  # Mine lease boundary
        "bench": [],        # Bench/pit boundaries
        "bench_kdtree": None,
        "bench_pts": None,
    }
    
    if not gpkg_path.exists():
        print(f"    WARNING: {gpkg_path.name} not found")
        return result

    print(f"    Loading geometries from {gpkg_path.name} …")
    con = sqlite3.connect(str(gpkg_path))

    # Extract each layer type from the GIS database
    for table in ["ob_dump", "haul_road", "ml_boundary", "bench"]:
        try:
            rows  = con.execute(f'SELECT geom FROM "{table}"').fetchall()
            geoms = []
            
            for (blob,) in rows:
                if blob:
                    coords = _parse_gpkg_wkb_linestring_2d(blob)
                    if len(coords) >= 2:  # Only keep valid geometries
                        geoms.append(coords)
            
            result[table] = geoms
            print(f"      {table}: {len(geoms)} geometries loaded")
        except Exception as e:
            print(f"      {table}: skipped ({e})")

    # Build KDTree for fast nearest-bench lookups
    if result["bench"]:
        # Use midpoints of bench line segments for spatial indexing
        bench_pts = np.vstack([
            (b[:-1] + b[1:]) / 2.0
            for b in result["bench"] if len(b) >= 2
        ])
        result["bench_kdtree"] = KDTree(bench_pts)
        result["bench_pts"]    = bench_pts
        print(f"      bench KDTree built from {len(bench_pts):,} segment midpoints")

    con.close()
    return result

#vectorized geometry helpers for fast point-in-polygon and distance calculations
def _vec_point_in_polygon(px, py, poly):
    """
    Vectorized point-in-polygon test using ray casting algorithm.
    Returns boolean array where True = point is inside polygon.
    Fast for many points at once.
    """
    inside = np.zeros(len(px), dtype=bool)
    n = len(poly)
    j = n - 1
    
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        
        # Ray casting: count boundary crossings
        intersect = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi
        )
        inside[intersect] = ~inside[intersect]  # Toggle for each crossing
        j = i
    
    return inside
def _vec_point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """
    Calculate squared distance from points (px, py) to line segment (ax,ay)-(bx,by).
    Uses projection formula with clamping to segment bounds.
    """
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    
    if len_sq < 1e-12:
        # Degenerate segment (start == end)
        return (px - ax) ** 2 + (py - ay) ** 2
    
    # Project point onto infinite line, clamp to segment
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len_sq, 0.0, 1.0)
    
    # Return squared distance to closest point on segment
    return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2

#geofencing step to classify GPS points by location (dump, road, boundary, etc)
def compute_geo_features_for_group(x_utm, y_utm, dt_time, geoms):
    """
    Compute geographic features for a single shift worth of GPS data.
    Features include: dump visits, time spent, road usage, boundary presence, etc.
    """
    n = len(x_utm)
    # Empty/default return for shifts with no GPS data
    empty = {
        "n_geo_dump_entries": 0,
        "time_in_dump_s": 0.0,
        "mean_road_offset_m": np.nan,
        "pct_on_road": np.nan,
        "in_ml_boundary_pct": np.nan,
        "mean_bench_dist_m": np.nan,
        "shift_centroid_x": np.nan,
        "shift_centroid_y": np.nan,
        "shift_start_x": np.nan,
        "shift_start_y": np.nan,
        "shift_end_x": np.nan,
        "shift_end_y": np.nan,
    }
    
    if n == 0:
        return empty

    #check if points are in dump sites
    in_dump = np.zeros(n, dtype=bool)
    for poly in geoms["ob_dump"]:
        if len(poly) < 3:
            continue
        
        # Quick bounding box check first (faster)
        mn_x, mn_y = poly[:, 0].min(), poly[:, 1].min()
        mx_x, mx_y = poly[:, 0].max(), poly[:, 1].max()
        cand = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)
        idx  = np.where(cand)[0]
        
        if len(idx):
            # Refine with actual point in polygon test for candidates
            in_dump[idx[_vec_point_in_polygon(x_utm[idx], y_utm[idx], poly)]] = True

    # Count how many times truck enters dump, and total time spent there
    dump_entries = int(np.sum((~in_dump[:-1]) & in_dump[1:]))  # Transitions from out to in
    time_in_dump = float(dt_time[in_dump].sum())  # Sum time intervals while in dump
    #distance to haul roads
    road_dist = np.full(n, np.nan)
    if geoms["haul_road"]:
        all_dists = np.full((n, len(geoms["haul_road"])), np.inf)
        
        for r_idx, road in enumerate(geoms["haul_road"]):
            if len(road) < 2:
                continue
            
            # Expand bbox by 500m buffer to find nearby candidates
            mn_x = road[:, 0].min() - 500
            mx_x = road[:, 0].max() + 500
            mn_y = road[:, 1].min() - 500
            mx_y = road[:, 1].max() + 500
            nearby  = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)
            
            min_dsq = np.full(n, np.inf)
            idx     = np.where(nearby)[0]
            
            if len(idx):
                # Find distance to nearest segment of this road
                for seg in range(len(road) - 1):
                    ax, ay = road[seg]
                    bx, by = road[seg + 1]
                    dsq = _vec_point_to_segment_dist_sq(
                        x_utm[idx], y_utm[idx], ax, ay, bx, by)
                    min_dsq[idx] = np.minimum(min_dsq[idx], dsq)
            
            all_dists[:, r_idx] = np.sqrt(min_dsq)
        
        # Keep minimum distance across all roads
        road_dist = all_dists.min(axis=1)

    # Calculate road-related metrics
    valid_road       = np.isfinite(road_dist)
    mean_road_offset = float(road_dist[valid_road].mean()) if valid_road.any() else np.nan
    pct_on_road      = float((road_dist[valid_road] <= ON_ROAD_THRESH_M).mean()) if valid_road.any() else np.nan

    # check if points are within mine lease boundary
    in_boundary = np.zeros(n, dtype=bool)
    if geoms["ml_boundary"]:
        poly = geoms["ml_boundary"][0]
        if len(poly) >= 3:
            mn_x, mn_y = poly[:, 0].min(), poly[:, 1].min()
            mx_x, mx_y = poly[:, 0].max(), poly[:, 1].max()
            cand = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)
            idx  = np.where(cand)[0]
            if len(idx):
                in_boundary[idx[_vec_point_in_polygon(x_utm[idx], y_utm[idx], poly)]] = True
    
    in_ml_pct = float(in_boundary.mean()) if n > 0 else np.nan

    #distance to nearest bench (pit boundary)
    mean_bench = np.nan
    if geoms["bench_kdtree"] is not None:
        dists, _ = geoms["bench_kdtree"].query(
            np.column_stack([x_utm, y_utm]), k=1, workers=-1)
        mean_bench = float(dists.mean())

    # Spatial fingerprint for route clustering 
    # Use centroid (center of gravity) and start/end points to characterize the shift route
    centroid_x = float(x_utm.mean())
    centroid_y = float(y_utm.mean())
    start_x    = float(x_utm[0])
    start_y    = float(y_utm[0])
    end_x      = float(x_utm[-1])
    end_y      = float(y_utm[-1])

    return {
        "n_geo_dump_entries": dump_entries,
        "time_in_dump_s":     time_in_dump,
        "mean_road_offset_m": mean_road_offset,
        "pct_on_road":        pct_on_road,
        "in_ml_boundary_pct": in_ml_pct,
        "mean_bench_dist_m":  mean_bench,
        "shift_centroid_x":   centroid_x,
        "shift_centroid_y":   centroid_y,
        "shift_start_x":      start_x,
        "shift_start_y":      start_y,
        "shift_end_x":        end_x,
        "shift_end_y":        end_y,
    }


def _process_single_geo_shift(name, group, mine_geoms):
    """
    Process geofencing for a single vehicle shift (parallelizable).
    Returns one row of geo features.
    """
    veh, date, shift = name
    mine_enc = int(group["mine_enc"].iloc[0]) if "mine_enc" in group.columns else -1
    geoms    = mine_geoms.get(mine_enc)
    
    # fallback row with NaNs if mine not found
    nan_row  = {
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
    if "x_utm" not in df.columns:
        return pd.DataFrame()
    
    print("    Computing geofencing features across all CPU cores …")
    
    # Process each shift in parallel
    records = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_process_single_geo_shift)(name, group, mine_geoms)
        for name, group in df.groupby(["vehicle", "adj_date", "shift_key"], sort=False)
    )
    
    geo_df = pd.DataFrame(records)
    print(f"    Geo feature rows: {len(geo_df):,}")
    return geo_df

#definf utility functions for loading data, normalizing vehicle names, safely accessing columns, and computing shifts/dates from timestamps

def norm_veh(s):
    """Normalize vehicle names: lowercase, remove special characters."""
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

def load_parquets(file_list, label):
    """
    Load and concatenate multiple parquet files into one DataFrame.
    Skip missing files with warning, normalize column names to lowercase.
    """
    seen, dfs = set(), []
    
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
    Safely get a column if it exists, otherwise return a Series of default values.
    Prevents KeyError when column is missing.
    """
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

def compute_shift_and_date(ts_series):
    """
    Determine shift (A/B/C) and adjusted date from timestamps.
    Shift A = 6-14h, B = 14-22h, C = 22-6h (midnight shift)
    Adjusted date = calendar date for shifts A/B, next day for shift C.
    """
    # Convert to Asia/Kolkata timezone
    if ts_series.dt.tz is None:
        ts = ts_series.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts_series.dt.tz_convert("Asia/Kolkata")
    
    hour = ts.dt.hour
    date = ts.dt.date
    
    # Assign shift based on hour of day
    shift    = np.select(
        [hour >= 22, hour < 6, (hour >= 6) & (hour < 14)],
        ["C", "C", "A"],
        default="B",
    )
    
    # For night shift (C), advance date to next calendar day
    next_date = (ts + pd.Timedelta(days=1)).dt.date
    dpr_date  = np.where(hour >= 22, next_date.astype(str), date.astype(str))
    
    return (
        pd.Series(dpr_date, index=ts_series.index, name="adj_date"),
        pd.Series(shift,    index=ts_series.index, name="shift_key"),
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
    rf["litres"]  = pd.to_numeric(rf["litres"], errors="coerce").fillna(0)
    
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

#temetry feature engineering to extract shift-level features from raw GPS/sensor data
# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY FEATURE ENGINEERING: Extract driving patterns from GPS/sensor data
# ─────────────────────────────────────────────────────────────────────────────
def get_advanced_features(df, label="", fleet_df=None):
    """
    Extract comprehensive shift-level features from raw telemetry data.
    Handles time gaps, calculates speeds, accelerations, cycle identification,
    laden/unladen segments, and aggregates to shift level.
    
    Returns:
      - agg_df: One row per shift with all engineered features
      - df_slim: Minimal telemetry data needed for geofencing
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), None

    print(f"  Processing {label} telemetry ({len(df):,} rows) …")
    df = df.copy()
    df["vehicle"] = norm_veh(df["vehicle"])
    df["ts"]      = pd.to_datetime(df["ts"])
    df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)

    #  Time deltas (seconds between successive points)
    raw_dt = df.groupby("vehicle")["ts"].diff().dt.total_seconds().fillna(30)
    df["dt_time"]  = raw_dt.clip(0, MAX_GAP_TIME)      # Cap large gaps for time-based features
    df["dt_accel"] = raw_dt.clip(1, MAX_GAP_ACCEL)     # Separate cap for acceleration

    #  Calculate shift and date from timestamps 
    calc_date, calc_shift = compute_shift_and_date(df["ts"])
    df["adj_date_calc"] = calc_date
    df["shift_calc"]    = calc_shift

    # Use provided date/shift if available, otherwise use calculated
    if "date_dpr" in df.columns and "shift_dpr" in df.columns:
        df["adj_date"]  = df["date_dpr"].astype(str).replace("nan", np.nan).fillna(df["adj_date_calc"])
        df["shift_key"] = df["shift_dpr"].astype(str).replace("nan", np.nan).fillna(df["shift_calc"])
    else:
        df["adj_date"]  = df["adj_date_calc"]
        df["shift_key"] = df["shift_calc"]

    # Remove rows with invalid date/shift
    df = df[df["adj_date"].notna() & df["shift_key"].notna()].copy()

    #  Convert GPS to UTM coordinates (for distance calculations) 
    if "longitude" in df.columns and "latitude" in df.columns:
        df["x_utm"], df["y_utm"] = transformer.transform(
            df["longitude"].values, df["latitude"].values)

    #  Altitude processing 
    # Smooth altitude to remove noise, then calculate elevation changes
    df["smooth_alt"] = df.groupby("vehicle")["altitude"].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).mean())
    df["delta_alt"]  = df.groupby("vehicle")["smooth_alt"].diff().fillna(0)

    #  Speed and movement 
    df["speed"]      = df["speed"].fillna(0).clip(lower=0)
    df["ignition"]   = safe_col(df, "ignition", 1).fillna(1)

    #  Acceleration and jerk (rate of acceleration change) 
    df["acceleration"] = (
        df.groupby("vehicle")["speed"]
        .transform(lambda x: x.diff().fillna(0)) / df["dt_accel"]
    )
    df["jerk"] = (
        df.groupby("vehicle")["acceleration"]
        .transform(lambda x: x.diff().fillna(0)) / df["dt_accel"]
    )

    #  Operating time 
    df["idle_sec"] = ((df["speed"] < SPEED_MOVING) & (df["ignition"] == 1)).astype(float) * df["dt_time"]
    df["run_sec"]  = (df["ignition"] == 1).astype(float) * df["dt_time"]
    df["dist_m"]   = safe_col(df, "disthav", 0).fillna(0).clip(lower=0)

    #  Dump signal detection 
    # Analog signal from dump hydraulics: when > threshold, truck is dumping
    has_analog    = "analog_input_1" in df.columns
    df["is_dump_sig"] = (
        (df["analog_input_1"].fillna(0) > DUMP_V_THRESH).astype(int)
        if has_analog else 0
    )

    #  Add mine info from fleet database 
    if fleet_df is not None:
        mine_lut = fleet_df[["vehicle", "mine_enc"]].drop_duplicates("vehicle")
        df = df.merge(mine_lut, on="vehicle", how="left")
        df["mine_enc"] = df["mine_enc"].fillna(-1).astype(int)
    else:
        df["mine_enc"] = -1

    #  Cycle (haul trip) identification 
    # A cycle = load, haul to dump, dump, return to load pit
    # Identified by either: (1) dump signal peaks, or (2) distance heuristic
    method_map = {}

    def map_cycles(group):
        """
        Identify haul cycles using either dump signal or distance heuristic.
        Returns cumulative cycle count for each point.
        """
        is_dump = group["is_dump_sig"].values
        veh     = group.name
        
        # Method 1: Use analog dump signal if available and reliable
        if has_analog and is_dump.sum() > 10:
            method_map[veh] = "signal"
            # Detect rising edges (0→1) in dump signal
            dump_events = (is_dump[1:] == 1) & (is_dump[:-1] == 0)
            return pd.Series(
                np.cumsum(np.concatenate(([0], dump_events))).astype(np.int32),
                index=group.index)
        
        # Method 2: Use distance traveled heuristic
        method_map[veh] = "heuristic"
        n = len(group)
        cycles = np.zeros(n, dtype=np.int32)
        cycle, dist_since_dump = 0, 0.0
        
        # Use mine-specific distance threshold
        dist_thresh = MINE_DIST_THRESH.get(int(group["mine_enc"].iloc[0]), DEFAULT_DIST_THRESH)
        is_stop = (group["speed"].values < SPEED_MOVING).astype(int)
        dists   = group["dist_m"].values
        
        # Increment cycle when truck stops after traveling enough distance
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

    #  Laden/unladen classification 
    # Laden = truck is carrying load (first 80% of cycle distance assumed)
    df["cum_dist"] = df.groupby(["vehicle", "cycle_id"])["dist_m"].cumsum()
    cycle_max      = df.groupby(["vehicle", "cycle_id"])["dist_m"].transform("sum")
    df["is_laden"] = (df["cum_dist"] > cycle_max * 0.20).astype(int)

    #  Segment distances and climbs 
    df["laden_dist"]    = df["dist_m"] * df["is_laden"]
    df["unladen_dist"]  = df["dist_m"] * (1 - df["is_laden"])
    df["laden_climb"]   = df["delta_alt"].clip(lower=0) * df["is_laden"]
    df["unladen_climb"] = df["delta_alt"].clip(lower=0) * (1 - df["is_laden"])
    df["moving_speed"]  = df["speed"].where(df["speed"] >= SPEED_MOVING)

    #  Aggregate to shift level 
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
        # Distance and movement
        total_dist_km    = ("dist_m",        lambda x: x.sum() / 1000),
        n_stops          = ("speed",         lambda x: ((x.shift(1) >= SPEED_MOVING) & (x < SPEED_MOVING)).sum()),
        
        # Engine time
        idle_h           = ("idle_sec",      lambda x: x.sum() / 3600),
        run_h            = ("run_sec",       lambda x: x.sum() / 3600),
        
        # Trip characteristics
        trip_count       = ("cycle_id",      "nunique"),
        shift_duration_h = ("ts",            lambda x: (x.max() - x.min()).total_seconds() / 3600),
        
        # Driving behavior
        accel_variance   = ("acceleration",  lambda x: x[x > 0].var() if (x > 0).any() else 0.0),
        hard_braking     = ("acceleration",  lambda x: (x < -2.0).sum()),
        n_dump_events    = ("is_dump_sig",   lambda x: (x.diff() == 1).sum()),
        
        # Terrain
        mean_alt         = ("smooth_alt",    "mean"),
        std_alt          = ("smooth_alt",    "std"),
        net_lift         = ("smooth_alt",    lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) >= 2 else 0.0),
        uphill_m         = ("delta_alt",     lambda x: x[x > 0].sum()),
        downhill_m       = ("delta_alt",     lambda x: (-x[x < 0]).sum()),
        
        # Data quality
        n_points         = ("ts",            "count"),
        
        # Laden/unladen breakdown
        laden_dist_km    = ("laden_dist",    lambda x: x.sum() / 1000),
        laden_climb_m    = ("laden_climb",   "sum"),
        unladen_dist_km  = ("unladen_dist",  lambda x: x.sum() / 1000),
        unladen_climb_m  = ("unladen_climb", "sum"),
        
        # Speed distribution
        mean_speed       = ("moving_speed",  "mean"),
        speed_std        = ("moving_speed",  "std"),
        max_speed        = ("moving_speed",  "max"),
        speed_p75        = ("moving_speed",  lambda x: x.quantile(0.75) if x.notna().any() else 0.0),
        
        # Jerk (driving smoothness)
        mean_jerk        = ("jerk",          lambda x: np.abs(x).mean()),
        max_jerk         = ("jerk",          lambda x: np.abs(x).max()),
        
        # Cycle segmentation method used
        cycle_method     = ("cycle_method",  lambda x: x.mode()[0] if len(x) > 0 else "heuristic"),
    ).reset_index()

    # Attach cycle-level features
    agg_df = agg_df.merge(cycle_dist_per_shift, on=grp, how="left")
    agg_df = agg_df.merge(cycle_dur_per_shift,  on=grp, how="left")

    #  Fill missing values with 0 for metrics that should be 0 when absent 
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

    #  Derived features (ratios, products, etc) 
    # These capture important relationships between base features
    agg_df["laden_ratio"]       = agg_df["laden_dist_km"]  / (agg_df["total_dist_km"] + 1e-6)
    agg_df["laden_climb_rate"]  = agg_df["laden_climb_m"]  / (agg_df["laden_dist_km"] * 1000 + 1e-6)
    agg_df["idle_ratio"]        = agg_df["idle_h"]         / (agg_df["run_h"] + 1e-6)
    agg_df["laden_climb_work"]  = agg_df["laden_dist_km"]  * agg_df["laden_climb_rate"]
    agg_df["alt_variability"]   = agg_df["std_alt"]        / (agg_df["mean_alt"].abs() + 1e-6)
    agg_df["terrain_intensity"] = (agg_df["uphill_m"] + agg_df["downhill_m"]) / (agg_df["total_dist_km"] * 1000 + 1e-6)
    agg_df["work_proxy"]        = agg_df["laden_climb_m"]  * agg_df["laden_dist_km"]
    agg_df["dist_per_runhrs"]   = agg_df["total_dist_km"]  / (agg_df["run_h"] + 1e-6)
    agg_df["idle_per_dist"]     = agg_df["idle_h"]         / (agg_df["total_dist_km"] + 1e-6)

    print(f"    Feature rows produced: {len(agg_df):,}")

    #  Prepare slim dataframe for geofencing 
    keep_cols = ["vehicle", "adj_date", "shift_key", "mine_enc", "dt_time"]
    if "x_utm" in df.columns:
        keep_cols += ["x_utm", "y_utm"]
    
    return agg_df, df[keep_cols].copy()

#vehicle history features to capture patterns in fuel consumption for each vehicle
def add_time_aware_vehicle_history(train_df, global_mean, global_std):
    """
    Create time-aware historical features for each vehicle.
    Uses expanding windows to avoid data leakage:
      - veh_mean_fuel: mean fuel from all previous shifts for this vehicle
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
    df["veh_lag1_fuel"]  = df.groupby("vehicle")["actual_fuel"].shift(1).fillna(global_mean)
    df["veh_lag2_fuel"]  = df.groupby("vehicle")["actual_fuel"].shift(2).fillna(global_mean)
    
    # Rolling average of previous 3 shifts
    df["veh_roll3_fuel"] = (
        df.groupby("vehicle")["actual_fuel"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(global_mean)
    )
    
    # Mean fuel for this vehicle in this shift type (across history)
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

#runhrs leakage check to ensure we're not using a feature that is too correlated with the target variable, which could lead to data leakage and overly optimistic model performance
def audit_runhrs_leakage(train_df):
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
    
    if corr > RUNHRS_LEAK_CORR_CAP:
        print(f"  [B1] WARNING: corr > {RUNHRS_LEAK_CORR_CAP} — runhrs_feat DROPPED")
        return False
    
    print(f"  [B1] Correlation acceptable — keeping runhrs_feat")
    return True

#feaute definitions to specify which features we will use for modeling, which are categorical, and which are for route-level benchmarks
# Core features from telemetry aggregation
BASE_FEATURES = [
    "vehicle_enc", "shift_enc", "mine_enc", "dump_switch_enc",
    "runhrs_feat",
    "run_h", "idle_h", "idle_ratio",
    "total_dist_km", "laden_dist_km", "unladen_dist_km",
    "laden_ratio", "laden_climb_m", "unladen_climb_m",
    "laden_climb_rate", "laden_climb_work",
    "mean_alt", "std_alt", "net_lift", "uphill_m", "downhill_m",
    "alt_variability", "terrain_intensity",
    "mean_speed", "speed_std", "max_speed", "speed_p75",
    "accel_variance", "hard_braking",
    "n_stops", "trip_count", "n_dump_events",
    "shift_duration_h",
    "dayofweek", "month", "dayofmonth", "is_weekend",
    "elapsed_days",
    "tankcap",
    "refuel_litres", "refuel_count",
    "n_points",
    "mean_jerk", "max_jerk",
    "mean_cycle_dist_km", "mean_cycle_dur_h",
    "work_proxy", "dist_per_runhrs", "idle_per_dist",
]

# Geographic/location features from geofencing
GEO_FEATURES = [
    "n_geo_dump_entries", "time_in_dump_s",
    "mean_road_offset_m", "pct_on_road",
    "in_ml_boundary_pct", "mean_bench_dist_m",
]

# Historical features from past shifts
HISTORY_FEATURES = [
    "veh_mean_fuel", "veh_std_fuel",
    "veh_lag1_fuel", "veh_lag2_fuel", "veh_roll3_fuel",
    "veh_shift_mean_fuel",
]

# Combined feature set for modeling
ALL_FEATURES_BASE = BASE_FEATURES + GEO_FEATURES + HISTORY_FEATURES

# Which features are categorical (need special handling in some models)
CAT_FEATURE_NAMES = ["vehicle_enc", "shift_enc", "mine_enc", "dump_switch_enc"]

# Features for building route-level benchmarks (vehicle-independent)
ROUTE_FEATURES = [
    "total_dist_km", "laden_dist_km", "unladen_dist_km",
    "laden_ratio", "laden_climb_m", "unladen_climb_m",
    "laden_climb_rate", "laden_climb_work",
    "mean_alt", "std_alt", "net_lift", "uphill_m", "downhill_m",
    "alt_variability", "terrain_intensity",
    "mean_speed", "speed_std", "max_speed", "speed_p75",
    "n_stops", "trip_count", "shift_duration_h",
    "mean_cycle_dist_km", "mean_cycle_dur_h",
    "work_proxy", "dist_per_runhrs",
    "n_geo_dump_entries", "time_in_dump_s",
    "mean_road_offset_m", "pct_on_road",
    "in_ml_boundary_pct", "mean_bench_dist_m",
    "mine_enc", "shift_enc",
    "dayofweek", "month", "dayofmonth", "is_weekend",
    "tankcap",
    "runhrs_feat", "run_h", "idle_h", "idle_ratio",
]

#model builders to create instances of different machine learning algorithms with predefined hyperparameters, which can be used for training and prediction
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

#ensemble optimization to find the best way to combine predictions from multiple models
def optimize_ensemble_weights(oof_preds, y_true):
    """
    Use SLSQP optimization to find ensemble weights that minimize RMSE.
    Ensures weights sum to 1 and each weight is between 0 and 1.
    """
    model_names = list(oof_preds.keys())
    P = np.column_stack([oof_preds[m] for m in model_names])

    def neg_rmse(w):
        return np.sqrt(mean_squared_error(y_true, P @ w))

    n      = len(model_names)
    result = minimize(
        neg_rmse, np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    
    weights    = result.x
    best_rmse  = np.sqrt(mean_squared_error(y_true, P @ weights))
    equal_rmse = np.sqrt(mean_squared_error(y_true, P.mean(axis=1)))
    
    print("\n  [SLSQP] Optimized ensemble weights:")
    for name, w in zip(model_names, weights):
        print(f"    {name:12s}: {w:.4f}")
    print(f"  Equal-weight RMSE: {equal_rmse:.2f} L  →  SLSQP RMSE: {best_rmse:.2f} L")
    
    return dict(zip(model_names, weights))
#TWO STAGE TRAIINING APPROACH
def train_two_stage(X, y, feat_names, date_groups, cat_feat_idx):
    """
    Two-stage training approach:
     Stage 1: Train classifier to predict working vs. non-working shifts
    (working = fuel consumption > ZERO_FUEL_THRESH)   
    Stage 2: Train ensemble of regressors on working shifts only
    Final prediction = P(working) * fuel_predictioN
    This helps capture zero-fuel shifts (errors/missing data/non-operational)
    separately from operational shifts where fuel consumption is variable.
    """
    kf         = GroupKFold(n_splits=N_CV_SPLITS)
    is_working = (y > ZERO_FUEL_THRESH).astype(int)
    print(f"  Working shifts: {is_working.sum():,} / {len(y):,} ({is_working.mean()*100:.1f}%)")

    #  Stage 1: Classify working shifts 
    print("\n  [Stage 1] Training working-shift classifier …")
    oof_prob   = np.zeros(len(y))
    clf_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, is_working, date_groups)):
        m_c = make_hgbt_clf()
        m_c.fit(X[tr_idx], is_working[tr_idx])
        oof_prob[val_idx] = m_c.predict_proba(X[val_idx])[:, 1]
        clf_models.append(m_c)
        print(f"    Clf fold {fold+1} stopped at {m_c.n_iter_} iterations")

    clf_acc = ((oof_prob > 0.5).astype(int) == is_working).mean()
    print(f"  Classifier OOF accuracy: {clf_acc*100:.1f}%")

    #  Stage 2: Regress fuel on working shifts 
    print("\n  [Stage 2] Training fuel regressor on working shifts only …")
    work_idx = np.where(is_working == 1)[0]
    X_work   = X[work_idx]
    y_work   = y[work_idx]
    g_work   = date_groups[work_idx]

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

    reg_models    = {k: [] for k in oof_reg_preds}
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

        # CatBoost (requires DataFrame input with categorical types specified)
        X_tr_cat  = pd.DataFrame(X_tr,  columns=feat_names)
        X_val_cat = pd.DataFrame(X_val, columns=feat_names)
        for idx in cat_feat_idx:
            col_name = feat_names[idx]
            X_tr_cat[col_name]  = X_tr_cat[col_name].astype(int).astype(str)
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

    #  Optimize ensemble weights using holdout set 
    # Split training data by date: recent dates = holdout, older = fit
    work_dates    = date_groups[work_idx]
    unique_dates  = np.unique(work_dates)
    n_holdout     = max(1, int(len(unique_dates) * WEIGHT_HOLDOUT_FRAC))
    holdout_dates = set(unique_dates[-n_holdout:])  # Most recent dates
    holdout_mask  = np.array([d in holdout_dates for d in work_dates])
    fit_mask      = ~holdout_mask

    print(f"\n  [L3] Fitting SLSQP on {fit_mask.sum():,} rows, validating on {holdout_mask.sum():,} holdout rows …")
    weights = optimize_ensemble_weights(
        {k: v[fit_mask] for k, v in oof_reg_preds.items()}, y_work[fit_mask])

    # Evaluate on holdout (truly unseen dates)
    P_holdout    = np.column_stack([oof_reg_preds[k][holdout_mask] for k in oof_reg_preds])
    holdout_rmse = np.sqrt(mean_squared_error(
        y_work[holdout_mask], P_holdout @ np.array([weights[k] for k in oof_reg_preds])))
    print(f"  [L3] Holdout RMSE (unseen): {holdout_rmse:.2f} L")

    # Overall OOF performance on working shifts
    P_all        = np.column_stack([oof_reg_preds[k] for k in oof_reg_preds])
    w_arr        = np.array([weights[k] for k in oof_reg_preds])
    reg_oof_rmse = np.sqrt(mean_squared_error(y_work, P_all @ w_arr))
    print(f"  Overall regressor OOF RMSE (working shifts) = {reg_oof_rmse:.2f} L")

    # Overall two-stage OOF (including zero-fuel shifts)
    full_oof           = np.zeros(len(y))
    full_oof[work_idx] = oof_prob[work_idx] * (P_all @ w_arr)
    two_stage_rmse     = np.sqrt(mean_squared_error(y, full_oof))
    print(f"  Two-stage OOF RMSE (all shifts incl. zeros) = {two_stage_rmse:.2f} L")

    # Show top feature importances from XGBoost fold 1
    if reg_models["xgb"]:
        fi = pd.Series(reg_models["xgb"][0].feature_importances_, index=feat_names)
        print("\n  Top 15 feature importances (XGBoost fold 1):")
        print(fi.sort_values(ascending=False).head(15).to_string())

    return {
        "clf_models":    clf_models,
        "reg_models":    reg_models,
        "ridge_scalers": ridge_scalers,
        "weights":       weights,
        "oof_rmse":      two_stage_rmse,
        "reg_oof_rmse":  reg_oof_rmse,
        "holdout_rmse":  holdout_rmse,
        "feat_names":    feat_names,
    }

#prediction function to generate fuel consumption predictions for new data using the trained two-stage ensemble model
def predict_ensemble(bundle, X, feat_names_override=None):
    """
    Generate predictions using the two-stage ensemble.
    Averages predictions across CV folds for each model type,
    blends using optimized weights.
    """
    clf_models = bundle["clf_models"]
    reg_models = bundle["reg_models"]
    weights    = bundle["weights"]
    scalers    = bundle["ridge_scalers"]
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
        reg_preds["rf"]  = np.mean([m.predict(X) for m in reg_models["rf"]],  axis=0)

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
#SHARED HELPER FUNCTIONS
def encode_df(df, vehicle_map, shift_map):
    """
    Encode categorical features (vehicle, shift) to integers.
    Extract date components (day of week, month, etc).
    """
    df = df.copy()
    df["vehicle_enc"] = df["vehicle"].map(vehicle_map).fillna(-1).astype(int)
    df["shift_enc"]   = df["shift_key"].map(shift_map).fillna(-1).astype(int)
    df["date_dt"]     = pd.to_datetime(df["adj_date"], errors="coerce")
    df["dayofweek"]   = df["date_dt"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"]       = df["date_dt"].dt.month
    df["dayofmonth"]  = df["date_dt"].dt.day
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)  # Saturday/Sunday
    return df

def add_elapsed_days(df, start_date_str):
    """Add feature: days elapsed since training start."""
    start = pd.to_datetime(start_date_str)
    df["elapsed_days"] = (
        pd.to_datetime(df["adj_date"], errors="coerce") - start
    ).dt.days.fillna(0).clip(lower=0)
    return df

def attach_fleet(df, fleet_df):
    """Join vehicle metadata from fleet database (mine, dump switch, tank capacity)."""
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
    df["refuel_count"]  = df["refuel_count"].fillna(0)
    return df

def fill_metadata(df):
    """Fill missing metadata with sensible defaults."""
    df["mine_enc"]        = df["mine_enc"].fillna(-1).astype(int)        if "mine_enc"        in df.columns else pd.Series(-1,     index=df.index).astype(int)
    df["dump_switch_enc"] = df["dump_switch_enc"].fillna(0).astype(int)  if "dump_switch_enc" in df.columns else pd.Series(0,      index=df.index).astype(int)
    df["tankcap"]         = df["tankcap"].fillna(1311.0)                 if "tankcap"         in df.columns else pd.Series(1311.0, index=df.index)
    return df

# SO1: ROUTE CLUSTERING AND ROUTE-LEVEL BENCHMARKS
def compute_route_clusters(combined_geo: pd.DataFrame, n_clusters: int = N_ROUTE_CLUSTERS) -> pd.DataFrame:
    """
    Cluster shifts into route types based on their spatial footprint.
    Uses (centroid, start, end) coordinates to capture both location and direction.
    Returns route_id for each shift.
    """
    coord_cols = ["shift_centroid_x", "shift_centroid_y",
                  "shift_start_x",    "shift_start_y",
                  "shift_end_x",      "shift_end_y"]

    valid = combined_geo.dropna(subset=coord_cols).copy()
    if len(valid) < n_clusters:
        print(f"    WARNING: only {len(valid)} shifts with GPS, reducing clusters to {len(valid)//2}")
        n_clusters = max(2, len(valid) // 2)

    print(f"    Clustering {len(valid):,} shifts into {n_clusters} route clusters …")

    # Normalize each coordinate to [0,1] so no axis dominates
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
    route_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a vehicle-agnostic fuel benchmark for each route.
    This helps evaluate how efficiently each vehicle performs on a given route.
    Process:
      1. Zero out vehicle identity and history features
      2. Predict fuel using the ensemble (as if "unknown vehicle")
      3. Average predictions by route_id
      4. Compare vs actual mean fuel
    """
    print("\n  [SO1] Computing route-level fuel benchmarks …")

    train_rt = train_data.copy()
    train_rt["route_id"] = train_rt["route_id"].fillna(-1).astype(int)

    # Create anonymous feature matrix (remove vehicle-specific info)
    X_anon = train_data[ALL_FEATURES].fillna(global_feat_meds).values.astype(np.float64)
    feat_idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    # Zero out vehicle identity
    if "vehicle_enc" in feat_idx:
        X_anon[:, feat_idx["vehicle_enc"]] = -1
    
    # Zero out vehicle history (use global mean instead)
    for hf in ["veh_mean_fuel", "veh_lag1_fuel", "veh_lag2_fuel",
               "veh_roll3_fuel", "veh_shift_mean_fuel"]:
        if hf in feat_idx:
            X_anon[:, feat_idx[hf]] = global_feat_meds.get(hf, 0.0)
    if "veh_std_fuel" in feat_idx:
        X_anon[:, feat_idx["veh_std_fuel"]] = global_feat_meds.get("veh_std_fuel", 0.0)

    # Predict using anonymous features (generic benchmark for route)
    route_pred_fuel = predict_ensemble(bundle, X_anon)
    train_rt["route_pred_fuel"] = route_pred_fuel

    # Aggregate to route level
    route_bench = (
        train_rt.groupby("route_id")
        .agg(
            n_shifts                = ("vehicle",            "count"),
            n_unique_vehicles       = ("vehicle",            "nunique"),
            mean_dist_km            = ("total_dist_km",      "mean"),
            mean_uphill_m           = ("uphill_m",          "mean"),
            mean_downhill_m         = ("downhill_m",        "mean"),
            mean_laden_dist_km      = ("laden_dist_km",      "mean"),
            mean_terrain_intensity  = ("terrain_intensity", "mean"),
            mean_pct_on_road        = ("pct_on_road",       "mean"),
            route_benchmark_fuel_L  = ("route_pred_fuel",   "mean"),
            actual_mean_fuel_L      = ("actual_fuel",       "mean"),
        )
        .reset_index()
    )
    
    # Calculate accuracy of benchmark
    route_bench["benchmark_accuracy_pct"] = (
        100 * (1 - abs(route_bench["route_benchmark_fuel_L"] - route_bench["actual_mean_fuel_L"])
               / (route_bench["actual_mean_fuel_L"] + 1e-6))
    ).clip(0, 100)

    print(f"    Route benchmark built for {len(route_bench)} routes.")
    print(route_bench[["route_id", "n_shifts", "mean_dist_km",
                        "route_benchmark_fuel_L", "actual_mean_fuel_L",
                        "benchmark_accuracy_pct"]].to_string(index=False))
    
    return route_bench, train_rt

# SO2: DUMPER EFFICIENCY RANKING

def compute_dumper_efficiency(
    predictions_df: pd.DataFrame,
    route_bench: pd.DataFrame,
    label: str = "train",
) -> pd.DataFrame:
    """
    Calculate efficiency of each vehicle by comparing actual/predicted fuel
    against the route benchmark.
    
    efficiency_ratio = predicted_fuel / route_benchmark_fuel
    Ratio < 1 = better than benchmark, > 1 = worse than benchmark
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

    # Calculate efficiency (predicted / benchmark)
    eff["efficiency_ratio"] = (
        eff["predicted_fuel"] / (eff["route_benchmark_fuel_L"] + 1e-6)
    ).clip(0, 5)  # Cap at 5x to avoid outliers

    # Per-vehicle mean efficiency
    veh_eff = (
        eff.groupby("vehicle")["efficiency_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"efficiency_ratio": "veh_mean_efficiency"})
    )
    
    # Rank vehicles: 1 = most efficient (lowest ratio)
    veh_eff["veh_efficiency_rank"] = veh_eff["veh_mean_efficiency"].rank(method="min").astype(int)
    eff = eff.merge(veh_eff, on="vehicle", how="left")

    # If training data, also compute actual efficiency
    if label == "train" and "actual_fuel" in eff.columns:
        eff["actual_efficiency_ratio"] = (
            eff["actual_fuel"] / (eff["route_benchmark_fuel_L"] + 1e-6)
        ).clip(0, 5)

    # Prepare output columns
    out_cols = ["vehicle", "adj_date", "shift_key", "route_id",
                "predicted_fuel", "route_benchmark_fuel_L", "efficiency_ratio",
                "veh_mean_efficiency", "veh_efficiency_rank"]
    if "actual_fuel" in eff.columns:
        out_cols += ["actual_fuel", "actual_efficiency_ratio"]

    print(f"    Top 10 most efficient vehicles:")
    top10 = veh_eff.sort_values("veh_mean_efficiency").head(10)
    print(top10.to_string(index=False))

    return eff[out_cols]

# SO3: CYCLE SEGMENTATION DETAILS

def build_cycle_segmentation_output(
    train_feats: pd.DataFrame,
    test_feats:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Output shift-level details about how haul cycles were identified and characterized.
    Useful for understanding the segmentation quality and method effectiveness.
    """
    print("\n  [SO3] Building cycle segmentation output …")
    
    cols = ["vehicle", "adj_date", "shift_key", "cycle_method",
            "trip_count", "mean_cycle_dist_km", "mean_cycle_dur_h",
            "laden_dist_km", "unladen_dist_km", "laden_climb_m",
            "n_dump_events", "n_stops"]

    def safe_select(df, label):
        """Select available columns and add split label."""
        present = [c for c in cols if c in df.columns]
        out     = df[present].copy()
        out["split"] = label
        return out

    # Combine train and test outputs
    combined = pd.concat([
        safe_select(train_feats, "train"),
        safe_select(test_feats,  "test"),
    ], ignore_index=True)

    print(f"    Cycle segmentation rows: {len(combined):,}")
    method_counts = combined["cycle_method"].value_counts() if "cycle_method" in combined.columns else {}
    print(f"    Method breakdown: {method_counts.to_dict()}")
    
    return combined

# SO4: DAILY FUEL CONSISTENCY

def build_daily_consistency(
    train_data:  pd.DataFrame,
    test_data:   pd.DataFrame,
    train_preds: np.ndarray,
    test_preds:  np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate shift-level predictions to daily level.
    Allows checking consistency of predictions across a day
    and comparing predicted vs actual daily totals.
    """
    print("\n  [SO4] Building daily fuel consistency output …")

    #  Train 
    train_daily = train_data[["vehicle", "adj_date", "shift_key", "actual_fuel"]].copy()
    train_daily["predicted_fuel"] = train_preds

    # Pivot to get A/B/C shift columns
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
    
    # Calculate daily total
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
    
    # Calculate residuals and errors
    train_out["daily_residual"] = train_out["predicted_daily_total"] - train_out["actual_daily_total"]
    train_out["abs_pct_error"]  = (
        100 * abs(train_out["daily_residual"]) / (train_out["actual_daily_total"] + 1e-6)
    ).clip(0, 500)
    train_out["split"] = "train"

    mae   = train_out["daily_residual"].abs().mean()
    mape  = train_out["abs_pct_error"].mean()
    print(f"    Train daily aggregation — MAE: {mae:.1f} L  |  MAPE: {mape:.1f}%")

    #  Test (no actual fuel available) 
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
    
    # Test has no actuals
    test_pivot["actual_daily_total"] = np.nan
    test_pivot["daily_residual"]     = np.nan
    test_pivot["abs_pct_error"]      = np.nan
    test_pivot["split"] = "test"

    # Combine
    combined = pd.concat([train_out, test_pivot], ignore_index=True)
    print(f"    Daily consistency rows: {len(combined):,} ({len(train_out)} train + {len(test_pivot)} test)")
    
    return combined

# MAIN PIPELINE

def main():
    """
    Complete fuel prediction pipeline:
    1. Load mine geometries and static data
    2. Load and process telemetry
    3. Compute geofencing and cycle features
    4. Train two-stage ensemble model
    5. Generate predictions
    6. Create secondary analysis outputs
    """
    print("=" * 65)
    print("  HaulMark v5.0 — Secondary Outputs Edition")
    print("=" * 65)

    #  Load mine geometries 
    print("\n[Step 0a] Loading mine geometries from GPKG …")
    mine_geoms: Dict = {}
    for mine_enc, gpkg_path in GPKG_FILES.items():
        mine_geoms[mine_enc] = load_gpkg_geometries(gpkg_path)

    #  Load static metadata 
    print("\n[Step 0b] Loading metadata …")
    id_map = pd.read_csv(RAW / "id_mapping_new.csv")
    id_map.columns = [c.lower().strip() for c in id_map.columns]
    id_map["vehicle"] = norm_veh(id_map["vehicle"])
    id_map.rename(columns={"date": "adj_date", "shift": "shift_key"}, inplace=True)
    id_map = id_map.drop_duplicates(subset=["id"])

    # Fleet data: vehicle attributes, mine assignments, tank capacities
    fleet = pd.read_csv(RAW / "fleet.csv")
    fleet.columns = [c.lower().strip() for c in fleet.columns]
    fleet_dumpers = fleet[fleet["fleet"].str.lower() == "dumper"][
        ["vehicle", "mine_anon", "tankcap", "dump_switch"]
    ].copy()
    fleet_dumpers["vehicle"]         = norm_veh(fleet_dumpers["vehicle"])
    mine_lut                         = {"mine001": 0, "mine002": 1, "mine003": 2}
    fleet_dumpers["mine_enc"]        = fleet_dumpers["mine_anon"].map(mine_lut).fillna(-1).astype(int)
    fleet_dumpers["dump_switch_enc"] = fleet_dumpers["dump_switch"].fillna(0).astype(int)
    fleet_dumpers["tankcap"]         = fleet_dumpers["tankcap"].fillna(1311.0)

    # Test dates mark rows that should be kept as test (not training)
    TEST_DATES = set(id_map["adj_date"].unique())

    #  Load summary (ground truth fuel consumption) 
    print("\n[Step 1] Loading summary files …")
    frames   = [pd.read_csv(RAW / fn) for fn in SUMMARY_FILES if (RAW / fn).exists()]
    all_smry = pd.concat(frames, ignore_index=True)
    all_smry.columns = [c.lower().strip() for c in all_smry.columns]
    all_smry["vehicle"] = norm_veh(all_smry["vehicle"])
    all_smry["date"]    = pd.to_datetime(all_smry["date"]).dt.date.astype(str)
    all_smry["acons"]   = all_smry["acons"].clip(lower=0)
    all_smry.rename(columns={
        "acons": "actual_fuel", "date": "adj_date",
        "shift": "shift_key",  "mine": "mine_name",
    }, inplace=True)
    
    # Separate train and test
    train_smry  = all_smry[~all_smry["adj_date"].isin(TEST_DATES)].copy()
    TRAIN_START = train_smry["adj_date"].min()
    print(f"  Train rows: {len(train_smry):,} | Zero-fuel: {(train_smry['actual_fuel'] < ZERO_FUEL_THRESH).sum():,}")
    print(f"  Training period: {TRAIN_START} → {train_smry['adj_date'].max()}")

    #  Load RFID refueling data 
    print("\n[Step 2] Loading RFID refuel transactions …")
    rfid_agg = load_rfid_refuels()

    #  Process telemetry 
    print("\n[Step 3] Processing train telemetry …")
    train_raw = load_parquets(TRAIN_TELEMETRY, "train")
    train_feats, train_raw_slim = get_advanced_features(train_raw, "train", fleet_df=fleet_dumpers)

    print("\n[Step 3b] Processing test telemetry …")
    test_raw = load_parquets(TEST_TELEMETRY, "test")
    test_feats, test_raw_slim = get_advanced_features(test_raw, "test", fleet_df=fleet_dumpers)

    #  Compute geofencing features 
    print("\n[Step 3c] Computing geo features (train) …")
    train_geo = (
        compute_geo_shift_features(train_raw_slim, mine_geoms)
        if train_raw_slim is not None else pd.DataFrame()
    )
    print("\n[Step 3d] Computing geo features (test) …")
    test_geo = (
        compute_geo_shift_features(test_raw_slim, mine_geoms)
        if test_raw_slim is not None else pd.DataFrame()
    )
    del train_raw_slim, test_raw_slim

    #  Route clustering (for SO1 benchmark) 
    print("\n[Step 3e] Route clustering for route-level benchmark (SO1) …")
    GEO_SPATIAL_COLS = ["vehicle", "adj_date", "shift_key",
                        "shift_centroid_x", "shift_centroid_y",
                        "shift_start_x", "shift_start_y",
                        "shift_end_x", "shift_end_y"]

    combined_geo = pd.DataFrame()
    if not train_geo.empty:
        avail_cols = [c for c in GEO_SPATIAL_COLS if c in train_geo.columns]
        tg = train_geo[avail_cols].copy()
        tg["split"] = "train"
        combined_geo = pd.concat([combined_geo, tg], ignore_index=True)
    if not test_geo.empty:
        avail_cols = [c for c in GEO_SPATIAL_COLS if c in test_geo.columns]
        tg2 = test_geo[avail_cols].copy()
        tg2["split"] = "test"
        combined_geo = pd.concat([combined_geo, tg2], ignore_index=True)

    if not combined_geo.empty:
        route_assignments = compute_route_clusters(combined_geo)
    else:
        print("    WARNING: No GPS data for route clustering — using mine_enc as route_id fallback")
        route_assignments = pd.DataFrame(columns=["vehicle", "adj_date", "shift_key", "route_id"])

    #  Build encoding maps 
    print("\n[Step 4] Building vehicle encodings …")
    all_vehicles = sorted(set(train_smry["vehicle"]) | set(id_map["vehicle"]))
    vehicle_map  = {v: i for i, v in enumerate(all_vehicles)}
    shift_map    = {"A": 0, "B": 1, "C": 2}

    #  Assemble training dataset 
    print("\n[Step 5] Assembling training dataset …")
    train_data = train_smry.merge(train_feats, on=["vehicle", "adj_date", "shift_key"], how="left")

    # Add geo features
    GEO_COLS_STANDARD = [c for c in train_geo.columns
                         if c not in ["vehicle", "adj_date", "shift_key"]] if not train_geo.empty else []
    if not train_geo.empty:
        train_data = train_data.merge(train_geo, on=["vehicle", "adj_date", "shift_key"], how="left")
    else:
        for f in GEO_FEATURES:
            train_data[f] = np.nan

    # Add encodings and metadata
    train_data = encode_df(train_data, vehicle_map, shift_map)
    train_data = add_elapsed_days(train_data, TRAIN_START)
    train_data = attach_fleet(train_data, fleet_dumpers)
    train_data = attach_rfid(train_data, rfid_agg)
    train_data = fill_metadata(train_data)

    # Add route assignments for SO1/SO2
    if not route_assignments.empty:
        train_data = train_data.merge(
            route_assignments[["vehicle", "adj_date", "shift_key", "route_id"]],
            on=["vehicle", "adj_date", "shift_key"], how="left"
        )
        train_data["route_id"] = train_data["route_id"].fillna(-1).astype(int)
    else:
        train_data["route_id"] = train_data.get("mine_enc", pd.Series(-1, index=train_data.index))

    #  Check for leakage and add runhrs feature 
    print("\n[B1 Audit] Checking runhrs for target leakage …")
    keep_runhrs = audit_runhrs_leakage(train_data)
    if keep_runhrs:
        train_data["runhrs_feat"] = train_data["runhrs"].fillna(
            train_data["run_h"] if "run_h" in train_data.columns else 0)
    else:
        train_data["runhrs_feat"] = 0.0

    #  Add vehicle history features 
    print("\n[Step 6] Computing time-aware vehicle history …")
    global_mean = train_data["actual_fuel"].mean()
    global_std  = train_data["actual_fuel"].std()
    train_data  = add_time_aware_vehicle_history(train_data, global_mean, global_std)

    #  Prepare feature matrix 
    print("\n[Step 7] Preparing feature matrices …")
    ALL_FEATURES = [f for f in ALL_FEATURES_BASE if f in train_data.columns or f == "runhrs_feat"]
    missing_feats = [f for f in ALL_FEATURES if f not in train_data.columns]
    for f in missing_feats:
        train_data[f] = 0.0

    X_all             = train_data[ALL_FEATURES].values.astype(np.float64)
    y_all             = train_data["actual_fuel"].values
    global_feat_meds  = pd.Series(np.nanmedian(X_all, axis=0), index=ALL_FEATURES).fillna(0)
    X_all_filled      = pd.DataFrame(X_all, columns=ALL_FEATURES).fillna(global_feat_meds).values
    cat_feat_idx      = [ALL_FEATURES.index(f) for f in CAT_FEATURE_NAMES if f in ALL_FEATURES]
    date_groups       = pd.to_datetime(train_data["adj_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown").values

    print(f"  Feature matrix: {X_all.shape}")
    print(f"  Target mean: {y_all.mean():.1f} L, std: {y_all.std():.1f} L")

    #  Train two-stage ensemble 
    print("\n[Step 8] Training two-stage ensemble …")
    bundle = train_two_stage(X_all_filled, y_all, ALL_FEATURES, date_groups, cat_feat_idx)

    #  Prepare test dataset 
    print("\n[Step 9] Assembling test predictions …")
    test_feats_dd = test_feats.drop_duplicates(subset=["vehicle", "adj_date", "shift_key"])
    test_data     = id_map.merge(test_feats_dd, on=["vehicle", "adj_date", "shift_key"], how="left")

    if not test_geo.empty:
        test_data = test_data.merge(test_geo, on=["vehicle", "adj_date", "shift_key"], how="left")
    else:
        for f in GEO_FEATURES:
            test_data[f] = np.nan

    test_data = encode_df(test_data, vehicle_map, shift_map)
    test_data = add_elapsed_days(test_data, TRAIN_START)
    test_data = attach_fleet(test_data, fleet_dumpers)
    test_data = attach_rfid(test_data, rfid_agg)
    test_data = fill_metadata(test_data)

    # Add route assignments
    if not route_assignments.empty:
        test_data = test_data.merge(
            route_assignments[["vehicle", "adj_date", "shift_key", "route_id"]],
            on=["vehicle", "adj_date", "shift_key"], how="left"
        )
        test_data["route_id"] = test_data["route_id"].fillna(-1).astype(int)
    else:
        test_data["route_id"] = test_data.get("mine_enc", pd.Series(-1, index=test_data.index))

    test_data["runhrs_feat"] = (
        test_data["run_h"].fillna(0)
        if keep_runhrs and "run_h" in test_data.columns
        else pd.Series(0.0, index=test_data.index)
    )
    for f in missing_feats:
        test_data[f] = 0.0
    
    test_data = apply_vehicle_history_test(test_data, train_data, global_mean, global_std)
    X_test    = test_data[ALL_FEATURES].fillna(global_feat_meds).values.astype(np.float64)

    #  Generate predictions 
    final_preds      = predict_ensemble(bundle, X_test)
    train_preds_full = predict_ensemble(bundle, X_all_filled)

    #  Apply floor and ceiling constraints 
    # Floor: for vehicles with historical non-zero shifts, don't predict below 2nd percentile
    veh_min       = (
        train_data[train_data["actual_fuel"] > ZERO_FUEL_THRESH]
        .groupby("vehicle")["actual_fuel"].quantile(0.02)
    )
    veh_has_zeros = set(train_data.loc[train_data["actual_fuel"] <= ZERO_FUEL_THRESH, "vehicle"])
    smart_floor   = test_data["vehicle"].map(veh_min).fillna(0).copy()
    smart_floor.loc[test_data["vehicle"].isin(veh_has_zeros).values] = 0.0
    final_preds   = np.maximum(final_preds, smart_floor.values)

    # Ceiling: don't predict above tank capacity (with margin)
    if "tankcap" in test_data.columns:
        ceiling  = (test_data["tankcap"].fillna(1311.0) * TANK_CAP_MARGIN).values
        n_capped = int((final_preds > ceiling).sum())
        final_preds = np.minimum(final_preds, ceiling)
        if n_capped:
            print(f"  Tank-cap applied to {n_capped} predictions")

    # Apply same constraints to train preds
    train_floor   = train_data["vehicle"].map(veh_min).fillna(0).copy()
    train_floor.loc[train_data["vehicle"].isin(veh_has_zeros).values] = 0.0
    train_preds_full = np.maximum(train_preds_full, train_floor.values)
    if "tankcap" in train_data.columns:
        tr_ceiling = (train_data["tankcap"].fillna(1311.0) * TANK_CAP_MARGIN).values
        train_preds_full = np.minimum(train_preds_full, tr_ceiling)

    #  Create primary submission 
    submission = pd.DataFrame({
        "id":          test_data["id"].values,
        "fuel_volume": np.round(final_preds, 2),
    })
    
    # Validate
    assert len(submission) == len(id_map)
    assert submission["fuel_volume"].isna().sum() == 0
    assert (submission["fuel_volume"] >= 0).all()
    
    out_path = RAW / "submission_v5_0.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n  Primary submission → {out_path}")

    #  Generate secondary outputs 
    print("\n[Step 12] Generating secondary outputs …")

    # SO1: Route-level fuel benchmark
    route_bench, train_with_route_preds = build_route_benchmark(
        train_data, bundle, ALL_FEATURES, global_feat_meds, route_assignments)
    route_bench_path = RAW / "secondary_route_benchmark.csv"
    route_bench.to_csv(route_bench_path, index=False)
    print(f"  [SO1] Route benchmark → {route_bench_path}")

    # SO2: Dumper efficiency (train)
    train_eff_df = train_data[["vehicle", "adj_date", "shift_key", "route_id", "actual_fuel"]].copy()
    train_eff_df["predicted_fuel"] = np.round(train_preds_full, 2)
    dumper_eff_train = compute_dumper_efficiency(train_eff_df, route_bench, label="train")

    # SO2: Dumper efficiency (test)
    test_eff_df = test_data[["vehicle", "adj_date", "shift_key", "route_id"]].copy()
    test_eff_df["predicted_fuel"] = np.round(final_preds, 2)
    dumper_eff_test = compute_dumper_efficiency(test_eff_df, route_bench, label="test")

    dumper_eff_all = pd.concat([
        dumper_eff_train.assign(split="train"),
        dumper_eff_test.assign(split="test"),
    ], ignore_index=True)
    eff_path = RAW / "secondary_dumper_efficiency.csv"
    dumper_eff_all.to_csv(eff_path, index=False)
    print(f"  [SO2] Dumper efficiency → {eff_path}")

    # SO3: Cycle segmentation
    cycle_seg = build_cycle_segmentation_output(train_feats, test_feats_dd)
    cycle_path = RAW / "secondary_cycle_segmentation.csv"
    cycle_seg.to_csv(cycle_path, index=False)
    print(f"  [SO3] Cycle segmentation → {cycle_path}")

    # SO4: Daily consistency
    daily_cons = build_daily_consistency(
        train_data, test_data, train_preds_full, final_preds)
    daily_path = RAW / "secondary_daily_consistency.csv"
    daily_cons.to_csv(daily_path, index=False)
    print(f"  [SO4] Daily consistency → {daily_path}")

    #  Final summary 
    print("\n" + "=" * 65)
    print("  All outputs saved:")
    print(f"   PRIMARY   → submission_v5_0.csv            ({len(submission)} rows)")
    print(f"   SO1       → secondary_route_benchmark.csv  ({len(route_bench)} routes)")
    print(f"   SO2       → secondary_dumper_efficiency.csv ({len(dumper_eff_all)} rows)")
    print(f"   SO3       → secondary_cycle_segmentation.csv ({len(cycle_seg)} rows)")
    print(f"   SO4       → secondary_daily_consistency.csv ({len(daily_cons)} rows)")
    print()
    print(f"   Two-stage OOF RMSE (all shifts):   {bundle['oof_rmse']:.2f} L")
    print(f"   Regressor OOF RMSE (working only): {bundle['reg_oof_rmse']:.2f} L")
    print(f"   Holdout RMSE (unseen 10% dates):   {bundle['holdout_rmse']:.2f} L")
    print()
    print(f"   Submission stats:")
    print(f"     Mean   : {submission['fuel_volume'].mean():.1f} L")
    print(f"     Median : {submission['fuel_volume'].median():.1f} L")
    print(f"     Min    : {submission['fuel_volume'].min():.1f} L")
    print(f"     Max    : {submission['fuel_volume'].max():.1f} L")
    print(submission.head(10).to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()