"""
Configuration and constants for HaulMark fuel prediction system.
Contains all hyperparameters, file paths, and feature definitions.
"""

from pathlib import Path

# ─── DATA PATHS ─────────────────────────────────────────────────────────────
RAW = Path(r"C:\Users\dogra\Desktop\haulmark hackathon\data")

GPKG_FILES = {
    0: RAW / "mine_001_anonymized.gpkg",
    1: RAW / "mine_002_anonymized.gpkg",
}

TRAIN_TELEMETRY = [
    "telemetry_2026-01-01_2026-01-10.parquet",
    "telemetry_2026-01-11_2026-01-20.parquet",
    "telemetry_2026-02-01_2026-02-10.parquet",
    "telemetry_2026-02-11_2026-02-20.parquet",
    "telemetry_2026-03-01_2026-03-11.parquet",
]

TEST_TELEMETRY = [
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
]

SUMMARY_FILES = [
    "smry_jan_train_ordered.csv",
    "smry_feb_train_ordered.csv",
    "smry_mar_train_ordered.csv",
]

RFID_FILE = "rfid_refuels_2026-01-01_2026-03-31.parquet"

# ─── HYPERPARAMETERS ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Telemetry processing
DUMP_V_THRESH = 2.5  # Analog voltage threshold for dump detection
MAX_GAP_TIME = 300  # seconds - cap large time gaps
MAX_GAP_ACCEL = 120  # seconds - separate cap for acceleration
SPEED_MOVING = 1.0  # km/h - speed threshold for "moving"
ON_ROAD_THRESH_M = 50.0  # meters - distance threshold for on-road classification

# Cycle segmentation (haul trips)
MINE_DIST_THRESH = {0: 1200, 1: 600}  # meters - distance to complete one cycle
DEFAULT_DIST_THRESH = 800  # fallback threshold

# Constraints
TANK_CAP_MARGIN = 1.05  # Don't predict above this % of tank capacity
ZERO_FUEL_THRESH = 1.0  # liters - below this = non-working shift
RUNHRS_LEAK_CORR_CAP = 0.95  # Max correlation before dropping runhrs feature

# Cross-validation
N_CV_SPLITS = 5
WEIGHT_HOLDOUT_FRAC = 0.10  # Reserve this fraction of unique dates for holdout

# Model selection
USE_RF = False  # Include RandomForest in ensemble (slower)
N_ROUTE_CLUSTERS = 20  # Number of route clusters for benchmarking

# ─── FEATURE DEFINITIONS ────────────────────────────────────────────────────
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

GEO_FEATURES = [
    "n_geo_dump_entries", "time_in_dump_s",
    "mean_road_offset_m", "pct_on_road",
    "in_ml_boundary_pct", "mean_bench_dist_m",
]

HISTORY_FEATURES = [
    "veh_mean_fuel", "veh_std_fuel",
    "veh_lag1_fuel", "veh_lag2_fuel", "veh_roll3_fuel",
    "veh_shift_mean_fuel",
]

ALL_FEATURES_BASE = BASE_FEATURES + GEO_FEATURES + HISTORY_FEATURES

CAT_FEATURE_NAMES = ["vehicle_enc", "shift_enc", "mine_enc", "dump_switch_enc"]

# Route-level features (vehicle-independent for benchmarking)
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
