"""
Geometry parsing and geofencing functionality.
Handles mine geometries (dump sites, haul roads, boundaries, benches)
from GPKG files and provides spatial analysis for GPS data.
"""

import struct
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import KDTree


def _parse_gpkg_wkb_linestring_2d(blob: bytes) -> np.ndarray:
    """
    Parse WKB geometry from GPKG format.
    Handles linestrings, polygons, and multigeometries.
    Returns coordinates as (n_points, 2) array.
    """
    try:
        # GPKG uses WKB format with envelope info at the start
        flags = blob[3]
        env_indicator = (flags >> 1) & 0x07
        env_size = [0, 32, 48, 48, 64][min(env_indicator, 4)]
        wkb = blob[8 + env_size:]  # Skip GPKG envelope, get WKB part

        # Determine byte order (endianness)
        byte_order = wkb[0]
        bo = "<" if byte_order == 1 else ">"

        # Read geometry type and check if 2D or 3D
        wkb_type = struct.unpack_from(bo + "I", wkb, 1)[0]
        wkb_type_2d = wkb_type % 1000  # Strip Z/M info
        offset = 5

        # Extract coordinates based on geometry type
        if wkb_type_2d == 2:  # Linestring
            n_pts = struct.unpack_from(bo + "I", wkb, offset)[0]
            offset += 4
            coords = np.frombuffer(wkb[offset: offset + n_pts * 16], dtype=np.float64)
            if byte_order == 0:
                coords = coords.byteswap()
            return coords.reshape(n_pts, 2)

        elif wkb_type_2d == 3:  # Polygon (use outer ring only)
            _ = struct.unpack_from(bo + "I", wkb, offset)[0]  # num_rings
            offset += 4
            n_pts = struct.unpack_from(bo + "I", wkb, offset)[0]
            offset += 4
            coords = np.frombuffer(wkb[offset: offset + n_pts * 16], dtype=np.float64)
            if byte_order == 0:
                coords = coords.byteswap()
            return coords.reshape(n_pts, 2)

        elif wkb_type_2d in (1002, 1003):  # Multilinestring / Multipolygon
            n_pts = struct.unpack_from(bo + "I", wkb, offset)[0]
            offset += 4
            pts = []
            for _ in range(n_pts):
                x, y = struct.unpack_from(bo + "dd", wkb, offset)
                pts.append((x, y))
                offset += 24
            return np.array(pts, dtype=np.float64)

    except Exception:
        pass

    return np.empty((0, 2), dtype=np.float64)


def load_gpkg_geometries(gpkg_path: Path) -> Dict:
    """
    Load all mine geometry layers from GPKG spatial database file.

    Returns dict with:
     - "ob_dump": List of overburden dump site polygons
     - "haul_road": List of haul road linestrings
     - "ml_boundary": List of mine lease boundaries
     - "bench": List of bench/pit boundaries
     - "bench_kdtree": KDTree for fast nearest-bench lookup
     - "bench_pts": Array of bench segment midpoints
    """
    result = {
        "ob_dump": [],
        "haul_road": [],
        "ml_boundary": [],
        "bench": [],
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
            rows = con.execute(f'SELECT geom FROM "{table}"').fetchall()
            geoms = []

            for (blob,) in rows:
                if blob:
                    coords = _parse_gpkg_wkb_linestring_2d(blob)
                    if len(coords) >= 2:
                        geoms.append(coords)

            result[table] = geoms
            print(f"      {table}: {len(geoms)} geometries loaded")
        except Exception as e:
            print(f"      {table}: skipped ({e})")

    # Build KDTree for fast nearest-bench lookups
    if result["bench"]:
        bench_pts = np.vstack([
            (b[:-1] + b[1:]) / 2.0
            for b in result["bench"] if len(b) >= 2
        ])
        result["bench_kdtree"] = KDTree(bench_pts)
        result["bench_pts"] = bench_pts
        print(f"      bench KDTree built from {len(bench_pts):,} segment midpoints")

    con.close()
    return result


def _vec_point_in_polygon(px, py, poly):
    """
    Vectorized point-in-polygon test using ray casting algorithm.
    Returns boolean array: True = point inside polygon.
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
        inside[intersect] = ~inside[intersect]
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
        # Degenerate segment
        return (px - ax) ** 2 + (py - ay) ** 2

    # Project point onto infinite line, clamp to segment
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len_sq, 0.0, 1.0)

    # Return squared distance to closest point on segment
    return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2


def compute_geo_features_for_group(x_utm, y_utm, dt_time, geoms, on_road_thresh_m=50.0):
    """
    Compute geographic features for a single shift.
    Features: dump visits, time in dump, road distance, boundary presence, etc.

    Args:
        x_utm, y_utm: UTM coordinates of GPS points
        dt_time: Time intervals between points (seconds)
        geoms: Dictionary with geometry arrays from load_gpkg_geometries()
        on_road_thresh_m: Distance threshold for on-road classification

    Returns:
        Dict with geo features for this shift
    """
    n = len(x_utm)
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

    # Check if points are in dump sites
    in_dump = np.zeros(n, dtype=bool)
    for poly in geoms["ob_dump"]:
        if len(poly) < 3:
            continue

        # Quick bounding box check first
        mn_x, mn_y = poly[:, 0].min(), poly[:, 1].min()
        mx_x, mx_y = poly[:, 0].max(), poly[:, 1].max()
        cand = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)
        idx = np.where(cand)[0]

        if len(idx):
            in_dump[idx[_vec_point_in_polygon(x_utm[idx], y_utm[idx], poly)]] = True

    # Count dump entries and time in dump
    dump_entries = int(np.sum((~in_dump[:-1]) & in_dump[1:]))
    time_in_dump = float(dt_time[in_dump].sum())

    # Distance to haul roads
    road_dist = np.full(n, np.nan)
    if geoms["haul_road"]:
        all_dists = np.full((n, len(geoms["haul_road"])), np.inf)

        for r_idx, road in enumerate(geoms["haul_road"]):
            if len(road) < 2:
                continue

            # Buffer bbox by 500m to find candidates
            mn_x = road[:, 0].min() - 500
            mx_x = road[:, 0].max() + 500
            mn_y = road[:, 1].min() - 500
            mx_y = road[:, 1].max() + 500
            nearby = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)

            min_dsq = np.full(n, np.inf)
            idx = np.where(nearby)[0]

            if len(idx):
                # Find distance to nearest segment
                for seg in range(len(road) - 1):
                    ax, ay = road[seg]
                    bx, by = road[seg + 1]
                    dsq = _vec_point_to_segment_dist_sq(
                        x_utm[idx], y_utm[idx], ax, ay, bx, by)
                    min_dsq[idx] = np.minimum(min_dsq[idx], dsq)

            all_dists[:, r_idx] = np.sqrt(min_dsq)

        road_dist = all_dists.min(axis=1)

    # Road metrics
    valid_road = np.isfinite(road_dist)
    mean_road_offset = float(road_dist[valid_road].mean()) if valid_road.any() else np.nan
    pct_on_road = float((road_dist[valid_road] <= on_road_thresh_m).mean()) if valid_road.any() else np.nan

    # Check if points are within mine lease boundary
    in_boundary = np.zeros(n, dtype=bool)
    if geoms["ml_boundary"]:
        poly = geoms["ml_boundary"][0]
        if len(poly) >= 3:
            mn_x, mn_y = poly[:, 0].min(), poly[:, 1].min()
            mx_x, mx_y = poly[:, 0].max(), poly[:, 1].max()
            cand = (x_utm >= mn_x) & (x_utm <= mx_x) & (y_utm >= mn_y) & (y_utm <= mx_y)
            idx = np.where(cand)[0]
            if len(idx):
                in_boundary[idx[_vec_point_in_polygon(x_utm[idx], y_utm[idx], poly)]] = True

    in_ml_pct = float(in_boundary.mean()) if n > 0 else np.nan

    # Distance to nearest bench
    mean_bench = np.nan
    if geoms["bench_kdtree"] is not None:
        dists, _ = geoms["bench_kdtree"].query(
            np.column_stack([x_utm, y_utm]), k=1, workers=-1)
        mean_bench = float(dists.mean())

    # Spatial fingerprint for route clustering
    centroid_x = float(x_utm.mean())
    centroid_y = float(y_utm.mean())
    start_x = float(x_utm[0])
    start_y = float(y_utm[0])
    end_x = float(x_utm[-1])
    end_y = float(y_utm[-1])

    return {
        "n_geo_dump_entries": dump_entries,
        "time_in_dump_s": time_in_dump,
        "mean_road_offset_m": mean_road_offset,
        "pct_on_road": pct_on_road,
        "in_ml_boundary_pct": in_ml_pct,
        "mean_bench_dist_m": mean_bench,
        "shift_centroid_x": centroid_x,
        "shift_centroid_y": centroid_y,
        "shift_start_x": start_x,
        "shift_start_y": start_y,
        "shift_end_x": end_x,
        "shift_end_y": end_y,
    }
