#!/usr/bin/env python3
"""
AnchorGK_final_clean.py

A cleaned, argument-driven pipeline to:
  1) Load a chosen dataset (UScoast/NDBC, Tehiku, Shenzhen)
  2) Build feature-wise subgraphs (via MOESTKF_functions)
  3) Construct polygons and fine-grained partitions per (king-station, feature)
  4) Derive adjacency matrices using (weighted) correlations within subgraphs
  5) Produce inverse-distance-weighted (IDW) interpolations
  6) Package outputs (values, adjacencies, ground-truth) and persist as .npy

Notes
-----
- This file intentionally removes long commented blocks and exposes configuration
  via CLI flags or a JSON/YAML config passed with --input.
- It keeps the original dependency on MOESTKF_functions for:
    - Toy_generation
    - get_complete_stations
    - Feature_wise_Subgraph
- The pipeline focuses on graph construction and data packaging. Model training
  (TensorFlow/Torch Geometric) is deliberately excluded for clarity and because
  environments differ.
- Station/value array conventions:
    station_info: (N, 3) -> [station_id, lon, lat]
    station_value: (N, T, F)

Outputs (all under --out-dir):
  - value_all_store_tensor_reshape.npy   # (K_sel, F, T, K+1)
  - adj_all_store_reshaped.npy           # (K_sel, F, K+1, K+1)
  - true_store_reshaped.npy              # (K_sel, F, T)
  - meta.json                            # Shapes, chosen indices, params, columns

Example
-------
python AnchorGK_final_clean.py \
  --dataset UScoast \
  --ndbc-values data/NDBC/all.npy \
  --ndbc-stations data/NDBC/Station_info_edit.csv \
  --K 5 --num-subdivisions 5 --king-select 5 \
  --out-dir outputs/UScoast_run1

# Or with a config file containing the same keys (JSON/YAML):
python AnchorGK_final_clean.py --input configs/uscoast.json
"""
from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from shapely.geometry import Point, Polygon

# Import project-specific utilities
from MOESTKF_functions import (
    Toy_generation,
    get_complete_stations,
    Feature_wise_Subgraph,
)

# -----------------------------
# Dataclasses / Config
# -----------------------------

@dataclass
class DataPaths:
    # UScoast/NDBC
    ndbc_values: Optional[str] = None
    ndbc_stations: Optional[str] = None
    # Tehiku
    tehiku_coord_xlsx: Optional[str] = None
    tehiku_value_dir: Optional[str] = None
    # Shenzhen
    shenzhen_value_dir: Optional[str] = None
    shenzhen_station_csv: Optional[str] = None

@dataclass
class HParams:
    K: int = 5  # # of neighbours per subgraph (excluding the king station)
    num_subdivisions: int = 5  # grid splits along x/y for fine polygons
    king_select: int = 5  # # of king stations to select for packaging
    weight_scalar: float = 0.2  # temporal weight for weighted correlation
    seed: int = 42

@dataclass
class RunConfig:
    dataset: str = "UScoast"  # one of: UScoast, NDBC, Tehiku, Shenzhen
    paths: DataPaths = DataPaths()
    hparams: HParams = HParams()
    out_dir: str = "outputs/run"

# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


# -----------------------------
# Dataset loaders
# -----------------------------

def load_uscoast_ndbc(values_path: Path, stations_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the NDBC/UScoast variant used in your experiments.

    Returns
    -------
    station_info : (N, 3) -> [station_id(str), lon(float), lat(float)]
    station_value: (N, T, F)
    """
    arr = np.load(values_path)  # original code: (T?, N?, F?) or similar
    # Original snippet: station_value = np.load(file_path); .transpose(2,0,1); [:,:,5:13]
    # We standardise to (N, T, F)
    station_value = arr.transpose(2, 0, 1)  # (N, T, F_total)
    station_value = station_value[:, :, 5:13]  # keep 8 features as per your note

    # Station info CSV without header; columns: [id, lon, lat]
    station_info = pd.read_csv(stations_csv, header=None).values

    # Normalise types: id -> str, lon/lat -> float
    station_info = np.stack([
        station_info[:, 0].astype(str),
        station_info[:, 1].astype(float),
        station_info[:, 2].astype(float),
    ], axis=1)
    return station_info, station_value


def load_tehiku(coords_xlsx: Path, value_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    # station_info
    station_info = pd.read_excel(coords_xlsx).values  # expects [id, lon, lat]
    station_info = np.stack([
        station_info[:, 0].astype(str),
        station_info[:, 1].astype(float),
        station_info[:, 2].astype(float),
    ], axis=1)

    # values: iterate *.xlsx -> for each file: stack sheet numeric columns
    xlsx_files = sorted([p for p in value_dir.iterdir() if p.suffix.lower() == ".xlsx"])
    data_list: List[np.ndarray] = []
    for p in xlsx_files:
        xl = pd.ExcelFile(p)
        sheet_arrays = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            # adapt to your numeric columns if needed
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                continue
            sheet_arrays.append(df[numeric_cols].values.astype(np.float32))
        if sheet_arrays:
            stacked = np.stack(sheet_arrays, axis=0)  # (S, T, F)
            data_list.append(stacked)
    if not data_list:
        raise RuntimeError("No valid Tehiku Excel files.")
    tensor_data = np.stack(data_list, axis=0)  # (Files, S, T, F)
    station_value = tensor_data.squeeze(axis=1)  # (Files, T, F) -> treat Files as stations
    return station_info, station_value


def load_shenzhen(value_dir: Path, station_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    station_info = pd.read_csv(station_csv).values  # expects [id, lon, lat]
    station_info = np.stack([
        station_info[:, 0].astype(str),
        station_info[:, 1].astype(float),
        station_info[:, 2].astype(float),
    ], axis=1)

    csv_files = sorted([p for p in value_dir.iterdir() if p.suffix.lower() == ".csv"])
    data_list: List[np.ndarray] = []
    for p in csv_files:
        df = pd.read_csv(p)
        numeric = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        data_list.append(numeric)
    if not data_list:
        raise RuntimeError("No valid Shenzhen CSV files.")
    station_value = np.stack(data_list, axis=0)  # (N, T, F)
    return station_info, station_value


# -----------------------------
# Graph building utilities
# -----------------------------

def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return [points[idx] for idx in hull.vertices]


def create_polygon_matrix(complete_sub_matrix: np.ndarray,
                          station_coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Return array (K_stations, F) of shapely Polygons covering each subgraph."""
    out = np.empty((complete_sub_matrix.shape[0], complete_sub_matrix.shape[1]), dtype=object)
    for i in range(complete_sub_matrix.shape[0]):
        for j in range(complete_sub_matrix.shape[1]):
            stations = complete_sub_matrix[i, j].tolist()
            coords = [station_coords[sid] for sid in stations if sid in station_coords]
            if len(coords) > 2:
                hull_coords = convex_hull(coords)
                out[i, j] = Polygon(hull_coords)
            else:
                out[i, j] = Polygon(coords)
    return out


def subdivide_polygon(polygon: Polygon, num_subdivisions: int) -> List[Polygon]:
    minx, miny, maxx, maxy = polygon.bounds
    width = (maxx - minx) / num_subdivisions if num_subdivisions > 0 else (maxx - minx)
    height = (maxy - miny) / num_subdivisions if num_subdivisions > 0 else (maxy - miny)
    sub_polygons: List[Polygon] = []
    if width == 0 or height == 0:
        return [polygon]
    for i in range(num_subdivisions):
        for j in range(num_subdivisions):
            sub_poly = Polygon([
                (minx + i * width,     miny + j * height),
                (minx + (i+1) * width, miny + j * height),
                (minx + (i+1) * width, miny + (j+1) * height),
                (minx + i * width,     miny + (j+1) * height),
            ])
            if sub_poly.intersects(polygon):
                sub_polygons.append(sub_poly.intersection(polygon))
    return sub_polygons or [polygon]


def create_fine_grained_polygons(poly_mat: np.ndarray, num_subdivisions: int) -> np.ndarray:
    fine = np.empty_like(poly_mat, dtype=object)
    for i in range(poly_mat.shape[0]):
        for j in range(poly_mat.shape[1]):
            poly: Polygon = poly_mat[i, j]
            fine[i, j] = subdivide_polygon(poly, num_subdivisions) if poly.is_valid else [poly]
    return fine


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    w = weights
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov_xy = np.average((x - wx) * (y - wy), weights=w)
    cov_xx = np.average((x - wx) ** 2, weights=w)
    cov_yy = np.average((y - wy) ** 2, weights=w)
    denom = math.sqrt(cov_xx * cov_yy) if cov_xx > 0 and cov_yy > 0 else 0.0
    return float(cov_xy / denom) if denom > 0 else 0.0


def generate_adj_by_corr(fine_polys: np.ndarray,
                          complete_sub_matrix: np.ndarray,
                          station_info: np.ndarray,
                          station_value: np.ndarray,
                          K: int,
                          weight_scalar: float) -> np.ndarray:
    """Return (K_stations, F) array of lists, each list containing adjacency matrices
    (len = #fine-polygons for that (i,j)). Each adjacency is (K+1, K+1).
    """
    # Map station id -> row index
    id_to_idx: Dict[str, int] = {str(s[0]): i for i, s in enumerate(station_info)}

    out = np.empty_like(fine_polys, dtype=object)
    for i in range(fine_polys.shape[0]):
        for j in range(fine_polys.shape[1]):
            polys: List[Polygon] = fine_polys[i, j]
            if not polys:
                out[i, j] = []
                continue
            stations: List[str] = [str(s) for s in complete_sub_matrix[i, j]]  # length K+1 (king + K)
            idxs: List[int] = [id_to_idx[sid] for sid in stations if sid in id_to_idx]
            if len(idxs) < 2:
                out[i, j] = [np.eye(len(idxs), dtype=float)]
                continue
            # time series for this feature j
            series = station_value[idxs, :, j]  # (K+1, T)
            T = series.shape[1]
            w = np.full(T, float(weight_scalar), dtype=float)
            # Build one adjacency per fine polygon (same correlation template per (i,j))
            base_adj = np.zeros((len(idxs), len(idxs)), dtype=float)
            for a in range(len(idxs)):
                for b in range(len(idxs)):
                    if a == b:
                        base_adj[a, b] = 1.0
                    else:
                        base_adj[a, b] = weighted_correlation(series[a], series[b], w)
            out[i, j] = [base_adj.copy() for _ in polys]
    return out


def idw_interpolate(target_xy: Tuple[float, float],
                    all_xy: np.ndarray,  # (K+1, 2)
                    all_series: np.ndarray  # (K+1, T)
                    ) -> np.ndarray:
    # distances to stations
    dists = np.linalg.norm(all_xy - np.asarray(target_xy, dtype=float), axis=1)
    dists = np.where(dists == 0.0, 1e-10, dists)
    w = 1.0 / dists
    w /= w.sum()
    return (w.reshape(-1, 1) * all_series).sum(axis=0)


# -----------------------------
# Pipeline
# -----------------------------

def build_and_package(station_info: np.ndarray,
                      station_value: np.ndarray,
                      K: int,
                      num_subdivisions: int,
                      king_select: int,
                      weight_scalar: float,
                      out_dir: Path) -> None:
    set_seed(42)
    ensure_dir(out_dir)

    # 1) Identify stations with complete data and build feature-wise subgraphs
    complete_stations, complete_indices = get_complete_stations(station_info, station_value)
    subgraph_matrix = Feature_wise_Subgraph(station_info, station_value,
                                            complete_stations, complete_indices, K)
    # complete_sub_matrix: insert the king station id at position 0 across features
    complete_sub_matrix = np.empty((subgraph_matrix.shape[0], subgraph_matrix.shape[1], K+1), dtype=object)
    for i, king_id in enumerate(complete_stations):
        for j in range(subgraph_matrix.shape[1]):
            neighbours = list(subgraph_matrix[i, j])[:K]  # safety slice
            complete_sub_matrix[i, j] = np.array([king_id] + neighbours, dtype=object)

    # 2) Polygons
    station_coords: Dict[str, Tuple[float, float]] = {
        str(row[0]): (float(row[1]), float(row[2])) for row in station_info
    }
    poly_mat = create_polygon_matrix(complete_sub_matrix, station_coords)
    fine_polys = create_fine_grained_polygons(poly_mat, num_subdivisions)

    # 3) Adjacency matrices using weighted correlations
    adj_lists = generate_adj_by_corr(
        fine_polys, complete_sub_matrix, station_info, station_value, K, weight_scalar
    )

    # 4) IDW interpolation & packaging per selected king stations
    #    Select king stations that contain a random target in their coarse polygon
    F = station_value.shape[2]
    all_king_rows = list(range(poly_mat.shape[0]))

    # Pick a random station as global target
    target_index = random.choice(range(station_info.shape[0]))
    target_xy = (float(station_info[target_index, 1]), float(station_info[target_index, 2]))
    target_point = Point(target_xy)

    selected_rows: List[int] = []
    # 1) Prefer rows whose coarse polygon (feature 0) contains the target
    for r in all_king_rows:
        poly0: Polygon = poly_mat[r, 0]
        if poly0.contains(target_point):
            selected_rows.append(r)
        if len(selected_rows) >= king_select:
            break
    # 2) If not enough, top-up by nearest centroids
    if len(selected_rows) < king_select:
        dists: List[Tuple[int, float]] = []
        for r in all_king_rows:
            if r in selected_rows:
                continue
            c = poly_mat[r, 0].centroid
            dists.append((r, Point(c.x, c.y).distance(target_point)))
        dists.sort(key=lambda x: x[1])
        for r, _ in dists:
            if r not in selected_rows:
                selected_rows.append(r)
            if len(selected_rows) >= king_select:
                break

    # Build lookup id->row index for value extraction
    id_to_idx = {str(s[0]): i for i, s in enumerate(station_info)}

    value_all_store: List[np.ndarray] = []  # will be (T, K+1) per (row, feature)
    adj_all_store: List[np.ndarray] = []    # will be (K+1, K+1) per (row, feature)
    true_store: List[np.ndarray] = []       # will be (T,) per (row, feature)

    for r in selected_rows:
        for f in range(F):
            station_ids = [str(s) for s in complete_sub_matrix[r, f]]  # (K+1,)
            idxs = [id_to_idx[sid] for sid in station_ids if sid in id_to_idx]
            if len(idxs) == 0:
                continue
            xy = np.array([(station_info[i, 1], station_info[i, 2]) for i in idxs], dtype=float)
            series = station_value[idxs, :, f]  # (K+1, T)

            # Choose one of the K+1 as the target for interpolation error (here: the king at idx 0)
            target_series = series[0]
            target_xy_local = tuple(xy[0])

            # IDW using all nodes (including target); you may exclude target by slicing 1: if desired
            idw_est = idw_interpolate(target_xy_local, xy, series)

            # Retrieve one adjacency for this (r,f) (first fine polygon)
            adj = adj_lists[r, f][0] if isinstance(adj_lists[r, f], list) and len(adj_lists[r, f]) else np.eye(len(idxs))

            # Package (T, K+1)
            merged = series.transpose(1, 0)  # (T, K+1)

            value_all_store.append(merged)
            adj_all_store.append(adj)
            true_store.append(target_series)

    if not value_all_store:
        raise RuntimeError("No packaged data was produced; check selections and inputs.")

    # Stack/reshape to tensors
    # shapes per entry: merged (T, K+1), adj (K+1, K+1), true (T,)
    K_sel = len(selected_rows)
    F_used = F
    T_len = value_all_store[0].shape[0]
    Kp1 = value_all_store[0].shape[1]

    value_tensor = np.stack(value_all_store, axis=0).reshape(K_sel, F_used, T_len, Kp1)
    adj_tensor = np.stack(adj_all_store, axis=0).reshape(K_sel, F_used, Kp1, Kp1)
    true_tensor = np.stack(true_store, axis=0).reshape(K_sel, F_used, T_len)

    # Save
    np.save(out_dir / "value_all_store_tensor_reshape.npy", value_tensor)
    np.save(out_dir / "adj_all_store_reshaped.npy", adj_tensor)
    np.save(out_dir / "true_store_reshaped.npy", true_tensor)

    meta = {
        "selected_rows": selected_rows,
        "target_index": int(target_index),
        "shapes": {
            "value": list(value_tensor.shape),
            "adj": list(adj_tensor.shape),
            "true": list(true_tensor.shape),
        },
        "params": {
            "K": K,
            "num_subdivisions": num_subdivisions,
            "king_select": king_select,
            "weight_scalar": weight_scalar,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


# -----------------------------
# Argparse / Config IO
# -----------------------------

def load_config_from_file(cfg_path: Path) -> Dict:
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # optional
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="AnchorGK final cleaned pipeline")
    p.add_argument("--input", type=str, default=None,
                   help="Path to JSON/YAML config containing keys: dataset, paths, hparams, out_dir")

    # Direct CLI (overrides config if provided)
    p.add_argument("--dataset", type=str, default=None,
                   choices=["UScoast", "NDBC", "Tehiku", "Shenzhen"],
                   help="Dataset name")

    # Paths (optional; only those relevant to the chosen dataset are used)
    p.add_argument("--ndbc-values", type=str, default=None)
    p.add_argument("--ndbc-stations", type=str, default=None)

    p.add_argument("--tehiku-coord-xlsx", type=str, default=None)
    p.add_argument("--tehiku-value-dir", type=str, default=None)

    p.add_argument("--shenzhen-value-dir", type=str, default=None)
    p.add_argument("--shenzhen-station-csv", type=str, default=None)

    # HParams
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--num-subdivisions", type=int, default=5)
    p.add_argument("--king-select", type=int, default=5)
    p.add_argument("--weight-scalar", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-dir", type=str, default="outputs/run")

    args = p.parse_args()

    # Load config file if given
    cfg = RunConfig()
    if args.input is not None:
        cfg_dict = load_config_from_file(Path(args.input))
        # Loose parsing to dataclasses
        cfg = RunConfig(**{
            **asdict(cfg),
            **cfg_dict,
        })

    # CLI overrides
    if args.dataset is not None:
        cfg.dataset = args.dataset
    cfg.paths = DataPaths(
        ndbc_values=args.ndbc_values or (cfg.paths.ndbc_values if cfg.paths else None),
        ndbc_stations=args.ndbc_stations or (cfg.paths.ndbc_stations if cfg.paths else None),
        tehiku_coord_xlsx=args.tehiku_coord_xlsx or (cfg.paths.tehiku_coord_xlsx if cfg.paths else None),
        tehiku_value_dir=args.tehiku_value_dir or (cfg.paths.tehiku_value_dir if cfg.paths else None),
        shenzhen_value_dir=args.shenzhen_value_dir or (cfg.paths.shenzhen_value_dir if cfg.paths else None),
        shenzhen_station_csv=args.shenzhen_station_csv or (cfg.paths.shenzhen_station_csv if cfg.paths else None),
    )
    cfg.hparams = HParams(
        K=args.K if args.K is not None else cfg.hparams.K,
        num_subdivisions=args.num_subdivisions if args.num_subdivisions is not None else cfg.hparams.num_subdivisions,
        king_select=args.king_select if args.king_select is not None else cfg.hparams.king_select,
        weight_scalar=args.weight_scalar if args.weight_scalar is not None else cfg.hparams.weight_scalar,
        seed=args.seed if args.seed is not None else cfg.hparams.seed,
    )
    cfg.out_dir = args.out_dir or cfg.out_dir

    return cfg


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.hparams.seed)

    ds = cfg.dataset.lower()
    paths = cfg.paths

    if ds in ("uscoast", "ndbc"):
        if not paths.ndbc_values or not paths.ndbc_stations:
            raise ValueError("For UScoast/NDBC, provide --ndbc-values and --ndbc-stations")
        station_info, station_value = load_uscoast_ndbc(Path(paths.ndbc_values), Path(paths.ndbc_stations))
    elif ds == "tehiku":
        if not paths.tehiku_coord_xlsx or not paths.tehiku_value_dir:
            raise ValueError("For Tehiku, provide --tehiku-coord-xlsx and --tehiku-value-dir")
        station_info, station_value = load_tehiku(Path(paths.tehiku_coord_xlsx), Path(paths.tehiku_value_dir))
    elif ds == "shenzhen":
        if not paths.shenzhen_value_dir or not paths.shenzhen_station_csv:
            raise ValueError("For Shenzhen, provide --shenzhen-value-dir and --shenzhen-station-csv")
        station_info, station_value = load_shenzhen(Path(paths.shenzhen_value_dir), Path(paths.shenzhen_station_csv))
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    out_dir = Path(cfg.out_dir)
    build_and_package(
        station_info=station_info,
        station_value=station_value,
        K=cfg.hparams.K,
        num_subdivisions=cfg.hparams.num_subdivisions,
        king_select=cfg.hparams.king_select,
        weight_scalar=cfg.hparams.weight_scalar,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
