# #!/usr/bin/env python3
# """
# Pre-processing for US coast NDBC-style hourly text files.
#
# Script location (save as):
#     Pre_processing/pre_processing_UScoast.py
#
# Default data layout:
#     Raw:       data\\UScoast_raw\\41001h2012.txt\\41001h2012.txt
#     Processed: data\\UScoast_processed\\41001h2012.txt\\41001h2012.txt
# """
# import argparse
# import math
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import Optional
#
# COLS = [
#     "YY", "MM", "DD", "hh", "mm",
#     "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
#     "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
# ]
#
# DEFAULT_INPUT  = Path("../data/UScoast_raw/41001h2012.txt/41001h2012.txt")
# DEFAULT_OUTPUT = Path("../data/UScoast_processed/41001h2012.txt/41001h2012.txt")
# DEFAULT_MWD_FALLBACK = 171.563213
# DEFAULT_ROLLING_WINDOW = 3
#
# def _to_float_df(df: pd.DataFrame) -> pd.DataFrame:
#     for c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     return df
#
# def _replace_invalid_with_nan(df: pd.DataFrame) -> pd.DataFrame:
#     sentinels = {99, 99.0, 999, 999.0, 9999, 9999.0}
#     for col in df.columns:
#         df[col] = df[col].apply(lambda x: np.nan if x in sentinels else x)
#     return df
#
# def _circular_mean_deg(series: pd.Series) -> Optional[float]:
#     vals = series.dropna().to_numpy(dtype=float)
#     if vals.size == 0:
#         return np.nan
#     ang = np.deg2rad(vals)
#     x = np.cos(ang).mean()
#     y = np.sin(ang).mean()
#     mean = np.degrees(np.arctan2(y, x))
#     if mean < 0:
#         mean += 360.0
#     return float(mean)
#
# def _impute_mwd(df: pd.DataFrame, window: int, default_mwd: Optional[float]) -> pd.DataFrame:
#     if "MWD" not in df:
#         return df
#     if window and window > 1:
#         rolled = (
#             df["MWD"]
#             .rolling(window=window, min_periods=1, center=False)
#             .apply(lambda x: _circular_mean_deg(pd.Series(x)), raw=False)
#         )
#         df["MWD"] = df["MWD"].fillna(rolled)
#     if default_mwd is not None:
#         df["MWD"] = df["MWD"].fillna(float(default_mwd))
#     return df
#
# def load_ndbc_txt(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(
#         path,
#         delim_whitespace=True,
#         comment="#",
#         header=None,
#         names=COLS,
#         dtype=str,
#         engine="python",
#     )
#     df = _to_float_df(df)
#     df = _replace_invalid_with_nan(df)
#     return df
#
# def format_like_exemplar(df: pd.DataFrame) -> pd.DataFrame:
#     return df[COLS].astype(float)
#
# def save_space_separated(df: pd.DataFrame, out_path: Path) -> None:
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     lines = []
#     for _, row in df.iterrows():
#         fields = ["NaN" if pd.isna(v) else f"{float(v):.6f}" for v in row]
#         lines.append(" ".join(fields))
#     out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="US coast pre-processing for NDBC hourly text files.")
#     p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to raw .txt file")
#     p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write processed table")
#     p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW,
#                    help="Rolling window for circular mean of MWD")
#     p.add_argument("--default-mwd", type=float, default=DEFAULT_MWD_FALLBACK,
#                    help="Fallback value for MWD when unknown")
#     return p.parse_args()
#
# def main():
#     args = parse_args()
#     df = load_ndbc_txt(args.input)
#     df = _impute_mwd(df, window=args.rolling_window, default_mwd=args.default_mwd)
#     df = format_like_exemplar(df)
#     save_space_separated(df, args.output)
#     with pd.option_context("display.max_columns", None, "display.width", 160):
#         print(df.head(10))
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Pre-processing for US coast NDBC-style hourly text files.

This script can run in:
1) Single-file mode (default)
2) Batch mode: process all *.txt under a given root folder, mirroring the structure in an output folder.

Example batch mode mapping:
  data/UScoast_raw/41001h2012.txt/41001h2012.txt
  ->
  data/UScoast_processed/41001h2012.txt/41001h2012.txt
"""
#!/usr/bin/env python3
"""
Pre-processing for US coast NDBC-style hourly text files.

This script can run in:
1) Single-file mode (default)
2) Batch mode: process all *.txt under a given root folder, mirroring the structure in an output folder.

Example batch mode mapping:
  data/UScoast_raw/41001h2012.txt/41001h2012.txt
  ->
  data/UScoast_processed/41001h2012.txt/41001h2012.txt
"""
#!/usr/bin/env python3
"""
Pre-processing for US coast NDBC-style hourly text files.

This script can run in:
1) Single-file mode (default)
2) Batch mode: process all *.txt under a given root folder, mirroring the structure in an output folder.

Example batch mode mapping:
  data/UScoast_raw/41001h2012.txt/41001h2012.txt
  ->
  data/UScoast_processed/41001h2012.txt/41001h2012.txt
"""
#!/usr/bin/env python3
"""
Pre-processing for US coast NDBC-style hourly text files.

Behaviour (no flags needed):
- By default, recursively scan the raw root folder for all *.txt files and write processed
  outputs to the mirrored structure under the processed root folder.
- No --input/--output single-file mode is used.

Example mapping:
  <project>/data/UScoast_raw/41001h2012.txt/41001h2012.txt
  ->
  <project>/data/UScoast_processed/41001h2012.txt/41001h2012.txt
"""
import argparse
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # assumes this file is at <project>/Pre_processing/
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/UScoast_raw"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/UScoast_processed"

# -----------------------------------------------------------------------------
# Data schema
# -----------------------------------------------------------------------------
COLS = [
    "YY", "MM", "DD", "hh", "mm",
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]

DEFAULT_MWD_FALLBACK = 171.563213
DEFAULT_ROLLING_WINDOW = 3

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _to_float_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _replace_invalid_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Column-agnostic replacement of literal sentinels with NaN
    sentinels = {99, 99.0, 999, 999.0, 9999, 9999.0}
    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if x in sentinels else x)
    return df


def _circular_mean_deg(series: pd.Series) -> Optional[float]:
    vals = series.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan
    ang = np.deg2rad(vals)
    x = np.cos(ang).mean()
    y = np.sin(ang).mean()
    mean = np.degrees(np.arctan2(y, x))
    if mean < 0:
        mean += 360.0
    return float(mean)


def _impute_mwd(df: pd.DataFrame, window: int, default_mwd: Optional[float]) -> pd.DataFrame:
    if "MWD" not in df:
        return df
    if window and window > 1:
        rolled = (
            df["MWD"]
            .rolling(window=window, min_periods=1, center=False)
            .apply(lambda x: _circular_mean_deg(pd.Series(x)), raw=False)
        )
        df["MWD"] = df["MWD"].fillna(rolled)
    if default_mwd is not None:
        df["MWD"] = df["MWD"].fillna(float(default_mwd))
    return df


def load_ndbc_txt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        header=None,
        names=COLS,
        dtype=str,
        engine="python",
    )
    df = _to_float_df(df)
    df = _replace_invalid_with_nan(df)
    return df


def format_like_exemplar(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLS].astype(float)


def save_space_separated(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, row in df.iterrows():
        fields = ["NaN" if pd.isna(v) else f"{float(v):.6f}" for v in row]
        lines.append(" ".join(fields))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one(in_path: Path, out_path: Path, rolling_window: int, default_mwd: Optional[float]) -> None:
    df = load_ndbc_txt(in_path)
    df = _impute_mwd(df, window=rolling_window, default_mwd=default_mwd)
    df = format_like_exemplar(df)
    save_space_separated(df, out_path)


def iter_txt_files(root: Path) -> Iterable[Path]:
    # Recursively yield all *.txt files under root
    yield from root.rglob("*.txt")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US coast pre-processing for NDBC hourly text files (batch by default).")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Root folder containing raw *.txt trees")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Root folder to mirror outputs into")
    p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW,
                   help="Rolling window for circular mean of MWD")
    p.add_argument("--default-mwd", type=float, default=DEFAULT_MWD_FALLBACK,
                   help="Fallback value for MWD when unknown")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()

    count_ok, count_fail = 0, 0
    for in_file in iter_txt_files(raw_root):
        if not in_file.is_file():
            continue
        rel = in_file.relative_to(raw_root)
        out_file = out_root / rel
        try:
            process_one(in_file, out_file, args.rolling_window, args.default_mwd)
            count_ok += 1
        except Exception as e:
            count_fail += 1
            print(f"[FAIL] {in_file} -> {out_file}: {e}")
    print(f"Processed OK: {count_ok}, Failed: {count_fail}, Root: {raw_root} -> {out_root}")


if __name__ == "__main__":
    main()
