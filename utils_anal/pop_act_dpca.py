"""
Aggregate exported pop_act (T, N) by fg digit (0–9) × fg sector (0–8).

Mean activity per (digit, sector) cell → ``pop_act_dpca.npy`` shape (N, 10, 9), plus counts / meta.

Input: directory or explicit paths to pop_act.npy and labels.tsv from export_pop_act.py.
Sector mapping matches MC_RNN_Dataset / pop_act_umap (3×3 from fg_char_x, fg_char_y).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def sector_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    num_sectors: int = 9,
) -> np.ndarray:
    grid_size = int(np.sqrt(num_sectors))
    if grid_size * grid_size != num_sectors:
        raise ValueError(f"num_sectors={num_sectors} must be a perfect square")
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    col = np.clip((xf / max(width - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
    row = np.clip((yf / max(height - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
    return (row * grid_size + col).astype(np.int64)


def load_digit_and_xy(labels_tsv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    digits: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    with open(labels_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames is None:
            raise ValueError("Empty labels.tsv")
        for col in ("fg_char_id", "fg_char_x", "fg_char_y"):
            if col not in r.fieldnames:
                raise ValueError(f"labels.tsv missing {col}, got {r.fieldnames}")
        for row in r:
            digits.append(int(float(row["fg_char_id"])))
            xs.append(float(row["fg_char_x"]))
            ys.append(float(row["fg_char_y"]))
    return (
        np.asarray(digits, dtype=np.int64),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    )


def aggregate_pop_act_digit_sector(
    pop_act: np.ndarray,
    digits: np.ndarray,
    sectors: np.ndarray,
    n_digit: int = 10,
    n_sector: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        mean_n_ds: (N, n_digit, n_sector), float32; cells with no samples are NaN.
        counts_ds: (n_digit, n_sector), int64 — number of frames per cell.
    """
    if pop_act.ndim != 2:
        raise ValueError(f"pop_act must be (T, N), got {pop_act.shape}")
    T, N = pop_act.shape
    if digits.shape[0] != T or sectors.shape[0] != T:
        raise ValueError("digits/sectors length must match pop_act rows")

    n_bin = n_digit * n_sector
    sum_flat = np.zeros((n_bin, N), dtype=np.float64)
    cnt_flat = np.zeros(n_bin, dtype=np.int64)

    d_clamped = np.clip(digits, 0, n_digit - 1)
    s_clamped = np.clip(sectors, 0, n_sector - 1)
    idx = (d_clamped * n_sector + s_clamped).astype(np.int64, copy=False)
    pop64 = pop_act.astype(np.float64, copy=False)

    for t in range(T):
        b = int(idx[t])
        sum_flat[b] += pop64[t]
        cnt_flat[b] += 1

    sum_v = sum_flat.reshape(n_digit, n_sector, N)
    cnt = cnt_flat.reshape(n_digit, n_sector)
    mean_ds = np.full((n_digit, n_sector, N), np.nan, dtype=np.float64)
    mask = cnt > 0
    mean_ds[mask] = sum_v[mask] / cnt[mask][:, np.newaxis]
    mean_n_ds = np.transpose(mean_ds.astype(np.float32, copy=False), (2, 0, 1))
    return mean_n_ds, cnt


def aggregate_from_dir(
    pop_act_dir: str,
    frame_height: int,
    frame_width: int,
    num_sectors: int,
    out_mean_name: str = "pop_act_dpca.npy",
    out_counts_name: str = "pop_act_digitxsector_counts.npy",
    out_meta_name: str = "pop_act_digitxsector_meta.json",
) -> tuple[str, str, str]:
    pop_path = os.path.join(pop_act_dir, "pop_act.npy")
    lbl_path = os.path.join(pop_act_dir, "labels.tsv")
    if not os.path.isfile(pop_path):
        raise FileNotFoundError(pop_path)
    if not os.path.isfile(lbl_path):
        raise FileNotFoundError(lbl_path)

    pop = np.load(pop_path)
    digits, x, y = load_digit_and_xy(lbl_path)
    sectors = sector_from_xy(x, y, frame_height, frame_width, num_sectors)

    mean_n_ds, counts = aggregate_pop_act_digit_sector(
        pop, digits, sectors, n_digit=10, n_sector=num_sectors
    )

    out_mean = os.path.join(pop_act_dir, out_mean_name)
    out_counts = os.path.join(pop_act_dir, out_counts_name)
    out_meta = os.path.join(pop_act_dir, out_meta_name)
    np.save(out_mean, mean_n_ds)
    np.save(out_counts, counts)

    meta = {
        "shape_mean": list(mean_n_ds.shape),
        "shape_counts": list(counts.shape),
        "description": "mean_n_ds[i,d,s] = mean pop_act[:,i] over frames with fg digit d and sector s; NaN if empty",
        "pop_act_path": os.path.abspath(pop_path),
        "labels_path": os.path.abspath(lbl_path),
        "frame_height": frame_height,
        "frame_width": frame_width,
        "num_sectors": num_sectors,
        "n_nonempty_cells": int(np.sum(counts > 0)),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved mean (N,10,9) = {mean_n_ds.shape} -> {out_mean}")
    print(f"Saved counts (10,9) -> {out_counts}")
    print(f"Saved meta -> {out_meta}")
    return out_mean, out_counts, out_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean pop_act over fg digit × fg sector → (hidden, 10, 9)."
    )
    p.add_argument(
        "--pop_act_dir",
        type=str,
        required=True,
        help="Directory containing pop_act.npy and labels.tsv.",
    )
    p.add_argument("--frame_height", type=int, default=96)
    p.add_argument("--frame_width", type=int, default=96)
    p.add_argument("--num_sectors", type=int, default=9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_from_dir(
        args.pop_act_dir,
        args.frame_height,
        args.frame_width,
        args.num_sectors,
    )


if __name__ == "__main__":
    main()
