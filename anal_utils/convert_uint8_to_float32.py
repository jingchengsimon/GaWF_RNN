#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path

import numpy as np
import numpy.lib.format as npfmt


def convert_npy_uint8_to_float32(src: Path, dst: Path, chunk_size: int = 4096):
    """
    Convert a large uint8 .npy file to float32 using memmap + chunked copy.

    src: original uint8 .npy path
    dst: new float32 .npy path
    chunk_size: number of frames per chunk along axis 0
    """
    print(f"[NPY] Converting {src} -> {dst}")
    arr = np.load(src, mmap_mode="r")
    if arr.dtype != np.uint8:
        print(f"  - dtype is {arr.dtype}, not uint8, skip conversion (copy as is).")
        # Just copy file if you want identical dtype; or skip entirely.
        shutil.copy2(src, dst)
        return

    shape = arr.shape
    print(f"  - shape: {shape}, dtype: {arr.dtype}")
    # Create target .npy memmap with same shape but float32
    # NOTE: must use numpy.lib.format.open_memmap so that the file
    # is a valid .npy (np.load can read it later).
    dst_mm = npfmt.open_memmap(
        dst,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    n = shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f"  - chunk {start}:{end} ...", end="\r")
        dst_mm[start:end] = arr[start:end].astype(np.float32)

    # Flush to disk
    del dst_mm
    print(f"\n  - done.")


def convert_pair(stim_npy: Path, tsv: Path, suffix: str = "-float32"):
    """
    Given stimulus_reg-*.npy and corresponding .tsv, create float32 versions
    with suffix appended before extension.
    """
    # New file names: xxx.npy -> xxx-float32.npy, same for .tsv
    npy_dst = stim_npy.with_name(stim_npy.stem + suffix + stim_npy.suffix)
    tsv_dst = tsv.with_name(tsv.stem + suffix + tsv.suffix)

    # 1) Convert .npy
    convert_npy_uint8_to_float32(stim_npy, npy_dst)

    # 2) Copy .tsv
    if tsv.exists():
        print(f"[TSV] Copy {tsv} -> {tsv_dst}")
        shutil.copy2(tsv, tsv_dst)
    else:
        print(f"[TSV] Warning: {tsv} not found, skip.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert uint8 stimulus_reg-*.npy to float32 and copy TSV with -float32 suffix."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing stimulus_reg-*.npy and stimulus_reg-*.tsv",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="-float32",
        help="Suffix to append before extensions (default: -float32)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="stimulus_reg-*.npy",
        help="Glob pattern for stimulus npy files (default: stimulus_reg-*.npy)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Number of frames per chunk when converting (default: 4096)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir {data_dir} is not a directory")

    npy_files = sorted(data_dir.glob(args.pattern))
    if not npy_files:
        print(f"No files matched pattern {args.pattern} in {data_dir}")
        return

    print(f"Found {len(npy_files)} npy files.")
    for npy_path in npy_files:
        # Infer corresponding TSV: same stem, .tsv
        tsv_path = npy_path.with_suffix(".tsv")
        convert_pair(npy_path, tsv_path, suffix=args.suffix)


if __name__ == "__main__":
    main()