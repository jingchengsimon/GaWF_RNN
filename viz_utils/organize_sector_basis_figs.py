"""
One-off script to organize gawf_sector_basis_figs: move each sector_k_*.png and
digit_k_*.png into subdirs sector_k/ and digit_k/.
Run from repo root: python -m viz_utils.organize_sector_basis_figs [--figs_dir DIR]
"""

import argparse
import os
import re


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize sector/digit PNGs into subdirs.")
    parser.add_argument(
        "--figs_dir",
        type=str,
        default="./gawf_sector_basis_figs",
        help="Directory containing sector_* and digit_* PNG files.",
    )
    args = parser.parse_args()
    figs_dir = os.path.abspath(args.figs_dir)
    if not os.path.isdir(figs_dir):
        print(f"Not a directory: {figs_dir}")
        return

    # Match sector_0_*.png or digit_7_*.png (only at top level)
    pattern = re.compile(r"^(sector|digit)_(\d+)_(.+\.png)$")
    moved = 0
    for name in os.listdir(figs_dir):
        m = pattern.match(name)
        if not m:
            continue
        prefix, idx, rest = m.group(1), m.group(2), m.group(3)
        subdir_name = f"{prefix}_{idx}"
        subdir = os.path.join(figs_dir, subdir_name)
        src = os.path.join(figs_dir, name)
        if not os.path.isfile(src):
            continue
        os.makedirs(subdir, exist_ok=True)
        dst = os.path.join(subdir, name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        os.rename(src, dst)
        print(f"Moved {name} -> {subdir_name}/")
        moved += 1
    print(f"Done. Moved {moved} file(s).")


if __name__ == "__main__":
    main()
