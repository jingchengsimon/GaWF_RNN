#!/usr/bin/env python3
"""
One-time (or re-runnable) merge of legacy Phase3 layout:
  results/<train_data|train_figs>/gen_phase3[_short]_<scale>_<model>_ep<N>/
->results/<...>/gen_phase3[_short]_<scale>_ep<N>/
File names are distinct per model (stems include model type), so a single directory is enough.
Run from repo root. Default is dry-run; pass --apply to move files and remove empty old dirs.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict
from typing import DefaultDict, List, Tuple

# short: gen_phase3_short_4h_rnn_ep150
SHORT_RE = re.compile(
    r"^gen_phase3_short_(4h|10h|20h|40h)_(rnn|lstm|gru|gawf)_ep(\d+)$"
)
# full: gen_phase3_4h_rnn_ep100 (not gen_phase3_short_...)
FULL_RE = re.compile(r"^gen_phase3_(4h|10h|20h|40h)_(rnn|lstm|gru|gawf)_ep(\d+)$")


def _parse(name: str) -> Tuple[str, str, str, int] | None:
    m = SHORT_RE.match(name)
    if m:
        return "short", m.group(1), m.group(2), int(m.group(3))
    m = FULL_RE.match(name)
    if m:
        return "full", m.group(1), m.group(2), int(m.group(3))
    return None


def _target_name(mode: str, scale: str, ep: int) -> str:
    if mode == "short":
        return f"gen_phase3_short_{scale}_ep{ep}"
    return f"gen_phase3_{scale}_ep{ep}"


def _merge_group(
    base: str,
    members: List[str],
    target: str,
    apply: bool,
) -> None:
    target_path = os.path.join(base, target)
    print(f"  {len(members)} sources -> {target_path}")
    if not apply:
        for m in members:
            print(f"     from {os.path.join(base, m)}")
        return
    os.makedirs(target_path, exist_ok=True)
    log_chunks: List[bytes] = []
    for m in sorted(members):
        src_dir = os.path.join(base, m)
        for fn in list(os.listdir(src_dir)):
            sp = os.path.join(src_dir, fn)
            if not os.path.isfile(sp):
                print(f"     skip non-file: {sp}")
                continue
            if fn == "train.log":
                with open(sp, "rb") as f:
                    log_chunks.append(f.read())
                os.remove(sp)
                continue
            dp = os.path.join(target_path, fn)
            if os.path.exists(dp):
                raise RuntimeError(f"Refusing to overwrite existing file: {dp}")
            shutil.move(sp, dp)
        if os.listdir(src_dir):
            raise RuntimeError(f"Unexpected leftover in {src_dir}: {os.listdir(src_dir)}")
        os.rmdir(src_dir)
    if log_chunks:
        log_path = os.path.join(target_path, "train.log")
        pre_existing = os.path.isfile(log_path)
        with open(log_path, "ab" if pre_existing else "wb") as out:
            for i, block in enumerate(log_chunks):
                if pre_existing or i:
                    out.write(b"\n# --- merged log segment ---\n")
                out.write(block)


def run_root(base: str, apply: bool) -> int:
    if not os.path.isdir(base):
        print(f"Missing directory: {base}")
        return 0
    groups: DefaultDict[Tuple[str, str, int], List[str]] = defaultdict(list)
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        parsed = _parse(name)
        if not parsed:
            continue
        mode, scale, _model, ep = parsed
        key = (mode, scale, ep)
        tname = _target_name(mode, scale, ep)
        if name == tname:
            continue
        groups[key].append(name)
    n = 0
    for key, mem in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        target = _target_name(key[0], key[1], key[2])
        _merge_group(base, mem, target, apply)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Move files; without this flag, only print the plan",
    )
    ap.add_argument(
        "--train-data",
        default="results/train_data",
        help="Root containing gen_phase3_* subdirs",
    )
    ap.add_argument(
        "--train-figs",
        default="results/train_figs",
        help="Root containing gen_phase3_* subdirs (figures)",
    )
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo)
    for label, sub in [("train_data", args.train_data), ("train_figs", args.train_figs)]:
        print(f"=== {label} ({sub}) ===")
        n = run_root(sub, args.apply)
        print(f"Groups processed: {n}")


if __name__ == "__main__":
    main()
