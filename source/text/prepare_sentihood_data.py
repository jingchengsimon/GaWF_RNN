#!/usr/bin/env python3
"""One-time SentiHood query-pair tensor preparation.

Downloads rows from the Hugging Face Dataset Viewer by default, or reads local
``sentihood-{train,dev,test}.json`` files with ``--source_dir``. Each raw sentence
is expanded into query examples for the actual locations in that sentence and
the selected aspect set. The default aspect set is the four-aspect SentiHood
paper setting: general, price, transit-location, safety.

Outputs under ``<data_dir>/sentihood/``:
- ``sentihood_{split}_ids.pt``       ``(N_query, max_len)`` int64
- ``sentihood_{split}_lengths.pt``   ``(N_query,)`` int64
- ``sentihood_{split}_labels.pt``    ``(N_query,)`` int64, 0=None, 1=Positive, 2=Negative
- ``sentihood_{split}_{aspect,target,sentence}_ids.pt``
- ``vocab.json`` and ``sentihood_meta.json``
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.text_sentihood_data import PAPER_ASPECTS, prepare_sentihood_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=None, help="Base data dir; defaults like IMDB.")
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Optional directory with sentihood-train/dev/test JSON files. If omitted, fetch HF.",
    )
    parser.add_argument("--aspects", nargs="+", default=list(PAPER_ASPECTS))
    parser.add_argument("--vocab_size", type=int, default=12000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = prepare_sentihood_tensors(
        data_dir=args.data_dir,
        source_dir=args.source_dir,
        aspects=args.aspects,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        max_len=args.max_len,
    )
    print(
        "[prepare_sentihood] done "
        f"vocab={meta['vocab_size']} max_len={meta['max_len']} "
        f"raw={meta['n_raw']} query={meta['n_query']}"
    )


if __name__ == "__main__":
    sys.exit(main())
