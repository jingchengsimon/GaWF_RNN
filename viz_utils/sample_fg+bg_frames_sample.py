"""
Quick visualization script to inspect foreground/background labels.

功能：
- 对给定的 sample_indices（例如 0 1 2），画出每个 sample 的前 num_frames 帧。
- 图像排布为：行 = 样本数，列 = num_frames（默认 10），共 rows x cols 个子图。
- 每个子图标题标注对应的 fg_char_id 以及 bg_char_ids 列表，
  方便人工核对 label 与实际帧中的数字是否一致。

实现要点：
- 使用与训练/分析相同的 MC_RNN_Dataset / create_datasets 逻辑，
  但强制 predict_all_chars=True 以获得 fg+bg 的 digit 标签。
- 单帧图像来自原始 stimuli（stims_test），而不是 (T, C, H, W) 的堆叠帧。
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so train_rnn_updated and utils can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_rnn_updated import MC_RNN_Dataset  # noqa: E402
from utils.train_helpers import (  # noqa: E402
    create_datasets,
    get_base_path,
    load_raw_data,
    prepare_data_paths,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw frames for given sample indices with fg/bg digit labels "
            "(using predict_all_chars=True)."
        )
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Sample indices within the test split to visualize (default: 0 1 2).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of time steps (frames) per sample to visualize (default: 10).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/gate_sample/fg_bg_frames.png",
        help="Output path for the visualization figure.",
    )
    # Dataset-related options（与训练/导出脚本保持一致）
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Base directory for stimuli/labels. If empty, uses ENV (AIM3_STIMULI_PATH / "
            "FAW_RNN_DATA_PATH) or <repo>/stimuli."
        ),
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for stimulus_reg-* files (e.g. '40h'). "
            "Same semantics as train_rnn_updated.py."
        ),
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help="Load stimuli with numpy mmap_mode='r' (default: True).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def build_test_dataset_allchars(args: argparse.Namespace):
    """
    构建 test 集数据集（predict_all_chars=True），并返回：
        - test_ds: MC_RNN_Dataset
        - stims_test: 原始 stimuli (total_frames, H, W)
    """
    base_path = get_base_path(override=args.data_dir or None)
    paths = prepare_data_paths(base_path, data_suffix=args.data_suffix, splits=("test",))
    stims_test, lbls_test = load_raw_data(
        None,
        None,
        None,
        None,
        use_mmap=args.use_mmap,
        paths_tuple=paths,
    )

    test_ds, num_pos = create_datasets(
        None,
        None,
        None,
        None,
        use_sector_mode=True,  # 无关紧要；predict_all_chars=True 时不使用 sector 标签
        predict_all_chars=True,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    # test_ds 是一个长度为 1 的列表/tuple 或直接是 dataset，保持与 train_helpers 一致
    if isinstance(test_ds, (list, tuple)):
        test_ds = test_ds[0]
    return test_ds, stims_test


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    test_ds, stims_test = build_test_dataset_allchars(args)

    frame_num = test_ds.frame_num
    chan_num = test_ds.chan_num

    sample_indices: List[int] = []
    for idx in args.sample_indices:
        if 0 <= idx < len(test_ds):
            sample_indices.append(idx)
        else:
            print(f"[warn] sample index {idx} out of range [0, {len(test_ds)-1}], skipped.")
    if not sample_indices:
        raise IndexError(
            f"No valid sample indices in {args.sample_indices}; "
            f"test set length={len(test_ds)}"
        )

    n_rows = len(sample_indices)
    n_cols = min(args.num_frames, frame_num)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    for row, sidx in enumerate(sample_indices):
        frames, labels = test_ds[sidx]  # frames: (T, C, H, W), labels: (T, max_chars)
        # labels[t, 0] 是 fg_char_id；labels[t, 1:] 是 bg_char_ids（填充 -1）
        T, max_chars = labels.shape

        start_idx = (sidx * frame_num) + chan_num  # 对应 MC_RNN_Dataset.__len__ 中的定义

        for col in range(n_cols):
            t = col
            if t >= T:
                continue

            ax = axes[row][col]

            raw_idx = start_idx + t
            if raw_idx < 0 or raw_idx >= stims_test.shape[0]:
                ax.axis("off")
                continue

            frame_img = stims_test[raw_idx]  # (H, W)

            ax.imshow(frame_img, cmap="gray", origin="lower")
            ax.axis("off")

            fg = int(labels[t, 0])
            bg_all = labels[t, 1:]
            bg_ids = [int(x) for x in bg_all.tolist() if x >= 0]

            title = f"s={sidx}, t={t}\nfg={fg}, bg={bg_ids}"
            ax.set_title(title, fontsize=6)

    fig.suptitle(
        f"Foreground / background digits for samples={sample_indices}, first {n_cols} frames",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_dir = os.path.dirname(args.save_dir)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.save_dir, dpi=150)
    plt.close(fig)
    print(f"Saved fg/bg frame visualization to: {args.save_dir}")


if __name__ == "__main__":
    main()

