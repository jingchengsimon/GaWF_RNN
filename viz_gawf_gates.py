"""
Visualize GaWF gate matrices (gate_ih / gate_hh) as heatmaps.

新功能：
- 如果输入字典中包含多帧 gate（gate_ih_all / gate_hh_all），则在一行中画出多个 gate matrix。
- 在每个 gate matrix 下方，再画一行 3 个原始帧：对应 t-1, t, t+1（使用 export_gawf_gates.py 导出的 neighbor_frames）。

兼容：
- 若只包含单帧 gate（老格式），仍然画单个 heatmap，不画帧。

Example commands:

  python viz_gawf_gates.py --in ./gawf_gates.pt --outdir ./gawf_gate_figs
"""

import argparse
import os
from typing import List, Optional

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GaWF gate matrices (gate_ih / gate_hh) as heatmaps."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="./gawf_gates.pt",
        type=str,
        help="Path to gate dictionary file (torch.save from export_gawf_gates.py).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./gawf_gate_figs",
        help="Output directory for figures (default: ./gawf_gate_figs).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help=(
            "Optional clipping for color scale. "
            "If 0 < clip < 0.5, interpreted as upper tail percentile (e.g. 0.01 -> clip top 1%%). "
            "If clip >= 0.5, interpreted as absolute symmetric bound for normalized values."
        ),
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "zscore", "minmax"],
        help="Normalization mode before plotting: none / zscore / minmax (default: none).",
    )
    return parser.parse_args()


def normalize_matrix(mat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mat
    flat = mat.astype(np.float64).reshape(-1)
    if mode == "zscore":
        mean = float(flat.mean())
        std = float(flat.std())
        if std == 0.0:
            return mat * 0.0
        return (mat - mean) / std
    if mode == "minmax":
        vmin = -0.5  # float(flat.min())
        vmax = 0.5  # float(flat.max())
        if vmax <= vmin:
            return mat * 0.0
        return (mat - vmin) / (vmax - vmin)
    raise ValueError(f"Unknown normalize mode: {mode}")


def clip_matrix(mat: np.ndarray, clip: Optional[float], normalized: bool) -> np.ndarray:
    if clip is None:
        return mat
    flat = mat.reshape(-1)
    if 0.0 < clip < 0.5:
        # Percentile-based clipping: remove extreme upper (and lower if normalized) values.
        if normalized:
            abs_flat = np.abs(flat)
            bound = float(np.quantile(abs_flat, 1.0 - clip))
            if bound <= 0:
                return mat
            return np.clip(mat, -bound, bound)
        else:
            upper = float(np.quantile(flat, 1.0 - clip))
            lower = float(np.quantile(flat, clip))
            return np.clip(mat, lower, upper)
    else:
        # Absolute symmetric bound (mainly for zscore).
        if clip <= 0:
            return mat
        bound = float(clip)
        if normalized:
            return np.clip(mat, -bound, bound)
        return np.clip(mat, mat.min(), bound)


def plot_gates_with_frames(
    gates: np.ndarray,  # (N, H, W)
    main_title: str,
    subtitles: List[str],
    out_path: str,
    vlim: float,
    cmap: str,
    x_label_gate: str,
    y_label_gate: str,
    neighbor_frames: Optional[np.ndarray] = None,  # (N, 3, Hf, Wf) or (N, 3, C, Hf, Wf)
    gate_aspect: str = "auto",
) -> None:
    """
    上排：N 个 gate 矩阵（heatmap，一行多个）。
    下排：每个 gate 对应 3 个原始帧（t-1, t, t+1），共 N 组。

    若 neighbor_frames 为空，则仅画 gate（兼容老格式）。
    """
    N = gates.shape[0]
    use_frames = (
        neighbor_frames is not None
        and isinstance(neighbor_frames, np.ndarray)
        and neighbor_frames.shape[0] == N
    )

    if not use_frames:
        # 仅画 gate（老格式或没有邻近帧信息）
        fig, axes = plt.subplots(1, N, figsize=(5 * N, 4), squeeze=False)
        axes_flat = axes.ravel().tolist()
        im = None
        for i, (ax, mat, sub) in enumerate(zip(axes_flat, gates, subtitles)):
            im = ax.imshow(
                mat,
                aspect=gate_aspect,
                origin="lower",
                interpolation="nearest",
                vmin=-vlim,
                vmax=vlim,
                cmap=cmap,
            )
            ax.set_xlabel(x_label_gate)
            if i == 0:
                ax.set_ylabel(y_label_gate)
            ax.set_title(sub, fontsize=10)

        fig.suptitle(main_title, fontsize=12)
        fig.tight_layout(rect=[0.0, 0.0, 0.9, 0.95])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {out_path}")
        return

    # 带帧的布局：2 行 N 列，第一行 gate，第二行每列再细分为 1x3 子图（t-1, t, t+1）
    fig = plt.figure(figsize=(5 * N, 6))
    gs = fig.add_gridspec(2, N, height_ratios=[3, 2])

    im = None
    for i in range(N):
        # Gate 子图（上排）
        ax_gate = fig.add_subplot(gs[0, i])
        mat = gates[i]
        im = ax_gate.imshow(
            mat,
            aspect=gate_aspect,
            origin="lower",
            interpolation="nearest",
            vmin=-vlim,
            vmax=vlim,
            cmap=cmap,
        )
        ax_gate.set_xlabel(x_label_gate)
        if i == 0:
            ax_gate.set_ylabel(y_label_gate)
        ax_gate.set_title(subtitles[i], fontsize=10)

        # 帧子图（下排，每个 gate 对应 3 帧：t-1, t, t+1）
        sub_gs = gs[1, i].subgridspec(1, 3)
        for j in range(3):
            ax_f = fig.add_subplot(sub_gs[0, j])
            frame_img = neighbor_frames[i, j]
            # 支持 (H, W) 或 (C, H, W)
            if frame_img.ndim == 2:
                ax_f.imshow(frame_img, cmap="gray", origin="lower")
            elif frame_img.ndim == 3:
                # 若为 (C, H, W)，则取单通道或求平均
                if frame_img.shape[0] in (1, 3):
                    # C=1 或 3：简单转为 H x W 或 H x W x 3
                    if frame_img.shape[0] == 1:
                        ax_f.imshow(frame_img[0], cmap="gray", origin="lower")
                    else:
                        # 假设是 RGB
                        ax_f.imshow(
                            np.transpose(frame_img, (1, 2, 0)), origin="lower"
                        )
                else:
                    ax_f.imshow(frame_img.mean(axis=0), cmap="gray", origin="lower")
            else:
                ax_f.imshow(frame_img.squeeze(), cmap="gray", origin="lower")
            ax_f.axis("off")
            if i == 0:
                # 第一列标注 t-1, t, t+1 的顺序提示
                if j == 0:
                    ax_f.set_title("t-1", fontsize=8)
                elif j == 1:
                    ax_f.set_title("t", fontsize=8)
                else:
                    ax_f.set_title("t+1", fontsize=8)

    fig.suptitle(main_title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()

    gate_dict = torch.load(args.in_path, map_location="cpu")

    gate_ih = gate_dict.get("gate_ih", None)
    gate_hh = gate_dict.get("gate_hh", None)
    gate_ih_all = gate_dict.get("gate_ih_all", None)
    gate_hh_all = gate_dict.get("gate_hh_all", None)

    if gate_ih is None or gate_hh is None:
        raise KeyError("Input file must contain 'gate_ih' and 'gate_hh' tensors.")

    # 多帧 gate 优先；否则退化为单帧
    if gate_ih_all is not None and gate_hh_all is not None:
        if isinstance(gate_ih_all, torch.Tensor):
            gate_ih_np_all = gate_ih_all.detach().cpu().numpy()
        else:
            gate_ih_np_all = np.asarray(gate_ih_all)

        if isinstance(gate_hh_all, torch.Tensor):
            gate_hh_np_all = gate_hh_all.detach().cpu().numpy()
        else:
            gate_hh_np_all = np.asarray(gate_hh_all)
    else:
        if isinstance(gate_ih, torch.Tensor):
            gate_ih_np_all = gate_ih.detach().cpu().numpy()[None, ...]
        else:
            gate_ih_np_all = np.asarray(gate_ih)[None, ...]

        if isinstance(gate_hh, torch.Tensor):
            gate_hh_np_all = gate_hh.detach().cpu().numpy()[None, ...]
        else:
            gate_hh_np_all = np.asarray(gate_hh)[None, ...]

    # 邻近帧信息（可选）
    neighbor_frames = gate_dict.get("neighbor_frames", None)
    if isinstance(neighbor_frames, torch.Tensor):
        neighbor_frames_np = neighbor_frames.detach().cpu().numpy()
    elif neighbor_frames is not None:
        neighbor_frames_np = np.asarray(neighbor_frames)
    else:
        neighbor_frames_np = None

    split = gate_dict.get("split", "unknown")
    sample_index = gate_dict.get("sample_index", -1)
    # 新版导出脚本会保存整体使用到的 sample_indices / target_fg_digit（旧版则只有单个 sample_index）。
    sample_indices_meta = gate_dict.get("sample_indices", None)
    target_fg_digit = gate_dict.get("target_fg_digit", None)
    selected_fg_digit = gate_dict.get("selected_fg_digit", None)
    selected_frame_indices = gate_dict.get("selected_frame_indices", None)
    selected_global_indices = gate_dict.get("selected_global_indices", None)

    def _to_list(x) -> Optional[List[int]]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        arr = np.asarray(x)
        return arr.tolist()

    frame_idx_list = _to_list(selected_frame_indices)
    global_idx_list = _to_list(selected_global_indices)
    sample_indices_list = _to_list(sample_indices_meta)

    os.makedirs(args.outdir, exist_ok=True)

    # Center gate at 0: (gate - 0.5) -> range [-0.5, 0.5]; then optional normalize/clip.
    gate_ih_centered = gate_ih_np_all.astype(np.float64) - 0.5  # (N, H, W)
    gate_hh_centered = gate_hh_np_all.astype(np.float64) - 0.5  # (N, H, W)
    is_normalized = args.normalize != "none"
    gate_ih_proc = clip_matrix(
        normalize_matrix(gate_ih_centered, args.normalize), args.clip, normalized=is_normalized
    )
    gate_hh_proc = clip_matrix(
        normalize_matrix(gate_hh_centered, args.normalize), args.clip, normalized=is_normalized
    )

    # 共同的颜色范围 [-max_abs, max_abs]
    max_abs = float(
        max(
            np.abs(gate_ih_proc).max(initial=0.0),
            np.abs(gate_hh_proc).max(initial=0.0),
        )
    )
    if max_abs == 0.0:
        max_abs = 1e-8

    ih_shape = gate_ih_np_all.shape
    hh_shape = gate_hh_np_all.shape

    suffix = args.format.lower()
    # 文件名后缀：优先使用导出时保存的 target_fg_digit，方便按 digit 管理结果；
    # 若旧文件中没有该字段，则退回默认的数字 3。
    if target_fg_digit is not None:
        idx_suffix = int(target_fg_digit)
    else:
        idx_suffix = 3
    ih_path = os.path.join(args.outdir, f"gate_ih_{idx_suffix}.{suffix}")
    hh_path = os.path.join(args.outdir, f"gate_hh_{idx_suffix}.{suffix}")

    # 构造每个 gate 的子标题：包含 split / sample_index / t 索引 / digit
    N = gate_ih_proc.shape[0]
    subtitles: List[str] = []
    for i in range(N):
        if sample_indices_list is not None and i < len(sample_indices_list):
            sample_for_i = sample_indices_list[i]
        else:
            sample_for_i = sample_index

        parts = [f"split={split}", f"sample={sample_for_i}"]
        if frame_idx_list is not None and i < len(frame_idx_list):
            parts.append(f"t={frame_idx_list[i]}")
        if global_idx_list is not None and i < len(global_idx_list):
            parts.append(f"g={global_idx_list[i]}")
        if selected_fg_digit is not None:
            parts.append(f"fg={int(selected_fg_digit)}")
        subtitles.append(", ".join(parts))

    if sample_indices_list:
        samples_str = ",".join(str(s) for s in sample_indices_list)
    else:
        samples_str = str(sample_index)

    ih_title = (
        f"GaWF gate_ih (hidden x input) shape={ih_shape}, "
        f"split={split}, samples={samples_str}"
    )
    hh_title = (
        f"GaWF gate_hh (hidden x hidden) shape={hh_shape}, "
        f"split={split}, samples={samples_str}"
    )

    # gate_ih：上排 gate，下排 3 帧（若 neighbor_frames 可用）
    plot_gates_with_frames(
        gates=gate_ih_proc,
        main_title=ih_title,
        subtitles=subtitles,
        out_path=ih_path,
        vlim=max_abs,
        cmap="RdBu_r",
        x_label_gate="Input index",
        y_label_gate="Hidden index",
        neighbor_frames=neighbor_frames_np,
        gate_aspect="auto",
    )

    # gate_hh：同样布局（复用相同的帧）
    plot_gates_with_frames(
        gates=gate_hh_proc,
        main_title=hh_title,
        subtitles=subtitles,
        out_path=hh_path,
        vlim=max_abs,
        cmap="RdBu_r",
        x_label_gate="Hidden index",
        y_label_gate="Hidden index",
        neighbor_frames=neighbor_frames_np,
        # 确保 gate_hh 为严格的方形像素（长宽一致）
        gate_aspect="equal",
    )


if __name__ == "__main__":
    main()

