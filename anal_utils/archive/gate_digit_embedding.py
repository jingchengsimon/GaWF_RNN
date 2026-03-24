"""
Gate matrices similarity comparison: PCA -> UMAP embedding, colored by digit.

Uses export_gawf_gates.collect_gate_matrices_for_digits to generate gate matrices
for each digit (no pt file saved). For each gate type (gate_ih, gate_hh):
  - Apply G - 0.5
  - Take a single row vector (gate_ih row length 1152, gate_hh row length 256)
  - PCA -> UMAP(2) (or t-SNE if UMAP not available)
  - Plot scatter by digit
  - Compute silhouette score and within/between distance ratio

Row index is configurable via --row_index (default 0).

Output: single image with two subplots (gate_ih, gate_hh).

Example:
  python gate_similarity_embedding.py --out ./gate_similarity_embedding.png
  python gate_similarity_embedding.py --row_index 0 --num_per_digit 100
"""

import argparse
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

from export_gate_sample import (
    build_model_from_ckpt,
    build_test_dataset,
    collect_gate_matrices_for_digits,
    resolve_device,
)
from utils.train_helpers import set_seed

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate similarity embedding: PCA->UMAP, silhouette, within/between ratio."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_fb100_model.pth",
        help="Path to trained GaWFRNNConv checkpoint.",
    )
    parser.add_argument(
        "--num_per_digit",
        type=int,
        default=1000,
        help="Number of gate matrices per digit (default: 100).",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components before UMAP (default: 50).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./gate_similarity_embedding.png",
        help="Output image path (default: ./gate_similarity_embedding.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device: cpu / cuda (default: cpu).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Temperature tau for gate computation (default: 2.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--row_index",
        type=int,
        default=0,
        help="Row index of the gate matrix to use as vector (gate_ih: 256 rows x 1152 cols; gate_hh: 256 x 256). Default 0.",
    )
    # Dataset options (mirror export_gawf_gates)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    return parser.parse_args()


def gate_dict_to_vectors(
    gate_by_digit: Dict[int, List[np.ndarray]],
    row_index: int,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert gate matrices to row vectors (one row per matrix).

    Args:
        gate_by_digit: digit -> list of (H, W) matrices
        row_index: which row to take (0 .. H-1); gives vector of length W
        center: if True, subtract 0.5 from each matrix

    Returns:
        X: (N, W) array of row vectors
        labels: (N,) array of digit labels
    """
    vectors = []
    labels = []
    for digit, mats in gate_by_digit.items():
        for m in mats:
            m = np.asarray(m, dtype=np.float64)
            if center:
                m = m - 0.5
            vec = m[row_index, :].ravel()
            vectors.append(vec)
            labels.append(digit)
    X = np.stack(vectors, axis=0)
    labels = np.array(labels, dtype=np.int64)
    return X, labels


def reduce_pca_umap(
    X: np.ndarray,
    n_pca: int = 50,
    use_umap: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    if X.shape[1] > n_pca:
        pca = PCA(n_components=n_pca, random_state=random_state)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    if use_umap and HAS_UMAP:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs.*random_state.*")
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            X_2d = reducer.fit_transform(X_reduced)
    else:
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X) - 1))
        X_2d = tsne.fit_transform(X_reduced)
    return X_2d


def compute_silhouette_and_ratio(
    X_2d: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute silhouette score and within/between distance ratio.

    Args:
        X_2d: (N, 2) embedded points
        labels: (N,) digit labels

    Returns:
        silhouette: silhouette score
        within_between_ratio: mean within-cluster distance / mean between-cluster distance
    """
    silhouette = silhouette_score(X_2d, labels)

    unique_labels = np.unique(labels)
    centroids = []
    for d in unique_labels:
        mask = labels == d
        centroids.append(X_2d[mask].mean(axis=0))
    centroids = np.array(centroids)

    within_dists = []
    for i, d in enumerate(unique_labels):
        mask = labels == d
        pts = X_2d[mask]
        c = centroids[i]
        within_dists.append(np.mean(np.linalg.norm(pts - c, axis=1)))
    mean_within = np.mean(within_dists)

    between_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            between_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    mean_between = np.mean(between_dists) if between_dists else 1e-8

    within_between_ratio = mean_within / mean_between
    return float(silhouette), float(within_between_ratio)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Build test dataset
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test set size: {len(test_ds)} samples")

    # Build model
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    print(f"Loaded model from {args.ckpt}")

    # Collect gate matrices per digit (no file save)
    gate_ih_by_digit, gate_hh_by_digit = collect_gate_matrices_for_digits(
        test_ds=test_ds,
        model=model,
        device=device,
        tau=args.tau,
        num_per_digit=args.num_per_digit,
        verbose=False,
    )

    # Convert to row vectors (G - 0.5, take row at row_index)
    row_idx = int(args.row_index)
    # Validate row_index against gate shapes (both have 256 rows)
    first_ih = next(iter(gate_ih_by_digit.values()), [])
    first_hh = next(iter(gate_hh_by_digit.values()), [])
    if first_ih:
        n_rows = np.asarray(first_ih[0]).shape[0]
        if row_idx < 0 or row_idx >= n_rows:
            raise ValueError(f"row_index must be in [0, {n_rows - 1}], got {row_idx}")
    X_ih, labels_ih = gate_dict_to_vectors(gate_ih_by_digit, row_index=row_idx, center=True)
    X_hh, labels_hh = gate_dict_to_vectors(gate_hh_by_digit, row_index=row_idx, center=True)
    print(f"gate_ih: {X_ih.shape[0]} samples, row {row_idx} -> {X_ih.shape[1]} dims")
    print(f"gate_hh: {X_hh.shape[0]} samples, row {row_idx} -> {X_hh.shape[1]} dims")

    # PCA(50) -> UMAP(2) or t-SNE
    method = "UMAP" if HAS_UMAP else "t-SNE"
    print(f"Using {method} for 2D embedding")

    X_ih_2d = reduce_pca_umap(
        X_ih,
        n_pca=args.pca_components,
        use_umap=HAS_UMAP,
        random_state=args.seed,
    )
    X_hh_2d = reduce_pca_umap(
        X_hh,
        n_pca=args.pca_components,
        use_umap=HAS_UMAP,
        random_state=args.seed,
    )

    # Compute metrics
    sil_ih, ratio_ih = compute_silhouette_and_ratio(X_ih_2d, labels_ih)
    sil_hh, ratio_hh = compute_silhouette_and_ratio(X_hh_2d, labels_hh)
    print(f"gate_ih: silhouette={sil_ih:.4f}, within/between={ratio_ih:.4f}")
    print(f"gate_hh: silhouette={sil_hh:.4f}, within/between={ratio_hh:.4f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap = plt.cm.tab10
    for ax, X_2d, labels, title, sil, ratio in [
        (axes[0], X_ih_2d, labels_ih, "gate_ih", sil_ih, ratio_ih),
        (axes[1], X_hh_2d, labels_hh, "gate_hh", sil_hh, ratio_hh),
    ]:
        for d in range(10):
            mask = labels == d
            if mask.sum() == 0:
                continue
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[cmap((d + 0.5) / 10)],
                label=str(d),
                alpha=0.6,
                s=20,
            )
        ax.set_title(f"{title} (silhouette={sil:.3f}, w/b={ratio:.3f})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="upper right", ncol=2, fontsize=8)

    fig.suptitle(f"Gate embedding by digit, row_index={row_idx} (PCA({args.pca_components}) -> {method})", fontsize=12)
    fig.tight_layout()

    out_dir = os.path.dirname(args.save_dir)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.save_dir, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {args.save_dir}")


if __name__ == "__main__":
    main()
