"""
MNIST digit embedding: flatten -> PCA -> UMAP -> 2D scatter colored by digit.

Uses the same pipeline as gate_similarity_embedding.py but on raw MNIST images:
  - Load MNIST (torchvision), collect images per digit
  - Flatten each image: vec = img.reshape(-1)
  - PCA(n_components) -> UMAP(2)
  - Plot 2D scatter by digit, optional silhouette and within/between ratio

Example:
  python mnist_embedding.py --num_per_digit 500 --out ./mnist_embedding.png
"""

import argparse
import os
import warnings
from typing import Tuple

import numpy as np

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import torch
from torchvision.datasets import MNIST

from utils_anal.anal_paths import output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST embedding: flatten -> PCA -> UMAP, colored by digit."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./mnist_data_pytorch",
        help="Root directory for MNIST data (download if missing).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Use training set (default: True).",
    )
    parser.add_argument(
        "--no_train",
        action="store_false",
        dest="train",
        help="Use test set.",
    )
    parser.add_argument(
        "--num_per_digit",
        type=int,
        default=500,
        help="Max number of images per digit (default: 500).",
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
        default=str(
            output_dir("E_relevance_alignment", "mnist_embedding", "figs")
            / "mnist_embedding.png"
        ),
        help="Output image path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_mnist_by_digit(
    root: str,
    train: bool,
    num_per_digit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST and collect up to num_per_digit images per digit.

    Returns:
        X: (N, 784) float64, flattened images in [0, 1]
        labels: (N,) int64, digit labels 0-9
    """
    dataset = MNIST(root=root, train=train, download=True)

    digit_to_indices = {d: [] for d in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        d = int(label)
        if len(digit_to_indices[d]) < num_per_digit:
            digit_to_indices[d].append(idx)

    vectors = []
    labels = []
    for d in range(10):
        for idx in digit_to_indices[d]:
            img, _ = dataset[idx]
            arr = np.array(img, dtype=np.float64) / 255.0
            vec = arr.reshape(-1)
            vectors.append(vec)
            labels.append(d)

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
    silhouette = silhouette_score(X_2d, labels)
    unique_labels = np.unique(labels)
    centroids = np.array([X_2d[labels == d].mean(axis=0) for d in unique_labels])
    within_dists = [
        np.mean(np.linalg.norm(X_2d[labels == d] - centroids[i], axis=1))
        for i, d in enumerate(unique_labels)
    ]
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
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)

    print("Loading MNIST...")
    X, labels = load_mnist_by_digit(
        root=args.root,
        train=args.train,
        num_per_digit=args.num_per_digit,
    )
    split = "train" if args.train else "test"
    print(f"Loaded {X.shape[0]} images ({X.shape[1]} dims), split={split}")

    method = "UMAP" if HAS_UMAP else "t-SNE"
    print(f"Reducing with PCA({args.pca_components}) -> {method}(2)...")
    X_2d = reduce_pca_umap(
        X,
        n_pca=args.pca_components,
        use_umap=HAS_UMAP,
        random_state=args.seed,
    )

    sil, ratio = compute_silhouette_and_ratio(X_2d, labels)
    print(f"Silhouette={sil:.4f}, within/between={ratio:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.cm.tab10
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
            s=15,
        )
    ax.set_title(f"MNIST embedding (PCA({args.pca_components}) -> {method}), silhouette={sil:.3f}, w/b={ratio:.3f}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    out_dir = os.path.dirname(args.save_dir)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.save_dir, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.save_dir}")


if __name__ == "__main__":
    main()
