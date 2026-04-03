"""
Pluggable dimensionality reduction for population activity (N, D) -> (N, n_components).

Implementations expose fit_transform(X) so callers do not depend on UMAP vs PCA, etc.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PopulationReducer(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


class UMAPReducer:
    """Wrapper around umap.UMAP (optional dependency: umap-learn)."""

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 0,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        **umap_kw: Any,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self._umap_kw = umap_kw

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for UMAPReducer. Install with: pip install umap-learn"
            ) from e
        reducer = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            **self._umap_kw,
        )
        return np.asarray(reducer.fit_transform(X), dtype=np.float32)


class PCAReducer:
    """
    Linear PCA embedder. fit_transform returns the first n_components scores (default 3).

    After fit_transform, ``explained_variance_ratio_`` holds per-component ratios for all
    fitted components (up to fit_max_components), for scree / variance plots.
    """

    def __init__(
        self,
        n_components: int = 3,
        fit_max_components: int = 20,
        whiten: bool = False,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.fit_max_components = int(fit_max_components)
        self.whiten = whiten
        self.random_state = random_state
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            from sklearn.decomposition import PCA
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for PCAReducer. Install with: pip install scikit-learn"
            ) from e

        X64 = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X64.shape
        n_cap = max(1, min(n_samples - 1, n_features))
        n_fit = min(max(self.n_components, self.fit_max_components), n_cap)
        pca = PCA(
            n_components=n_fit,
            whiten=self.whiten,
            random_state=self.random_state if self.random_state is not None else None,
        )
        Z = pca.fit_transform(X64)
        self.explained_variance_ratio_ = np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32
        )
        return np.asarray(Z[:, : self.n_components], dtype=np.float32)
