"""Compare condition-level population representations across models.

Inputs are per-model ``pop_act_dpca.npy`` arrays with shape ``(H, 10, 9)``. The script
centers each hidden unit across the 90 digit-sector conditions, saves a 90x90 Euclidean
representational dissimilarity matrix (RDM) for each model, and computes cross-model
RDM Spearman RSA and centered Linear CKA matrices. Different models may have different
hidden widths because both comparisons operate in the shared condition space.

All arrays, CSV files, and JSON metadata are written below
``results/anal_data/5_pop_act_umap``. Plotting is handled by
``utils_viz/pop_act_representation_similarity.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
from scipy import stats


DEFAULT_MODELS = [
    "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model",
    "rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5_model",
    "lstm_sector_acc_h80_lr0.001_wd0.001_cdo0.0_rdo0.5_model",
    "gru_sector_acc_h105_lr0.005_wd0.001_cdo0.0_rdo0.5_model",
    "mamba_sector_acc_dmodel170_lr0.001_wd0.001_cdo0.0_rdo0.5_model",
    "s5_sector_acc_dmodel256_state128_lr0.001_wd0.0_cdo0.0_rdo0.5_model",
]

MODEL_LABELS = {
    DEFAULT_MODELS[0]: "GaWF",
    DEFAULT_MODELS[1]: "RNN",
    DEFAULT_MODELS[2]: "LSTM",
    DEFAULT_MODELS[3]: "GRU",
    DEFAULT_MODELS[4]: "Mamba",
    DEFAULT_MODELS[5]: "S5",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute condition-level RDM RSA and Linear CKA across models."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="results/anal_data/pop_act",
        help="Root containing <model>/pop_act_dpca.npy arrays.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/anal_data/5_pop_act_umap",
        help="Analysis-data root for per-model RDMs and cross-model similarity matrices.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--normalization",
        choices=["centered", "unit_zscore"],
        default="centered",
        help=(
            "Preprocessing across conditions: subtract each unit mean, or additionally "
            "divide each unit by its condition standard deviation."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical floor for zero-variance units and similarity denominators.",
    )
    return parser.parse_args()


def load_condition_representation(path: str) -> np.ndarray:
    """Load ``(H, 10, 9)`` means and return digit-major ``(90, H)`` float64 data."""
    array = np.load(path)
    if array.ndim != 3 or array.shape[1:] != (10, 9):
        raise ValueError(f"Expected {path} to have shape (H, 10, 9), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} contains non-finite condition means")
    return np.transpose(array, (1, 2, 0)).reshape(90, array.shape[0]).astype(np.float64)


def normalize_representation(
    representation: np.ndarray,
    mode: str,
    eps: float = 1e-8,
) -> np.ndarray:
    """Center units across conditions and optionally z-score each unit."""
    values = np.asarray(representation, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 90:
        raise ValueError(f"representation must have shape (90, H), got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    if mode == "centered":
        return centered
    if mode == "unit_zscore":
        scale = centered.std(axis=0, keepdims=True)
        scale = np.where(scale < float(eps), 1.0, scale)
        return centered / scale
    raise ValueError(f"Unknown normalization mode {mode!r}")


def compute_euclidean_rdm(representation: np.ndarray) -> np.ndarray:
    """Return the symmetric condition-by-condition Euclidean RDM."""
    values = np.asarray(representation, dtype=np.float64)
    squared_norm = np.sum(values**2, axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * values @ values.T
    rdm = np.sqrt(np.maximum(squared_distance, 0.0))
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def compute_rdm_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Compare two aligned RDM upper triangles with Spearman correlation."""
    first = np.asarray(rdm_a, dtype=np.float64)
    second = np.asarray(rdm_b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError(f"RDM shapes must match and be square, got {first.shape}, {second.shape}")
    upper = np.triu_indices(first.shape[0], k=1)
    correlation = stats.spearmanr(first[upper], second[upper]).statistic
    return float(correlation)


def compute_linear_cka(
    representation_a: np.ndarray,
    representation_b: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute centered Linear CKA for aligned conditions and arbitrary feature widths."""
    first = np.asarray(representation_a, dtype=np.float64)
    second = np.asarray(representation_b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError(
            "Representations must be 2D with the same number of conditions, got "
            f"{first.shape}, {second.shape}"
        )
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    gram_first = first @ first.T
    gram_second = second @ second.T
    numerator = float(np.sum(gram_first * gram_second))
    denominator = float(np.sqrt(np.sum(gram_first**2) * np.sum(gram_second**2)))
    if denominator <= float(eps):
        return float("nan")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _write_matrix_csv(path: str, matrix: np.ndarray, models: list[str]) -> None:
    """Write a labeled square similarity matrix."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *models])
        for model, row in zip(models, matrix):
            writer.writerow([model, *[float(value) for value in row]])


def analyze_representations(
    input_root: str,
    output_dir: str,
    models: list[str],
    normalization: str = "centered",
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute and save per-model RDMs plus cross-model RSA and Linear CKA."""
    os.makedirs(output_dir, exist_ok=True)
    representations: dict[str, np.ndarray] = {}
    rdms: dict[str, np.ndarray] = {}
    model_records = []

    for model in models:
        source_path = os.path.join(input_root, model, "pop_act_dpca.npy")
        representation = normalize_representation(
            load_condition_representation(source_path),
            normalization,
            eps=eps,
        )
        rdm = compute_euclidean_rdm(representation)
        representations[model] = representation
        rdms[model] = rdm

        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        rdm_path = os.path.join(model_dir, "condition_rdm.npy")
        meta_path = os.path.join(model_dir, "representation_similarity_meta.json")
        np.save(rdm_path, rdm.astype(np.float32, copy=False))
        record = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "source": os.path.abspath(source_path),
            "input_shape": [int(representation.shape[1]), 10, 9],
            "representation_shape": list(representation.shape),
            "normalization": normalization,
            "rdm": {
                "path": os.path.abspath(rdm_path),
                "shape": list(rdm.shape),
                "dtype": "float32",
                "metric": "euclidean",
            },
            "condition_order": "digit-major: flat_index = digit * 9 + sector",
        }
        with open(meta_path, "w") as file:
            json.dump(record, file, indent=2)
        model_records.append(record)
        print(f"Saved {rdm_path}")
        print(f"Saved {meta_path}")

    count = len(models)
    rsa = np.eye(count, dtype=np.float64)
    cka = np.eye(count, dtype=np.float64)
    for first_idx in range(count):
        for second_idx in range(first_idx + 1, count):
            first_model = models[first_idx]
            second_model = models[second_idx]
            rsa_value = compute_rdm_spearman(rdms[first_model], rdms[second_model])
            cka_value = compute_linear_cka(
                representations[first_model],
                representations[second_model],
                eps=eps,
            )
            rsa[first_idx, second_idx] = rsa[second_idx, first_idx] = rsa_value
            cka[first_idx, second_idx] = cka[second_idx, first_idx] = cka_value

    rsa_npy = os.path.join(output_dir, "representation_similarity_rsa_spearman.npy")
    cka_npy = os.path.join(output_dir, "representation_similarity_linear_cka.npy")
    rsa_csv = os.path.join(output_dir, "representation_similarity_rsa_spearman.csv")
    cka_csv = os.path.join(output_dir, "representation_similarity_linear_cka.csv")
    meta_path = os.path.join(output_dir, "representation_similarity_meta.json")
    np.save(rsa_npy, rsa.astype(np.float32))
    np.save(cka_npy, cka.astype(np.float32))
    _write_matrix_csv(rsa_csv, rsa, models)
    _write_matrix_csv(cka_csv, cka, models)
    payload = {
        "models": model_records,
        "normalization": normalization,
        "rsa": {
            "matrix": os.path.abspath(rsa_npy),
            "csv": os.path.abspath(rsa_csv),
            "method": "Spearman correlation of Euclidean RDM upper triangles",
        },
        "linear_cka": {
            "matrix": os.path.abspath(cka_npy),
            "csv": os.path.abspath(cka_csv),
            "method": "centered biased Linear CKA in condition Gram space",
        },
    }
    with open(meta_path, "w") as file:
        json.dump(payload, file, indent=2)
    for path in (rsa_npy, cka_npy, rsa_csv, cka_csv, meta_path):
        print(f"Saved {path}")
    return {"rsa": rsa, "linear_cka": cka, "metadata": payload}


def main() -> None:
    """Run the command-line analysis."""
    args = parse_args()
    analyze_representations(
        args.input_root,
        args.output_dir,
        list(args.models),
        normalization=args.normalization,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
