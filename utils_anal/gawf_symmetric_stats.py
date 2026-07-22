"""Pure statistics for symmetric GaWF relevance and switch-timing analyses.

Inputs are frame-by-unit activations, frame labels, gate-column summaries, and switch-aligned
event arrays. Outputs are NumPy dictionaries/arrays with explicit unit and context axes. Model
loading and plotting intentionally live in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats


NUM_SECTORS = 9
NUM_DIGITS = 10
NUM_CELLS = NUM_SECTORS * NUM_DIGITS


@dataclass(frozen=True)
class SelectivityResult:
    """Per-unit two-way selectivity and marginal tuning profiles."""

    eta_sector: np.ndarray
    eta_digit: np.ndarray
    eta_interaction: np.ndarray
    eta_residual: np.ndarray
    tuning_sector: np.ndarray
    tuning_digit: np.ndarray


def joint_design(labels: np.ndarray) -> dict[str, Any]:
    """Return a 9x10 frequency table and chi-square independence audit."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"labels must have shape (trials, 2), got {labels.shape}")
    digits, sectors = labels[:, 0], labels[:, 1]
    if np.any((digits < 0) | (digits >= NUM_DIGITS)):
        raise ValueError("digit labels must be in [0, 9]")
    if np.any((sectors < 0) | (sectors >= NUM_SECTORS)):
        raise ValueError("sector labels must be in [0, 8]")
    table = np.zeros((NUM_SECTORS, NUM_DIGITS), dtype=np.int64)
    np.add.at(table, (sectors, digits), 1)
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n_trials = int(labels.shape[0])
    cramers_v = float(np.sqrt(chi2 / (n_trials * min(NUM_SECTORS - 1, NUM_DIGITS - 1))))
    return {
        "n_trials": n_trials,
        "joint_frequency_sector_rows_digit_columns": table.tolist(),
        "sector_counts": table.sum(axis=1).tolist(),
        "digit_counts": table.sum(axis=0).tolist(),
        "chi_square": float(chi2),
        "chi_square_p_value": float(p_value),
        "chi_square_degrees_freedom": int(dof),
        "cramers_v": cramers_v,
        "minimum_expected_count": float(expected.min()),
        "independent_at_alpha_0_05": bool(p_value >= 0.05),
    }


def _zscore_levels(profile: np.ndarray) -> np.ndarray:
    """Z-score each unit across context levels, mapping constant profiles to zero."""

    mean = profile.mean(axis=0, keepdims=True)
    scale = profile.std(axis=0, keepdims=True)
    return np.divide(
        profile - mean,
        scale,
        out=np.zeros_like(profile, dtype=np.float64),
        where=scale > 0,
    )


def two_way_decomposition(activations: np.ndarray, labels: np.ndarray) -> SelectivityResult:
    """Compute an equal-cell two-way variance decomposition for every unit.

    All 90 sector x digit cells receive equal weight. Between-cell effects use the same
    marginalization as the existing hidden-state variance analysis. Residual variance is the
    mean within-cell squared residual, summed over cells, which keeps it on the same equal-cell
    scale as the three between-cell components.
    """

    values = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != labels.shape[0]:
        raise ValueError("activations must be (trials, units) and align with labels")
    design = joint_design(labels)
    counts = np.asarray(
        design["joint_frequency_sector_rows_digit_columns"], dtype=np.int64
    ).reshape(-1)
    if np.any(counts == 0):
        raise RuntimeError("All 90 sector x digit cells are required for equal-cell decomposition")
    codes = labels[:, 1] * NUM_DIGITS + labels[:, 0]
    sums = np.zeros((NUM_CELLS, values.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, codes, values)
    np.add.at(sums_sq, codes, np.square(values))
    cell_mean = (sums / counts[:, None]).reshape(NUM_SECTORS, NUM_DIGITS, -1)
    cell_var = (sums_sq / counts[:, None] - np.square(sums / counts[:, None])).reshape(
        NUM_SECTORS, NUM_DIGITS, -1
    )
    np.maximum(cell_var, 0.0, out=cell_var)

    grand = cell_mean.mean(axis=(0, 1), keepdims=True)
    sector_mean = cell_mean.mean(axis=1)
    digit_mean = cell_mean.mean(axis=0)
    sector_effect = sector_mean - grand.reshape(1, -1)
    digit_effect = digit_mean - grand.reshape(1, -1)
    interaction = (
        cell_mean
        - grand
        - sector_effect[:, None, :]
        - digit_effect[None, :, :]
    )
    ss_sector = NUM_DIGITS * np.square(sector_effect).sum(axis=0)
    ss_digit = NUM_SECTORS * np.square(digit_effect).sum(axis=0)
    ss_interaction = np.square(interaction).sum(axis=(0, 1))
    ss_residual = cell_var.sum(axis=(0, 1))
    total = ss_sector + ss_digit + ss_interaction + ss_residual

    def fraction(component: np.ndarray) -> np.ndarray:
        return np.divide(
            component,
            total,
            out=np.zeros_like(component, dtype=np.float64),
            where=total > 0,
        )

    return SelectivityResult(
        eta_sector=fraction(ss_sector),
        eta_digit=fraction(ss_digit),
        eta_interaction=fraction(ss_interaction),
        eta_residual=fraction(ss_residual),
        tuning_sector=_zscore_levels(sector_mean),
        tuning_digit=_zscore_levels(digit_mean),
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return Benjamini-Hochberg rejection flags and adjusted q-values."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be a 1D array in [0, 1]")
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = p_values.size
    adjusted_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted <= alpha, adjusted


def _permuted_target(
    target: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle target labels independently within every level of the held-fixed factor."""

    permuted = target.copy()
    for level in np.unique(strata):
        indices = np.flatnonzero(strata == level)
        permuted[indices] = target[rng.permutation(indices)]
    return permuted


def _eta_from_cell_moments(
    cell_sum: torch.Tensor,
    cell_sum_sq: torch.Tensor,
    cell_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sector and digit eta-squared from batched equal-cell moments."""

    means = (cell_sum / cell_count[..., None]).reshape(
        -1, NUM_SECTORS, NUM_DIGITS, cell_sum.shape[-1]
    )
    second = (cell_sum_sq / cell_count[..., None]).reshape_as(means)
    cell_var = torch.clamp(second - means.square(), min=0.0)
    grand = means.mean(dim=(1, 2), keepdim=True)
    sector_effect = means.mean(dim=2) - grand[:, :, 0, :]
    digit_effect = means.mean(dim=1) - grand[:, 0, :, :]
    interaction = (
        means
        - grand
        - sector_effect[:, :, None, :]
        - digit_effect[:, None, :, :]
    )
    ss_sector = NUM_DIGITS * sector_effect.square().sum(dim=1)
    ss_digit = NUM_SECTORS * digit_effect.square().sum(dim=1)
    ss_interaction = interaction.square().sum(dim=(1, 2))
    ss_residual = cell_var.sum(dim=(1, 2))
    total = ss_sector + ss_digit + ss_interaction + ss_residual
    return ss_sector / total.clamp_min(1e-30), ss_digit / total.clamp_min(1e-30)


def permutation_selectivity(
    activations: np.ndarray,
    labels: np.ndarray,
    observed: SelectivityResult,
    *,
    resamples: int,
    seed: int,
    device: torch.device,
    permutation_batch_size: int = 10,
    fdr_alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Estimate per-unit sector/digit nulls using stratified label permutations.

    Sector is shuffled within digit and digit within sector. Dense batched indicator matrix
    multiplication is used so all units share each valid label permutation without assuming a
    shared null distribution.
    """

    if resamples <= 0 or permutation_batch_size <= 0:
        raise ValueError("resamples and permutation_batch_size must be positive")
    values_np = np.asarray(activations, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    values = torch.as_tensor(values_np, device=device)
    values_sq = values.square()
    digits, sectors = labels[:, 0], labels[:, 1]
    rng = np.random.default_rng(seed)
    nulls = {
        "sector": np.empty((resamples, values_np.shape[1]), dtype=np.float32),
        "digit": np.empty((resamples, values_np.shape[1]), dtype=np.float32),
    }
    for factor, target, strata in (
        ("sector", sectors, digits),
        ("digit", digits, sectors),
    ):
        for start in range(0, resamples, permutation_batch_size):
            stop = min(start + permutation_batch_size, resamples)
            batch_codes = []
            for _ in range(start, stop):
                shuffled = _permuted_target(target, strata, rng)
                shuffled_sector = shuffled if factor == "sector" else sectors
                shuffled_digit = digits if factor == "sector" else shuffled
                batch_codes.append(shuffled_sector * NUM_DIGITS + shuffled_digit)
            codes = torch.as_tensor(np.stack(batch_codes), device=device, dtype=torch.int64)
            one_hot = torch.nn.functional.one_hot(codes, NUM_CELLS).to(values.dtype)
            design = one_hot.transpose(1, 2)
            counts = design.sum(dim=2)
            cell_sum = torch.matmul(design, values)
            cell_sum_sq = torch.matmul(design, values_sq)
            eta_sector, eta_digit = _eta_from_cell_moments(cell_sum, cell_sum_sq, counts)
            selected = eta_sector if factor == "sector" else eta_digit
            nulls[factor][start:stop] = selected.detach().cpu().numpy().astype(np.float32)

    output: dict[str, np.ndarray] = {}
    for factor, observed_eta in (
        ("sector", observed.eta_sector),
        ("digit", observed.eta_digit),
    ):
        null = nulls[factor].astype(np.float64)
        null_mean = null.mean(axis=0)
        null_std = null.std(axis=0, ddof=1)
        z_score = np.divide(
            observed_eta - null_mean,
            null_std,
            out=np.zeros_like(observed_eta, dtype=np.float64),
            where=null_std > 0,
        )
        p_value = (1.0 + np.count_nonzero(null >= observed_eta[None, :], axis=0)) / (
            resamples + 1.0
        )
        passed, q_value = benjamini_hochberg(p_value, alpha=fdr_alpha)
        output[f"null_{factor}"] = null.astype(np.float32)
        output[f"z_{factor}"] = z_score.astype(np.float32)
        output[f"p_{factor}"] = p_value.astype(np.float32)
        output[f"q_{factor}"] = q_value.astype(np.float32)
        output[f"passed_{factor}"] = passed
    return output


def interaction_dominant(selectivity: SelectivityResult) -> np.ndarray:
    """Flag units whose interaction eta-squared exceeds both main effects."""

    return (selectivity.eta_interaction > selectivity.eta_sector) & (
        selectivity.eta_interaction > selectivity.eta_digit
    )


def architecture_axis_variance(selectivity: SelectivityResult) -> dict[str, float]:
    """Quantify whether encoder eta-squared varies more across space or channels."""

    if selectivity.eta_sector.size != 32 * 6 * 6:
        raise ValueError("architecture-axis audit requires exactly 1152 encoder units")
    output: dict[str, float] = {}
    for factor, eta in (
        ("sector", selectivity.eta_sector),
        ("digit", selectivity.eta_digit),
    ):
        shaped = eta.reshape(32, 6, 6)
        spatial_variance = float(shaped.mean(axis=0).var())
        channel_variance = float(shaped.mean(axis=(1, 2)).var())
        output[f"{factor}_variance_across_36_positions"] = spatial_variance
        output[f"{factor}_variance_across_32_channels"] = channel_variance
        output[f"{factor}_spatial_to_channel_variance_ratio"] = (
            spatial_variance / channel_variance if channel_variance > 0 else float("inf")
        )
    return output


def gate_context_moments(
    gate_columns: np.ndarray,
    labels: np.ndarray,
    factor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return context x unit gate sums, squared sums, and trial counts."""

    if factor not in ("sector", "digit"):
        raise ValueError("factor must be 'sector' or 'digit'")
    values = np.asarray(gate_columns, dtype=np.float64)
    context = np.asarray(labels[:, 1] if factor == "sector" else labels[:, 0], dtype=np.int64)
    levels = NUM_SECTORS if factor == "sector" else NUM_DIGITS
    sums = np.zeros((levels, values.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, context, values)
    np.add.at(sums_sq, context, np.square(values))
    return sums, sums_sq, np.bincount(context, minlength=levels).astype(np.int64)


def relevance_masks(
    tuning: np.ndarray,
    eligible: np.ndarray,
    top_fraction: float,
) -> np.ndarray:
    """Select the top tuned eligible units independently for every context level."""

    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    tuning = np.asarray(tuning, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    indices = np.flatnonzero(eligible)
    if indices.size < 2:
        raise RuntimeError("At least two eligible units are required for relevance comparison")
    count = max(1, int(np.ceil(top_fraction * indices.size)))
    count = min(count, indices.size - 1)
    masks = np.zeros_like(tuning, dtype=bool)
    for level in range(tuning.shape[0]):
        order = indices[np.argsort(tuning[level, indices], kind="stable")]
        masks[level, order[-count:]] = True
    return masks


def cohens_d_from_moments(moment: np.ndarray) -> float:
    """Compute pooled-standard-deviation Cohen's d from six sufficient statistics."""

    sum_rel, sumsq_rel, n_rel, sum_other, sumsq_other, n_other = map(float, moment)
    if n_rel <= 1 or n_other <= 1:
        return float("nan")
    mean_rel, mean_other = sum_rel / n_rel, sum_other / n_other
    var_rel = max(0.0, (sumsq_rel - sum_rel * sum_rel / n_rel) / (n_rel - 1.0))
    var_other = max(
        0.0, (sumsq_other - sum_other * sum_other / n_other) / (n_other - 1.0)
    )
    pooled = np.sqrt(((n_rel - 1) * var_rel + (n_other - 1) * var_other) / (
        n_rel + n_other - 2
    ))
    return (mean_rel - mean_other) / pooled if pooled > 0 else float("nan")


def trial_relevance_moments(
    gate_columns: np.ndarray,
    contexts: np.ndarray,
    masks: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Build per-trial moments for relevant versus other eligible gate columns."""

    gates = np.asarray(gate_columns, dtype=np.float64)
    contexts = np.asarray(contexts, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    output = np.zeros((gates.shape[0], 6), dtype=np.float64)
    for level in range(masks.shape[0]):
        trial_mask = contexts == level
        relevant = masks[level]
        other = eligible & ~relevant
        selected = gates[trial_mask]
        output[trial_mask, 0] = selected[:, relevant].sum(axis=1)
        output[trial_mask, 1] = np.square(selected[:, relevant]).sum(axis=1)
        output[trial_mask, 2] = int(np.count_nonzero(relevant))
        output[trial_mask, 3] = selected[:, other].sum(axis=1)
        output[trial_mask, 4] = np.square(selected[:, other]).sum(axis=1)
        output[trial_mask, 5] = int(np.count_nonzero(other))
    return output


def bootstrap_d(
    trial_moments: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Bootstrap trials and return point Cohen's d plus bootstrap draws."""

    moments = np.asarray(trial_moments, dtype=np.float64)
    point = cohens_d_from_moments(moments.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for idx in range(resamples):
        sampled = rng.integers(0, moments.shape[0], size=moments.shape[0])
        draws[idx] = cohens_d_from_moments(moments[sampled].sum(axis=0))
    return point, draws


def relevance_label_null(
    gate_columns: np.ndarray,
    labels: np.ndarray,
    factor: str,
    tuning: np.ndarray,
    eligible: np.ndarray,
    top_fraction: float,
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Shuffle each unit's context tuning labels and recompute exact pooled d."""

    sums, sums_sq, counts = gate_context_moments(gate_columns, labels, factor)
    rng = np.random.default_rng(seed)
    null = np.empty(resamples, dtype=np.float64)
    for sample_idx in range(resamples):
        permutations = rng.random(tuning.shape).argsort(axis=0)
        shuffled = np.take_along_axis(tuning, permutations, axis=0)
        masks = relevance_masks(shuffled, eligible, top_fraction)
        moment = np.zeros(6, dtype=np.float64)
        for level in range(tuning.shape[0]):
            relevant = masks[level]
            other = eligible & ~relevant
            moment += np.asarray(
                [
                    sums[level, relevant].sum(),
                    sums_sq[level, relevant].sum(),
                    counts[level] * np.count_nonzero(relevant),
                    sums[level, other].sum(),
                    sums_sq[level, other].sum(),
                    counts[level] * np.count_nonzero(other),
                ],
                dtype=np.float64,
            )
        null[sample_idx] = cohens_d_from_moments(moment)
    return null


def cosine_alignment(
    activation_tuning: np.ndarray,
    gate_columns: np.ndarray,
    labels: np.ndarray,
    factor: str,
    eligible: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute context cross-alignment, diagonal contrast, and permutation p-value."""

    sums, _sums_sq, counts = gate_context_moments(gate_columns, labels, factor)
    gate_tuning = sums / counts[:, None]
    eligible = np.asarray(eligible, dtype=bool)
    activation = np.asarray(activation_tuning[:, eligible], dtype=np.float64)
    gate = np.asarray(gate_tuning[:, eligible], dtype=np.float64)
    activation = _zscore_rows(activation)
    gate = _zscore_rows(gate)
    matrix = _cosine_rows(activation, gate)
    point = diagonal_contrast(matrix)
    rng = np.random.default_rng(seed)
    null = np.empty(resamples, dtype=np.float64)
    for idx in range(resamples):
        null[idx] = diagonal_contrast(matrix[:, rng.permutation(matrix.shape[1])])
    p_value = float((1 + np.count_nonzero(null >= point)) / (resamples + 1))
    return {
        "matrix": matrix.astype(np.float32),
        "diagonal_minus_off_diagonal": point,
        "permutation_null": null.astype(np.float32),
        "permutation_p_value": p_value,
    }


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return np.divide(values - mean, scale, out=np.zeros_like(values), where=scale > 0)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def diagonal_contrast(matrix: np.ndarray) -> float:
    """Return mean diagonal minus mean off-diagonal for a square matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    diagonal = np.diag(matrix).mean()
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)].mean()
    return float(diagonal - off_diagonal)


def bootstrap_onset(
    event_values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Find first post frame above post1 using paired event bootstrap confidence intervals."""

    values = np.asarray(event_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("event_values must be (events, post_frames>=2)")

    def onset_for(sample: np.ndarray) -> int:
        delta = sample - sample[:, [0]]
        for time_idx in range(1, sample.shape[1]):
            boot_mean = delta[:, time_idx].mean()
            if boot_mean > 0:
                return time_idx + 1
        return sample.shape[1] + 1

    rng = np.random.default_rng(seed)
    deltas = values - values[:, [0]]
    lower = np.empty(values.shape[1], dtype=np.float64)
    upper = np.empty_like(lower)
    for time_idx in range(values.shape[1]):
        draws = np.empty(resamples, dtype=np.float64)
        for sample_idx in range(resamples):
            selected = rng.integers(0, values.shape[0], size=values.shape[0])
            draws[sample_idx] = deltas[selected, time_idx].mean()
        lower[time_idx], upper[time_idx] = np.quantile(draws, [0.025, 0.975])
    significant = np.flatnonzero((lower > 0) & (np.arange(values.shape[1]) > 0))
    point_onset = int(significant[0] + 1) if significant.size else values.shape[1] + 1

    onset_draws = np.empty(resamples, dtype=np.int64)
    for sample_idx in range(resamples):
        selected = rng.integers(0, values.shape[0], size=values.shape[0])
        onset_draws[sample_idx] = onset_for(values[selected])
    return {
        "onset_post_frame": point_onset,
        "onset_ci95": [float(x) for x in np.quantile(onset_draws, [0.025, 0.975])],
        "mean": values.mean(axis=0).astype(np.float32),
        "delta_ci95_lower": lower.astype(np.float32),
        "delta_ci95_upper": upper.astype(np.float32),
        "onset_bootstrap": onset_draws,
    }


def first_crossing(values: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Return 1-based first frame meeting a threshold, or NaN if never reached."""

    values = np.asarray(values)
    hits = values >= threshold
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    any_hit = hits.any(axis=1)
    output[any_hit] = hits[any_hit].argmax(axis=1) + 1
    return output


def first_negative_to_nonnegative(values: np.ndarray) -> np.ndarray:
    """Return first 1-based negative-to-nonnegative crossing after a negative frame.

    A trajectory that begins nonnegative is not counted as reconfigured at post1. It must first
    become negative and then cross back to zero or above. This enforces the causal-direction
    definition and exposes curves whose direction contradicts the assumed old-to-new transition.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape (events, time)")
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    for event_idx, event in enumerate(values):
        has_been_negative = bool(event[0] < 0)
        for time_idx in range(1, event.size):
            if has_been_negative and event[time_idx] >= 0:
                output[event_idx] = time_idx + 1
                break
            has_been_negative = has_been_negative or bool(event[time_idx] < 0)
    return output


def paired_lead_test(gate_frame: np.ndarray, readout_frame: np.ndarray) -> dict[str, Any]:
    """Test whether gate crossing precedes readout using paired finite event times."""

    gate = np.asarray(gate_frame, dtype=np.float64)
    readout = np.asarray(readout_frame, dtype=np.float64)
    valid = np.isfinite(gate) & np.isfinite(readout)
    difference = readout[valid] - gate[valid]
    if difference.size == 0:
        return {
            "n_paired_events": 0,
            "difference_definition": (
                "readout_frame_minus_gate_frame; positive means gate leads"
            ),
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "fraction_gate_leads": float("nan"),
            "fraction_tied": float("nan"),
            "wilcoxon_greater_statistic": float("nan"),
            "wilcoxon_greater_p_value": float("nan"),
            "differences": difference.astype(np.float32),
            "valid_mask": valid,
        }
    wilcoxon = stats.wilcoxon(difference, alternative="greater", zero_method="zsplit")
    return {
        "n_paired_events": int(difference.size),
        "difference_definition": "readout_frame_minus_gate_frame; positive means gate leads",
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "fraction_gate_leads": float(np.mean(difference > 0)),
        "fraction_tied": float(np.mean(difference == 0)),
        "wilcoxon_greater_statistic": float(wilcoxon.statistic),
        "wilcoxon_greater_p_value": float(wilcoxon.pvalue),
        "differences": difference.astype(np.float32),
        "valid_mask": valid,
    }
