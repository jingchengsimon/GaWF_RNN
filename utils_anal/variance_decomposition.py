"""Balanced two-factor variance decomposition for every GaWF representation.

The module consumes trial batches shaped ``(trials, units)`` plus integer labels shaped
``(trials, 2)`` in ``[digit, sector]`` order.  For gates, a unit is a **synapse**, not a
neuron.  Outputs include aggregate and per-unit condition-mean and trial-level fractions.

Every one of the 90 ``(sector, digit)`` cells must be subsampled to a shared ``n``.  With
equal cell counts, the trial-weighted grand mean and the unweighted mean of cell means
coincide, while sector and digit marginals are orthogonal and their sums of squares add
exactly.  Neither property holds for the historical unbalanced design; that is why its
hidden-state fractions summed to about 99.6% and why those results were not directly
comparable with gate decompositions.

The streaming accumulator stores only ``S1_u``, ``S2_u``, and ``C1_u,k``.  In particular,
it never materializes a ``(n_trials, n_synapses)`` tensor.  All accumulator and temporary
allocations are checked against a configurable memory budget before allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable

import numpy as np


NUM_SECTORS = 9
NUM_DIGITS = 10
NUM_CELLS = NUM_SECTORS * NUM_DIGITS
CM_FACTORS = ("sector", "digit", "interaction")
TRIAL_FACTORS = (*CM_FACTORS, "residual")
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class BalanceReport:
    """Metadata for repeated equal-cell subsampling."""

    n_per_cell: int
    total_trials_available: int
    trials_retained_per_draw: int
    trials_discarded_per_draw: int
    repeats: int
    seed: int


@dataclass(frozen=True)
class DecompositionResult:
    """All four decomposition cells and their sufficient statistics."""

    aggregate_cm: dict[str, float]
    aggregate_trial: dict[str, float]
    per_unit_cm: dict[str, np.ndarray]
    per_unit_trial: dict[str, np.ndarray]
    sum_squares: dict[str, np.ndarray]
    unweighted_per_unit_mean_cm: dict[str, float]
    unweighted_per_unit_mean_trial: dict[str, float]
    consistency: dict[str, float]
    n_per_cell: int
    total_trials: int


@dataclass(frozen=True)
class RepeatedDecomposition:
    """Memory-bounded repeated decomposition across unit blocks."""

    aggregate_cm: dict[str, np.ndarray]
    aggregate_trial: dict[str, np.ndarray]
    per_unit_cm: dict[str, np.ndarray]
    per_unit_trial: dict[str, np.ndarray]
    unweighted_per_unit_mean_cm: dict[str, np.ndarray]
    unweighted_per_unit_mean_trial: dict[str, np.ndarray]
    consistency: dict[str, np.ndarray]


def _require_memory(required_bytes: int, budget_bytes: int, allocation: str) -> None:
    if required_bytes > budget_bytes:
        mib = required_bytes / 1024**2
        budget_mib = budget_bytes / 1024**2
        raise MemoryError(
            f"Refusing {allocation}: {mib:.1f} MiB exceeds memory budget " f"{budget_mib:.1f} MiB"
        )


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("labels must have shape (trials, 2) in [digit, sector] order")
    digits, sectors = labels[:, 0], labels[:, 1]
    if np.any((digits < 0) | (digits >= NUM_DIGITS)):
        raise ValueError("digit labels must be in [0, 9]")
    if np.any((sectors < 0) | (sectors >= NUM_SECTORS)):
        raise ValueError("sector labels must be in [0, 8]")
    return labels


def balanced_subsample_indices(
    labels: np.ndarray,
    *,
    repeats: int = 20,
    seed: int = 0,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> tuple[list[np.ndarray], BalanceReport]:
    """Return fixed-seed indices for repeated balanced draws over all 90 cells."""

    labels = _validate_labels(labels)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    codes = labels[:, 1] * NUM_DIGITS + labels[:, 0]
    by_cell = [np.flatnonzero(codes == cell) for cell in range(NUM_CELLS)]
    counts = np.asarray([indices.size for indices in by_cell], dtype=np.int64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise RuntimeError(f"All 90 sector x digit cells are required; missing cells: {missing}")
    n_per_cell = int(counts.min())
    retained = n_per_cell * NUM_CELLS
    _require_memory(
        repeats * retained * np.dtype(np.int64).itemsize,
        memory_budget_bytes,
        "repeated balanced trial-index draws",
    )
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(repeats):
        selected = np.concatenate(
            [rng.choice(indices, size=n_per_cell, replace=False) for indices in by_cell]
        )
        draws.append(np.sort(selected).astype(np.int64, copy=False))
    return draws, BalanceReport(
        n_per_cell=n_per_cell,
        total_trials_available=int(labels.shape[0]),
        trials_retained_per_draw=retained,
        trials_discarded_per_draw=int(labels.shape[0] - retained),
        repeats=repeats,
        seed=seed,
    )


class StreamingMoments:
    """Accumulate second-order statistics for one balanced decomposition."""

    def __init__(
        self,
        num_units: int,
        *,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    ) -> None:
        if num_units <= 0 or memory_budget_bytes <= 0:
            raise ValueError("num_units and memory_budget_bytes must be positive")
        accumulator_bytes = (NUM_CELLS + 2) * num_units * np.dtype(np.float64).itemsize
        _require_memory(accumulator_bytes, memory_budget_bytes, "streaming accumulators")
        self.num_units = num_units
        self.memory_budget_bytes = memory_budget_bytes
        self.cell_sum = np.zeros((NUM_CELLS, num_units), dtype=np.float64)
        self.total_sum = np.zeros(num_units, dtype=np.float64)
        self.total_sum_sq = np.zeros(num_units, dtype=np.float64)
        self.cell_count = np.zeros(NUM_CELLS, dtype=np.int64)

    def update(self, values: np.ndarray, labels: np.ndarray) -> None:
        """Accumulate one trial batch without retaining it."""

        labels = _validate_labels(labels)
        raw = np.asarray(values)
        if raw.ndim < 2 or raw.shape[0] != labels.shape[0]:
            raise ValueError("values must be (trials, ...) and align with labels")
        flattened_units = int(np.prod(raw.shape[1:]))
        if flattened_units != self.num_units:
            raise ValueError(f"Expected {self.num_units} units, got {flattened_units}")
        temporary_bytes = raw.shape[0] * self.num_units * np.dtype(np.float64).itemsize
        accumulator_bytes = (NUM_CELLS + 2) * self.num_units * 8
        _require_memory(
            accumulator_bytes + temporary_bytes,
            self.memory_budget_bytes,
            "accumulators plus float64 input batch",
        )
        batch = np.asarray(raw.reshape(raw.shape[0], self.num_units), dtype=np.float64)
        if not np.all(np.isfinite(batch)):
            raise ValueError("values contain NaN or infinite entries")
        codes = labels[:, 1] * NUM_DIGITS + labels[:, 0]
        self.total_sum += batch.sum(axis=0, dtype=np.float64)
        self.total_sum_sq += np.einsum("ij,ij->j", batch, batch, dtype=np.float64)
        for cell in np.unique(codes):
            mask = codes == cell
            self.cell_sum[cell] += batch[mask].sum(axis=0, dtype=np.float64)
            self.cell_count[cell] += int(np.count_nonzero(mask))

    def finalize(self) -> DecompositionResult:
        """Compute all four cells, requiring a nonempty exactly balanced design."""

        counts = self.cell_count
        if np.any(counts == 0) or not np.all(counts == counts[0]):
            raise RuntimeError(
                "Balanced decomposition requires the same positive trial count in all 90 cells; "
                f"observed min={counts.min()}, max={counts.max()}"
            )
        return decomposition_from_moments(
            self.total_sum,
            self.total_sum_sq,
            self.cell_sum,
            counts,
        )


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=output, where=denominator > 0)


def decomposition_from_moments(
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
    cell_sum: np.ndarray,
    cell_count: np.ndarray,
) -> DecompositionResult:
    """Compute the exact balanced decomposition from second-order accumulators."""

    total_sum = np.asarray(total_sum, dtype=np.float64)
    total_sum_sq = np.asarray(total_sum_sq, dtype=np.float64)
    cell_sum = np.asarray(cell_sum, dtype=np.float64)
    cell_count = np.asarray(cell_count, dtype=np.int64)
    if cell_sum.shape != (NUM_CELLS, total_sum.size):
        raise ValueError("cell_sum must have shape (90, units)")
    if total_sum_sq.shape != total_sum.shape or cell_count.shape != (NUM_CELLS,):
        raise ValueError("incompatible sufficient-statistic shapes")
    if np.any(cell_count <= 0) or not np.all(cell_count == cell_count[0]):
        raise RuntimeError("decomposition_from_moments requires equal positive cell counts")

    n = int(cell_count[0])
    total_trials = int(cell_count.sum())
    means = (cell_sum / n).reshape(NUM_SECTORS, NUM_DIGITS, -1)
    grand = means.mean(axis=(0, 1))
    sector_effect = means.mean(axis=1) - grand
    digit_effect = means.mean(axis=0) - grand
    interaction = (
        means - grand[None, None, :] - sector_effect[:, None, :] - digit_effect[None, :, :]
    )
    ss_sector = n * NUM_DIGITS * np.square(sector_effect).sum(axis=0)
    ss_digit = n * NUM_SECTORS * np.square(digit_effect).sum(axis=0)
    ss_interaction = n * np.square(interaction).sum(axis=(0, 1))
    ss_total_cm = ss_sector + ss_digit + ss_interaction
    ss_total_trial = total_sum_sq - np.square(total_sum) / total_trials
    ss_between = np.square(cell_sum).sum(axis=0) / n - np.square(total_sum) / total_trials
    ss_residual = ss_total_trial - ss_total_cm
    tolerance = 1e-9 * np.maximum(1.0, np.abs(ss_total_trial))
    if np.any(ss_residual < -tolerance):
        raise RuntimeError(
            f"Negative residual SS exceeds tolerance: minimum {float(ss_residual.min())}"
        )
    ss_residual = np.maximum(ss_residual, 0.0)

    ss = {
        "sector": ss_sector,
        "digit": ss_digit,
        "interaction": ss_interaction,
        "residual": ss_residual,
        "total_cm": ss_total_cm,
        "total_trial": ss_total_trial,
        "between": ss_between,
    }
    per_unit_cm = {factor: _divide(ss[factor], ss_total_cm) for factor in CM_FACTORS}
    per_unit_trial = {factor: _divide(ss[factor], ss_total_trial) for factor in TRIAL_FACTORS}
    total_cm_scalar = float(ss_total_cm.sum())
    total_trial_scalar = float(ss_total_trial.sum())
    if total_cm_scalar <= 0 or total_trial_scalar <= 0:
        raise RuntimeError("Total condition-mean and trial variance must both be positive")
    aggregate_cm = {factor: float(ss[factor].sum() / total_cm_scalar) for factor in CM_FACTORS}
    aggregate_trial = {
        factor: float(ss[factor].sum() / total_trial_scalar) for factor in TRIAL_FACTORS
    }

    weighted_deviations = []
    renormalization_ratios = []
    nonresidual = 1.0 - aggregate_trial["residual"]
    for factor in CM_FACTORS:
        weighted = float(np.nansum(per_unit_cm[factor] * ss_total_cm) / total_cm_scalar)
        weighted_deviations.append(abs(weighted - aggregate_cm[factor]))
        renormalized = aggregate_trial[factor] / nonresidual
        renormalization_ratios.append(renormalized / aggregate_cm[factor])
    zero_sum = max(
        float(np.abs(sector_effect.sum(axis=0)).max()),
        float(np.abs(digit_effect.sum(axis=0)).max()),
        float(np.abs(interaction.sum(axis=0)).max()),
        float(np.abs(interaction.sum(axis=1)).max()),
    )
    consistency = {
        "aggregate_weighted_per_unit_max_abs_deviation": max(weighted_deviations),
        "condition_mean_trial_renormalization_max_abs_deviation_from_one": max(
            abs(ratio - 1.0) for ratio in renormalization_ratios
        ),
        "condition_mean_trial_renormalization_ratio_sector": renormalization_ratios[0],
        "condition_mean_trial_renormalization_ratio_digit": renormalization_ratios[1],
        "condition_mean_trial_renormalization_ratio_interaction": renormalization_ratios[2],
        "zero_sum_max_abs_violation": zero_sum,
        "between_vs_condition_mean_max_abs_deviation": float(
            np.abs(ss_between - ss_total_cm).max()
        ),
    }
    return DecompositionResult(
        aggregate_cm=aggregate_cm,
        aggregate_trial=aggregate_trial,
        per_unit_cm=per_unit_cm,
        per_unit_trial=per_unit_trial,
        sum_squares=ss,
        unweighted_per_unit_mean_cm={
            factor: float(np.nanmean(per_unit_cm[factor])) for factor in CM_FACTORS
        },
        unweighted_per_unit_mean_trial={
            factor: float(np.nanmean(per_unit_trial[factor])) for factor in TRIAL_FACTORS
        },
        consistency=consistency,
        n_per_cell=n,
        total_trials=total_trials,
    )


def unbalanced_condition_mean_bridge(
    total_sum: np.ndarray,
    cell_sum: np.ndarray,
    cell_count: np.ndarray,
) -> dict[str, float]:
    """Reproduce ``5_dpca_marginalized_variance.raw_marginalized_variance`` exactly.

    This diagnostic bridge uses an unweighted ``nanmean`` over nonempty condition means, as the
    historical pipeline does. It is not one of the four canonical balanced cells.
    """

    total_sum = np.asarray(total_sum, dtype=np.float64)
    cell_sum = np.asarray(cell_sum, dtype=np.float64)
    counts = np.asarray(cell_count, dtype=np.int64).reshape(NUM_SECTORS, NUM_DIGITS)
    if cell_sum.shape != (NUM_CELLS, total_sum.size) or not np.any(counts > 0):
        raise ValueError("The unbalanced bridge requires at least one nonempty cell moment")
    sums = cell_sum.reshape(NUM_SECTORS, NUM_DIGITS, -1)
    means = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts[..., None], out=means, where=counts[..., None] > 0)
    x_masked = means.transpose(2, 1, 0)
    mask = counts.T > 0
    grand = np.nanmean(x_masked, axis=(1, 2))
    digit_effect = np.nanmean(x_masked, axis=2) - grand[:, None]
    sector_effect = np.nanmean(x_masked, axis=1) - grand[:, None]
    digit_bc = np.broadcast_to(digit_effect[:, :, None], x_masked.shape)
    sector_bc = np.broadcast_to(sector_effect[:, None, :], x_masked.shape)
    interaction = x_masked - grand[:, None, None] - digit_bc - sector_bc
    valid = mask[None, :, :]
    components = {
        "sector": float(np.nansum(np.where(valid, sector_bc, np.nan) ** 2)),
        "digit": float(np.nansum(np.where(valid, digit_bc, np.nan) ** 2)),
        "interaction": float(np.nansum(np.where(valid, interaction, np.nan) ** 2)),
    }
    centered = x_masked - grand[:, None, None]
    total = float(np.nansum(np.where(valid, centered, np.nan) ** 2))
    if total <= 0:
        raise RuntimeError("Unbalanced bridge condition means have zero variance")
    fractions = {factor: float(value / total) for factor, value in components.items()}
    return {**fractions, "sum": float(sum(fractions.values()))}


def decompose_array(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    selected_indices: np.ndarray | None = None,
    batch_size: int = 32,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> DecompositionResult:
    """Decompose an array or mmap while only materializing one trial batch at a time."""

    raw = np.asarray(values)
    labels = _validate_labels(labels)
    if raw.ndim < 2 or raw.shape[0] != labels.shape[0]:
        raise ValueError("values and labels must align on the trial axis")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = (
        np.arange(labels.shape[0], dtype=np.int64)
        if selected_indices is None
        else np.asarray(selected_indices, dtype=np.int64)
    )
    if indices.ndim != 1 or np.any((indices < 0) | (indices >= labels.shape[0])):
        raise ValueError("selected_indices contains an invalid trial index")
    accumulator = StreamingMoments(
        int(np.prod(raw.shape[1:])), memory_budget_bytes=memory_budget_bytes
    )
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        accumulator.update(raw[batch_indices], labels[batch_indices])
    return accumulator.finalize()


def summarize_repeated(
    results: Iterable[DecompositionResult],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return mean and percentile 95% interval for every aggregate fraction."""

    collected = list(results)
    if not collected:
        raise ValueError("At least one decomposition result is required")
    summary: dict[str, dict[str, tuple[float, float, float]]] = {}
    for variant, factors in (("aggregate_cm", CM_FACTORS), ("aggregate_trial", TRIAL_FACTORS)):
        summary[variant] = {}
        for factor in factors:
            values = np.asarray([getattr(result, variant)[factor] for result in collected])
            low, high = np.quantile(values, [0.025, 0.975])
            summary[variant][factor] = (float(values.mean()), float(low), float(high))
    return summary


def decompose_repeated_blocks(
    read_block: Callable[[np.ndarray, slice], np.ndarray],
    labels: np.ndarray,
    selected_draws: list[np.ndarray],
    *,
    num_units: int,
    unit_block_size: int = 4096,
    trial_batch_size: int = 32,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> RepeatedDecomposition:
    """Run repeated balanced decompositions without constructing a trial-by-unit tensor.

    ``read_block(trial_indices, unit_slice)`` must return only the requested
    ``(trial, unit)`` block.  Each unit block is read once and shared across all repeated
    subsamples, keeping the 20-draw input-gate case within a bounded memory footprint.
    """

    labels = _validate_labels(labels)
    repeats = len(selected_draws)
    if repeats <= 0 or num_units <= 0 or unit_block_size <= 0 or trial_batch_size <= 0:
        raise ValueError("draws, unit counts, and block sizes must be positive")
    per_unit_bytes = repeats * num_units * len(TRIAL_FACTORS + CM_FACTORS) * 4
    largest_block = min(unit_block_size, num_units)
    block_bytes = (
        repeats * (NUM_CELLS + 2) * largest_block * 8
        + (NUM_CELLS + NUM_SECTORS + NUM_DIGITS + 8) * largest_block * 8
    )
    temporary_bytes = trial_batch_size * largest_block * 8
    membership_bytes = repeats * labels.shape[0] * np.dtype(bool).itemsize
    draw_bytes = sum(np.asarray(indices).nbytes for indices in selected_draws)
    _require_memory(
        per_unit_bytes + block_bytes + temporary_bytes + membership_bytes + draw_bytes,
        memory_budget_bytes,
        "repeated outputs, accumulators, trial membership, and input batch",
    )
    membership = np.zeros((repeats, labels.shape[0]), dtype=bool)
    for repeat, indices in enumerate(selected_draws):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any((indices < 0) | (indices >= labels.shape[0])):
            raise ValueError("selected_draws contains an invalid trial index")
        membership[repeat, indices] = True
    per_unit_cm = {
        factor: np.full((repeats, num_units), np.nan, dtype=np.float32) for factor in CM_FACTORS
    }
    per_unit_trial = {
        factor: np.full((repeats, num_units), np.nan, dtype=np.float32) for factor in TRIAL_FACTORS
    }
    ss_sums = {
        name: np.zeros(repeats, dtype=np.float64)
        for name in (*TRIAL_FACTORS, "total_cm", "total_trial")
    }
    weighted_numerators = {factor: np.zeros(repeats, dtype=np.float64) for factor in CM_FACTORS}
    zero_sum = np.zeros(repeats, dtype=np.float64)
    between_deviation = np.zeros(repeats, dtype=np.float64)

    all_trials = np.arange(labels.shape[0], dtype=np.int64)
    for unit_start in range(0, num_units, unit_block_size):
        unit_stop = min(unit_start + unit_block_size, num_units)
        unit_slice = slice(unit_start, unit_stop)
        block_units = unit_stop - unit_start
        accumulators = [
            StreamingMoments(block_units, memory_budget_bytes=memory_budget_bytes)
            for _ in range(repeats)
        ]
        for trial_start in range(0, labels.shape[0], trial_batch_size):
            trial_indices = all_trials[trial_start : trial_start + trial_batch_size]
            active = membership[:, trial_indices]
            if not np.any(active):
                continue
            values = np.asarray(read_block(trial_indices, unit_slice))
            if values.shape != (trial_indices.size, block_units):
                raise ValueError(
                    "read_block returned shape "
                    f"{values.shape}, expected {(trial_indices.size, block_units)}"
                )
            for repeat, accumulator in enumerate(accumulators):
                selected = active[repeat]
                if np.any(selected):
                    accumulator.update(values[selected], labels[trial_indices[selected]])
        for repeat, accumulator in enumerate(accumulators):
            result = accumulator.finalize()
            for factor in CM_FACTORS:
                per_unit_cm[factor][repeat, unit_slice] = result.per_unit_cm[factor].astype(
                    np.float32
                )
                weighted_numerators[factor][repeat] += float(
                    np.nansum(result.per_unit_cm[factor] * result.sum_squares["total_cm"])
                )
            for factor in TRIAL_FACTORS:
                per_unit_trial[factor][repeat, unit_slice] = result.per_unit_trial[factor].astype(
                    np.float32
                )
            for name in ss_sums:
                ss_sums[name][repeat] += float(result.sum_squares[name].sum())
            zero_sum[repeat] = max(
                zero_sum[repeat], result.consistency["zero_sum_max_abs_violation"]
            )
            between_deviation[repeat] = max(
                between_deviation[repeat],
                result.consistency["between_vs_condition_mean_max_abs_deviation"],
            )

    aggregate_cm = {factor: ss_sums[factor] / ss_sums["total_cm"] for factor in CM_FACTORS}
    aggregate_trial = {factor: ss_sums[factor] / ss_sums["total_trial"] for factor in TRIAL_FACTORS}
    unweighted_cm = {factor: np.nanmean(per_unit_cm[factor], axis=1) for factor in CM_FACTORS}
    unweighted_trial = {
        factor: np.nanmean(per_unit_trial[factor], axis=1) for factor in TRIAL_FACTORS
    }
    weighted_deviation = np.zeros(repeats, dtype=np.float64)
    renormalization_deviation = np.zeros(repeats, dtype=np.float64)
    renormalization_ratios: dict[str, np.ndarray] = {}
    for factor in CM_FACTORS:
        weighted = weighted_numerators[factor] / ss_sums["total_cm"]
        weighted_deviation = np.maximum(weighted_deviation, np.abs(weighted - aggregate_cm[factor]))
        ratios = (
            aggregate_trial[factor] / (1.0 - aggregate_trial["residual"]) / aggregate_cm[factor]
        )
        renormalization_ratios[factor] = ratios
        renormalization_deviation = np.maximum(renormalization_deviation, np.abs(ratios - 1.0))
    consistency = {
        "aggregate_weighted_per_unit_max_abs_deviation": weighted_deviation,
        "condition_mean_trial_renormalization_max_abs_deviation_from_one": (
            renormalization_deviation
        ),
        "zero_sum_max_abs_violation": zero_sum,
        "between_vs_condition_mean_max_abs_deviation": between_deviation,
        **{
            f"condition_mean_trial_renormalization_ratio_{factor}": values
            for factor, values in renormalization_ratios.items()
        },
    }
    return RepeatedDecomposition(
        aggregate_cm=aggregate_cm,
        aggregate_trial=aggregate_trial,
        per_unit_cm=per_unit_cm,
        per_unit_trial=per_unit_trial,
        unweighted_per_unit_mean_cm=unweighted_cm,
        unweighted_per_unit_mean_trial=unweighted_trial,
        consistency=consistency,
    )
