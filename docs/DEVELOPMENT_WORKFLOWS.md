# Development Workflows

Read this document before adding or changing training, analysis, or visualisation scripts. Public
names and output formats are defined in `CONVENTIONS.md`; architecture boundaries are defined in
`ARCHITECTURE.md`.

## General Python rules

- Use Python 3.10+ type hints on public functions and `from __future__ import annotations`.
- Start every module with a docstring describing purpose, inputs, and outputs.
- Keep lines within 100 characters and avoid wildcard imports.
- Use the project logger for training progress; `print()` is acceptable for standalone analysis
  progress and explicit CLI diagnostics.
- Create output directories before writing and save NumPy arrays as `float32` or `int64`.
- Preserve unrelated worktree changes and update all imports/call sites after public renames.

## Training changes

### Clutter

- Keep `utils/training/train_scripts/clutter.py` as orchestration; heavy loop logic belongs in
  `utils/clutter_train_engine.py`.
- Register model types through `utils/clutter_train_helpers.get_model_classes()`.
- Build losses through the factories in `clutter_train_sector.py` or
  `clutter_train_predict_all_chars.py`.
- Put recurrent computation in `utils/training/recurrent_cores/`, not task wrappers.
- Extend `AccelerationConfig` rather than adding acceleration branches to the training loop.
- Official curves use the train-eval and validation passes, not online batch averages.
- Standard long jobs pass `--checkpoint_interval_epochs 5 --auto_resume`. Checkpoints are atomic
  and occur only after a complete epoch, so interruption loses at most four completed epochs.
  Resume rejects model, optimizer, data-pipeline, seed, or hyperparameter mismatches. Dataset
  samples are deterministic, so restoring the loader and sampler generators is sufficient even
  though persistent worker process internals are recreated.
- A signal-triggered stop must retain the last periodic training checkpoint and exit without
  writing final `.pkl`, metrics JSON, or best-model artifacts from partial state.

### mmap and devices

- **Historical reproduction only:** the legacy `40h-float32` mmap pipeline used
  `num_workers=0` and `pin_memory=False`. Keep this configuration available when reproducing
  historical runs, but do not treat it as a general mmap requirement or the default for new
  experiments.
- **Current standard Clutter 40h configuration:** use `40h-uint8` with `--use_mmap`,
  `--input_cast_mode device`, `--frame_layout compact`, and `--shuffle_block_size -1` together
  with `AIM3_NUM_WORKERS=2` and `AIM3_PIN_MEMORY=1` on CUDA compute nodes. The loaders use
  persistent workers and `prefetch_factor=2`; the batch-sized block sampler preserves epoch
  coverage while reducing random shared-filesystem access.
- mmap and pinned memory are not inherently incompatible: mmap backs the CPU dataset, while
  pinning applies to the collated uint8 batches transferred asynchronously to CUDA. Do not raise
  the standard two-worker value without an endpoint-specific benchmark; more workers can increase
  page faults and shared-filesystem contention.
- Convert float64 inputs to float32 before MPS/CUDA transfer.
- Do not load the full dataset onto the accelerator.
- Scope `torch.no_grad()` to evaluation/inference blocks.
- For complex-parameter cores such as S5, AMP autocast may remain enabled while GradScaler and
  foreach clipping exclude unsupported complex gradients.

### Checkpoint loading

Use the canonical compatibility pattern and always report incompatibilities:

```python
state_dict = torch.load(ckpt_path, map_location=device)
state_dict = {key: value for key, value in state_dict.items() if key != "prev_feedback"}
incompatible = model.load_state_dict(state_dict, strict=False)
print("missing_keys:", incompatible.missing_keys)
print("unexpected_keys:", incompatible.unexpected_keys)
```

## Analysis scripts

New analysis belongs in `utils/analysis/` and exports one logical result set per invocation.

Required structure:

```python
"""One-line summary.

Describe inputs, computation, and every output with shape and dtype.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Reuse the canonical helpers:

```python
from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset
```

Analysis requirements:

- Resolve every analysis data/figure destination with
  `utils.analysis.anal_paths.output_dir(category, script_name, kind)`. Figures go directly to
  `results/figs/<CATEGORY>/`; data and the run manifest go to the parallel
  `results/data/analysis/<CATEGORY>/<script_name>/`. A task-specific workflow may explicitly
  route one curated final figure to `results/save/`; its structured data still uses `output_dir`.
  migration notes; do not recreate nested `data/` or `figs/` directories there.
- Each run writes `manifest.json` in its `anal_data` directory containing script path, commit,
  timestamp, category, both parallel roots, files written, and a flat dictionary of key numerical
  results.
- Accumulate averages/statistics in float64 and cast to float32 before saving.
- Use `.npy` for one array and `.npz` for related arrays.
- Save companion metadata with mode, selected index, frame/sample counts, model/input sizes,
  aggregation mode, absolute checkpoint path, and spatial/feature shapes.
- Print qualifying-sample progress every 200 samples.
- Raise `RuntimeError` when no frames match; do not silently emit empty outputs.
- Keep numeric analysis and plotting in the same `utils/analysis/` task module.

### Continuous switch trajectories and PCA

研究方法、八步分析顺序和当前图形的解释边界见
[DYNAMICAL_SYSTEM_ANALYSIS_PROTOCOL.md](DYNAMICAL_SYSTEM_ANALYSIS_PROTOCOL.md)。
本节维护实现与输出约定。

`utils.analysis.clutter.continuous_switch_pca` is the fixed-point preparation workflow for
joint-balanced CM-MNIST.  It performs inference only: start one raw movie at the canonical
zero state, retain RNN/GaWF runtime state across 32-frame compute chunks, and never reset at a
switch.  It stores all stream records separately from clean event-aligned windows.  The initial
PCA uses only raw recurrent state, is centred without per-unit standardization, and is fit and
reported independently for every model seed; digit/sector labels are retained but do not define
the first PCA grouping.  Its chronological event holdout removes each training window that shares
a raw frame with a held-out window and records the resulting event-index provenance.
Variance is summarized in one figure with model-wise seed means and sample SD (`ddof=1`),
showing both the first 20 PCs and full spectrum. `plot --variance_only` reuses saved seed PCA
arrays. Seed coordinates and bases remain independent; numerical aggregates are saved in the
data directory as `variance_<model>_mean_std.npz`.
Each 3D trajectory marks its start (purple circle), switch (red X), and end (yellow square),
with black arrows following successive states at quarter and three-quarter window positions.
For radius 50, the exact endpoints are -50 and +49; switch 0 is the first post record.
`plot --interactive_only` exports one `continuous_switch_pca_interactive.ipynb` from saved PCA
arrays. Its first code cell loops over GaWF seeds 1–10; the second loops over RNN seeds 1–10.
Each cell embeds all ten Plotly outputs. Interactive figures use only time-coloured points and
lines with one colorbar marked at -50, 0 and +49, plus red X markers at switch 0;
no event legend, endpoint markers or direction cones.
The code cells reload `interactive_figures.json` from the separate numeric data directory.
VS Code's Jupyter Notebook Renderers extension displays the saved output without a running
kernel or external browser. The exporter uses JSON and NumPy, without a Plotly Python dependency.
The Amarel `run_continuous_pca_preflight.sh` accepts optional fourth argument `install-deps`
to install missing analysis/test packages in `aim3_rnn` on its allocated compute node before
running the rollout tests and collection CLI import check. Existing packages are not explicitly
upgraded. An inference array must depend on successful completion of this preflight.

### Seed1 pre-digit PCA

`utils.analysis.clutter.pre_digit_pca` reuses full-stream seed1 RNN/GaWF records. Eligible
joint events have no other switch in -50…-1; no post-window or sector restriction is imposed.
It selects 22 temporally spread events per pre-digit from chronological candidate lists, sharing
the same 220 events across models. Each model fits one centered, unscaled PCA on all balanced
pre windows and uses identical axis ranges across its ten digit panels. Switch0 is displayed as
a red reference X, excluded from the PCA fit. The one exported notebook has exactly two code
cells: GaWF then RNN, each looping over digits0–9. Raw event states, labels, selection provenance,
PCA arrays and per-digit capture in the shared basis remain in the separate data leaf.

### RNN fixed-point inputs

`utils.analysis.clutter.rnn_fixed_points` reconstructs the historical one-layer tanh `nn.RNN`
from the checkpoint-compatible affine weights and freezes it. Autonomous input removes the
entire `W_ih*u + b_ih` term via a stateless
`bias_ih=0` override and zero features; recurrent `W_hh*h + b_hh` remains. The separate
zero-feature condition retaining `b_ih` is deliberately excluded. Real-input conditions fix
the actual two-frame CNN features at each clean switch's preceding frame and switch frame.
Each seed has 61 conditions, 64 shared trajectory-based initializations (half perturbed),
2000 Adam steps, and float64 damped Newton refinement. Roots require max-absolute residual
at most `1e-7`; RMS deduplication uses `1e-4` only within an input condition. Candidates and
their residuals, root Jacobians/eigenspectra and perturbation linearization errors are retained.
The numerical search is not an exhaustive count of all fixed points. No GaWF analysis is run.

### Unified GaWF variance decomposition

Use `utils.analysis.variance_decomposition` for encoder activation, input/recurrent gate synapses,
effective input/recurrent weights, hidden state, and feedback/readout vectors. Every run balances
all 90 sector-digit cells to a common `n`, repeats the subsample for 20 fixed-seed draws, and
reports aggregate plus per-unit condition-mean and trial-level fractions. Gate/effective-weight
unit axes index synapses, not neurons. Trial-level gate analysis must stream second-order moments
under an explicit memory budget; a trial-by-synapse array is forbidden. Object figures label each
aggregate bar with its 20-draw mean and show per-unit fractions as violins after averaging each
unit across the 20 draws; the black line marks the mean of those draw-averaged unit values.
The additional compact core-object aggregate figure reports condition-mean aggregate means only.
It uses two single-axis rows: input/recurrent gates in the first row and encoder/hidden activation
in the second, with the representation names on the x-axis and adjacent factor bars. It follows
the poster style below: one shared `Explained variance (%)` label, no numeric bar labels, one
shared legend, aligned `0`-to-`100` numeric ticks without repeated percent signs, adjacent bars
matching the physical width-to-height ratio of the GRU afferent-gate panel, a taller two-row canvas,
exactly `1.5` bar widths of clear space between the two object groups in each row, a compact legend
contained within the y-axis span, a high-contrast factor palette distinct from model/gate panels,
and no top or right spines.

The preliminary Figure 4 activation comparison is generated by
`utils.analysis.clutter.fig4_activation_anova`: it streams each of the six-model, ten-seed
checkpoints once, stores only the balanced condition-mean aggregate draws for encoder/input and
hidden activations, and renders a 1-by-2 panel. Each panel groups six model-coloured bars within
Sector, Digit, and Interaction; residual is excluded because condition-mean fractions sum to 100%.

The Figure 4 shuffle extension is generated by
`utils.analysis.clutter.fig4_shuffle_activation_anova`. It reuses the Figure 2 512-frame,
per-sample feedback schedule and permutation order for `baseline`, `shuffle_digit`, and
`shuffle_sector`, while grouping every representation by the unshuffled ground-truth labels. Its
activation plot is a 1-by-3 panel for default GaWF, shuffled-sector feedback, and shuffled-digit
feedback.
It excludes the first reset frame of every rollout, uses the same 20 fixed balanced draws for all
three conditions, and keeps only GPU-streamed aggregate sufficient statistics for encoder/hidden
activation and input/recurrent gate synapses. Its ten-seed delivery consists of a 120-row long
table, a mean-with-SEM summary, and the three-condition 1-by-3 hidden-activation panel. Each
panel shows a unified trial-level Sector, Digit, Interaction, and Residual decomposition: each
seed's first three components are multiplied by its own `1 - residual_frac` before the mean and
SEM are computed, while Residual uses the trial-level denominator directly;
the aggregate runner rejects any non-identical encoder result across conditions.

The Figure 3 half-mass audit is generated by
`utils.analysis.clutter.fig3_gate_half_mass`. It reconstructs gates from the retained float32
feedback trajectories and `U/V`, counts `abs(g - 0.5) < tolerance` and `[0.1, 0.9]` without
saving dense gates, and reports seed-level means with seed-level SEM. Keep reset
frames explicit: a zero-feedback first frame makes every gate exactly `sigmoid(0) = 0.5` and must
not be interpreted as a destination row with `U[j]` near zero. The point-excluded middle fraction
uses `(middle_count - half_count) / (total_count - half_count)`.
`utils.analysis.clutter.fig3_reset_excluded_gate_panel` is its visual companion: it removes the
zero-feedback reset mass from the saved ten-seed histograms and shows the remaining intermediate
mass with one point per training seed and mean ± SEM.

The Figure 6 spatial gate topology is preceded by
`utils.analysis.clutter.fig6_encoder_sector_patterns`. It excludes the first reset frame of every
32-frame held-out rollout, then streams equal-n encoder activation means for each Sector or Digit
condition and all ten GaWF seeds, retaining only compact
`(classes, 32, 6, 6)` means per seed. Each condition renders one combined canvas with the spatial
grid on the left and the channel grid on the right, sharing one activation color scale. Sector uses
3-by-3 spatial and 2-by-5 channel grids; Digit uses 2-by-5 spatial and 3-by-4 channel grids, with
the final occupied channel panel showing the respective condition-pattern similarity matrix. Do not
replace this pattern comparison with ANOVA, which would discard the condition-specific activation
maps needed to establish the prerequisite. The optional standalone Sector spatial grid reads the
saved ten-seed summary and uses the spatial maps' own sequential activation range.

`utils.analysis.clutter.fig6_sector_gate_weight_sign` creates the sign-split companion to
`Fig6_sector_gate_mean_sequential_equal_n_point_excluded_10seed`: it reuses each seed's original
equal-n sector selection and reconstructs the point-excluded sequential input gates from the saved
trajectory. The two 3-by-3 grids separately average synapses whose corresponding static
`weight_ih` is positive or negative, with a shared `0`/`0.5`/`1` gate scale.

`utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector` is the input-gate counterpart to
the Supplementary 3 recurrent sign-vs-magnitude check. Its formal workflow pools ten seeds and all
nine sectors, comparing each sector's 128 encoder sources in the matching 2-by-2 receptive-field
block against the remaining 1024 sources while keeping positive/negative static `weight_ih` curves
separate. Its gate and delta-gate figures use the same zoomed scatter, binned mean plus SEM, and
shared-|W| overlap convention. Descriptive overlap gaps and sign-specific slopes use the training
seed as the inference unit; the legacy Sector-0-only output remains available without
`--all_sectors`. The optional `plot --all_sectors --connection_stats_only` companion reports
per-seed and all-seed-pooled connection-row Welch gap tests and OLS `delta_g ~ |W|` slopes.
These p-values are diagnostic because rows share seeds, sectors, destinations, and physical
connections; interpretation prioritizes gap/slope magnitude and never replaces seed-level
inference with a tiny pooled p-value.

`utils.analysis.clutter.appendix_sign_magnitude_figures` reads the retained reset-excluded
Supplementary 2 and Supplementary 3 NPZ files and renders their ICLR-width vector PDFs without
rerunning a model. Supplementary 2 uses a compact matching/other 1-by-2 layout. Supplementary 3
uses Digit and Sector rows with TT/TR/RT/RR columns. Both retain a deterministic display-only
subsample of connection--condition points, compute each nine-bin curve within training seed, and
show the mean plus or minus seed-level SEM across the ten seeds.

The input-gate sign summary first averages overlap-band connection rows separately within each
Sector condition and then averages the nine condition means equally inside each training seed.
Its all-connections level uses the same condition-first average without sign or overlap-band
restriction. Observation-level slopes and the nine-bin display curves remain computed from their
connection--condition rows.

Formal Fig1 and Supplementary 1 target-switch recovery curves use ten training seeds. At every
offset from `pre10` through `pre1` and `post1` through `post10`, first compute one accuracy per
training seed, then plot the seed mean with seed-level SEM. Do not pool
switch events across seeds as the inference unit. Fig1 groups the ten curves separately for each
of the six models; Supplementary 1 does the same for each GaWF feedback-ablation condition.

Figure 7 sign-gap inference uses the training seed as the independent unit. For each group, sign,
and seed, first average the shared-overlap-band connection rows separately within every Digit or
Sector condition, then average the 10 Digit or 9 Sector condition means with equal weight. Compute
the paired positive-minus-negative difference from those seed-level values and test the ten seed
differences against zero with an exact two-sided sign-flip test. The all-connections bar uses the
same condition-first equal average without the sign or shared-support restriction. Connection-level
or pooled-across-seed p-values are diagnostic only and must not be shown as the Figure 7 p-value.
The formal ten-seed workflow retains compact per-condition recurrent-gate means and seed-specific
hidden tuned masks rather than full trial-by-connection gate caches. The retained Figure 7 does
not draw significance stars; structured raw, sign-flip, and 24-test Holm values remain diagnostic
records only. Supplementary 3 pools the same ten compact seed outputs for descriptive
sign-versus-magnitude curves and does not report pooled connection-level p-values. Its optional
seed-level sign/magnitude summary fits separate W-positive and W-negative OLS `delta_g ~ |W|`
slopes inside each seed/group shared overlap band and reports the ten-seed mean with seed-level SEM;
the companion overall delta level averages all rows in the same variable/group without an overlap
restriction.

`utils.analysis.clutter.fig6_net_recurrent_current` is the reset-excluded, Digit- or
Sector-conditioned current companion. For each selected frame it streams the exact recurrent product
`g_ij(t) * W_ij * h_j(t-1)` and immediately aggregates the context-specific TT/TR/RT/RR masks by
positive/negative `W`; it neither multiplies separately averaged gate and hidden activity nor keeps
a frame-by-synapse tensor. Every signed sum is divided by the selected destination-unit count.
The gate-only term uses the equal-weight mean gate across the selected condition while retaining
the actual prior hidden activity, so it is an instantaneous decomposition rather than a frozen-gate
counterfactual. The ten-seed summary and long table retain raw `I`, context-centered `ΔI`, and
`ΔI^gate` as the full condition-level numeric record. The historical Supplementary 4 renderer
shows all three quantities across Digits or Sectors with seed mean ± SEM. The historical Figure 8
renderer reads only those completed long tables and writes matched Digit/Sector vector PDFs, a
long CSV with `I`, `I_frozen = I - ΔI^gate`, and `ΔI^gate`, and caption statistics. Its
connection-level Figure 8 and destination-unit Supplementary 4 each retain only grouped current
bars, with Digit and Sector arranged as a Figure-7-style pair and one shared `W > 0` / `W < 0` /
`All` legend. Each condition is first summarized within its training seed; every gray point
is one seed's equal-weight mean across the 10 Digit or 9 Sector conditions, and the bar, SEM, and
uncorrected two-sided one-sample t-test against zero use those same ten seed values. The standalone
Figure 8 / Supplementary 4 renderers are deprecated after their required panels were merged into
the ICLR recurrent-gate figure; retained formal panels do not draw significance stars. The historical
second Supplementary 4 PDF retains the full E/I-plane record as a 2-by-2 unit/connection by
Digit/Sector grid. Its arrows are
plotted for every condition and group across all training seeds; group colour and frozen/observed
lightness are shared across all four panels.
The historical standalone Figure 8 PDFs use destination-unit normalization. The same script's `connection`
subcommand derives a separate Figure 8 companion from the retained raw current means and compact
seed masks, without rerunning models: it divides E and I by their corresponding nonzero recurrent
connection counts, while total is the signed sum across both signs divided by their combined count.
Its `fig8 --normalization connection`
mode writes `Fig8_recurrent_current_connection.pdf`; the destination-unit bars are written as
`Supple4_recurrent_current_unit.pdf`. Their connection-level E/I-plane limits are fixed at
`0.00`--`0.16` for Digit and `0.00`--`0.12` for Sector; their Panel C limits are respectively
`-0.02`--`0.09` and `-0.02`--`0.03`.
The ICLR recurrent-gate figure combines the Digit/Sector delta-g bars in its upper-left panels,
the connection-normalized current bars in its lower-left panels, and the two T-to-T delta-g versus
`|W|` curves in the right column. It omits per-bar significance stars.

For the cross-architecture Figure-03 comparison, GaWF connection gates are additionally projected
to destination units by taking the arithmetic mean of raw sigmoid gates over incoming input or
recurrent synapses on each frame. The balanced decomposition is then applied to those length-H
unit vectors exactly as for LSTM/GRU unit gates. This derived destination-unit view supplements,
and never replaces, the canonical synapse-level GaWF decomposition. Prefer the unified exporter's
saved float32 gate mmap arrays when available; otherwise reconstruct the same float32 gates in
batches from the compact trajectory's aligned `feedback`, `U`, and `V`, immediately reduce the
incoming-source axis, and never retain a trial-by-synapse array.

`utils/analysis/run_unified_variance_decomposition.py` reads saved mmap `.npy` representations,
including the input and recurrent gate tensors. A saved GaWF trajectory may supply labels,
feedback, and static weights only; the runner never reconstructs gates from `U/V`, reruns the
model, or regenerates activations. Missing trial-level representations are a hard failure. When
those saved representations do not yet exist, run
`utils/analysis/export_unified_variance_sources.py` once on a CUDA host with enough disk space. The
exporter loads the canonical checkpoint/test dataset, writes frame-major float32 mmap sources
without materializing a complete trial-by-synapse tensor, and emits the runner input manifest.
Use
`utils/analysis/migrate_analysis_outputs.py` to plan or apply the one-time legacy output move;
ambiguous mixed artifacts remain in place and appear in its migration report.

## Visualisation scripts

Plotting belongs in `utils/analysis/` beside its owning analysis and reads saved result files rather
than loading models independently.

`utils.analysis.clutter.partial_nonlinearity_training_curves_20260922 --complete` renders the
complete 10-seed, 150-epoch GaWF/RNN baseline-versus-no-tanh comparison from staged pickle
histories. It preserves the earlier partial snapshot, writes the complete long-form epoch table,
coverage, and summary below
`results/data/analysis/G_behaviour/partial_nonlinearity_training_curves_20260922/` with the unique
suffix `_complete_20260923`, and writes the matching PNG/PDF directly below
`results/figs/G_behaviour/`.

`utils.analysis.clutter.data_scale_comparison` summarizes the formal Clutter 4h/10h/20h/40h
comparison for the fixed best6 hyperparameters, six models, and seeds 1-10. It validates the
40h-uint8 evaluation protocol and 150 completed epochs, writes the 240 seed-level rows plus
mean-with-SEM and JSON summaries to the canonical `G_behaviour` analysis-data leaf, then renders
one curated 2-by-3 validation-accuracy, training-accuracy, and overfit-gap PDF from the saved
mean-with-SEM CSV into `results/save/`. The historical uppercase-category figure outputs remain
retained and are not regenerated.
When metrics are staged locally from Amarel, pass `--source-root` with the original remote result
root so the structured outputs retain durable provenance rather than the temporary staging path.

`utils.analysis.clutter.fig1b_feedback_controls_behavior_2x4` renders the CM-MNIST ten-model
behaviour comparison into `results/data/analysis/G_behaviour/fig1b_feedback_controls_behavior_2x4`
and the curated `results/save/cmmnist_feedback_controls_behavior_2x4.pdf`/`.png`. Rows are the
Location (sector) and Identity (character) readouts; columns are reset-excluded test accuracy,
validation loss, target-switch recovery, and the feedback-shuffle ablation. The four campaign
feedback controls are drawn solid with filled markers, and the six retained best-six Clutter models
(GaWF strict, RNN, LSTM, GRU, Mamba, S5) are drawn dashed with open markers and hatched bars; a
control reuses the palette of the baseline it ablates. Columns A-C cover all ten models, while
column D can only cover models that have a feedback-shuffle export under
`--reference-shuffle-root` (currently the four controls plus strict GaWF): the five open-loop
baselines were never run with that protocol, so their slots are labelled `not run`. Add a model to
column D only by producing its reset-excluded shuffle export first, never by mixing another
protocol.

### Remote-data-first result visualisation

- For updates to an already completed experiment whose structured ten-seed results are retained on
  `sjc-remote`, render from those exact remote results by default.  Synchronize only the requested
  PDF/PNG figure leaf back to the local `results/save/`; do not download JSON, NPZ, NPY, CSV, or
  other numerical source files solely to redraw a figure.
- Render from local structured results only when they are already present or when the user
  explicitly requests local numerical-data access. Do not submit an Amarel plotting job merely to
  change labels, limits, aggregation, smoothing, or figure layout.
- Use a compute-node plotting job only when the required source data cannot be safely or
  practicably transferred, or when the plot genuinely requires remote-only compute. In that case,
  combine source validation, rendering, output verification, and one scoped figure transfer into
  one workflow; do not iteratively submit or poll separate plotting jobs for routine revisions.
- Small analyses and visualisations do not need a training smoke test or an Amarel job. Run them
  locally from saved structured results, validate the output, and synchronize the exact figure
  only when a remote copy is requested.

- Call `matplotlib.use("Agg")` before importing pyplot.
- Default to 150 DPI and save with `bbox_inches="tight", pad_inches=0.06`.
- Close every figure immediately after saving.
- A bar chart aggregated across ten independent training seeds displays the seed mean ± SEM and,
  by default, one neutral-gray jittered point per seed above every bar.  Expose this overlay as a
  `--show-seed-points` / `--no-show-seed-points` switch; the latter suppresses points without
  changing the mean or SEM error bar.  Do not color seed points by the bar category.
- Use `RdBu_r` with symmetric limits for diverging heatmaps and `viridis` for sequential data
  unless the CLI provides another colormap.
- Load feature order from `channel_order_by_cosine_similarity.npy` and hidden-unit order from
  `sorted_npz_order.npy`; gracefully fall back to natural order.
- Draw boundary/highlight lines in red with linewidth 0.7.
- For N components plus sum and full panels, use three columns and
  `ceil((N + 2) / 3)` rows; hide unused axes.
- Use the task-specific figure workflows in `utils/analysis/` for saved training metrics unless a
  custom figure is explicitly requested. They dispatch clutter `.pkl` histories and Atari
  `metrics_history.jsonl` to their task-specific plotting modules. For a multi-task Atari run,
  the default figure contains one `episodic_return_100` curve per environment; use
  `--include_combined` only for diagnostic plots that intentionally pool episodes with different
  score scales.
- For multi-task visualisations, render `environment_steps` as the sole default x-axis. Do not
  render a `global_step` variant unless the request explicitly asks for it.

### Poster and multi-panel figure style

Figures intended for posters must remain readable at viewing distance. Use approximately 13 pt
for tick labels, legends, and body text, 15 pt for column titles and shared row labels, and 16 pt
for axis labels. Treat these values as a coordinated baseline: scale the full hierarchy together
when the physical figure size changes, and never shrink one crowded panel independently.

- Hide the top and right spines by default. Keep the left and bottom spines, use a light y-axis
  grid where it aids comparison, and avoid complete boxes, nested axes, or duplicated frames.
- Set axis limits and tick intervals explicitly when they carry scientific meaning. Equivalent
  panels must use aligned plot rectangles and consistent axis widths.
- Keep multi-row and multi-column layouts compact. Reduce unused outer margins, `hspace`, and
  `wspace` without allowing labels, titles, or legends to collide.
- Omit a composite main title unless it adds information not already present in the column titles
  and legend.
- Use one shared row label for each semantic row rather than repeating a y-axis label on every
  subplot. Compute its vertical position from the actual subplot bounding boxes in that row.
- Use one title for each semantic column and compute its horizontal position from the actual
  subplot bounding boxes in that column. Do not align row labels or column titles with fixed,
  visually estimated offsets.
- In a multi-row figure, show x-axis text and the semantic x-axis label only on the bottom row.
  Keep the upper-row tick marks, but suppress the upper-row tick labels and x-axis label.
- Use one shared legend above the panels when the same series appear throughout the figure. Keep
  model order, names, and colors consistent across panels. Summary legends use line-only handles
  unless markers themselves encode a scientific variable.
- Sparse tick labels or highlighted markers must not subsample the plotted data. Draw the complete
  time series and use labels or markers only at interpretable checkpoints. Target-switch recovery
  figures, for example, draw every frame from `pre10` through `post10` while highlighting
  `pre10`, `switch`, `post4`, and `post10`.
- For multi-seed bar plots, show the mean bar, sample-SD error bar, and individual seed points when
  space permits. Keep bars within a grouped category adjacent; reserve visible spacing for
  category boundaries. Add numerical mean labels only when direct aggregate lookup is part of the
  figure's purpose.
- Prefer plotting from numeric CSV, NPZ, NPY, or PKL outputs. A validated raster figure is a
  temporary fallback only when the underlying numeric results are unavailable. When reusing a
  raster, remove its old axes, titles, legend, and frame; map its data extent exactly; extend grids
  through any newly exposed axis range; and verify that all panel widths remain aligned. Document
  the fallback in code so that it can be replaced when the numeric source becomes available.
- For curated `results/save/` outputs, save `Fig*` as PDF only and `Supple*` as PNG only; never
  create a same-basename companion in the other format. Do not sync PDFs to the external
  `6-Writing/Aim3/Figures` publication tree automatically: `--publication_fig_dir` (and
  `AIM3_PUBLICATION_FIGURES_DIR`) are opt-in extra-copy destinations only, off by default.
  ICLR-width alternate layouts go under `results/save/iclr_figs/` and leave their canonical
  `results/save/` counterparts untouched. Figure 3 uses `--gate_weight_layout 1x4` for this
  wide-layout export while retaining the canonical `2x2` rendering.
  Visually inspect the saved output for typography, row/column alignment, legend placement, axis
  ranges and ticks, spine/grid continuity, complete curves, and consistent rendering before
  accepting the figure.

## Shell launchers

- Use non-interactive commands where possible and fail early with `set -euo pipefail` (bash) or
  the equivalent zsh options.
- Preserve a preset `CUDA_VISIBLE_DEVICES`.
- Keep reusable launchers tracked, generated Slurm files ignored, and one-off recovery scripts
  clearly marked and removed before synchronization.
- Result suffixes must encode protocol-changing settings, including Pong frame skip and stack.

## Review checklist

- [ ] Read the owning architecture/convention/runbook document.
- [ ] Updated all imports and call sites after public changes.
- [ ] Preserved dependency direction and shared recurrent cores.
- [ ] Added/updated tests for changed behavior and compatibility.
- [ ] Kept dataset memory and device handling safe.
- [ ] Preserved result names, dtypes, metadata, and checkpoint loading behavior.
- [ ] Closed figures and handled empty analysis selections explicitly.
- [ ] Updated the owning documentation without duplicating it elsewhere.
- [ ] Added an `EXPERIMENT_LOG.md` entry only if the change is research-significant and confirmed.
