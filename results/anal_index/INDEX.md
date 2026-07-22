# Analysis output index

This directory is the human-maintained analysis index and migration-notes location. Categorized
outputs use parallel figure and data trees:

```text
results/anal_figs/<CATEGORY>/
results/anal_data/<CATEGORY>/<script_name>/
results/anal_data/<CATEGORY>/<script_name>/manifest.json
```

Categories:

- `A_raw_gate`: raw gate values without condition labels
- `B_gate_by_context`: raw gate values partitioned by condition
- `C_delta_gate`: per-synapse grand-mean-subtracted gate values
- `D_variance_decomposition`: variance apportioned to factors
- `E_relevance_alignment`: activation-derived relevance and alignment
- `F_timing`: post-switch frame or event-latency analyses
- `G_behaviour`: task performance without gate internals
- `H_controls`: confound, convergence, and invariance controls

The unified seven-object decomposition is written directly below
`results/anal_figs/D_variance_decomposition/` and
`results/anal_data/D_variance_decomposition/unified/` directories. Gate and effective-weight unit
axes index synapses, not neurons. Existing legacy entries directly below `anal_figs/` and
`anal_data/` remain unclassified until a later review.
