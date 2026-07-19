# Analysis output index

This is the only analysis output tree. Every run uses:

```text
results/anal_index/<CATEGORY>/<script_name>/figs/
results/anal_index/<CATEGORY>/<script_name>/data/
results/anal_index/<CATEGORY>/<script_name>/manifest.json
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

The unified seven-object decomposition is written below
`D_variance_decomposition/unified/`. Gate and effective-weight unit axes index synapses, not
neurons.

Use `python utils_anal/migrate_analysis_outputs.py` for a dry-run migration report and add
`--apply` only after reviewing every ambiguous file.
