# labml-nn HyperLSTM provenance

- Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations
- Pinned commit: `33ab02281c2b928e6b32792909cc79cbdcfe1d6a`
- Upstream files: `labml_nn/hypernetworks/hyper_lstm.py` and
  `labml_nn/lstm/__init__.py`
- License: MIT; retained in `LICENSE`.

The vendored subset removes all `labml` runtime dependencies by inlining only the
small `LSTMCell` used by HyperLSTM. Equations, parameterization, state order, gate
order, layer normalization, and initialization remain those of the pinned source.
The repository adapter handles only batch/time transposition and the Clutter
post-core `LayerNorm -> ReLU -> dropout` contract.

Equation audit against Ha et al. (2017) Section 3.2:

- The implementation uses the paper's row-wise scaling
  `d(z) * (W h)` for hidden and input matrices and dynamically generated bias.
- It uses the updated hyper hidden state `h_hat_t` to generate `z_h`; Equation (11)
  prints `h_hat_(t-1)`. Upstream documents this as a likely paper typo, and the
  authors' TensorFlow implementation also uses the current hyper state.
- It applies layer normalization to the hyper cell, main gate preactivations, and
  main cell state. The paper describes layer normalization in the implementation
  details but it is not explicit in every displayed recurrence equation.
- It does not implement the paper's candidate-state recurrent dropout term. The
  shared Clutter adapter instead retains the established post-core output dropout.
