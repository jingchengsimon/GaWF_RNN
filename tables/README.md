# Appendix table files

Each `.tex` file contains one complete `table` or `table*` environment. Upload this directory
beside `appendix.tex`; the Appendix already inserts every retained table with `\input{tables/...}`.

The revised Appendix retains four tables. Original identifiers 14--16 remain in the filenames so
their provenance is clear; LaTeX assigns consecutive displayed table numbers from inclusion order.

| Order | File | Label | Appendix insertion point |
|---:|---|---|---|
| 1 | `01_formal_model_configurations.tex` | `tab:formal-config` | Section B, after the single hyperparameter-selection paragraph |
| 2 | `14_input_gate_sign.tex` | `tab:input-gate-sign` | Section H, after the matching-versus-other source result paragraph |
| 3 | `15_recurrent_sign_gap.tex` | `tab:recurrent-sign-gap` | Section I, after the recurrent-gate sign-dependence paragraph |
| 4 | `16_recurrent_current.tex` | `tab:recurrent-current` | Section I, after the connection-normalized recurrent-current paragraph |
