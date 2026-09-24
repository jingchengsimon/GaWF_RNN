# Clutter / Amarel

Canonical Slurm launchers remain under `experiments/clutter/amarel/`. The current GaWF/RNN
ablation family uses:

- `submit_clutter_nonlinearity_ablation.sh` and `run_clutter_nonlinearity_ablation.sh`
- `submit_clutter_gawf_core_variants.sh` and `run_clutter_gawf_core_variants.sh`
- `submit_clutter_rnn_inloop_notanh.sh` and `run_clutter_rnn_inloop_notanh.sh`
- `submit_clutter_ablation_behavior.sh` and `run_clutter_ablation_behavior.sh`

Other retained Clutter Slurm launchers are indexed by the same directory. Do not copy them here;
their submitter/runner relative paths, safety tests, and artifact roots depend on that canonical
location.
