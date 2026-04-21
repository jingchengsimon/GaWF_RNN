# Sourced by phase3_train_*_short.sh (and orchestrators that need the same naming).
# NUM_EPOCHS: max training epochs for train_model.py (--num_epochs).
# CSV_TAG: suffix for phase3_summary CSVs and plot_generalization --csv_tag base (default _short).
# TAG: full artifact tag "${CSV_TAG}_ep${NUM_EPOCHS}" e.g. _short_ep50.
: "${NUM_EPOCHS:=50}"
: "${CSV_TAG:=_short}"
TAG="${CSV_TAG}_ep${NUM_EPOCHS}"
