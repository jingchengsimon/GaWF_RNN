# Pre-ICLR experiment archive

This directory preserves launch-time checks and regression tests for recurrent definitions that
were removed from the anonymous-release runtime on 2026-09-24. Files use the `.py.txt` suffix so
pytest and public entry points do not collect or import historical model paths.

The archive covers nonlinearity-placement variants, additive/concatenated feedback controls, and
the mLSTM/HyperLSTM/BRIMs reviewer baselines. It is provenance, not an executable API.
