# Pre-ICLR recurrent-core archive

This directory preserves the implementations that existed immediately before the
2026-09-24 anonymous-release cleanup. They include historical nonlinearity-placement,
feedback-off, additive-feedback, and reviewer-baseline branches.

The files are provenance only. Active code must not import from this directory. The public
Clutter runtime uses only `gawf.py`, `rnn.py`, `mamba.py`, and `s5.py` in the parent directory;
`paper_lstm.py` remains active only for the separate MiniGrid paper protocol.
