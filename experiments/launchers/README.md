# Launcher catalog

This directory groups operational entry points first by task family and then by execution
backend:

```text
launchers/
  clutter/{amarel,dsw}/
  text/{amarel,dsw}/
  rl/{amarel,dsw}/
```

The existing task-owned scripts remain canonical. This directory must not contain copied training
implementations that can drift from those scripts. Backend README files list the canonical paths;
small wrappers and campaign-control utilities live here when they span multiple task-owned files
or hosts.

`clutter/dsw/pause_campaigns.py` is the coordinated checkpoint-pause trigger for the current
`gawf_legacy_notanh` and `rnn_inloop_notanh` campaigns. It is inert unless invoked with both
`--execute` and the exact confirmation token. The checked-in example config contains no endpoint;
the ignored `.local.json` holds machine-specific SSH commands.
