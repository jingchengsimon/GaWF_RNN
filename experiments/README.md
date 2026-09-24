# Experiment layout

`experiments/` contains task-specific protocols, backend launchers, launcher indexes, and remote
run manifests. Training entry points and reusable implementations remain at the repository root
and in `utils/`.

| Directory | Role |
|---|---|
| `clutter/` | Final Clutter protocol and curation notes; no active launcher is retained. |
| `text/` | IMDB/SentiHood task definitions and `amarel/` launchers. |
| `rl/atari/` | Atari definitions and `amarel/` launchers. |
| `rl/minigrid/` | MiniGrid definitions and `amarel/` launchers. |
| `launchers/` | Host-neutral launcher catalog plus task/backend-grouped operational entry points. |

Cross-host synchronization and run manifests live under `experiments/remote/`, not this directory.
Canonical task launchers remain in their owning task directories. `experiments/launchers/`
indexes or wraps them rather than copying their implementation.
