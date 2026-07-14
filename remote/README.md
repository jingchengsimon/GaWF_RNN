# Remote Workflow Wrappers

These wrappers synchronize committed code, update the configured remote checkout, run commands,
and fetch result files. General SSH and environment rules are in
`docs/operations/REMOTE_EXECUTION.md`.

## Local setup

```bash
cp remote/config.example.sh remote/config.sh
```

Fill in the ignored `remote/config.sh` with the SSH target, remote project root, branch, result
paths, and optional activation command. Do not put real endpoints in the tracked example.

The wrapper never runs `git add` or `git commit`. It stops if local edits make synchronization
unsafe and prints the required manual action.

## Synchronize code

```bash
./remote/sync_code.sh --push
```

## Run commands

Run in the foreground and fetch files created after the run began:

```bash
./remote/run.sh --push -- python train_model.py --help
```

Run a long command in remote tmux:

```bash
./remote/run.sh --push --detach my_session -- \
  bash experiments/local/run_hparam_full_grid_2gpu.sh --scale 4
```

After a detached launch succeeds, record its run ID, tmux session, remote root, exact logs,
results, and validity conditions with the project-local registry in
`experiments/monitoring/README.md`. This makes the same run discoverable from Mac and Mac mini
without an external Dashboard.

The wrapper prints a marker for detached runs. Fetch only files newer than that marker:

```bash
./remote/fetch_results.sh --since <marker-file>
```

Fetch one result subdirectory:

```bash
./remote/fetch_results.sh train_data/<result-suffix>
```

Use `--all` only when the full remote history is intentionally required.

## Connections

The example enables SSH ControlMaster so one authenticated connection can be reused briefly by
sync, run, and fetch operations. Configure SSH keys and aliases outside the repository. Keep real
usernames, addresses, and project paths in `remote/config.sh` and `.agents/local.md`.
