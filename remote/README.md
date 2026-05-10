# Remote Workflow Wrappers

These wrappers sync committed local code through GitHub, update the remote checkout, and run
commands on the remote machine.

## Setup

```bash
cp remote/config.example.sh remote/config.sh
```

`remote/config.sh` is ignored by git. It currently targets:

- host: `sjc@172.26.48.213`
- remote project: `/G/MIMOlab/Codes/aim3_RNN`
- branch: `master`

Conda activation is intentionally disabled by default. Keep remote environment changes manual, or
set `REMOTE_ACTIVATE` in `remote/config.sh` later.

## Sync Code

```bash
./remote/sync_code.sh --push
```

The sync wrapper never runs `git add` or `git commit`. If local edits are uncommitted, it stops and
prints suggested commands.

## Run Commands

Foreground command with result fetch for files created after this run starts:

```bash
./remote/run.sh --push -- python train_model.py --help
```

Long command in remote tmux:

```bash
./remote/run.sh --push --detach my_session -- bash experiments/local/run_hparam_full_grid_2gpu.sh --scale 4
```

After a tmux run finishes, fetch only files newer than that run's marker:

```bash
./remote/fetch_results.sh --since .remote_wrapper_<timestamp>_<pid>.marker
```

The wrapper prints the exact marker name when it starts a tmux session.

## Fetch Results

Avoid fetching all results on the first run unless you really want the full remote history:

```bash
./remote/fetch_results.sh --all
```

Fetch one subdirectory:

```bash
./remote/fetch_results.sh train_data/gen_phase3_short_4h_ep100
```

## SSH Passwords

`SSH_OPTS` uses SSH ControlMaster so one successful login can be reused for sync, run, and rsync for
about 10 minutes. For passwordless login, configure an SSH key manually:

```bash
ssh-keygen -t ed25519 -C "FAW_RNN remote"
ssh-copy-id sjc@172.26.48.213
ssh sjc@172.26.48.213
```
