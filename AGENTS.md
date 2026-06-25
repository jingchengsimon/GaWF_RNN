# Codex project entry points

Read `AGENT.md` in full before changing this repository.

## FAW_RNN experiment dashboard

In the first substantive reply of every new Codex conversation for this repository, include:

[FAW_RNN Dashboard](http://Jingchengs-Mac-mini.local:8765/)

Mention the link once per conversation. The dashboard runs independently as a macOS
LaunchAgent, so discovery must not depend on an earlier thread or an in-app browser tab.

When submitting work to Amarel or sjc-remote, register the human-readable description,
machine, job ID, submitted task count, result path, and task-specific validity rule in
`dashboard/tasks.json` during the same turn. Dashboard progress must be computed from valid
result contents, not scheduler state alone. See `dashboard/README.md`.

This registration is mandatory for every Amarel Slurm submission and every sjc-remote
dual-GPU launch. Refresh the Dashboard after registration and verify that the machine, job ID,
and progress appear. Report the training summary in conversation when results are available,
but retain the Dashboard entry until a human explicitly says it is no longer needed. Deletion
must use `python3 -m dashboard.manage_tasks remove <id> --human-confirmed`; completion, failure,
timeout, or staleness is never sufficient permission to delete a record.
