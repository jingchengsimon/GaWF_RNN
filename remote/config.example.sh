#!/usr/bin/env bash
# Copy to remote/config.sh and replace placeholders with local-only values.
# remote/config.sh is ignored by Git.

REMOTE_HOST="<ssh-alias-or-user-at-host>"
REMOTE_PROJECT_DIR="<absolute-remote-project-path>"
REMOTE_BRANCH="<branch-name>"

LOCAL_RESULTS_DIR="./results"
REMOTE_RESULTS_DIR="${REMOTE_PROJECT_DIR}/results"

SSH_OPTS=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath=~/.ssh/cm-%r@%h:%p
)

# Example: source <conda-init-path> && conda activate aim3_rnn
REMOTE_ACTIVATE=""
