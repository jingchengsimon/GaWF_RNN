#!/usr/bin/env bash
# Copy this file to remote/config.sh and edit local-only values there.
# remote/config.sh is ignored by git.

REMOTE_HOST="sjc@172.26.48.213"
REMOTE_PROJECT_DIR="/G/MIMOlab/Codes/aim3_RNN"
REMOTE_BRANCH="master"

LOCAL_RESULTS_DIR="./results"
REMOTE_RESULTS_DIR="${REMOTE_PROJECT_DIR}/results"

# ControlMaster reuses one SSH connection for several minutes. This reduces
# repeated password prompts during sync + run + rsync, but the first connection
# still needs a password unless SSH keys are configured.
SSH_OPTS=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath=~/.ssh/cm-%r@%h:%p
)

# Keep conda changes manual for now. If you later want automatic activation,
# set this in remote/config.sh, for example:
# REMOTE_ACTIVATE="source ~/.bashrc && conda activate aim3_rnn"
REMOTE_ACTIVATE=""
