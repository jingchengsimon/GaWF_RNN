#!/usr/bin/env bash
# Shared helpers for remote workflow wrappers.

set -euo pipefail

REMOTE_WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${REMOTE_WRAPPER_DIR}/.." && pwd)"
CONFIG_PATH="${REMOTE_WRAPPER_DIR}/config.sh"
CONFIG_EXAMPLE_PATH="${REMOTE_WRAPPER_DIR}/config.example.sh"

load_remote_config() {
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing ${CONFIG_PATH}."
    echo "Create it with:"
    echo "  cp ${CONFIG_EXAMPLE_PATH} ${CONFIG_PATH}"
    echo "  ${EDITOR:-vi} ${CONFIG_PATH}"
    exit 2
  fi

  # shellcheck source=/dev/null
  source "${CONFIG_PATH}"

  : "${REMOTE_HOST:?REMOTE_HOST must be set in remote/config.sh}"
  : "${REMOTE_PROJECT_DIR:?REMOTE_PROJECT_DIR must be set in remote/config.sh}"

  REMOTE_BRANCH="${REMOTE_BRANCH:-master}"
  LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-./results}"
  REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-${REMOTE_PROJECT_DIR}/results}"
  REMOTE_ACTIVATE="${REMOTE_ACTIVATE:-}"

  if ! declare -p SSH_OPTS >/dev/null 2>&1; then
    SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=30)
  fi
}

quote_args() {
  local quoted=""
  printf -v quoted "%q " "$@"
  printf "%s" "${quoted% }"
}

ssh_command_string() {
  local cmd="ssh"
  local opt
  for opt in "${SSH_OPTS[@]}"; do
    cmd+=" $(printf "%q" "${opt}")"
  done
  printf "%s" "${cmd}"
}

run_remote_bash() {
  local body="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc $(printf "%q" "${body}")"
}

remote_body_with_activation() {
  local body="$1"
  if [[ -n "${REMOTE_ACTIVATE}" ]]; then
    printf "%s && %s" "${REMOTE_ACTIVATE}" "${body}"
  else
    printf "%s" "${body}"
  fi
}

local_results_abs() {
  if [[ "${LOCAL_RESULTS_DIR}" = /* ]]; then
    printf "%s" "${LOCAL_RESULTS_DIR}"
  else
    printf "%s/%s" "${PROJECT_ROOT}" "${LOCAL_RESULTS_DIR#./}"
  fi
}
