#!/usr/bin/env bash
# Sync committed code to the remote checkout, then run an arbitrary remote command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=remote/lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage: ./remote/run.sh [options] -- <command> [args...]

Options:
  --push              Push committed local changes before remote pull.
  --no-push           Fail if committed local changes are not pushed.
  --allow-dirty       Allow uncommitted local changes. They will not be synced.
  --skip-sync         Do not run sync_code.sh before executing.
  --detach <session>  Run command in a remote tmux session and return immediately.
  --tmux <session>    Alias for --detach.
  --no-fetch          Do not fetch results after a foreground command.
  --fetch-all         Fetch all remote results after a foreground command.
  -h, --help          Show this help.

Examples:
  ./remote/run.sh --push -- python train_model.py --help
  ./remote/run.sh --detach hparam_4h -- bash experiments/amarel/check_hparam_4h_5epoch_test_status.sh
  ./remote/run.sh --push -- bash experiments/local/run_hparam_full_grid_2gpu.sh --scale 4

This script never runs git add or git commit.
EOF
}

load_remote_config

sync_args=()
skip_sync=false
fetch_mode="since"
detach=false
session_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push|--no-push|--allow-dirty)
      sync_args+=("$1")
      shift
      ;;
    --skip-sync)
      skip_sync=true
      shift
      ;;
    --detach|--tmux)
      detach=true
      session_name="${2:-}"
      if [[ -z "${session_name}" ]]; then
        echo "$1 requires a tmux session name."
        exit 2
      fi
      shift 2
      ;;
    --no-fetch)
      fetch_mode="none"
      shift
      ;;
    --fetch-all)
      fetch_mode="all"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing remote command."
  usage
  exit 2
fi

if [[ "${skip_sync}" != true ]]; then
  "${SCRIPT_DIR}/sync_code.sh" "${sync_args[@]}"
fi

timestamp="$(date +"%Y%m%d_%H%M%S")"
marker_name=".remote_wrapper_${timestamp}_$$.marker"
log_name="run_${timestamp}.log"

remote_dir_q="$(quote_args "${REMOTE_PROJECT_DIR}")"
remote_results_q="$(quote_args "${REMOTE_RESULTS_DIR}")"
marker_path_q="$(quote_args "${REMOTE_RESULTS_DIR}/${marker_name}")"
log_path_q="$(quote_args "${REMOTE_RESULTS_DIR}/${log_name}")"
cmd_q="$(quote_args "$@")"

command_body="
  set -euo pipefail
  cd ${remote_dir_q}
  mkdir -p ${remote_results_q}
  touch ${marker_path_q}
  ${cmd_q} 2>&1 | tee ${log_path_q}
"
remote_command="$(remote_body_with_activation "${command_body}")"

echo "Remote command:"
printf "  %q " "$@"
echo
echo "Remote log: ${REMOTE_RESULTS_DIR}/${log_name}"

if [[ "${detach}" == true ]]; then
  session_q="$(quote_args "${session_name}")"
  tmux_shell_cmd="bash -lc $(printf "%q" "${remote_command}")"
  tmux_shell_cmd_q="$(quote_args "${tmux_shell_cmd}")"

  echo "Starting remote tmux session: ${session_name}"
  run_remote_bash "tmux new-session -d -s ${session_q} ${tmux_shell_cmd_q}"
  echo "Started."
  echo
  echo "Useful commands:"
  echo "  ssh ${REMOTE_HOST} 'tmux attach -t ${session_name}'"
  echo "  ssh ${REMOTE_HOST} 'tmux ls'"
  echo "  ./remote/fetch_results.sh --since ${marker_name}"
  exit 0
fi

set +e
run_remote_bash "${remote_command}"
remote_exit=$?
set -e

if [[ "${fetch_mode}" == "since" ]]; then
  "${SCRIPT_DIR}/fetch_results.sh" --since "${marker_name}"
elif [[ "${fetch_mode}" == "all" ]]; then
  "${SCRIPT_DIR}/fetch_results.sh" --all
else
  echo "Skipping result fetch."
fi

exit "${remote_exit}"
