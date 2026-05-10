#!/usr/bin/env bash
# Fetch result files from the remote machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=remote/lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage: ./remote/fetch_results.sh [options] [subdir]

Options:
  --all              Fetch the whole remote results directory.
  --since <marker>   Fetch only files newer than a marker under remote results.
  --dry-run          Show rsync changes without writing files.
  -h, --help         Show this help.

Examples:
  ./remote/fetch_results.sh --since .remote_wrapper_20260510_230000_123.marker
  ./remote/fetch_results.sh train_data/gen_phase3_short_4h_ep100
  ./remote/fetch_results.sh --all
EOF
}

load_remote_config

mode="subdir"
since_marker=""
dry_run=false
subdir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      mode="all"
      shift
      ;;
    --since)
      mode="since"
      since_marker="${2:-}"
      if [[ -z "${since_marker}" ]]; then
        echo "--since requires a marker name or path."
        exit 2
      fi
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
    *)
      if [[ -n "${subdir}" ]]; then
        echo "Only one subdir can be provided."
        exit 2
      fi
      subdir="${1%/}"
      shift
      ;;
  esac
done

local_results_dir="$(local_results_abs)"
mkdir -p "${local_results_dir}"

rsync_args=(-avz --progress --update)
if [[ "${dry_run}" == true ]]; then
  rsync_args+=(--dry-run)
fi

ssh_cmd="$(ssh_command_string)"

case "${mode}" in
  all)
    echo "Fetching full remote results. This can be large."
    rsync "${rsync_args[@]}" -e "${ssh_cmd}" \
      "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" \
      "${local_results_dir}/"
    ;;
  since)
    tmp_files="$(mktemp)"
    trap 'rm -f "${tmp_files}"' EXIT

    remote_results_q="$(quote_args "${REMOTE_RESULTS_DIR}")"
    marker_q="$(quote_args "${since_marker}")"
    run_remote_bash "
      set -euo pipefail
      cd ${remote_results_q}
      if [[ ! -e ${marker_q} ]]; then
        echo \"Marker not found under ${REMOTE_RESULTS_DIR}: ${since_marker}\" >&2
        exit 2
      fi
      find . -type f -newer ${marker_q} -print
    " > "${tmp_files}"

    if [[ ! -s "${tmp_files}" ]]; then
      echo "No remote result files are newer than ${since_marker}."
      exit 0
    fi

    echo "Fetching files newer than ${since_marker}."
    rsync "${rsync_args[@]}" --files-from="${tmp_files}" -e "${ssh_cmd}" \
      "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" \
      "${local_results_dir}/"
    ;;
  subdir)
    if [[ -z "${subdir}" ]]; then
      echo "No subdir provided. Use --all to fetch the whole results directory."
      usage
      exit 2
    fi

    mkdir -p "${local_results_dir}/${subdir}"
    echo "Fetching remote results subdir: ${subdir}"
    rsync "${rsync_args[@]}" -e "${ssh_cmd}" \
      "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/${subdir}/" \
      "${local_results_dir}/${subdir}/"
    ;;
esac

echo "Fetch complete: ${local_results_dir}"
