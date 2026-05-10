#!/usr/bin/env bash
# Push committed local changes if requested, then fast-forward the remote checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=remote/lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage: ./remote/sync_code.sh [options]

Options:
  --push          Push local committed changes to origin/<branch> without prompting.
  --no-push       Do not push; fail if local commits are not on origin/<branch>.
  --allow-dirty   Allow uncommitted local changes. They will not be synced.
  -h, --help      Show this help.

This script never runs git add or git commit.
EOF
}

load_remote_config

push_mode="prompt"
allow_dirty=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      push_mode="yes"
      shift
      ;;
    --no-push)
      push_mode="no"
      shift
      ;;
    --allow-dirty)
      allow_dirty=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "${REMOTE_BRANCH}" ]]; then
  echo "Local branch is '${current_branch}', expected '${REMOTE_BRANCH}'."
  echo "Suggested command:"
  echo "  git checkout ${REMOTE_BRANCH}"
  exit 2
fi

dirty_status="$(git status --porcelain --untracked-files=all)"
if [[ -n "${dirty_status}" && "${allow_dirty}" != true ]]; then
  echo "Local working tree has uncommitted changes. They are not synced by this wrapper."
  echo
  git status --short
  echo
  echo "Suggested commands:"
  echo "  git add <files>"
  echo "  git commit -m \"describe your change\""
  echo "  ./remote/sync_code.sh --push"
  echo
  echo "To sync only already committed changes while keeping local edits:"
  echo "  ./remote/sync_code.sh --push --allow-dirty"
  exit 2
fi

echo "[1/3] Fetching origin/${REMOTE_BRANCH}..."
git fetch origin "${REMOTE_BRANCH}"

behind_count="$(git rev-list --count "HEAD..origin/${REMOTE_BRANCH}")"
if [[ "${behind_count}" -gt 0 ]]; then
  echo "Local ${REMOTE_BRANCH} is behind origin/${REMOTE_BRANCH} by ${behind_count} commit(s)."
  echo "Suggested command:"
  echo "  git pull --ff-only origin ${REMOTE_BRANCH}"
  exit 2
fi

unpushed_count="$(git rev-list --count "origin/${REMOTE_BRANCH}..HEAD")"
if [[ "${unpushed_count}" -gt 0 ]]; then
  echo "Local ${REMOTE_BRANCH} has ${unpushed_count} committed change(s) not on origin/${REMOTE_BRANCH}."
  case "${push_mode}" in
    yes)
      echo "[2/3] Pushing committed changes..."
      git push origin "${REMOTE_BRANCH}"
      ;;
    no)
      echo "Suggested command:"
      echo "  git push origin ${REMOTE_BRANCH}"
      exit 2
      ;;
    prompt)
      read -r -p "Run 'git push origin ${REMOTE_BRANCH}' now? [y/N] " answer
      if [[ "${answer}" =~ ^[Yy]$ ]]; then
        echo "[2/3] Pushing committed changes..."
        git push origin "${REMOTE_BRANCH}"
      else
        echo "Suggested command:"
        echo "  git push origin ${REMOTE_BRANCH}"
        exit 2
      fi
      ;;
  esac
else
  echo "[2/3] No unpushed local commits."
fi

remote_dir_q="$(quote_args "${REMOTE_PROJECT_DIR}")"
remote_branch_q="$(quote_args "${REMOTE_BRANCH}")"

echo "[3/3] Fast-forwarding remote checkout..."
run_remote_bash "
  set -euo pipefail
  cd ${remote_dir_q}
  current_branch=\$(git branch --show-current)
  if [[ \"\${current_branch}\" != ${remote_branch_q} ]]; then
    echo \"Remote branch is '\${current_branch}', expected ${REMOTE_BRANCH}.\" >&2
    exit 2
  fi
  git fetch origin ${remote_branch_q}
  git pull --ff-only origin ${remote_branch_q}
  git status --short
"

echo "Remote checkout is up to date."
