#!/usr/bin/env bash
# Validate a detached Git worktree without requiring the git executable.

assert_execution_snapshot_commit() {
  local root="${1:?worktree root is required}"
  local expected="${2:?expected commit is required}"
  local git_pointer git_dir actual

  [[ -f "$root/.git" ]] || {
    echo "Expected detached worktree metadata at $root/.git" >&2
    return 1
  }
  IFS= read -r git_pointer < "$root/.git"
  [[ "$git_pointer" == "gitdir: "* ]] || {
    echo "Malformed detached worktree metadata at $root/.git" >&2
    return 1
  }
  git_dir="${git_pointer#gitdir: }"
  [[ "$git_dir" == /* ]] || git_dir="$root/$git_dir"
  IFS= read -r actual < "$git_dir/HEAD"
  [[ "$actual" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Execution snapshot HEAD is not detached: $actual" >&2
    return 1
  }
  [[ "$actual" == "$expected" ]] || {
    echo "Source commit changed after submission: expected $expected, found $actual" >&2
    return 1
  }
}
