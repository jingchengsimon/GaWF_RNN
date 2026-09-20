#!/usr/bin/env bash
# Compute-node source-identity guard for Amarel run_*.sh launchers.
#
# Amarel compute nodes do not reliably provide git: several GPU nodes have no `git` on PATH and no
# `/usr/bin/git` either. A runtime source guard therefore must not depend on a git executable or on
# a login-node absolute path. The submitter stamps the resolved commit into the run's status
# directory at submit time; at run time this helper resolves the checked-out commit from git
# metadata (git first, then shell builtins) and otherwise accepts that submit-time stamp.
#
# Callers must treat a non-zero return as a stop condition and write their own fail marker before
# exiting, because an `exit` inside a `|| { ...; }` list does not trigger the `ERR` trap.

# Print a usable git executable, or return non-zero when the node provides none.
amarel_git_bin() {
  local candidate="${AIM3_GIT_BIN:-}"
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  candidate="$(command -v git 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

# Resolve the checked-out commit with shell builtins only (no git, no external tools).
# Supports a normal checkout, a linked worktree (`.git` file), detached HEAD, and packed refs.
amarel_head_commit_without_git() {
  local root="$1" gitdir entry ref value commons packed
  gitdir="$root/.git"
  if [[ -f "$gitdir" ]]; then
    IFS= read -r entry < "$gitdir" || return 1
    gitdir="${entry#gitdir: }"
    [[ "$gitdir" != "$entry" && -n "$gitdir" ]] || return 1
    [[ "$gitdir" = /* ]] || gitdir="$root/$gitdir"
  fi
  [[ -d "$gitdir" && -f "$gitdir/HEAD" ]] || return 1
  IFS= read -r entry < "$gitdir/HEAD" || return 1
  if [[ "$entry" != ref:* ]]; then
    [[ -n "$entry" ]] || return 1
    printf '%s\n' "$entry"
    return 0
  fi
  ref="${entry#ref: }"
  if [[ -f "$gitdir/$ref" ]]; then
    IFS= read -r value < "$gitdir/$ref" || return 1
    [[ -n "$value" ]] || return 1
    printf '%s\n' "$value"
    return 0
  fi
  commons="$gitdir"
  if [[ -f "$gitdir/commondir" ]]; then
    IFS= read -r commons < "$gitdir/commondir" || return 1
    [[ -n "$commons" ]] || return 1
    [[ "$commons" = /* ]] || commons="$gitdir/$commons"
  fi
  packed="$commons/packed-refs"
  if [[ -f "$packed" ]]; then
    while IFS=' ' read -r value ref_name; do
      if [[ "$ref_name" == "$ref" && -n "$value" ]]; then
        printf '%s\n' "$value"
        return 0
      fi
    done < "$packed"
  fi
  return 1
}

# Resolve the checked-out commit, preferring git and falling back to git-metadata parsing.
amarel_resolve_head_commit() {
  local root="$1" git_bin="" value=""
  if git_bin="$(amarel_git_bin)"; then
    value="$("$git_bin" -C "$root" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "$value" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  fi
  amarel_head_commit_without_git "$root"
}

# Verify the checked-out commit against the submitted commit, with the submit-time stamp as the
# documented fallback when the compute node exposes no git metadata at all.
amarel_require_source_commit() {
  local root="$1" expected="$2" stamp="${3:-}" actual=""
  if [[ -z "$expected" ]]; then
    printf 'AIM3_SOURCE_COMMIT is empty; cannot verify source identity\n' >&2
    return 1
  fi
  actual="$(amarel_resolve_head_commit "$root" || true)"
  if [[ -n "$actual" ]]; then
    if [[ "$actual" != "$expected" ]]; then
      printf 'Source commit changed after submission: expected %s, found %s\n' \
        "$expected" "$actual" >&2
      return 1
    fi
    return 0
  fi
  if [[ -n "$stamp" && -s "$stamp" ]]; then
    actual=""
    IFS= read -r actual < "$stamp" || actual=""
    if [[ "$actual" != "$expected" ]]; then
      printf 'Submit-time commit stamp does not match AIM3_SOURCE_COMMIT: expected %s, stamp %s has %s\n' \
        "$expected" "$stamp" "${actual:-nothing}" >&2
      return 1
    fi
    printf 'note: no usable git metadata at %s; source identity accepted from submit-time stamp %s\n' \
      "$root" "$stamp" >&2
    return 0
  fi
  printf 'Cannot verify source identity: no git metadata at %s and no submit-time stamp at %s\n' \
    "$root" "${stamp:-<unset>}" >&2
  return 1
}
