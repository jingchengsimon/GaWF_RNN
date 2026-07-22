"""Refuse to start a run that could exhaust the shared /scratch soft quota.

Running out of quota while an mmap replay buffer is mapped does not surface as a
clean ``ENOSPC``. Touching an already-mapped page raises ``SIGBUS``, which kills
the process without a useful traceback and leaves a truncated replay that then
fails the resume geometry check -- so the unit is neither finished nor
resumable. Refusing to start is much cheaper than diagnosing that afterwards.

Amarel serves one ``/scratch`` namespace from two GPFS clusters: ``DSSP`` on the
login node and the ``gpu###`` nodes, ``DSSK`` on the ``gpuk###`` and ``halk###``
nodes. Both see identical file content -- verified by writing from one class and
reading the same checksum from the other -- but they report different fileset
accounting for it, and the ``DSSK`` view additionally prints a placeholder row
whose usage column is the literal string ``none``:

    scratch    root       USR            none                             DSSK.amarel
    scratch    scratch    USR       776405392 1073741824 2147483648 ...   DSSK.amarel

An earlier version of this guard assumed one row with numeric columns and died
on ``int('none')``, which failed the job it was meant to protect. So: parse every
row that has numeric usage, ignore the rest, and take the most pessimistic view,
because a task may land on either cluster and be enforced by either.

Failure policy is deliberately asymmetric. A parse problem or a missing
``mmlsquota`` fails **open** with a warning -- a guard that cannot read the quota
must not take down a whole array. Only a successfully measured shortage fails
**closed**.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

KIB_PER_GIB = 1024 * 1024

EXIT_OK = 0
EXIT_BLOCKED = 4


def parse_quota_rows(text: str, filesystem: str) -> list[tuple[int, int]]:
    """Return ``(used_kb, soft_kb)`` for every parsable row of ``filesystem``.

    Rows whose usage or quota column is not an integer -- notably the ``none``
    placeholder row -- are skipped rather than treated as an error.
    """
    rows: list[tuple[int, int]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != filesystem:
            continue
        try:
            used_kb = int(fields[3])
            soft_kb = int(fields[4])
        except ValueError:
            continue
        if soft_kb <= 0:
            continue
        rows.append((used_kb, soft_kb))
    return rows


def most_pessimistic_free_gib(rows: list[tuple[int, int]]) -> float | None:
    """Smallest headroom across the reporting clusters, in GiB."""
    if not rows:
        return None
    return min((soft_kb - used_kb) / KIB_PER_GIB for used_kb, soft_kb in rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--filesystem", default="scratch")
    parser.add_argument(
        "--required_gib",
        type=float,
        required=True,
        help="Space one unit needs; the guard demands --headroom_factor times this.",
    )
    parser.add_argument(
        "--headroom_factor",
        type=float,
        default=3.0,
        help="Multiplier over one unit: this run, a concurrent peer, and final artifacts.",
    )
    parser.add_argument(
        "--marker_path",
        default=None,
        help="Write a blocked-status marker here when refusing to start.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        output = subprocess.run(
            ["mmlsquota", "-u", args.user, args.filesystem],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except Exception as error:  # noqa: BLE001 - any failure must fail open
        print(f"quota check skipped ({type(error).__name__}: {error})")
        return EXIT_OK

    rows = parse_quota_rows(output, args.filesystem)
    free_gib = most_pessimistic_free_gib(rows)
    if free_gib is None:
        print("quota check skipped: no parsable quota row")
        print(output.strip())
        return EXIT_OK

    needed_gib = args.required_gib * args.headroom_factor
    views = ", ".join(
        f"{(soft - used) / KIB_PER_GIB:.1f}" for used, soft in rows
    )
    if free_gib < needed_gib:
        message = (
            f"Refusing to start: {free_gib:.1f} GiB below the {args.filesystem} "
            f"soft quota, need {needed_gib:.1f} GiB "
            f"({args.required_gib:.1f} x {args.headroom_factor:g}). "
            f"Per-cluster headroom: {views} GiB."
        )
        print(message, file=sys.stderr)
        if args.marker_path:
            with open(args.marker_path, "w", encoding="utf-8") as handle:
                handle.write(
                    f"status=blocked_quota free_gib={free_gib:.1f} "
                    f"needed_gib={needed_gib:.1f}\n"
                )
        return EXIT_BLOCKED

    print(
        f"quota ok: {free_gib:.1f} GiB headroom (need {needed_gib:.1f} GiB); "
        f"per-cluster headroom: {views} GiB"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
