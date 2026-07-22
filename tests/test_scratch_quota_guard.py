"""Parsing tests for the /scratch quota guard.

The fixtures below are verbatim `mmlsquota -u js3269 scratch` output captured
from both Amarel GPFS clusters on 2026-07-22. The DSSK variant is the one that
crashed an earlier inline version of this guard.
"""

from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.amarel.scratch_quota_guard import (
    most_pessimistic_free_gib,
    parse_quota_rows,
)

# Login node and gpu### nodes.
DSSP_OUTPUT = """                         Block Limits                                               |     File Limits
Filesystem Fileset    type             KB      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
scratch    root       USR       389857920 1073741824 2147483648     163840     none |   165026       0        0       20     none DSSP.amarel
"""

# gpuk### and halk### nodes: note the placeholder row whose usage is "none".
DSSK_OUTPUT = """                         Block Limits                                               |     File Limits
Filesystem Fileset    type             KB      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
scratch    root       USR            none                                        DSSK.amarel
scratch    scratch    USR       776405392 1073741824 2147483648    1153712     none |   108911       0        0      102     none DSSK.amarel
"""


class ParseQuotaRowsTest(unittest.TestCase):
    def test_dssp_single_row(self) -> None:
        rows = parse_quota_rows(DSSP_OUTPUT, "scratch")
        self.assertEqual(rows, [(389857920, 1073741824)])

    def test_dssk_placeholder_row_is_skipped(self) -> None:
        """The 'none' row must be ignored, not crash the guard."""
        rows = parse_quota_rows(DSSK_OUTPUT, "scratch")
        self.assertEqual(rows, [(776405392, 1073741824)])

    def test_headers_are_ignored(self) -> None:
        self.assertEqual(parse_quota_rows(DSSP_OUTPUT.splitlines()[0], "scratch"), [])

    def test_other_filesystem_is_ignored(self) -> None:
        self.assertEqual(parse_quota_rows(DSSK_OUTPUT, "home"), [])

    def test_zero_soft_limit_is_ignored(self) -> None:
        text = "scratch    root       USR       1000 0 0 0 none\n"
        self.assertEqual(parse_quota_rows(text, "scratch"), [])

    def test_empty_and_garbage_input(self) -> None:
        self.assertEqual(parse_quota_rows("", "scratch"), [])
        self.assertEqual(parse_quota_rows("mmlsquota: command failed\n", "scratch"), [])


class FreeHeadroomTest(unittest.TestCase):
    def test_no_rows_returns_none_so_caller_fails_open(self) -> None:
        self.assertIsNone(most_pessimistic_free_gib([]))

    def test_dssp_headroom(self) -> None:
        free = most_pessimistic_free_gib(parse_quota_rows(DSSP_OUTPUT, "scratch"))
        self.assertAlmostEqual(free, 652.2, places=1)

    def test_dssk_headroom_is_much_tighter(self) -> None:
        free = most_pessimistic_free_gib(parse_quota_rows(DSSK_OUTPUT, "scratch"))
        self.assertAlmostEqual(free, 283.6, places=1)

    def test_takes_the_smallest_headroom_across_clusters(self) -> None:
        """A task may land on either cluster, so the tighter view binds."""
        rows = parse_quota_rows(DSSP_OUTPUT, "scratch") + parse_quota_rows(
            DSSK_OUTPUT, "scratch"
        )
        self.assertAlmostEqual(most_pessimistic_free_gib(rows), 283.6, places=1)

    def test_eight_concurrent_units_fit_but_twelve_do_not(self) -> None:
        """Pins the concurrency cap to the measured DSSK headroom."""
        free = most_pessimistic_free_gib(parse_quota_rows(DSSK_OUTPUT, "scratch"))
        gib_per_unit = 28.2 * 1000**3 / 1024**3  # 28.2 GB advertised, in GiB
        self.assertLess(8 * gib_per_unit, free)
        self.assertGreater(12 * gib_per_unit, free)


if __name__ == "__main__":
    unittest.main()
