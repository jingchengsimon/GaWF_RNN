"""Small checks for structured Skiing behavior-audit summaries."""

from utils.analysis.rl.atari.evaluate_skiing_behavior import _episode_summary


def test_episode_summary_measures_progress_gaps_and_action_counts() -> None:
    summary = _episode_summary(
        episode=2,
        rewards=[-1.0] * 6,
        actions=[0, 1, 1, 17, 17, 17],
        legal_actions=[0, 0, 0, 8, 8, 8],
        progress_changed=[True, False, False, True, False, False],
        terminated=False,
        truncated=True,
        end_reason="stalled",
        final_info={"stall_steps": 450, "course_progress_events": 2},
    )

    assert summary["return"] == -6.0
    assert summary["course_progress_events"] == 2
    assert summary["max_consecutive_no_progress_steps"] == 2
    assert summary["final_consecutive_no_progress_steps"] == 2
    assert summary["canonical_action_counts"][17] == 3
    assert summary["legal_action_counts"][8] == 3
    assert summary["wrapper_stall_steps"] == 450
