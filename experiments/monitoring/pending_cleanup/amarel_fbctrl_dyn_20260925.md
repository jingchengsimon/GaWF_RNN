# Amarel old fbctrl / dyn cleanup — pending exact-target approval

The user retired the old feedback-control and dynamic-baseline protocols to development
history. Their four held jobs (61726137, 61726138, 61731757, 61731758) were cancelled on
2026-09-25 EDT and then absent from `squeue`; no matching trainer process was observed.
**No result or artifact directory below has been deleted.** Automated approval rejected the
proposed 17-leaf deletion because the user's broad authorization did not explicitly cover
these exact preflight, versioned artifact, and log paths.

Inventory was captured on 2026-09-25 EDT. All listed leaves were directories with zero
symbolic links. The 17 leaves contain 732 files and 771,402,238 file bytes in total.
Approve this exact list before deletion; recheck job and writer absence and the full inventory
immediately before carrying it out.

| Exact leaf under `/scratch/js3269/results/` | Files | File bytes |
|---|---:|---:|
| `data/clutter/runs/dynamic_weight_baselines/clutter_dynamic_weight_baselines_ep150_v1` | 63 | 272,769,911 |
| `data/clutter/runs/preflight/dynamic_weight_baselines_2epoch_seed1_v1` | 16 | 9,441,851 |
| `data/clutter/runs/feedback_controls/clutter_feedback_controls_ep150_v1` | 194 | 474,539,145 |
| `data/analysis/dynamic_weight_baselines_reset_excluded_test_10seed_v1` | 2 | 858 |
| `data/analysis/feedback_controls_reset_excluded_test_10seed_v1` | 38 | 16,144 |
| `data/analysis/feedback_controls_shuffle_resetexcluded_10seed_v1` | 114 | 8,690,416 |
| `artifacts/clutter_dynamic_weight_baselines_ep150_v1` | 0 | 0 |
| `artifacts/clutter_dynamic_weight_baselines_ep150_v2_chain` | 1 | 41 |
| `artifacts/clutter_dynamic_weight_baselines_ep150_v3_chain` | 90 | 388,801 |
| `artifacts/dynamic_weight_baselines_2epoch_seed1_v1` | 8 | 476 |
| `artifacts/dynamic_weight_baselines_2epoch_seed1_v2_gitpath` | 8 | 164 |
| `artifacts/dynamic_weight_baselines_2epoch_seed1_v3` | 1 | 41 |
| `artifacts/dynamic_weight_baselines_2epoch_seed1_v4_chain` | 9 | 1,013 |
| `artifacts/dynamic_weight_baselines_2epoch_seed1_v5` | 21 | 34,839 |
| `artifacts/clutter_feedback_controls_ep150_v1_amarel` | 2 | 119 |
| `artifacts/clutter_feedback_controls_ep150_v1_amarel_retry1` | 2 | 149 |
| `artifacts/clutter_feedback_controls_ep150_v1_amarel_retry2` | 163 | 5,518,270 |

The two expected aggregate leaves
`data/analysis/dynamic_weight_baselines_formal_10seed_v1` and
`data/analysis/feedback_controls_formal_10seed_v1` did not exist in the inventory.
Preserve `data/analysis/feedback_controls_prerequisites_v1` (input data), all source
worktrees, registry manifests, and the new `clutter_output_only_330_v1` campaign.
