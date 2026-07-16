# Pending Atari Result Deletions

Confirmed by the user on 2026-07-13. These results used the rejected
`frame_skip=4, frame_stack=1` protocol and must not be used as the replacement
for strict one-frame Pong. Keep them only until their strict `fs1/stack1`
replacements have been validated.

## Pending groups

1. Historical single-layer 70-unit sweep
   - Remote root: Amarel project configured in `.agents/local.md`
   - Results: `results/train_data/atari_dqn_pong_fs4_stack1_*`
   - Artifacts: `experiments/amarel/artifacts/atari_pong_fs4_stack1/`
   - Delete only after the strict `fs1/stack1` 70-unit replacement is 70/70 valid.

2. Historical depth-2 10-unit sweep
   - Remote root: alternate Amarel worktree configured in `.agents/local.md`
   - Results: `results/train_data/atari_dqn_pong1f_*depth2match*`
   - Artifacts: `experiments/amarel/artifacts/atari_pong_depth2/`
   - Delete only after the strict `fs1/stack1` depth-2 replacement is 10/10 valid.

3. Historical smoke runs
   - Remote root: Amarel project configured in `.agents/local.md`
   - Results: `results/train_data/smoke_cnn`, `smoke_gawf`, and `smoke_mamba`
   - Delete with the historical single-layer cleanup after replacement validation.

## Deletion rule

Do not delete any path solely because it is listed here. At deletion time,
verify the replacement counts and obtain the user's final confirmation.
