# MiniGrid experiments

MiniGrid DQN/DRQN、通用 PPO 和 paper-protocol PPO 分别由 `train_minigrid_dqn.py`、
`train_minigrid_ppo.py` 与 `train_minigrid_ppo_paper.py` 启动。共享 recurrent core 位于
`utils/recurrent_cores/`，任务专用 encoder/head 位于 `utils/minigrid_*`。

Amarel paper PPO 由 `experiments/amarel/submit_minigrid_ppo_paper.sh` 提交；本地双 GPU
smoke 使用 `experiments/local/run_minigrid_ppo_paper_smoke_2gpu.sh`。远端输出必须显式
使用 `AIM3_RESULTS_PATH`，不依赖旧 MiniGrid worktree。
