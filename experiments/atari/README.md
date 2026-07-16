# Atari experiments

Atari A2C 使用 `train_atari.py`；DQN/DRQN 使用 `train_atari_dqn.py`。本目录保存任务定义，
包括 `atari_ssm_param_match.py`。Slurm 和双 GPU wrappers 仍按执行环境放在
`experiments/amarel/` 与 `experiments/local/`。

Pong 名称必须明确写为 `pong_fs1_stack1` 或 `pong_fs4_stack1`。正式运行记录 commit hash、
frame skip、frame stack、feedback mode 和结果 suffix，不依赖旧 Atari worktree 路径。
