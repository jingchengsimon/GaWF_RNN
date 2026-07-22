# Atari experiments

Atari A2C 使用 `train_atari.py`；DQN/DRQN 使用 `train_atari_dqn.py`。本目录保存任务定义，
包括 `atari_ssm_param_match.py`。Slurm 和双 GPU wrappers 仍按执行环境放在
`experiments/amarel/` 与 `experiments/local/`。

Pong 名称必须明确写为 `pong_fs1_stack1` 或 `pong_fs4_stack1`。正式运行记录 commit hash、
frame skip、frame stack、feedback mode 和结果 suffix，不依赖旧 Atari worktree 路径。

严格单任务 Pong 的 checkpoint 视频使用 `utils_anal/evaluate_atari_dqn_video.py` 在
`render_mode=rgb_array` 下执行 greedy evaluation，并由 OpenCV 直接编码 MP4，避免依赖
Gymnasium 的可选 MoviePy 录制组件。正式视频同时保存 metadata JSON，注明训练 seed、
evaluation seed、逐 episode return、被选中的最佳 episode、frame protocol 和 checkpoint。
