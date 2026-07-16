# Clutter experiments

Clutter 的训练入口为 `train_model.py`。train-scale / fixed-40h validation、1024-run grid、
parameter matching、GaWF feedback LR grid 与 fixed-best multi-seed 定义目前集中在
`experiments/generalization/`；Amarel 与双 GPU wrappers 分别位于 `experiments/amarel/`
和 `experiments/local/`。

输出使用 `results/train_data/<result_suffix>/`，聚合表写入实验自己的 `artifacts/`。
新 launcher 不得绑定 `FAW_RNN-clutter*` 等旧 worktree 路径。
