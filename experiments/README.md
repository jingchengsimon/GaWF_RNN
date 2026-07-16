# Experiment layout

长期维护只使用一个项目仓库。任务定义按研究对象归档，执行器按运行环境归档：

| 目录 | 职责 |
|---|---|
| `clutter/` | Clutter 协议入口与专项说明 |
| `atari/` | Atari 参数匹配与协议说明 |
| `minigrid/` | MiniGrid PPO/DQN 协议说明 |
| `text/` | IMDB/SentiHood grids 与协议说明 |
| `generalization/` | Clutter train-scale / fixed-40h validation 流程 |
| `amarel/` | Slurm submit/run/check wrappers |
| `local/` | 本地或 sjc 双 GPU wrappers |
| `monitoring/` | 项目内 run manifests；历史执行根目录保持原样作为 provenance |

任务代码、模型和结果不再依赖长期 worktree。正式实验应记录明确 commit hash；需要隔离
运行代码时使用由该 commit 生成的只读 snapshot，而不是再维护任务命名的仓库副本。
