# 三端仓库收敛记录（2026-07-16）

## 目标

本地、sjc 和 Amarel 各只维护一个长期仓库，统一目录名为 `aim3_gawf_rnn`。
任务分支保留 Git history，但不再永久绑定独立目录；正式实验以 commit hash 为准。

## 本地合并来源

| 原目录 | 分支 | 审计结论 |
|---|---|---|
| `FAW_RNN-atari` | `codex/atari-final-accel-validation` | Atari/MiniGrid、共享 core 与 acceleration 已合并 |
| `FAW_RNN-clutter` | `codex/clutter-mnist` | Clutter GaWF feedback grid 修正已合并 |
| `FAW_RNN-text` | `codex/text-imdb-sentihood` | IMDB/SentiHood 内容已合并并迁移到统一 depth 接口 |
| `FAW_RNN-clutter-best6-10seed` | `codex/clutter-best6-10seed` | chan=1、multi-seed 与 monitoring 更新已合并 |
| `FAW_RNN-gawf-core-accel` | `codex/gawf-core-feedback-accel` | core acceleration 已被统一实现覆盖；MiniGrid paper 文件逐项核对并保留新版本 |
| `FAW_RNN-monitoring` | `codex/pong-depth2-unified-cores` | 历史已连接，内容由当前实现覆盖 |

主工作区合并前的 dirty/untracked 状态保存在命名 stash
`pre-unification-main-worktree-2026-07-16`。在最终验证和人类确认前不删除该恢复点。
`FAW_RNN-gawf-core-accel` 的 untracked 状态另存为
`pre-removal-gawf-core-worktree-untracked-2026-07-16`。六个旧 worktree 均已在逐项 dry-run
并确认内容已合并或保存后移除；本地只保留主工作树。

## 迁移原则

- 新任务目录为 `experiments/clutter/`、`experiments/atari/`、
  `experiments/minigrid/`、`experiments/text/`。
- `experiments/amarel/` 与 `experiments/local/` 只表示执行环境，不代表独立代码副本。
- monitoring 中旧 `remote_root` 是真实运行 provenance，不重写成未实际使用的新路径；
  新提交不得再使用这些任务 worktree 根目录。
- 当前仍在执行的远端 run 可继续读取原只读 snapshot，结束后不再复用。
- 数据与结果通过根目录 `stimuli`、`results` 的本地 symlink 接入；两者由 Git 忽略。

## 远端状态

- sjc 长期仓库：`/G/MIMOlab/Codes/aim3_gawf_rnn`。旧 Atari 目录被交互 shell 占用，
  旧主目录和 Clutter snapshot 被活动训练依赖；待进程结束后再逐项清理。
- Amarel 长期仓库：`/cache/home/js3269/projects/aim3_gawf_rnn`。非活动执行快照已移动到
  `.aim3_gawf_rnn_execution_snapshots/20260716/`；旧主目录被登录进程占用，两个 Slurm
  snapshot 仍在运行，均暂不移动或删除。
- Amarel 旧代码、Git 元数据和独立结果已分别归档到
  `/scratch/js3269/code_archives/aim3_gawf_rnn/20260716/` 和
  `/scratch/js3269/result_archives/aim3_gawf_rnn/20260716/`，并完成 SHA-256 与压缩包校验。
- sjc 旧任务代码归档于 `/G/MIMOlab/Codes/.aim3_gawf_rnn_archives/20260716/`；独立结果已用
  非删除式、checksum 校验的同步方式并入统一结果根目录。

## 后续清理门槛

任何剩余旧目录只能在对应训练、Slurm job 和 shell cwd 全部退出后处理。每次必须重新核对
精确路径、活动进程、归档或结果副本，并先 dry-run；不得批量递归删除父目录，也不得使用
`rsync --delete` 清理仓库或结果根目录。GitHub 仓库由人类改名后，再更新三端 `origin` URL。
