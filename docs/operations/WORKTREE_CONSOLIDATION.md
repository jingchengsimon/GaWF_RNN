# 本地 worktree 合并审计（2026-07-16）

## 目标

本地长期维护入口收敛到 `FAW_RNN`。任务分支保留 Git history，但不再永久绑定独立目录；
正式实验以 commit hash 为准。旧目录本轮只归档，不删除，等待人类确认。

## 已审计来源

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

## 迁移原则

- 新任务目录为 `experiments/clutter/`、`experiments/atari/`、
  `experiments/minigrid/`、`experiments/text/`。
- `experiments/amarel/` 与 `experiments/local/` 只表示执行环境，不代表独立代码副本。
- monitoring 中旧 `remote_root` 是真实运行 provenance，不重写成未实际使用的新路径；
  新提交不得再使用这些任务 worktree 根目录。
- 当前仍在执行的远端 run 可继续读取原只读 snapshot，结束后不再复用。

## 删除门槛

旧目录当前为待确认 archive。只有在统一分支测试通过、三端同步与最终项目改名完成、
且人类明确确认后，才可逐一移除 worktree。删除前再次列出每个目录的 branch、HEAD、
dirty/untracked 状态并验证 recovery copy；不得批量递归删除父目录。
