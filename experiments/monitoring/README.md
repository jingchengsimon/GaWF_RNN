# Experiment Monitoring Registry

该目录是项目内的轻量实验检测层，用来快速定位 Amarel 和 sjc-remote job。它不依赖
Dashboard，不定义模型、实验协议或研究结论，也不属于项目的关键创新。所有文件使用
Python 标准库，可在 Mac 和 Mac mini 的同一项目版本中使用。

## 三个组件

1. `jobs/<id>.json`：每个实验一个持久 manifest，保存 job/run ID、remote root、日志、
   结果路径、有效完成条件和备注。单 job 单文件可以减少多端同时登记时的合并冲突。
2. `active_jobs.json` 与 `JOBS.md`：前者是当前活跃实验的机器索引，后者是默认不清空的
   人类可搜索历史。两者均可从 manifests 重建。
3. `progress.py`：根据 ID、名称或结果路径解析 manifest，并按 host 合并为一次 SSH，读取
   scheduler/tmux、GPU、精确日志、`.done/.fail`、`metrics.json`、
   `metrics_history.jsonl` 和 checkpoint 证据。

## 提交后立即登记

简单 job 可以直接登记：

```bash
python -m experiments.monitoring.job_registry new \
  --id pong-example-12345678 \
  --description "Pong example sweep" \
  --host amarel \
  --remote-root /absolute/remote/worktree \
  --conda-init /absolute/path/to/conda.sh \
  --scheduler-type slurm \
  --job-id 12345678 \
  --log-glob 'experiments/amarel/artifacts/example/*.out' \
  --result-path 'results/train_data/example_*' \
  --expected-units 7
```

需要严格逐单元验证时，复制 `run_manifest.template.json`，填写 `tracking.units`，然后：

```bash
python -m experiments.monitoring.job_registry register /tmp/my-run-manifest.json
```

提交 job 的同一轮工作必须完成登记。manifest 至少记录：

- 人类可读描述、logical host、remote root、`aim3_rnn` Conda 初始化路径；
- 所有 Slurm job ID，或 sjc run ID、tmux session、process pattern；
- 精确日志 glob、status 目录和 result path/prefix；
- expected units，以及能够确认结果有效的 metrics/checkpoint 条件。

## 快速搜索进度

```bash
# 无参数：检查全部 active jobs；如果没有 active job，则检查最新历史记录
python -m experiments.monitoring.progress

# 任何稳定标识均可搜索
python -m experiments.monitoring.progress 58145944
python -m experiments.monitoring.progress fscompare1m
python -m experiments.monitoring.progress sjc-pong-fscompare1m-c219996

# 一次检查某 host 的所有 active jobs
python -m experiments.monitoring.progress --active --host amarel

# 机器可读结果
python -m experiments.monitoring.progress fscompare1m --json
```

检查器不会递归搜索远端 home。它只访问 manifest 中记录的 remote root、日志和结果路径。
同一 host/Conda 配置的多个 job 会合并到一个前台 SSH 会话。只有 manifest 明确设置
`tracking.auto_complete=true` 且严格满足全部 expected units 时，检查器才会把非终态记录
自动更新为 `completed`；其他状态不会被猜测。简单 CLI 登记默认关闭自动完成。

如果某台 Mac 使用不同 SSH alias，可临时覆盖：

```bash
python -m experiments.monitoring.progress --active --ssh-alias sjc-remote=my-sjc-alias
```

## 状态、同步与保留

```bash
python -m experiments.monitoring.job_registry list
python -m experiments.monitoring.job_registry list --active
python -m experiments.monitoring.job_registry show 58145944
python -m experiments.monitoring.job_registry set-status <id> running
python -m experiments.monitoring.job_registry rebuild
```

- `jobs/*.json`、`JOBS.md`、`active_jobs.json` 都保存在项目中，随项目版本在 Mac 和
  Mac mini 间同步。
- 每台机器仍分别维护忽略的 `.agents/local.md` 和 SSH config；密码和私钥不得写入
  manifest。
- manifests 是事实来源。如果生成的 `JOBS.md` 或 `active_jobs.json` 发生合并冲突，
  保留双方 `jobs/*.json` 后运行 `rebuild`。
- 两个生成索引不写每次检查时间，因此相同 manifests 会得到确定性内容，减少 Mac 与
  Mac mini 的无意义同步冲突。
- 历史 backfill 如果不能确认精确日志或结果路径，应留空并写入 note，不得用递归整个
  artifacts/home 的宽泛 glob 代替事实。
- completed、failed、cancelled 记录都继续保留。只有人类明确要求清理时才允许：

```bash
python -m experiments.monitoring.job_registry remove <id> --human-confirmed
```

不得因为完成、失败、超时或过期自动删除记录。
