# Experiment Monitoring Registry

该目录是项目内的轻量实验检测层，用来快速定位 Amarel 和 sjc-remote job。它不依赖
外部任务服务，不定义模型、实验协议或研究结论，也不属于项目的关键创新。所有文件使用
Python 标准库，可在 Mac 和 Mac mini 的同一项目版本中使用。

## 三个组件

1. `jobs/<id>.json`：每个实验一个持久 manifest，保存跨端统一的简短 experiment ID、
   Slurm/tmux execution ID、remote root、日志、结果路径、有效完成条件和备注。
   `id` 不包含 host 或执行器 ID；host 位于 `host`，Amarel 的 Slurm ID 位于 `job_ids`，SJC
   tmux/run ID 位于 `run_ids`。单 job 单文件可以减少多端同时登记时的合并冲突。
2. `active_jobs.json` 与 `JOBS.md`：前者是当前活跃实验的轻量机器索引，后者是默认不清空的
   人类和 agent 可搜索历史。两者均可从 manifests 重建，且都不取代 manifest 的事实来源地位。
3. `progress.py`：只接受完整 experiment ID，并只读取对应 manifest。随后按 host 合并为
   一次 SSH，读取
   scheduler/tmux、GPU、精确日志、`.done/.fail`、`metrics.json`、`metrics_history.jsonl`
   和 checkpoint 证据。

## 提交后立即登记

简单 job 可以直接登记：

```bash
python -m experiments.monitoring.job_registry new \
  --id pong-example-12345678 \
  --description "Pong example sweep" \
  --host amarel \
  --remote-root /absolute/experiments/remote/worktree \
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
- expected units，以及能够定位对应结果的 metrics/checkpoint 文件名或 glob。

默认使用 `artifacts` 判定：若 `result_globs` 匹配结果目录，则可解析的最终
`metrics.json` 与明确要求的 checkpoint 数量构成 valid 证据；若它匹配精确结果文件
（例如每 seed 的 `.npz`），该文件本身就是该 unit 的 valid artifact，不再要求不存在的
`metrics.json`。`expected` 元数据差异与缺少 `.done` marker 仍会显示在诊断输出中，
但不会否决结果；早期失败尝试遗留的 `.fail` marker 也不会覆盖已经完整生成的结果
artifacts。确实需要逐字段、marker 都完全一致的实验，
可以在 `tracking` 或单个 unit 中设置 `"validation_mode": "strict"`。

## 快速搜索进度

```bash
# 只使用完整 experiment ID：不会读取或校验无关历史 manifest
python -m experiments.monitoring.progress \
  rl-atari-multitask-5task18-l3-eps500k-lrpertask-20m-lstm-gru-s1-2 --no-update

# 机器可读结果
python -m experiments.monitoring.progress \
  rl-atari-multitask-5task18-l3-eps500k-lrpertask-20m-lstm-gru-s1-2 --json
```

检查器不会递归搜索远端 home。它只访问 manifest 中记录的 remote root、日志和结果路径。
完整 ID 的本地解析不会遍历 `jobs/`，因此无关旧 JSON 的本地读取/校验错误不会阻断该查询；
自然语言只用于人工或 agent 在 `JOBS.md` / `active_jobs.json` 中确定唯一完整 ID，不能作为
checker 参数。单任务查询没有扫描完整历史的入口。每次
`progress` 对 alias-only endpoint 会先执行 `ssh -O check <alias>`；失败会原样报告 socket
错误且不新建 SSH。DSW 等没有稳定 alias/ControlMaster 的 endpoint，可以在忽略的
`.agents/local.md` 中按 ``- DSW 5000: `ssh ... root@host` `` 记录完整命令；logical host
`dsw-5000` 会自动匹配该配置并执行一次显式 direct foreground SSH，而不是在 socket 失败后
静默降级。`--ssh-alias` 显式覆盖仍优先，并继续要求可用的 ControlMaster socket。成功后，
同一 host/Conda 配置的多个 job 会合并到一个前台 SSH 会话。即使用 `--no-update`，若全部
expected units 都具有有效 artifacts，输出也会报告 `completed (verified)`，不会再把
已经完成的文件型分析显示为 `0/N`。只有 manifest 明确设置
`tracking.auto_complete=true` 且不传 `--no-update` 时，检查器才会将非终态记录写为
`completed`；这应在完成交接而非纯 status-only 查询中执行。`job_registry new` 对带有
`--expected-units` 和 `--result-path` 的新登记默认启用自动完成，可用
`--no-auto-complete` 明确关闭。

如果某台 Mac 使用不同 SSH alias，可临时覆盖：

```bash
python -m experiments.monitoring.progress <完整实验ID> \
  --ssh-alias sjc-remote=my-sjc-alias --no-update
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
