# Atari experiments

Atari A2C 使用 `python run_task.py atari-a2c`；DQN/DRQN 使用
`python run_task.py atari-dqn`。本目录保存任务定义，
包括 `atari_ssm_param_match.py`。Slurm 和双 GPU wrappers 仍按执行环境放在
`amarel/`。

Pong 名称必须明确写为 `pong_fs1_stack1` 或 `pong_fs4_stack1`。正式运行记录 commit hash、
frame skip、frame stack、feedback mode 和结果 suffix，不依赖旧 Atari worktree 路径。

严格单任务 Pong（6-action）和 Breakout（4-action）的 checkpoint 视频使用
`utils/analysis/rl/atari/evaluate_dqn_video.py` 在 `render_mode=rgb_array` 下执行 greedy evaluation，
并由 OpenCV 直接编码 MP4，避免依赖 Gymnasium 的可选 MoviePy 录制组件。正式视频同时保存
metadata JSON，注明训练 seed、evaluation seed、逐 episode return、被选中的最佳 episode、
frame protocol 和 checkpoint。

five-task full-18 representative videos use
`amarel/submit_atari_5task_18action_videos.sh`. The compute job first evaluates all three
training seeds for each GaWF/LSTM-task pair on the same fixed greedy suite, selects the median
seed by mean return, then records the first episode from that same fixed evaluation seed. It
writes ten MP4s, per-video metadata, the 30-episode selection metadata, and
`selected_seeds.json` below `results/videos/5task_18action/`.

## Five-task full-18 pilot

固定 five-task 协议使用 Pong、Breakout、Assault、Seaquest 和 Skiing，所有任务共享
ALE canonical 18-action output，模型不接收 task ID。Collection 在 episode boundary 按累计
environment steps 选择最少的 task；replay batch 的非整除 remainder 在 task 间轮转。
训练须等每个 task 至少收集 20k valid environment steps 后才开始 update，scheduler
task counts/cursor 随 checkpoint 恢复。
Skiing 的 ALE legal action set 只有 9 个动作，因此 unsupported fire variants 映射到对应的
legal non-fire movement，standalone FIRE 映射到 NOOP；模型仍保持 18-dim output，且不使用
task-specific action mask。

历史 global-step decay 的 Amarel 提交入口是
`experiments/rl/atari/amarel/submit_atari_5task_18action_l3_pilot.sh`。当前正式 five-task
protocol 使用 `submit_atari_5task_18action_l3_lrpertask_pilot.sh`：每个 task 的 LR 在其自身
达到 1M environment steps 后才 decay；该 5M global-step pilot 因而不会触发 decay。它不设
smoke gate，提交 5-model × 3-seed × 5M-step array，三个较慢的 GaWF seeds 优先，最大并发为 5。
每个 task 使用独立的 0.5M mmap replay（每个 unit 总容量 2.5M transitions）；任务完成后清理
replay。Amarel 请求为单 GPU、16 CPUs、64G memory、30 小时 walltime，且不固定 GPU 型号，
以提高 backfill 机会。structured results 和后续 figures 分别写入
`results/data/rl/atari/5task_18action/{parameter_match,smoke,pilot,figs}/`；此 protocol 的 pilot
结果根为 `pilot/per_task_buf500k/`，不得写入或覆盖 `multitask_18action`。

Five-task DQN 的 epsilon 默认在固定的 500k global steps 内由 1.0 线性衰减到 0.01，与
总训练 budget 无关；五任务均衡收集时约为每个 task 100k environment steps。仅复现实验时可
显式传入历史 `--exploration_fraction`，新 launcher 不得再把 fraction 作为默认协议。

10M formal GRU/LSTM 扩展使用
`amarel/submit_atari_5task_18action_formal_10m.sh`。它固定五个 task、full18、fs4/stack4、
five 个独立 1M mmap partitions、10M global steps（每 task 2M）、GRU L3/h458 和当前验证的
LSTM L3/h373，各三个 seeds。它先运行 500-step structural/recovery smoke：checkpoint 必须包含
五个 replay partitions，受控 `SIGUSR1` 后自动 requeue/resume；仅 smoke 成功才通过
`afterok` 释放 six-unit formal array。成功的 smoke result/artifact leaves 由独立 compute
cleanup job 精确删除；正式 run 成功时只由训练入口删除本 unit 的 replay。若需重试且必须保留
失败 smoke 证据，使用 `--run-tag <tag>`；它会同时创建独立的 result 与 artifact leaves。

20M five-task/full18 L3 formal protocol 的 GaWF seed 1/2 使用相同的 500K per-task replay、
per-task 1M LR decay 和固定 500K global-step epsilon decay。追加 GRU/LSTM/GaWF 的 seed 3/4/5
时，使用 `amarel/submit_atari_5task_18action_l3_formal_20m_seeds3_5.sh`；它以
`6-14%2` 数组映射 nine units，保持每个 job 为单 Ada Lovelace GPU、16 CPUs、64G、72 小时及
可恢复 requeue。runner 在 walltime 前 10 分钟向 training step 直接发送 `SIGUSR1`；训练保存
checkpoint 后由 runner requeue。runner 的可选 `SEED_OFFSET` 为连续 seed 添加偏移；未设置时仍保持连续
`1..SEED_COUNT` 的既有行为。

## SJC two-task L3 GRU comparison

The SJC comparison launcher `experiments/remote/run_sjc_atari_multitask_l3_gru.sh` evaluates a
GRU L3 h458, seed 42 protocol using Pong plus Breakout (4M global steps) or a corresponding
single-task control (2M steps). All use full18, `fs4/stack4`, per-task 1M mmap replay and a
per-task 1M LR decay; the two-task run additionally uses `transition_balanced` collection and
`task_balanced` replay. Its smoke is fixed at 25k steps and writes to a separate `_smoke` result
leaf. These results belong under `results/data/rl/atari/multitask_18action/`, never the fixed
five-task namespace.

## Skiing stall/actionfix weights-only adaptation

`skiing-stall-actionfix-v1` 是独立于 historical five-task formal baseline 的新 MDP。模型仍
输出 18 个 Q-values；Pong、Breakout、Assault、Seaquest 的 ALE full action set 保持 identity，
Skiing 只做一次 18-to-9 non-FIRE legal-action mapping。Skiing 使用 ALE RAM 86:94 的 course
object y slots 变化作为下坡/赛程进展；连续 450 agent steps 无变化时返回
`truncated=True`、`info["end_reason"]="stalled"`，并用一次性 reward adjustment 将总 raw
return 限制到不高于 -30,000。该 truncation 重置 episode/recurrent state，但 TD target 继续
bootstrap；自然 terminal 仍停止 bootstrap。

SJC launcher `experiments/remote/run_sjc_atari_skiing_warmstart_l3.sh` 默认只接受完成的 20M
five-task final `state_dict`，不直接接受 resumable checkpoint。诊断性实验可显式使用
`--allow-incomplete-source`，但必须先从稳定只读复制的 checkpoint 中仅提取 model
`state_dict`，并在 metadata 与 result leaf 中记录精确 source step。两种路径均只加载 model
weights，fresh 初始化 optimizer、replay、global step、epsilon/LR schedules，并固定 seed1、
fs4/stack4、L3、full18。25k smoke 验收后，分别运行 LSTM h373、GRU h458、GaWF h604 的
1M single-Skiing adaptation。所有 leaf 均位于
`results/data/rl/atari/5task_18action/formal_20m_4mpertask_raw_seeds/`，不得写回 20M source
leaf，也不得建立平行 result parent。

三模型 single-Skiing return comparison 复用
`utils.analysis.rl.atari.atari_5task_raw_learning_curves` 的 model colors、rolling-100 return
和 provenance manifest。传入 `--task-only Skiing --x-axis environment_steps` 时只生成一个
Skiing return panel，不生成 shared TD loss panel；running histories 可作为明确标注时间点的
snapshot 重复渲染。

完成模型的无视频 behavior audit 使用
`utils.analysis.rl.atari.evaluate_skiing_behavior`。固定 greedy evaluation suite 保存逐 episode
return、length、stall reason、course-progress 间隔和 action counts 到 JSON，并将逐 step 的
canonical/legal action、reward、RAM 86:94 marker、Q-values 和 termination flags 保存为压缩
NPZ。Amarel 三模型 array 入口为 `amarel/submit_atari_skiing_behavior_audit.sh`；输入必须是只读
复制的 completed final metrics/state_dict，输出使用独立 result leaf，不写回训练结果。
默认 audit 使用 BF16；设置 `AUDIT_AMP_DTYPE=none` 和独立 `AUDIT_TAG` 可在相同固定 episode
suite 上运行 FP32-only inference，用于隔离 autocast quantization 对 greedy action ties 的影响。

累计 2M diagnostic extension 对完成的 1M 模型使用 `--extend-from-skiing-1m`：只加载 final
model weights，新增 phase 训练 1M steps，fresh 初始化 optimizer/replay/phase-local global
step，并固定 epsilon=0.01、LR=1e-5。仍在运行且保留 resumable checkpoint/replay 的 unit 可用
显式 budget-extension 开关把 target 从 1M 单向增加到 2M；除 `total_timesteps` 外的 resume
协议字段继续严格匹配。两种路径必须在 manifest 中分别标注 continuous resume 与
weights-only extension，不能把后者描述为严格续训。

## Skiing unclipped-reward / gamma 0.999 comparison

`amarel/submit_atari_skiing_unclipped_l3.sh` 提交 seed1 的 LSTM/GRU/GaWF：三个 25k
smoke 验收成功后，以 `afterok` 释放各完整 4M environment steps 的新训练阶段。
人类明确要求不做 smoke 时传入 `--skip-smoke`，直接提交三模型 4M array，并在 manifest
记录该授权；提交安全测试仍在计算节点执行。2026-09-05 本次运行采用该显式豁免。
复用上述 weights-only launcher，通过 `--gamma 0.999 --no-reward-clip` 恢复 TD reward
幅度；不额外缩放 reward，不改 stall adjustment、bootstrap 或 action mapping，仍用 BF16。
LSTM/GRU 初始化于原曲线的 20M source，GaWF 初始化于其 19.45M source；三者均 fresh
optimizer/replay，epsilon 1→0.01（500k steps），LR 1e-4 在 1M steps 后降到 1e-5。
每 unit 单 Ada GPU、16 CPUs、64G，formal 72h；walltime 前发送 SIGUSR1，保存后 requeue。
smoke 不生成视频，final metrics 记录 `gamma` 和 `reward_clip`；新 run 保留 replay。

该独立协议通过 `--result-parent` 写入
`results/data/rl/atari/5task_18action/single_skiing/unclipped_gamma0p999_4m_seed1/{smoke,formal}/`。
此显式新协议目录是上文 historical adaptation parent 约束的例外；既有 data/checkpoint
路径不迁移、不覆盖。只读初始化输入与新代码 snapshot 分开保存，并验证 SHA256。

本地 figures 已合并到 `results/figs/rl/atari/5task_18action/`：
`formal_20m_4mpertask_raw_seeds/` 保存 Amarel 五任务正式曲线；并列的 `single_skiing/`
保存原 SJC 单 Skiing 的 `01_skiing_learning_curves_environment_steps.png` 及 CSV/JSON
分析输入。Amarel 旧 pilot 位于 `pilot_amarel/`，原 `pilot/` 保留不覆盖。

已登记的 Seaquest/Skiing 进度图使用
`python -m utils.analysis.rl.atari.atari_registered_task_curves`：显式传入完整
`--experiment-id`（可重复）、`--snapshot-root`、`--task` 与 `--output-dir`。
只从 manifest 指定的 result leaves 同步 metrics/history 到带日期的本地 snapshot，
不读取 checkpoint/replay。图表区分 completed、in-progress 与 not-started，并保存
`curves.npz`、`summary.csv`、带输入 SHA256 和 experiment IDs 的 `manifest.json`。
Seaquest 图位于 `results/figs/rl/atari/5task_18action/seaquest_diagnostics/`；本次 Skiing
图位于 `5task_18action/single_skiing/unclipped_gamma0p999_4m_seed1/`，保留原历史曲线。

两次 Skiing 单任务比较使用
`PYTHONDONTWRITEBYTECODE=1 python -B -m utils.analysis.rl.atari.compare_skiing_protocols`，
显式传入 `--old-inputs`（旧 `figure_inputs.json`）、`--new-manifest`（新曲线
`manifest.json`）与 `--output-dir`；可同时传 `--old-audit`、`--new-audit`，各根目录
包含三个 model 的 `summary.json` / `step_trace.npz`。输出训练比较 PNG、Q-tie PNG、
`curves.npz` 和 `comparison.json`，位于 `single_skiing/protocol_comparison/`。
Q 分析验证 trace SHA256、finite values 和保存值的 BF16 可表示性；原 evaluator snapshot
未在 summary 记录 AMP，精度依据已登记 launcher 的 BF16 参数与 forward autocast 实现，
不能把 NPZ 的 FP32 存储 dtype 当成 forward 精度。legal-group tie 先按既定 18→9 mapping
对每组取最大 Q，再判断是否有多个不同 legal groups 并列最优。
旧方案为分段 weights-only extensions，新方案为一个可恢复的 4M phase，因此训练曲线
比较不能作为仅改变 reward clip / gamma 的严格 ablation。

## Skiing A/C/D/E/F seed1 diagnostics

`amarel/run_atari_skiing_acdef_l3.sh` runs one independent array task using
`amarel/skiing_acdef_diagnostic.py`; submit separate five-task arrays for LSTM/GRU/GaWF.
Array indices 0–4 map to A/C/D/E/F, with no dependencies between models or variants.
Each unit accepts its own isolated 25k smoke before starting a fresh 4M phase. Source weights
are the same immutable five-task seed1 inputs used by the unclipped/gamma=.999 comparison,
not that comparison's final Skiing weights. All units preserve BF16, unclipped reward,
gamma=.999, single canonical mapping, stall boundary, 500k replay and eight sequences/batch.

| Variant | Exploration steps / final epsilon | Initial LR / decay step | seq_len |
|---|---|---|---|
| A | 500k / .01 | 1e-4 / 1M | 16 |
| C | 1M / .05 | 1e-4 / 1M | 16 |
| D | 500k / .01 | 1e-4 / 3M | 16 |
| E | 500k / .01 | 3e-5 / 1M | 16 |
| F | 500k / .01 | 1e-4 / 1M | 64 |

All LR decays multiply by .1. F processes four times as many sequence positions per update;
it does not change the eight sequences/batch or update frequency. Clutter's 32-frame sample
window is distinct from Atari's baseline replay seq_len=16 and batch_size=32.

Results use `5task_18action/single_skiing/acdef_4m_seed1_20260907/{smoke,formal,evaluations}`.
Formal training saves immutable model-only snapshots every 500k, in addition to its resumable
50k checkpoint. After training, the same GPU evaluates all eight snapshots with BF16 greedy,
20 episodes and eval_seed=20260904. Evaluation does not alter training RNG or replay.
Each evaluation step retains summary.json, step_trace.npz, SHA256 and statistics.json containing
unclipped wrapper return (including the existing stall adjustment), stall rate, top-Q ties across
legal groups, canonical ties, margins, and canonical/legal action fractions. Final metrics serve
as model construction metadata for every snapshot; statistics.json records the actual snapshot
training_step separately. No video is generated. Interrupted audit attempts remain preserved.
User protection of checkpoint/result/replay data keeps accepted smoke outputs in this campaign.

For A/C/D/E/F training curves, use `atari_registered_task_curves --task Skiing --skiing-acdef`
with the three complete model experiment IDs and a timestamped local snapshot. This renders
three model panels with consistent variant colors and training-completion labels, plus numeric
NPZ/CSV and source SHA256 provenance. Store each dated output under
`results/figs/rl/atari/5task_18action/single_skiing/acdef_4m_seed1_20260907/`.
These online rolling-100 training returns are distinct from the fixed-seed greedy audits;
the plot's `done` label denotes completed training, not audit completion.

## Seaquest A/C seeds 1–4 on Amarel

`amarel/run_atari_seaquest_ac_l3.sh` runs independent array cells 0–15 using
`amarel/seaquest_ac_diagnostic.py`: 0–3 LSTM A, 4–7 LSTM C, 8–11 GaWF A, 12–15 GaWF C,
each in seed order 1–4. All cells use fresh initialization, isolated 25k smoke and fresh 3M
formal training, with no dependencies between cells. Commands and validation reuse the SJC
Seaquest diagnostics: full18, fs4/stack4, L3, LSTM h373/GaWF h605, BF16, clipped reward,
gamma .99, replay 1M, seq_len 16, eight sequences per batch, LR 1e-4 decaying by .1.
A uses epsilon 1→.01 over 300k and LR decay at 1M; C uses 1→.05 over 1M and decay at 2M.
Results live under `5task_18action/seaquest_diagnostics/ac_3m_seeds1_4_20260908/`
with separate `smoke` and `formal` leaves. Existing SJC seed2 results remain untouched.
Replay and smoke evidence are retained; Slurm warning flushes replay and saves training state
before requeue. Resume resets the environment and recurrent state as in the existing trainer.
The initial 16-cell quota budget is 450 GiB with 20% headroom; individual starts require
27 GiB with 20% headroom. Each cell requests one Ada GPU, 16 CPUs, 64G, and 72 hours.

### Seaquest A/C four-seed progress plots

Use `utils.analysis.rl.atari.atari_registered_task_curves --task Seaquest` with
`--seaquest-ac-multiseed` and the exact registered experiment ID. Four panels show
LSTM/GaWF × A/C, individual seed curves and the four-seed mean only over their
common step coverage (linear interpolation, no extrapolation). Numeric NPZ includes
individual curves, mean and sample SD; summary CSV records each unit’s progress.
Save each update to a new timestamped snapshot under `5task_18action/seaquest_diagnostics`.

### GaWF Riverraid five-task C profiling

The candidate tasks are Pong/Breakout/Assault/Seaquest/Riverraid. Use C optimization
with epsilon 1→.05 over 5M global steps (about 1M/task), LR 1e-4 decayed ×.1
when every task reaches 2M, gamma .99 and clipped rewards. Retain per-task 500k
mmap replay (2.5M total), transition-balanced collection, task-balanced replay,
BF16, fs4/stack4, L3, seq_len16 and eight sequences/batch. The formal budget and
multi-actor implementation remain undecided; no A control is requested.

`atari-dqn --profile_stages --record_timing` captures a 256-step CPU/CUDA trace
1,000 global steps after replay warm-up allows optimization. `profile/stages.json`
contains CUDA-synchronized host phase timings and collection task coverage;
`optimization` includes replay sampling, whose nested `aim3/replay_sample` range
is in the trace. `operators.txt` gives operator timings. These measurements include
instrumentation overhead. `throughput.json` separately records post-trace normal
throughput; this excludes replay warm-up and profiler serialization. The profiling
window need not cover collection from every task, although replay remains task-balanced.
Initial short diagnostics use seed1 and 200k steps; this is not the formal budget,
and the partly filled replay cannot establish full-buffer mmap I/O performance.

### One-allocation GaWF actor selection

`amarel/run_atari_actor_benchmark.sh` runs compute-node tests and
`amarel/actor_benchmark.py` on one allocated Ada GPU (16 CPUs, 64G). The candidate
set is one original synchronous collector versus five subprocess collectors, one
fixed game per subprocess; both learn one shared GaWF model from all five games.
Five workers are not five independent training runs. Replay remains 500k/task and
never mixes multiple actor trajectories within a task partition. Optimizer cadence
is one update per four aggregate transitions, including multiple updates after a
five-transition collection round. C schedules count aggregate/per-task experience
as documented above. Formal training budget is not chosen by this benchmark.

Both candidates run isolated 25k smoke checks, then fresh 300k trials in the order
1,5,5,1 on the same GPU, using seed1 throughout. Every trial keeps results/replay and
records GPU samples, stage traces and post-trace throughput. `recommendation.json`
selects five only when its median throughput improves by at least 10% and both
paired repeats improve; otherwise it selects one. It includes all raw trial summaries
and a compute-time projection. This selection applies to this five-task allocation;
short partly filled buffers do not establish full-buffer I/O or long-run score parity.
No additional A optimization control or later actor sweep is required by this workflow.

### Breakout depth comparison for slides

Run `python -B -m utils.analysis.rl.atari.atari_breakout_depth_slides` to generate
`layer1_vs_layer3_slides.{png,svg,npz,json}` under
`results/figs/rl/atari/breakout_4action/fs4_stack4_l3_10seed_lrdecay/`.
The two panels preserve the original 1-layer five-seed/1M and 3-layer ten-seed/3M
histories and SD conventions, with a shared model legend and y scale. This is a
presentation comparison, not a matched-budget depth-only ablation. Existing output
files are protected against overwrite; numeric curves and source hashes are retained.
