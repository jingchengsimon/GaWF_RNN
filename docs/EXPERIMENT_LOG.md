# 实验日志（Experiment Log）

本文档用于帮助人类快速回顾项目中的模型演进、实验协议和关键结论。每条记录保持精简，
统一使用 **改动（Change）**、**原因（Reason）**、**证据（Evidence）** 和
**现状（Current）** 四个字段。

后续记录必须以中文叙述，并保留必要的英文技术术语、模型名、CLI 参数、metric 名称、
tensor shape 和数学公式，例如 `GaWF`、`--num_layers`、`val_acc_at_best`、`(B,H,I)`、
`base_lr * gawf_feedback_lr_scale`。不要翻译代码 identifier，也不要在本文档中记录详细命令、
机器路径、scheduler job ID 或普通工程重构；这些内容应放在实验 README 或结果文件中。

## 2026-06-14 — GaWF feedback 泛化

- **改动（Change）：** 初始 Clutter GaWF 将 19 个 task logits 直接反馈给 gate。随后加入
  linear projector，将反馈投影到与 task output 解耦的维度（如 `dz=8`）；早期 multi-layer
  实现使用独立的 `gawf_multi` model，并为每层配置一组 U/V。
- **原因（Reason）：** 直接使用 `dz=19` 会绑定 10 个 character classes 和 9 个 sectors；
  task head 改变后，这种 feedback representation 无法原样复用。
- **证据（Evidence）：** Projected runs 保留原有 `U * fb * V` gating，只改变 feedback
  representation；没有采用 CFL paper 中的 LoRA design。
- **现状（Current）：** `--dz > 0` 控制 projected feedback。独立 `gawf_multi` public
  interface 已于 2026-07-12 被 `gawf --num_layers N` 取代。

## 2026-06-18 — 降低 GaWF gated matmul 的显存占用

- **改动（Change）：** 使用代数等价的 `torch.einsum` contraction，替代显式构造的
  per-sample gated weight tensors `(B,H,I)` 和 `(B,H,H)`。
- **原因（Reason）：** Single-layer GaWF 在 `hidden_size=512` 的 full grid 中发生 OOM；
  原因是 peak activation memory，而不是 parameter count。
- **证据（Evidence）：** 固定 seed 的 5-step A/B test 最大差异为 `4.77e-7`，重复执行优化
  路径的差异为 `0`。在 `hidden_size=512`、`batch_size=256`、AMP 条件下，peak allocated
  memory 从 2603.7 MB 降至 1709.5 MB。
- **现状（Current）：** einsum path 是标准 GaWF 实现，并使 h=512 sweep 可以完成。

## 2026-06-18 / 2026-07-12 — 统一 GaWF optimizer groups

- **改动（Change）：** 删除旧的 multi-layer-only LR knobs，统一不同 depth 的 optimizer
  parameter grouping。
- **原因（Reason）：** 两个叠乘的 LR scales 会使实际 base LR 和 feedback LR 难以解释。
- **证据（Evidence）：** Base parameters 使用指定或搜索得到的 learning rate 和 weight
  decay；U、V 和 feedback projectors 进入 no-weight-decay group。
- **现状（Current）：** Feedback parameters 使用
  `base_lr * --gawf_feedback_lr_scale`，default 为 `1.0`。历史规则
  `gawf_multi_feedback_lr_scale=0.1` 已废弃。

## 2026-06-25 — GaWF data scale / hidden size full grid

- **改动（Change）：** 在 4h、10h、20h、40h training scales 上分别搜索 hidden size、
  learning rate 和 weight decay，并统一使用 40h validation set。
- **原因（Reason）：** 将 training data scale 的影响与 validation distribution 的变化分离，
  同时避免假设所有 data scales 都对应同一个最优 model capacity。
- **证据（Evidence）：** 256 个 GaWF units 均具备 metrics、pickle 和 checkpoint companion。
  按 validation character accuracy 选择的最佳结果如下：

| Train scale | Hidden | LR | WD | Val char | Val sector |
|---|---:|---:|---:|---:|---:|
| 4h | 512 | 0.001 | 0.001 | 72.34 | 86.59 |
| 10h | 256 | 0.005 | 0.0001 | 80.51 | 89.77 |
| 20h | 512 | 0.001 | 0.0001 | 86.26 | 92.01 |
| 40h | 256 | 0.005 | 0.001 | 90.09 | 93.64 |

- **现状（Current）：** Performance 随 training scale 增长，但最佳 hidden size 依赖 data
  scale；40h reference model 使用 `hidden_size=256`。

## 2026-06-27 / 2026-07-06 — 六模型 parameter-matched comparison

- **改动（Change）：** 将 RNN/LSTM/GRU/Mamba/S5 的 middle-path parameter count 匹配到
  GaWF h=256，并在独立 40h test set 上评估六个选定 checkpoints。
- **原因（Reason）：** 早期 comparison 允许 recurrent baselines 使用 h=512，导致 model
  family 与 capacity 混杂。
- **证据（Evidence）：** 所有模型与约 587K parameters 的目标相差不超过约 0.3%：

| Model | Params | Val char | Test char | Char gap |
|---|---:|---:|---:|---:|
| GaWF | 586,067 | 90.09 | 85.62 | 3.24 |
| Mamba | 587,275 | 86.57 | 82.67 | 5.06 |
| GRU | 586,905 | 84.83 | 80.17 | 3.93 |
| RNN | 586,865 | 84.15 | 79.67 | 7.15 |
| LSTM | 584,675 | 83.61 | 79.81 | 6.34 |
| S5 | 587,475 | 80.00 | 75.39 | 6.17 |

- **现状（Current）：** GaWF 的 validation/test character accuracy 最高，且 char gap 最小；
  Mamba 排名第二。S5 使用 `state_size=128`，取代参数偏多的历史 `state_size=189`。
  详细 test metrics 位于
  `results/train_data/clutter_best_6model_param_matched_40h/test_acc_40h_eval.json`。

## 2026-07-03 — Fair evaluation 与 switch-window metrics

- **改动（Change）：** 官方 train/validation curves 改为使用相同 protocol 的完整
  evaluation pass。Sector data 可额外记录 strict global accuracy，以及 foreground switch
  附近的 `pre5` / `post5` windows。
- **原因（Reason）：** Online batch mean 依赖 sampling/order，且会掩盖 target identity
  切换前后的即时性能。
- **证据（Evidence）：** 当 label TSV 包含 `fg_switch` 时，新 metrics 与 legacy curves
  同时保存；`predict_all_chars` protocol 不变。
- **现状（Current）：** Model selection 和 early stopping 使用 fair validation character
  accuracy；global 与 switch-window metrics 是 transition analysis 的首选指标。

## 2026-07-10 — Atari DRQN 扩展

- **改动（Change）：** Atari DQN variants 共享 Nature-DQN convolutional stack 和 Q head，
  readout slot 分别使用 ANN/RNN/GRU/LSTM/GaWF/S5/Mamba。GaWF 使用 detached previous-step
  Q values 调制 recurrence。
- **原因（Reason）：** 在保留 classic DQN visual observation 与 action-value objective 的
  同时，尽量隔离 recurrent model family 的差异。
- **证据（Evidence）：** Recurrent input 只包含 convolutional features；Q feedback 仅用于
  GaWF。Episode reset 会同时清空 recurrent state 和 previous Q values。
- **现状（Current）：** Public feedback modes 为 `none` 和 GaWF `qvalues`。这与 A2C 不同：
  A2C 的 LSTM/GaWF 接收 previous action/reward，GaWF 可额外使用 policy/value output feedback。

## 2026-07-12 — 统一 recurrent depth

- **改动（Change）：** RNN/GRU/LSTM/GaWF 和 ANN 统一使用 `--num_layers`。GaWF 以一个
  public model type `gawf` 同时覆盖 single-layer 和 multi-layer。
- **原因（Reason）：** Depth-specific classes 和 flags 会重复 training/analysis logic，
  也使 cross-task comparison 不一致。
- **证据（Evidence）：** Direct multi-layer GaWF 中，non-final layer 接收相邻 upper layer
  的 detached previous hidden state，final layer 接收 detached previous task output；
  projected mode 为每层使用独立的 U/V/projector。
- **现状（Current）：** 新 checkpoints 使用 `_L<N>` 编码 depth，并可附加 `_dz<N>`；
  历史 `gawf_multi_` checkpoints 仍可读取。

## 2026-07-13 — 修正 Pong frame protocol

- **改动（Change）：** 已完成的 five-seed sweep 被确认采用
  `frame_skip=4, frame_stack=1`，并重命名为 `pong_fs4_stack1`；严格替代实验使用
  `pong_fs1_stack1`。
- **原因（Reason）：** 旧标签 `pong1f` 只表达每次 decision 提供一个 observed frame，
  没有说明每个 action 实际推进了 4 个 ALE frames。
- **证据（Evidence）：** Historical metrics 已记录恢复出的 protocol；result suffix 同时
  编码 frame skip 和 frame stack。
- **现状（Current）：** `fs4/stack1` 结果仅作为历史记录。在 strict `fs1/stack1` sweeps
  达到 expected valid-result counts 之前，不据此形成替代实验结论。

## 2026-07-13 — Phase0 multi-task task-balanced replay

- **改动（Change）：** Atari DQN 新增 `--replay_sampling`，默认
  `task_balanced`，并保留 `global_uniform` 作为对照。Replay 记录外部 `task_id` 作为
  sampling/loss metadata；transition batch 对各任务等配额，recurrent batch 使用
  task-pure windows，TD loss 再对各任务的 mean loss 等权平均。`task_id` 不进入模型。
- **原因（Reason）：** Episode round-robin 只能平衡 episode 数，不能平衡 transitions。
  Pong 与 Breakout 的 episode 长度不同，因此 global uniform 会把较长 episode 对应的任务
  赋予更多 gradient updates。
- **证据（Evidence）：** 已完成的 2M-step ANN Phase0 两个任务各 364 episodes，但有效
  environment steps 分别为 Pong 1,686,676、Breakout 312,596；旧 replay 因而约 84% 来自
  Pong。Smoke tests 覆盖等配额 transition sampling、保留的 global-uniform path、task-pure
  sequence windows，以及 autoreset 后首个有效 observation 的二次 hidden-state reset。
- **现状（Current）：** 已完成的 `frame_stack=1, frame_skip=1` Phase0 结果仍标记为历史
  `global_uniform`，不能与新默认混为同一 protocol。后续 metrics 会显式保存
  `replay_sampling`；模型输入仍只有 image observation，不含 task cue/embedding。

## 2026-07-13 — Phase0 transition-balanced collection

- **改动（Change）：** Multi-task Atari collection 默认改为 `transition_balanced`：仍然
  只在 episode boundary 选择任务，但优先选择累计 environment steps 最少的任务，并以
  cyclic order 处理相同步数。保留历史 `round_robin` 作为对照；不再要求 per-task episode
  数一致。
- **原因（Reason）：** `task_balanced` replay 只能平衡训练时的 sampling/update 数，不能
  为短 episode 任务补充新的 transitions。若按 episode round-robin，短任务会产生更少且
  被更频繁重复抽取的数据，导致任务间 experience coverage 不公平。
- **证据（Evidence）：** 已完成的 `stack4/skip4` GRU Phase0 虽然 Pong 与 Breakout 均为
  708 episodes，但有效 environment steps 分别为 1,759,600 与 238,984；Breakout 仅获得
  约 12% 的 collected transitions。Task-balanced replay 不会改变这一采集差异。
- **现状（Current）：** Collection balance 和 replay balance 作为两个独立协议字段
  记录。新实验使用 `transition_balanced` collection + `task_balanced` replay；外部
  `task_id` 仅用于 environment scheduling、metrics、replay sampling 和 loss aggregation，
  不进入模型输入。

## 2026-07-14 — Clutter six-model 10-seed confirmation

- **改动（Change）：** 固定 40h validation 选出的 GaWF/RNN/LSTM/GRU/Mamba/S5 最佳
  hyperparameters，以 seeds 1--10 训练 60 个独立 runs；每个 run 固定 150 epochs，
  `patience=0`，使用 40h train/validation 与 sector objective。
- **原因（Reason）：** 单 seed test result 不能提供 model-family generalization 的
  uncertainty；需要 independent-seed distribution 支持 test accuracy/loss error bars。
- **协议（Protocol）：** 使用 acceleration path；119 GB training array 通过 mmap 读取，
  因此 DataLoader 固定 `num_workers=0`、`pin_memory=False`。训练完整运行 150 epochs，最终
  checkpoint 保存完整 trajectory 中 validation accuracy 最佳的 state。
- **现状（Current）：** 初始单一 60-task array `58171730` 在全部 tasks 尚未启动时取消。
  正式实验改为 10 个独立 Slurm jobs `58172415`--`58172424`，每个 seed 对应一个包含六个
  models 的 array。新旧提交均保留在 `experiments/monitoring/JOBS.md`；有效完成要求
  metrics、checkpoint、pickle 同时存在，且 `actual_epochs=150`、seed/protocol fields
  全部匹配。
