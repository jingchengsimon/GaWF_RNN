# 实验日志（Experiment Log）

本文档供人类快速回顾模型演进、实验协议和关键结论。每条记录只保留
**改动（Change）**、**原因（Reason）**、**证据（Evidence）**、**现状（Current）**。
后续必须使用中文叙述，保留必要的 English 术语、identifier、metric、tensor shape 和公式；
不记录命令、机器路径、scheduler job ID、任务状态或普通工程重构。

## 2026-06-14 — GaWF feedback 泛化

- **改动（Change）：** Clutter GaWF 从 19 个 task logits 直接反馈（`dz=19`），演进为可选
  projector（如 `dz=8`）；早期 multi-layer 版本曾使用独立 `gawf_multi`。
- **原因（Reason）：** 将 feedback representation 与 10-class + 9-sector task head 解耦。
- **证据（Evidence）：** Projected mode 保留 `U * fb * V` gating，仅改变 feedback 表示；
  未采用 CFL paper 的 LoRA design。
- **现状（Current）：** `--dz > 0` 启用 projected feedback；`gawf_multi` 已由统一接口
  `gawf --num_layers N` 替代。

## 2026-06-18 — GaWF gated matmul 显存优化

- **改动（Change）：** 用等价 `torch.einsum` contraction 替代显式构造 `(B,H,I)`、
  `(B,H,H)` per-sample gated weights。
- **原因（Reason）：** `hidden_size=512` 的 full grid 因 peak activation memory OOM。
- **证据（Evidence）：** 固定 seed 的 5-step A/B 最大误差 `4.77e-7`；在 h=512、
  batch=256、AMP 下 peak memory 从 2603.7 MB 降至 1709.5 MB。
- **现状（Current）：** einsum path 为标准实现，并支持 h=512 sweep。

## 2026-06-18 / 2026-07-12 — 统一 GaWF optimizer groups

- **改动（Change）：** 统一不同 recurrent depth 的 optimizer parameter grouping。
- **原因（Reason）：** 旧 multi-layer-only LR knobs 会使实际 learning rate 难以解释。
- **证据（Evidence）：** Base parameters 使用指定 LR/WD；U、V、projector 不使用
  weight decay。
- **现状（Current）：** Feedback parameters 使用
  `base_lr * gawf_feedback_lr_scale`，默认 scale 为 `1.0`。

## 2026-06-25 — GaWF data-scale full grid

- **改动（Change）：** 在 4h/10h/20h/40h train scales 上分别搜索 hidden size、LR、WD，
  并固定使用 40h validation。
- **原因（Reason）：** 分离 training scale 与 validation distribution，并允许 capacity 随
  data scale 改变。
- **证据（Evidence）：** 256 个有效 runs 的最佳 `Val char` 为：

| Train | Hidden | LR | WD | Val char | Val sector |
|---|---:|---:|---:|---:|---:|
| 4h | 512 | 0.001 | 0.001 | 72.34 | 86.59 |
| 10h | 256 | 0.005 | 0.0001 | 80.51 | 89.77 |
| 20h | 512 | 0.001 | 0.0001 | 86.26 | 92.01 |
| 40h | 256 | 0.005 | 0.001 | 90.09 | 93.64 |

- **现状（Current）：** Performance 随 data scale 增长；40h reference 使用 h=256。

## 2026-06-27 / 2026-07-06 — 六模型 parameter-matched comparison

- **改动（Change）：** 将 RNN/LSTM/GRU/Mamba/S5 的 middle-path parameters 匹配到
  GaWF h=256，并在独立 40h test set 上评估。
- **原因（Reason）：** 避免 model family 与 capacity 混杂。
- **证据（Evidence）：** 参数量均接近 587K；`Val/Test char`：GaWF 90.09/85.62，
  Mamba 86.57/82.67，GRU 84.83/80.17，RNN 84.15/79.67，LSTM 83.61/79.81，
  S5 80.00/75.39。
- **现状（Current）：** GaWF 排名第一、Mamba 第二；S5 使用 `state_size=128`，替代旧 189。

## 2026-07-03 — Fair evaluation 与 switch-window metrics

- **改动（Change）：** Train/validation 统一为完整 evaluation pass，并加入 strict global
  accuracy 与 foreground-switch `pre5/post5`。
- **原因（Reason）：** Online batch mean 受 sampling/order 影响，且掩盖 transition transient。
- **证据（Evidence）：** 含 `fg_switch` 的 sector labels 同时保存 global、window 和 legacy
  curves；`predict_all_chars` 不变。
- **现状（Current）：** Model selection/early stopping 使用 fair validation char accuracy；
  transition analysis 优先使用 global 与 switch-window metrics。

## 2026-07-10 — Atari DRQN 方法扩展

- **改动（Change）：** DQN variants 共享 Nature encoder/Q head，readout 为
  ANN/RNN/GRU/LSTM/GaWF/S5/Mamba；GaWF 使用 detached previous-Q feedback。
- **原因（Reason）：** 固定 visual encoder 与 action-value objective，隔离 recurrent family。
- **证据（Evidence）：** `task_id` 不进入模型；episode reset 同时清空 recurrent state 和 Q。
- **现状（Current）：** DQN feedback 为 `none/qvalues`；A2C 为 LSTM/GaWF +
  `none/output`，二者不可混写。

## 2026-07-12 — 统一 recurrent depth 与 GaWF

- **改动（Change）：** RNN/GRU/LSTM/GaWF/ANN 统一使用 `--num_layers`；GaWF 仅保留
  public model type `gawf`。
- **原因（Reason）：** 消除 depth-specific classes/flags 和重复分析逻辑。
- **证据（Evidence）：** Direct mode 中 non-final layer 接收 upper-layer previous hidden，
  final layer 接收 previous output；projected mode 每层独立 U/V/projector。
- **现状（Current）：** 新 checkpoint 使用 `_L<N>`（可附 `_dz<N>`）；旧
  `gawf_multi_` checkpoint 保持兼容。

## 2026-07-13 — Pong frame protocol 纠正

- **改动（Change）：** 历史 sweep 明确命名为 `pong_fs4_stack1`；严格替代实验使用
  `pong_fs1_stack1`。
- **原因（Reason）：** 旧名 `pong1f` 未说明每次 action 实际推进 4 个 ALE frames。
- **证据（Evidence）：** Historical metrics 已补充恢复后的 `frame_skip/frame_stack`。
- **现状（Current）：** `fs4/stack1` 仅作历史记录；严格 sweep 完成前不写替代性能结论。

## 2026-07-13 — Phase0 task-balanced replay

- **改动（Change）：** Multi-task Atari DQN 默认 `task_balanced` replay；transition batch
  按 task 等配额，sequence batch 使用 task-pure windows，per-task TD loss 等权聚合。
- **原因（Reason）：** Episode round-robin 不等于 transition balance，长 episode task 会
  获得更多 gradient updates。
- **证据（Evidence）：** 一个历史 2M-step run 的两任务 episode 数相同，但有效 steps 为
  1,686,676 vs 312,596，旧 replay 约 84% 来自 Pong。
- **现状（Current）：** 历史结果标记 `global_uniform`；`task_id` 仅作 sampling/loss metadata，
  不进入模型输入。

## 2026-07-13 — Phase0 transition-balanced collection

- **改动（Change）：** Collection 默认 `transition_balanced`：仅在 episode boundary 切换，
  优先选择累计 environment steps 最少的 task。
- **原因（Reason）：** Balanced replay 无法弥补短 episode task 的 experience coverage 不足。
- **证据（Evidence）：** 一个历史 stack4/skip4 run 的两任务 episode 数同为 708，但有效
  steps 为 1,759,600 vs 238,984，短任务仅约 12%。
- **现状（Current）：** 新实验组合为 `transition_balanced` collection + `task_balanced`
  replay；collection/replay 分别记录。

## 2026-07-14 — Clutter 六模型 10-seed confirmation

- **改动（Change）：** 固定六模型最佳 hyperparameters，以 seeds 1–10 训练 60 个独立 runs；
  每个 run 为 150 epochs、`patience=0`、40h train/validation。
- **原因（Reason）：** 用 independent-seed distribution 取代单 seed 排名，支持 uncertainty。
- **证据（Evidence）：** 有效输出要求 metrics/checkpoint/pickle 齐全，且 seed、protocol、
  `actual_epochs=150` 一致。
- **现状（Current）：** 最佳 checkpoint 从完整 trajectory 中按 validation accuracy 选择；
  最终多 seed 统计写入结果摘要，不在此记录 scheduler provenance。

## 2026-07-14 — Foreground-switch transient trajectories

- **改动（Change）：** 从连续 test `pop_act.npy` 直接提取 `fg_switch`-aligned trials，window
  为 half-open `[-8,20)`；比较全部 768 trials 与无 `bg_switch` 的 439-trial subset。
- **原因（Reason）：** 避免先按 digit/sector averaging 丢失切换瞬态，并控制背景切换混杂。
- **证据（Evidence）：** 两组 mean trajectories 共用一个 3D PCA basis，可直接比较坐标。
- **现状（Current）：** 六个 selected models 均保存 trial tensors、shared-PCA coordinates、
  metadata 和交互图。

## 2026-07-14 — S5 non-fused Adam 兼容路径

- **改动（Change）：** Atari S5 固定使用 `torch.optim.Adam(fused=False)`；其他兼容模型仍可
  使用 fused Adam。
- **原因（Reason）：** CUDA fused Adam 不接受 S5 的 complex-valued parameters。
- **证据（Evidence）：** 旧部署中 real-parameter models 可训练，而 S5 在训练前因 optimizer
  compatibility 失败；BF16/TF32、replay 和 recurrent scan acceleration 与此无关。
- **现状（Current）：** 这是 S5-only compatibility exception；GaWF U/V group 与 feedback
  LR scale 规则保持不变。
