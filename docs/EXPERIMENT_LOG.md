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
- **现状（Current）：** Projected-feedback 实现仅为历史 checkpoint 兼容保留；当前主设计
  不使用 project layer。`gawf_multi` 已由统一接口 `gawf --num_layers N` 替代。

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

## 2026-07-16 — 暂停 GaWF projected feedback (`dz`)

- **改动（Change）：** 当前 GaWF 主设计不再使用 `dz/project layer`，实验统一采用 direct
  feedback；projected-feedback 结果移入 archive。
- **原因（Reason）：** `dz` 增加 feedback representation 与 projector capacity，但没有形成
  稳定的 held-out improvement，反而增加过拟合风险。
- **证据（Evidence）：** `dz=8/16/32/64` 与 control 系列的 learning curves、train/validation
  gap 显示更强拟合没有转化为可靠 validation/generalization 收益；该批实验结论为加入
  `dz` 会过拟合。
- **现状（Current）：** 暂不开展 project-layer sweep；实现仅为历史 checkpoint 兼容保留，
  主实验和结论均以无 projector 的 direct-feedback GaWF 为准。

## 2026-07-17 — Clutter 40h 数据默认改为 uint8 存储

- **改动（Change）：** 标准 40h train/validation/test 从现有 float32 文件逐块精确转换为
  `40h-uint8`；新生成数据和未来 Clutter 实验默认读取 uint8 存储。
- **原因（Reason）：** uint8 将共享文件系统读取量降低 75%；模型输入仍在 Dataset 边界转换
  为 float32，因此不改变网络数值类型或训练目标。
- **证据（Evidence）：** 转换仅接受 `[0,255]` 内的有限整数，并逐块验证
  `uint8_target.astype(float32) == float32_source`；TSV 进行 SHA-256 一致性校验。
- **现状（Current）：** 历史 float32 文件、旧实验 worktree 和专用
  `40h-float32-jointswitch-balanced` 测试集保留；DataLoader 批量转换优化另行基准测试。

## 2026-07-17 — Clutter uint8 DataLoader 跨端优化

- **改动（Change）：** 标准 40h pipeline 固定为 `uint8 mmap + device cast + compact window +
  batch-sized block shuffle + 2 workers + pinned memory`；`sample/stacked/global` 保留为显式
  historical reproduction path，node-local staging 仅作可选优化。
- **原因（Reason）：** float32 共享盘读取量大；逐 sample 转换破坏 pinned uint8 transfer；
  全局随机 mmap 访问产生大量 page faults，host 侧 stacked window 还会重复展开相邻帧。
- **证据（Evidence）：** sjc cold-cache loader 中 float32/global 为 26.72 samples/s，
  uint8/global 为 45.06 samples/s；block-local variants 达 374–552 samples/s。GPU 负载下
  device+compact transfer 为 107.80 batches/s，batch-CPU cast 为 11.85 batches/s；RNNConv
  AMP end-to-end 为 34.40 vs legacy 34.66 batches/s，说明 compute-bound 路径无回退。
  30GB node-local staging 可达 1161 samples/s，但首次复制需 121 s，且两端空间条件不同。
  Amarel 真实 shared-scratch CPU smoke 为 88.27→765.30 samples/s（8.67×）；Volta AMP
  e2e 为 57.09→61.27 batches/s、peak CUDA memory 相同，并完成 production entry 一轮训练。
- **现状（Current）：** 两端共用相同 CLI/defaults；默认不 staging。所有新 metrics 记录
  `input_cast_mode`、`frame_layout` 和 `shuffle_block_size`，以区分 sampling protocol。

## 2026-07-18 — GaWF symmetric relevance 与 switch timing

- **改动（Change）：** 以 validation activation 定义 encoder/hidden unit 的 sector、digit 与
  interaction selectivity，再在 held-out test 上检验 input/recurrent gate relevance；Part 2
  同时报告排除和包含 interaction-dominant units 的版本，Part 3 使用严格
  `negative -> nonnegative` switch crossing。
- **原因（Reason）：** 固定空间 proxy 不能对 sector/digit 做对称检验，且把正值起点算作
  crossing 会虚构 gate-leads-readout 的因果时序。
- **证据（Evidence）：** Validation/test 的 sector×digit joint design 均不独立
  (`Cramér's V=0.0816/0.0654`)；interaction-dominant 比例为 encoder `61.72%`、hidden
  `2.34%`。Top-20 interaction contrast 在排除/包含版本均为正 (`0.888/0.907`)，但
  recurrent-sector 与 recurrent-digit relevance 均为负，且严格 event-paired crossing
  没有支持 gate 领先 readout；sequence split-half 结论一致。
- **现状（Current）：** Input gate 的 sector relevance 得到支持；“recurrent gate 更偏
  digit”及 gate 先于 readout 的完整 dissociation/causal claim 不成立。所有 marginal
  解释必须保留 joint-design confounding 限定。

## 2026-07-19 — GaWF gate axis、activation confound 与 robustness audit

- **改动（Change）：** 对 corrected group-mean `Delta g` 做 provenance reconciliation；对
  gate matrix 同时报告 SOURCE（按 destination 平均）和 DESTINATION（按 source 平均）
  relevance；加入 activation linear/stratified controls、digit/sector leave-one-out，以及
  128–8192 synapse CI convergence。
- **原因（Reason）：** Variance fraction 显示 recurrent digit dominance，而 `|Delta g|>0.5`
  tail 显示 sector dominance；旧 relevance 仅保留 source axis，负 `d` 也可能被 hidden
  activation magnitude 混杂。
- **证据（Evidence）：** 两个 headline 实现均满足 per-synapse、group-first、trial-weighted
  定义，独立重算与旧值一致。Recurrent digit 在 moderate thresholds 更强，但 sector 在
  `t≈0.443` 后具有更重 extreme tail。Top-20 interaction-excluded recurrent relevance 为
  sector SOURCE/DESTINATION `-0.623/+0.267`（sign flip），digit `-0.440/-0.797`；SOURCE
  negative `d` 经 linear 与 activation-quintile controls 后保留。任何单一 digit/sector
  removal 均未使 headline fraction 移动超过 3 percentage points；四个 headline CI width
  在 2048 synapses 后 plateau。
- **现状（Current）：** “Selective hidden units 的 outgoing recurrent connections 被抑制”
  仅适用于 SOURCE view；sector-selective units 在 DESTINATION view 反而 receive more gated
  recurrent input。Variance/tail 差异是 distribution crossing，不是计算不一致；最终 CI
  使用 8192-synapse、1000-draw trial bootstrap 并以 exact full-gate point recenter。
