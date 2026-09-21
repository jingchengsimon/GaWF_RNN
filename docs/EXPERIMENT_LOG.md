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

## 2026-07-19 — Continuous alignment 改用双侧 permutation test

- **改动（Change）：** Part 2 continuous alignment 的 `diag-offdiag` significance 从只检验
  正向 alignment 的 upper-tail permutation test 改为基于 `|null| >= |observed|` 的双侧
  permutation test，并保留 `(1 + exceedances) / (B + 1)` finite-sample correction。
- **原因（Reason）：** recurrent gate 的 `diag-offdiag` 为负；单侧正向检验只能给出
  `p=1`，不能判断 negative alignment/anti-alignment 是否显著。
- **证据（Evidence）：** 在 `B=1000` 下，input-sector、input-digit、recurrent-sector 与
  recurrent-digit 的双侧 permutation p-value 均为 `1/1001=0.000999`。
- **现状（Current）：** 四个 alignment contrast 均显著偏离 permutation null；input gate
  为显著正向 diagonal alignment，recurrent gate 为显著负向 diagonal alignment。

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

## 2026-07-19 — GaWF/LSTM/GRU unit-level gate context decomposition

- **改动（Change）：** 将 GaWF Figure 03 的 balanced sector×digit variance decomposition
  迁移到 seed42 parameter-matched LSTM/GRU；LSTM 分析 input/forget/output gates，GRU
  分析 reset/update gates，candidate activations 不计入 gate。另将 GaWF raw sigmoid
  input/recurrent connection gates 沿 incoming-source axis 取 arithmetic mean，形成每个
  destination hidden unit 一个标量的 derived projection，并用完全相同的 pooled-SS-first
  decomposition 分析。
- **原因（Reason）：** LSTM/GRU gate 是每个 hidden unit 一个标量，不能沿用 GaWF
  `(destination, source)` connection-level gate 的解释；图题和 legend 因此明确标为
  `unit-level gates`。
- **证据（Evidence）：** 在同一 continuous test 的 57,568 frames 上使用 90 个
  sector×digit cells、每格 `n=521`。手工 gate recurrence 与 PyTorch native outputs 的
  max absolute difference 为 LSTM `2.74e-6`、GRU `2.75e-5`。Condition-mean sector fraction
  为 LSTM input/forget/output `40.45/62.42/67.96%`，GRU reset/update
  `63.27/56.84%`；对应 digit fraction 为 `45.87/24.74/19.00%` 与 `15.45/27.16%`。GaWF
  destination-unit input-mean gate 的 sector/digit/interaction fraction 为
  `92.66/4.21/3.13%`，recurrent-mean gate 为 `19.05/75.64/5.31%`。
- **现状（Current）：** LSTM input gate 的 condition-mean digit component 略高于 sector；
  其余四个 unit-level gates 均以 sector component 为主。Trial-total decomposition 仍由
  residual 主导：LSTM/GRU 为 `69.22–84.19%`，GaWF destination-unit input/recurrent
  projections 为 `81.04/60.57%`。因此 condition-mean fractions 不应解释为单 trial
  variance 的占比；derived GaWF projection 也不替代 canonical synapse-level 结果。

## 2026-07-19 — Sector input-gate mean 改用 sequential equal-n protocol

- **改动（Change）：** 新增基于真实 pre-step feedback trajectory 的 input-gate sector mean，
  对 9 个 sector 各抽取相同数量的 frames，并分别报告保留/排除 `0.5 point mass` 的
  `3×3` spatial maps；历史 `fig2_sector_gate_mean` 保留为 one-step/reset 对照，max-gate
  view 不再生成。
- **原因（Reason）：** one-step/reset gate 由当前 frame 的 zero-state output 构造，并非模型
  正常 recurrent inference 在该 timestep 实际应用的 gate；原始 sector frame counts 也不等量。
- **证据（Evidence）：** sequential trajectory 在每个 32-frame sequence 内传递 hidden state
  与 feedback，只有初始化 frame 的 applied gate 固定为 `0.5`；equal-n protocol 消除不同
  sector frame 数对 raw mean 的直接加权差异。
- **现状（Current）：** sequential equal-n included/excluded maps 作为更新后的主要 sigmoid
  mean view；legacy one-step/reset mean 仅用于 protocol comparison。

## 2026-07-22 — Atari DQN 可恢复训练与 mmap replay

- **改动（Change）：** `train_atari_dqn.py` 新增周期性原子 checkpoint、`--resume_from` /
  `--auto_resume` 续跑和抢占信号处理；replay buffer 增加 `mmap` backing，六个数组落在
  `<save_dir>/replay/`，checkpoint 只保存 position、task counts、sampler RNG 与
  model/target/optimizer/scaler 状态。默认 `--checkpoint_interval_steps 0` 且
  `--replay_backing memory`，不传新参数时执行路径与历史实验完全一致。
- **原因（Reason）：** DQN 的 checkpoint 离开 replay buffer 没有意义，而 fs4/stack4 的
  buffer 约 28 GB，无法随 checkpoint 反复写盘；丢弃 buffer 续跑则会让被抢占的 run 在
  epsilon 已退火到 0.01 时用空 buffer 重新填充，replay 分布与其他 seed 不可比。
  历史代价是真实的：strict Pong fs1/stack1 的 70 个 unit 中 50 个因抢占或失败作废，
  只能整段重跑。
- **证据（Evidence）：** amarel GPU 端到端验证（job 58890853）在 SIGUSR1 抢占后落盘并
  自动续跑至 60k steps，`resume_count=1`、history 单调无重复、replay 与 scaffolding
  checkpoint 均被回收。单元测试覆盖 memmap 重开后采样逐元素一致、两次独立 resume 的
  最终权重逐参数相同，以及 protocol 不匹配（`hidden_size`/`total_timesteps`/
  `format_version`）时拒绝续跑。
- **现状（Current）：** ALE 环境与 recurrent state 按设计重置，因此续跑是统计上有效的
  continuation 而非逐位重放；每次中断记入 metrics 的 `resume_count` 与 `resumed_at_steps`。
  仅新实验启用；已完成的 Pong 70+50+35 个 run 不受影响也不重跑。mmap replay 受
  `/scratch` 1 TiB soft quota 约束，fs4/stack4 数组并发上限为 12，并在启动前做配额守卫。

## 2026-07-22 — GaWF feedback-component 消融（zero + shuffle）

- **改动（Change）：** 对冻结的 GaWF（h=256）在 40h test set 上做 inference-time feedback
  消融，只作用于 recurrent feedback vector（不动 frame input）。两个 lesion family 各 4
  条件：zero 系（`clear_digit` 置零 `[0:10]`、`clear_sector` 置零 `[10:19]`、`clear_all`
  置零整向量→gate=sigmoid(0)=0.5）；shuffle 系（先录无消融 baseline 的 next-feedback
  日程 `fb[b,t,:]`，再对每个序列独立抽一个 time permutation 沿时间重排目标切片，
  `shuffle_digit`/`shuffle_sector` 各打乱一个切片、`shuffle_all` 用同一 perm 打乱整向量，
  rollout 时覆盖 live feedback）。跨 **10 个 GaWF seeds**（best6_multiseed_40h_ep150）汇报
  mean ± SD。
- **原因（Reason）：** 单模型 + 纯 zero 消融把"该分量携带 task 信息"与"该分量数值幅度大"
  混为一谈；shuffle 保留每个 channel 的边际分布（幅度）、只破坏与当前帧刺激的时间对齐
  （且序列内 foreground identity 会 switch，随机帧的反馈通常是错误身份），因此是
  magnitude-controlled 的信息移除 control，`shuffle_all` 为 `clear_all` 的幅度受控对照。
- **证据（Evidence）：** digit-readout 掉落（相对 baseline 85.9，Δ pp，10 seeds 均值）：
  `clear_digit` −7.1、`clear_sector` −13.2、`clear_all` −20.9；`shuffle_digit` −0.5、
  `shuffle_sector` −10.7、`shuffle_all` −7.1。sector-readout 同向。核心不对称在 zero 与
  shuffle 下都稳定（SD ≤ 1.3，散点几乎重叠）：**打乱/移除 sector 反馈伤害大，digit 反馈几乎
  无害**（`shuffle_digit` ≈ 0）。`shuffle_all`（−7.1）落在两个单分量 shuffle 之间且比
  `shuffle_sector` 单独更轻，因为它保留了每时刻自洽的 (digit,sector) 联合向量，而单独打乱
  sector 会制造 digit/sector 互相矛盾的反馈，对 digit readout 反而更伤。
- **现状（Current）：** 论点"consume sector 反馈重要、digit 反馈不重要"在 10 seeds 上、
  且在 magnitude-controlled shuffle 对照下均成立。实现见
  `utils/analysis/feedback_ablation.py`（`--shuffle` 追加三个 shuffle 条件），汇总图
  `utils/analysis/` 中对应 Figure workflow（上 zero / 下 shuffle 两行、按 readout 分组、
  跨 seed error bar + 散点，PNG-only）。

## 2026-08-07 — Atari GaWF BF16 acceleration acceptance baseline

- **改动（Change）：** 为 Atari GaWF 建立 acceleration acceptance protocol：固定 BF16、
  `allow_tf32=True`、L3、Breakout、`B=8`、`T=16`，以两条 matched baseline 和两条 candidate
  的完整 RL trajectory 比较；acceptance 不再要求 eager/optimized 的逐位或固定 `rtol/atol`
  相等，而要求 candidate 不扩大 baseline 的跨 run/GPU numerical dispersion，且不出现超出
  baseline envelope 的系统性偏移。固定输入的 Q/state/gradient diagnostic 仅用于定位差异来源。
- **原因（Reason）：** BF16 Inductor/CUDA Graph 等优化可在不改变 Python formula 的情况下改变
  kernel fusion、归约次序或 capture execution；固定输入 tolerance 不能区分这种改变和 RL
  自然跨次波动，也不能证明 end-to-end learning trajectory 保持可比。
- **证据（Evidence）：** `compile_full_gawf_scan` 在 BF16 fixed-input diagnostic 中相对 eager
  出现 Q/state/gradient 差异，并在 40k-step Breakout trajectory test 中超出 baseline envelope，
  因此拒绝。B=1,T=1 online CUDA Graph replay 也在实际 RL trajectory 中产生偏移，拒绝。gate-only
  `torch.compile` 同样依赖 BF16 kernel fusion，不满足“保留 GPU arithmetic path”的前提，未作为
  candidate。FP32 full-scan 与保持 BF16 的实验约束冲突，未采用。
- **现状（Current）：** 所有后续 acceleration/optimization 首先保持 sampling、batch content、
  loss、update cadence、模型结构与 GPU arithmetic path；随后执行 matched repeated end-to-end
  variance acceptance。对 replay batch 的 pinned CPU/CUDA staging buffer（每个 batch field 独立
  buffer，避免同 shape/dtype 字段覆盖）进行了完整测试：四条 trajectory 的
  `loss`、`q_values_mean`、`episodic_return_100` 均为 0 差异，数值上通过；但 `20k→40k`
  的 `B=8,T=16` update 段从 582.21 s 变为 586.76 s（`0.992×`，`−0.78%`），无实际加速，
  因而不用于正式训练。该开关保持 default-off，仅作为未来 data-transfer optimization 的
  对照基线。

## 2026-08-10 — SJC Atari multi-task 每卡多 job contention variance

- **改动（Change）：** 在 sjc-remote 两张 RTX A6000 上，以相同的 120,028-step warm-up
  checkpoint 与 seed 9101，完成三轮 50k-step continuation；比较每卡 `1/2/4/8 job`，其中每个
  job 固定为五任务 GaWF L3、`per_task` 500k mmap replay、task-balanced sampling。
- **证据（Evidence）：** 三轮共 90/90 unit 完成。汇总 `per_gpu1`（n=6）与 `per_gpu8`
  （n=48）：final `loss` 的均值差为 −0.00150（−0.09 baseline SD），`q_values_mean` 为
  +0.03096（+0.38 SD），`episodic_return_100` 为 −125.69（−0.94 SD）；`per_gpu8` 的各项
  离散度与 baseline 接近。
- **结论（Conclusion）：** 每卡 8 job 未显示超出 baseline envelope 的系统性数值偏移，作为
  GPU-contention 下的 execution variance 可接受。该检验固定 warm-up checkpoint/seed，只证明
  并发配置的数值可比性，不替代跨 seed 的训练泛化评估。

## 2026-08-16 — Figure 7 改为分组报告 raw seed-level gap p

- **改动（Change）：** Figure 7 的 Digit/Sector × TT/TR/RT/RR 各组分别报告 10-seed paired
  sign gap 的 uncorrected exact sign-flip p-value；不再在图中显示 positive/negative bar
  相对零的 significance stars。Structured summary 继续保留全部 raw p 与 24-test Holm
  diagnostic，Supplementary 3 不变。
- **原因（Reason）：** 八个 sign-gap 被定义为分别解释的研究问题，而不是一个需要共同控制
  family-wise error 的联合结论；主图只呈现每组直接对应的 gap test。

## 2026-08-16 — Recovery 改用 joint-balanced 10-digit test protocol

- **改动（Change）：** Fig1 六模型与 Supplementary 1 GaWF feedback-ablation recovery
  统一改用 `40h-float32-jointswitch-balanced-10digit-unique` test dataset，并以 seeds 1–10
  在 `pre10`–`pre1`、`post1`–`post10` 的每个 offset 报告 mean 与 95% t-CI。
- **原因（Reason）：** 先前 runner 实际传入默认 `40h-uint8`，没有保证 foreground 与
  background 同时 switch，也没有保证 switch-event 的 90 个 digit×sector cells 平衡。
- **证据（Evidence）：** dataset metadata 记录 2430 次 joint switches、每个 digit×sector
  cell 27 次；每帧 digits 0–9 各一次，每个 clutter onset 覆盖全部九个 sectors。修正后的
  60/60 Fig1 model-seed 与 10/10 Supplementary 1 GaWF-seed outputs 均通过 finite、offset、
  frame-count 和条件完整性检查；最终两张 PDF 均保留 95% t-CI，并移除标题中的
  `10-seed mean` 字样。

## 2026-08-17 — Figure 3 的 0.5 点质量来自 sequence reset

- **改动（Change）：** 对十个 GaWF seeds 的 retained Figure 3 trajectories 直接重建 input /
  recurrent gates，并以 $|g-0.5|<10^{-6}$ 审计 0.5 点质量、reset 外点质量、剔除点质量后的
  `[0.1, 0.9]` 比例，以及 `U[j]` 的零行。
- **证据（Evidence）：** input / recurrent 的 0.5 点质量分别为 3.125026% / 3.125063%，而每条
  32-frame sequence 的首帧 reset 恰占 3.125%；reset 外仅余 0.0000257% / 0.0000633%。十个
  seed 均无 $\max_r|U[j,r]|<10^{-6}$ 的 destination unit。剔除点质量后，中间区间仍占
  14.8352% / 33.3198%。
- **结论（Conclusion）：** 0.5 spike 是 zero-feedback reset 的全行瞬时现象，不是
  `U[j]≈0` 的死反馈通路。input gate 的 binary tendency 仍较强，但 recurrent gate 保留约
  三分之一的 graded mass；后续文字不能把两类 gate 统一概括为普遍 binary。

## 2026-08-19 — Figure 4 raw-synapse gate ANOVA 排除 reset frame

- **改动（Change）：** 用 SJC 保留的十个 feedback trajectories 和对应 GaWF checkpoints，
  在每个 32-frame window 排除 zero-feedback `t=0` 后重算 raw input/recurrent synapse 的
  20-draw balanced Sector × Digit ANOVA；Fig4 core PDFs 随之更新为 10-seed mean ± SEM 并显示
  每个 seed 点。
- **证据（Evidence）：** 十个 seeds 各排除 1,799 个 reset frames，保留 55,769 frames；Input
  gate 的 Sector / Digit / Interaction 为 76.7069% / 15.9906% / 7.3025%，recurrent gate 为
  23.2512% / 71.7333% / 5.0155%（均为 10-seed mean）。
- **结论（Conclusion）：** §4.4–4.5 现与其他正式 gate 统计共用 reset-excluded 口径；不得再以
  旧 `unified_multiseed` 的含-reset synapse decomposition 引用这些数字。

## 2026-08-26 — Skiing stall boundary 与 single canonical mapping protocol

- **改动（Change）：** 新增独立 `skiing-stall-actionfix-v1`，不改变 historical baseline。
  所有 task 仍使用 18-output Q head；Pong、Breakout、Assault、Seaquest 的 full ALE action set
  保持 identity，Skiing 仅做一次 18-to-9 non-FIRE legal-action mapping。Skiing 以 ALE RAM
  86:94 course-object y slots 的变化作为 course progress；连续 450 agent steps 无进展时返回
  `truncated=True` 与 `end_reason=stalled`，并将 raw episode return 限制到不高于 -30,000。
  stall boundary 重置 episode/recurrent state，但 TD target 保留 final-observation bootstrap。
- **证据（Evidence）：** unit tests 覆盖正常下滑、阈值恰好触发、natural terminal 优先级、
  reward/flags/info、baseline 默认路径、18-to-9 mapping、四任务 identity、weights-only loading
  与 truncation bootstrap。真实 ALE probe 在持续撞边时于 475 steps 结束，其中恰有连续 450
  steps 无 progress，return=-30,000。SJC 25k GRU seed1 weights-only smoke 达到 finite
  loss=0.0602963；训练 20 episodes 均自然结束，return100=-16,311.4、length100=1,204.2；固定
  greedy video 的 3 episodes 均由 stall 截断，returns 全为 -30,000，lengths 为
  964/964/889。另一个从已完成 1M GRU weights 启动的 25k extension smoke 固定
  `epsilon=0.01`、`LR=1e-5` 且不再 decay，得到 finite loss=0.0375964；训练与 greedy video
  的 episodes 均以 `stalled` 截断、returns 为 -30,000，证明 fresh extension phase 的
  artifacts 与 boundary metadata 完整。
- **现状（Current）：** 该协议改变 MDP，不把派生训练当作 resume。LSTM/GRU seed1 从完成的
  20M final model `state_dict` 初始化；GaWF seed1 从稳定只读复制的 19.45M resumable
  checkpoint 中仅提取 model `state_dict`。三者均 fresh 初始化 optimizer、replay、global step、
  RNG trajectory、epsilon/LR schedules，并已在 SJC 提交 1M single-Skiing diagnostic run；所有
  新 leaf 均位于唯一 parent `formal_20m_4mpertask_raw_seeds`，GaWF leaf 记录 source step
  `19450000`。后续累计 2M exposure 采用两种明确区分的语义：仍有 resumable checkpoint 的
  GaWF 连续保留 optimizer/replay/global step/RNG/epsilon/LR；已 finalization 的 GRU/LSTM
  分别从其 1M final `state_dict` 启动独立 fresh 1M phase，固定末端 `epsilon=0.01` 与
  `LR=1e-5`，不把 phase-local step 冒充 strict resume step。累计 2M 完成后的三个 source
  均已按 finalization 回收 resumable checkpoint/replay；因此累计 4M 使用各自 2M final
  `state_dict` 再启动独立 fresh 2M phase，继续固定末端 epsilon/LR，并在分析中为该 phase
  增加 2M x-axis offset。

## 2026-09-07 — Skiing unclipped reward / gamma 0.999 未消除 BF16 Q ties

- **改动（Change）：** 从匹配的五任务 source weights 开始，seed1 的三个模型完成新的
  4M single-Skiing phase，取消 reward clip、gamma=0.999，保留 BF16 forward 和既定
  stall boundary / 18→9 single mapping。
- **原因（Reason）：** 检验恢复 reward 幅度与延长折扣 horizon 后，Q 分辨与行为是否改善。
- **证据（Evidence）：** 每模型以 eval_seed=20260904 起始的 20 个 greedy episodes
  审计最终 checkpoint；trace SHA256 校验通过。旧→新 top-Q exact tie rate：LSTM
  93.35%→100%、GRU 94.40%→100%、GaWF 99.86%→100%；新方案跨不同 legal groups
  的最优并列率也均为 100%，并非仅同一 legal action 的 aliases 并列。全部 18 个 Q
  相等的比例分别为 100%、83.49%、0%；median |Q| 分别为 1032、6048、1056。
  LSTM 所有 trace steps 的 18 个 Q 均为 -1032，argmax 全程选 NOOP；GaWF 99.89%
  steps 选 NOOP。LSTM/GRU/GaWF 的 greedy mean return 为 -8994/-30000/-15875，
  stalled episodes 为 0/20、20/20、0/20。
- **现状（Current）：** 新方案仍无法区分最优动作，return 改善或自然结束不能证明学会
  有效 steering；BF16 表示下 Q 尺度扩大、绝对量化间距增大，但本次未测量量化前的
  action gaps，不能把全部 ties 归因于单一数值原因。旧方案分段重建 optimizer/replay，
  新方案为一个可恢复的 4M phase，故不能单独归因于 clip 或 gamma。

## 2026-09-21 — GaWF 核心语义修正为 `nn.RNN` + 逐元素门控

- **改动（Change）：** `GaWFCore` 改为严格对齐 `nn.RNN`：递推、两个 bias、recurrence 内激活
  （`rnn_activation`，默认 `tanh`，可选 `relu`；`identity` 为唯一的非内置分支）与 state 约定
  全部沿用内置实现，只把 `W_ih`、`W_hh` 换成逐元素门控版本 `gate_ih⊙W_ih`、`gate_hh⊙W_hh`；
  回灌状态为 `activation(preactivation)`，外层 `LayerNorm→ReLU→dropout` 只作用于读出；多层
  GaWF 按 `nn.RNN` 方式逐层堆叠、层间施加该 wrap；`gawf_additive` 以及 Atari A2C/DQN、
  MiniGrid PPO 的 GaWF 包装层同步切换。旧实现冻结为 `gawf_legacy.GaWFCoreLegacy`
  （模型类型 `gawf_legacy`）。
- **原因（Reason）：** 旧实现把 `LayerNorm→ReLU→dropout` 放在循环内部并把 wrap 后的值回灌，
  因此并非"`nn.RNN` 加上门控"：同一组权重下其回灌状态与 `tanh(preactivation)` 最大相差
  0.80，且 train 模式下 dropout 直接作用于回灌状态，使递推本身随机化。这类实现偏离会
  让 GaWF 与 RNN/LSTM/GRU 基线的对照混入"LN 位置"这一额外变量。
- **证据（Evidence）：** 门饱和到 1 时，逐层门控核心与 `nn.RNN`（多层为逐层 `nn.RNN` +
  层间 wrap）逐步输出一致：单层 max|Δ|=5.96e-08、多层 max|Δ|=1.86e-07（float32 归约顺序）；
  门控关掉时 `step_no_feedback` 轨迹与 `nn.RNN` **逐位相同**（Δ=0.00e+00），
  `forward_no_feedback` 与 `relu(LN(nn.RNN 输出))` 也逐位相同；新核心与旧核心在
  `output_wrap="none"` 时逐位相同（Δ=0.00e+00），而单步回灌状态相差 0.93。
- **已完成结果的语义归属：** 所有以旧实现训练的 GaWF 均属 legacy 语义，包括 Clutter best6
  十 seeds（Fig1–Fig7、Supple1–3 及全部派生分析）、Atari GaWF A2C/DQN、MiniGrid GaWF，
  以及 feedback-control 里的 `gawf_additive`。分析侧已按"历史 `gawf_*` stem 走 legacy 核心、
  `gawf_rnncore_*` stem 走新核心"隔离，旧图件仍可复现；需要按新语义重跑时使用新 commit
  与新结果叶子。同时新增非线性位置消融类型（`*_nowrap`、`gawf_notanh`、`rnn_notanh`）与
  40-unit Amarel 数组（job 61735045）在新核心上开跑。
- **附带测量（feedback controls，10 seeds，validation accuracy @ identity-selected
  checkpoint）：** 旧语义下 `gawf_additive` vs `rnn_fb` 的 identity 差 +0.982±0.207 pt
  （paired t=+4.74），location 差 −0.379±0.117 pt（paired t=−3.24）；即二者虽只差一个
  trainable bias 与参数化写法，旧语义下仍是可分辨的两个系统，这正说明"状态归一化位置"
  不是可忽略的实现细节。
