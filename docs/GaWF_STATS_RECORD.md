# GaWF 正文与 Appendix A–I 实现/统计记录

更新日期：2026-09-18

## 口径与范围

本记录是正文与 Appendix A–I 共用的事实源。实现口径以当前工作副本为准；第一阶段
reset-exclusion 加固对应 commit `5d0d4ba`。数值只记录当前本地结构化结果或此前已经逐文件
核验的正式结果；仅有 PDF、历史说明或当前无法访问的远端路径不算新的数值核验。

除非另有说明，数值格式为 **10 个 training seeds 的均值 ± seed-level SEM**，其中
SEM 为 $s/\sqrt{10}$。所有 retained bar error bars 与 recovery shading 都使用
seed-level SEM；bar 上每个灰点代表一个 training seed。Supplementary 2/3 的有色 raw
scatter 是 connection/context rows，不是 seed points。accuracy 与 variance fraction 使用百分数。
门中位数先在每个 seed 内求得，再报告十个 seed median 的均值 ± SEM。除 §6.15 的指定
interaction test 外，不报告 p value。

Appendix 写作采用**按量类别固定精度**：accuracy、accuracy drop、generalization gap 与
variance fraction 均保留 1 位小数；$\Delta g$、sign gap、slope 与
$\Delta I^{\mathrm{gate}}$ 均保留 3 位小数。内部结构化文件和本记录的审计表可以保留更多
小数，不据此增加论文正文的显示精度。

数据审计发现：

- Fig1 test accuracy 现使用六个模型各 10 seeds 的 reset-excluded CSV：每个 32-frame rollout
  的 `t=0` 都从 accuracy 分母中剔除（每个 seed 保留 55,769/57,568 frames）。
- Fig1 target-switch recovery 使用独立的
  `40h-float32-jointswitch-balanced-10digit-unique` test protocol；每个 32-frame rollout 的
  `t=0` 同样剔除。Supplementary 1 保持其原有 512-frame rollout protocol，并剔除其每个
  512-frame window 的 `t=0`。每个 offset 的阴影是 training-seed mean 的 SEM，不是 pooled
  switch-event uncertainty。
- Fig4 activation、formal Fig7 及其他此前已经逐文件核验的 compact summaries，其固定统计值
  保留在本记录后半部分；本轮没有因 SSH 不可用而把旧远端路径重新宣称为当前可访问。
- Amarel 的 `multiseed_1_10/seed01`–`seed10` 含完整 GaWF synapse-level unified
  decomposition；顶层 manifest 只是较早的单-checkpoint 结果，不能代表该 multiseed 子树。
- `results/save_data/supple2/gate_context_specificity/` 是单 trajectory 结果，不能用于本文
  10-seed 数字。
- §5.9 使用本次在 sjc-remote 完成的 10-seed × 9-sector compact analysis；training seed 是
  inference unit，未使用 pooled connection-level 显著性检验。
- Fig6 的正式 encoder maps 以 sjc-remote 的
  `fig6_encoder_patterns_resetexcluded_gawf_10seed/` 为唯一数据源：Sector/Digit 各有 10 个
  reset-excluded compact outputs；`results/save/` 的正式 Sector PDF 已由此重绘。早期
  `fig6_encoder_tuning_gawf_10seed/` 路径不再是任何正式图或统计的数据源，故不作为缺失项追踪。
- `results/save/` 的正式 Fig3 分布图及 Fig6 maps 不显示跨-seed uncertainty，因而不需要改变
  图形误差条；Fig3 histograms 与 Fig6 encoder maps 均使用 reset-excluded 数据。

### 正式图的 reset / window 口径总表

本表只覆盖 `results/save/` 的当前正式交付物；**全部**已排除每个 recurrent window 的 `t=0`。
`results/save/archive/` 仅保留文件历史，不是正式结果或数据源。所有 32/512 标记都是 rollout
window，而非总 frame 数。

| 图 | reset-excluded 新结果 | window | 说明 |
|---|---|---:|---|
| Fig1（test accuracy、target-switch recovery） | 是 | 32 | test 与 recovery 都逐 window 排除 `t=0`。 |
| Fig2（feedback-shuffle ablation） | 是 | 512 | baseline / shuffle-sector / shuffle-digit 同口径。 |
| Fig3（主 gate/weight distribution PDF） | 是 | 32 | `fig3_gate_distribution` 在写入每个 seed 的 histogram 前已通过 `feedback != 0` 排除 reset；PDF 从这些 reset-excluded histograms 汇总。 |
| Fig4（six-model activation） | 是 | 32 | reset-excluded 60 model-seed compact summaries。 |
| Fig4（core objects：input/recurrent gate + activations） | 是 | 32 | 本次重算 raw synapse gate 后更新，含 10-seed points。 |
| Fig4（shuffle activation / gate ANOVA） | 是 | 512 | §4.12 / §2 shuffle-ablation protocol。 |
| Fig5（unit-gate marginalization） | 是 | 32 | GaWF 是 raw synapse 的 destination-unit projection。 |
| Fig6（encoder pattern maps） | 是 | 32 | Sector/Digit 各 10 seeds 均在 equal-n selection 前排除 1,799 个 `t=0`，各保留 55,769 frames；正式 Sector PDF 已更新。 |
| Fig6（sequential gate / sign maps） | 是 | 32 | 以 `feedback != 0` 排除 reset frame。 |
| Fig7（recurrent gate sign gaps） | 是 | 32 | 使用 reset-excluded r3 compact caches。 |
| Supplementary 1（feedback ablation recovery） | 是 | 512 | ablation 系列统一用 512-frame window。 |
| Supplementary 2（input-gate sign/magnitude） | 是 | 32 | 九个 sectors 的 reset-excluded compact analysis。 |
| Supplementary 3（recurrent-gate sign/magnitude） | 是 | 32 | Fig7 同一 reset-excluded compact caches。 |
| Supplementary 4（旧 standalone net recurrent-current） | deprecated | 32 | 已并入 ICLR Fig7；不再作为独立正式图或编号依据。保留的合并图代码不绘制 significance stars。 |

## Appendix A — GaWF 架构与参数化

### A.1 Clutter 模型的数据流

| 部件 | 当前实现 |
|---|---|
| 输入 | 每个 recurrent step 输入相邻两帧 grayscale CM-MNIST，tensor shape 为 `(B, T, 2, 96, 96)`。 |
| CNN encoder | `Conv2d(2,32,5,padding=same)` → `MaxPool2d(2)` → `LayerNorm(32,48,48)` → ReLU；再经 `Conv2d(32,64,3,padding=1)` → `MaxPool2d(4)` → `LayerNorm(64,12,12)` → `1×1 Conv(64,32)` → ReLU → `AdaptiveAvgPool2d(6,6)`，得到固定 `(32,6,6)`，按 channel-major contiguous order flatten 为 **1152** features。 |
| recurrent core | 正式 GaWF 使用 hidden size $H=256$、一层 recurrent core、`LayerNorm(H)`、ReLU 与 recurrent-output dropout $p=0.5$；encoder dropout 为 0。 |
| readouts | 同一个 hidden state 分别进入 10-class Digit linear head 与 9-class Sector linear head。 |
| feedback | 上一时刻两个 head 的 raw logits 按 `[digit logits; sector logits]` 拼接成 $F=19$；不经过 softmax。正式 non-projected 模式下整个 feedback vector detached。每个 rollout 的初始 hidden state与 feedback 都置零。 |

### A.2 GaWF 方程和共享低秩因子

对 $x_t\in\mathbb{R}^{1152}$、$h_{t-1}\in\mathbb{R}^{256}$、detached feedback
$z_{t-1}\in\mathbb{R}^{19}$，当前实现先将 feedback clamp 到 $[-10,10]$，再计算

$$M_t=(U\odot z_{t-1})V,\qquad G_t=\sigma(M_t/0.5),$$

其中 $U\in\mathbb{R}^{256\times19}$，$V\in\mathbb{R}^{19\times(1152+256)}$；`U` 和 `V`
均以 `randn * 0.01` 初始化。$G_t$ 沿 source 维拆成 $G_t^x$ 与 $G_t^h$，并逐连接调制
静态 `weight_ih` 与 `weight_hh`：

$$a_t=(G_t^x\odot W_{ih})x_t+(G_t^h\odot W_{hh})h_{t-1}+b_{ih}+b_{hh},$$
$$h_t=\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{LayerNorm}(\tanh a_t))).$$

输入与 recurrent pathway **共享同一个 $U$**，而 $V$ 的相应列分别服务于两条 pathway。
当前 factor 参数数为 $256\times19+19\times(1152+256)=31{,}616$。若两条 pathway 各自
拥有一套 $U/V$，则为 36,480；共享节省 4,864 个 factor 参数（13.33%）。

### A.3 参数量口径

匹配口径是完整 trainable Clutter model（共享 CNN encoder + recurrent core + 两个 heads），
同时由于 encoder 共享，recurrent middle path 也近似匹配。当前代码可直接解析的六个模型为：

| Model | Width | Encoder | Recurrent core（含 core LayerNorm） | Heads | Total |
|---|---:|---:|---:|---:|---:|
| RNN | 275 | 188,096 | 393,525 | 5,244 | **586,865** |
| LSTM | 80 | 188,096 | 395,040 | 1,539 | **584,675** |
| GRU | 105 | 188,096 | 396,795 | 2,014 | **586,905** |
| GaWF | 256 | 188,096 | 393,088 | 4,883 | **586,067** |
| Mamba | 170 | 188,096 | 395,930 | 3,249 | **587,275** |
| S5 | 256（state size 128） | 188,096 | 394,496 | 4,883 | **587,475** |

以上不是把“约 586K”仅写成 GaWF core 参数量；GaWF core 本身为 393,088。Mamba/S5 的
current-worktree 结构分别为：Mamba 一层 `mamba` block，`d_state=16`、`d_conv=4`、
`expand=2`、residual on、inter-layer dropout 0、output dropout 0.5；S5 一层、state size 128、
residual on、inter-layer dropout 0、output dropout 0.5。六模型均不使用 bidirectional core，
baseline 均无 feedback。

本轮重新计算时，本机 `aim3_rnn` 缺少 `mamba_ssm`，S5 实例化则在第三方 HiPPO 初始化处
触发本机 OpenMP/segfault；因此上表两项按当前 wrapper 与对应第三方 class 的 parameter shapes
逐项重算，而不是本轮成功 runtime-instantiation 的输出。Mamba core 分解为 input projection
196,010 + Mamba-v1 block 199,580 + LayerNorm 340；S5 core 分解为 input projection 295,168 +
S5 layer 98,816 + LayerNorm 512。正式稿可使用这些当前实现计数；若要求以训练节点实际安装的
dependency version 为 provenance，仍应从正式 metrics 的 `core_param_count/total_param_count`
做一次交叉核验。

各模型都复用同一个 `ClutterCNNEncoder` class 和 PyTorch 默认初始化规则，但参数 tensor 并不
跨模型共享。两个 readout 的 label semantics 均为 Digit=10、Sector=9；由于 core width 不同，
head weight shapes 随模型变化，不能写成“heads 完全相同”。

## Appendix B — Hyperparameter search、正式配置与跨 scale 表现

### B.1 搜索顺序与搜索空间

Appendix B 应依次给出三张配置表，而不是把搜索与正式训练混成一张：

**Table B-1：GaWF search。** 每个 scale 搜索 hidden `{64,128,256,512}`、LR
`{1e-4,5e-4,1e-3,5e-3}`、WD `{0,1e-5,1e-4,1e-3}`；seed 42、最多 100 epochs、
patience 15，共 256 个完成组合。下列是 scale-specific 最优结果，不与 10-seed formal
mean ± SEM 混报。

| Train scale | Best hidden | Best LR | Best WD | Val Digit (%) | Val Sector (%) |
|---|---:|---:|---:|---:|---:|
| 4h | 512 | 0.001 | 0.001 | 72.34 | 86.59 |
| 10h | 256 | 0.005 | 0.0001 | 80.51 | 89.77 |
| 20h | 512 | 0.001 | 0.0001 | 86.26 | 92.01 |
| 40h | 256 | 0.005 | 0.001 | 90.09 | 93.64 |

**Table B-2：other-model search。** 全部使用 seed 42、最多 100 epochs、patience 15，
并固定同一个 40h validation set。

| Model | Fixed parameter-matched width/state | LR search | WD search | 特殊规则 |
|---|---|---|---|---|
| RNN | H=275 | `{1e-4,5e-4,1e-3,5e-3}` | `{0,1e-5,1e-4,1e-3}` | — |
| LSTM | H=80 | `{1e-4,5e-4,1e-3,5e-3}` | `{0,1e-5,1e-4,1e-3}` | — |
| GRU | H=105 | `{1e-4,5e-4,1e-3,5e-3}` | `{0,1e-5,1e-4,1e-3}` | — |
| Mamba | d_model=170 | `{1e-4,5e-4,1e-3,5e-3,1e-2}` | `{0,1e-5,1e-4,1e-3}` | — |
| S5 | d_model=256，state=128 | `{1e-4,5e-4,1e-3,5e-3,1e-2}` | `{0,1e-5,1e-4,1e-3}` | core LR=`0.1 × base_lr`。 |

**Table B-3：formal configuration。** 六模型均在 4h/10h/20h/40h 上使用冻结配置，
seeds 1–10、150 epochs、patience 0，不再 search。

| Model | Width/state | LR | WD | 特殊 optimizer group |
|---|---|---:|---:|---|
| RNN | H=275 | 0.001 | 0.00001 | — |
| LSTM | H=80 | 0.001 | 0.001 | — |
| GRU | H=105 | 0.005 | 0.001 | — |
| GaWF | H=256 | 0.005 | 0.001 | `U/V/projector` 无 WD，LR=`base_lr × gawf_feedback_lr_scale`（正式 scale=1）。 |
| Mamba | d_model=170 | 0.001 | 0.001 | — |
| S5 | d_model=256，state=128 | 0.001 | 0 | S5 core LR=`0.1 × base_lr`。 |

### B.2 跨 scale 的 10-seed 结果

下表来自当前本地结构化文件 `data_scale_summary_10seed.json` / `data_scale_mean_sem_10seed.csv`。
每个 readout 的 validation accuracy 取自该 readout 自身达到最大 validation accuracy 的 epoch，
`Gap = train accuracy − validation accuracy` 也在该 epoch 计算。Identity 的 epoch 对应实际保存的
best checkpoint；Location 的最佳 epoch 只作 summary，可与 Identity epoch 不同。所有单元均为
mean ± seed-level SEM（percentage points）。2×3 panel 第一行为 Location 的
train/validation/gap，第二行为 Identity 的 train/validation/gap。当前 plotting entry point 输出
输出单一 2×3 图：
`results/save/data_scale_performance_2x3_10seed.pdf`；第三列是 accuracy
generalization gap，不是 loss gap。

| Scale | Model | Digit train | Digit val | Digit gap | Sector train | Sector val | Sector gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 4h | RNN | 92.8610 ± 0.8249 | 60.4061 ± 0.3747 | 32.4560 ± 0.9457 | 91.8574 ± 0.2714 | 84.7072 ± 0.1271 | 7.1500 ± 0.2866 |
| 4h | LSTM | 93.3681 ± 0.7118 | 60.3387 ± 0.3130 | 33.0300 ± 0.6670 | 92.8900 ± 0.2682 | 85.7998 ± 0.0718 | 7.0890 ± 0.2788 |
| 4h | GRU | 94.2353 ± 0.4444 | 53.5814 ± 0.5245 | 40.6540 ± 0.6982 | 93.9892 ± 0.2914 | 86.8207 ± 0.1243 | 7.1670 ± 0.3728 |
| 4h | GaWF | 95.3370 ± 0.3520 | 69.6158 ± 0.1999 | 25.7210 ± 0.4326 | 93.5992 ± 0.5381 | 86.5998 ± 0.1662 | 6.9990 ± 0.5183 |
| 4h | Mamba | 91.2566 ± 1.1718 | 63.4296 ± 0.4301 | 27.8270 ± 1.0557 | 93.4778 ± 0.3343 | 85.7337 ± 0.1212 | 7.7460 ± 0.3224 |
| 4h | S5 | 83.8468 ± 0.9789 | 55.6318 ± 0.4996 | 28.2150 ± 0.8838 | 91.3506 ± 0.4564 | 79.5728 ± 0.2815 | 11.7780 ± 0.5704 |
| 10h | RNN | 92.0163 ± 0.7173 | 72.4661 ± 0.2487 | 19.5500 ± 0.6474 | 94.7881 ± 0.2165 | 88.1765 ± 0.1234 | 6.6110 ± 0.2866 |
| 10h | LSTM | 92.2556 ± 0.6736 | 71.3827 ± 0.2195 | 20.8740 ± 0.7832 | 93.6210 ± 0.4160 | 88.4745 ± 0.0930 | 5.1470 ± 0.3473 |
| 10h | GRU | 89.9286 ± 0.7895 | 68.2408 ± 0.2981 | 21.6890 ± 0.7535 | 93.3579 ± 0.2563 | 88.8942 ± 0.0678 | 4.4630 ± 0.2428 |
| 10h | GaWF | 94.9035 ± 0.2872 | 79.3703 ± 0.1658 | 15.5320 ± 0.3614 | 96.0240 ± 0.3238 | 89.7628 ± 0.1186 | 6.2600 ± 0.3510 |
| 10h | Mamba | 92.2263 ± 0.3524 | 74.4259 ± 0.1640 | 17.8010 ± 0.3422 | 94.6540 ± 0.2598 | 89.0795 ± 0.1258 | 5.5750 ± 0.2782 |
| 10h | S5 | 82.9249 ± 0.6112 | 65.9946 ± 0.3598 | 16.9310 ± 0.5502 | 91.2307 ± 0.2664 | 84.6607 ± 0.1156 | 6.5710 ± 0.2916 |
| 20h | RNN | 92.9281 ± 0.4807 | 79.0717 ± 0.2099 | 13.8570 ± 0.4885 | 95.3496 ± 0.1723 | 90.6055 ± 0.0617 | 4.7430 ± 0.1503 |
| 20h | LSTM | 91.1820 ± 0.4233 | 78.2430 ± 0.1740 | 12.9390 ± 0.4158 | 94.8264 ± 0.2774 | 90.4077 ± 0.0787 | 4.4180 ± 0.2801 |
| 20h | GRU | 90.0809 ± 0.4720 | 76.1128 ± 0.3122 | 13.9660 ± 0.4355 | 94.0369 ± 0.2242 | 90.3590 ± 0.1102 | 3.6790 ± 0.2271 |
| 20h | GaWF | 95.2182 ± 0.1163 | 84.4493 ± 0.1395 | 10.7670 ± 0.2050 | 96.4430 ± 0.0805 | 91.7836 ± 0.0638 | 4.6580 ± 0.0641 |
| 20h | Mamba | 92.4340 ± 0.3418 | 80.8372 ± 0.1665 | 11.5960 ± 0.3507 | 95.8545 ± 0.2283 | 91.4440 ± 0.0947 | 4.4100 ± 0.1814 |
| 20h | S5 | 84.7816 ± 0.4820 | 71.9742 ± 0.1809 | 12.8060 ± 0.4726 | 91.2958 ± 0.2620 | 86.9950 ± 0.1614 | 4.3010 ± 0.1704 |
| 40h | RNN | 91.5411 ± 0.3405 | 83.8777 ± 0.1694 | 7.6630 ± 0.4476 | 94.3953 ± 0.2095 | 92.4209 ± 0.0441 | 1.9750 ± 0.2030 |
| 40h | LSTM | 90.8182 ± 0.2964 | 83.5344 ± 0.1603 | 7.2830 ± 0.4047 | 94.2908 ± 0.1855 | 91.9660 ± 0.0536 | 2.3250 ± 0.1761 |
| 40h | GRU | 90.3687 ± 0.2874 | 82.8165 ± 0.1567 | 7.5530 ± 0.3766 | 93.3215 ± 0.1304 | 92.0646 ± 0.0686 | 1.2580 ± 0.1778 |
| 40h | GaWF | 93.8971 ± 0.1236 | 89.4640 ± 0.0953 | 4.4340 ± 0.1011 | 95.5297 ± 0.0642 | 93.7352 ± 0.0638 | 1.7930 ± 0.0650 |
| 40h | Mamba | 92.9817 ± 0.2293 | 86.7878 ± 0.1026 | 6.1930 ± 0.2357 | 96.4966 ± 0.2278 | 93.5100 ± 0.0586 | 2.9860 ± 0.2264 |
| 40h | S5 | 84.5983 ± 0.6845 | 78.1985 ± 0.1511 | 6.4000 ± 0.6080 | 91.8990 ± 0.2606 | 90.2288 ± 0.1316 | 1.6700 ± 0.2869 |

## Appendix C — CM-MNIST 生成与 evaluation datasets

### C.1 Standard train/validation/test

三套 split 使用**同一生成算法与随机分布**，但不是同一随机 realization。生成器先设
`np.random.seed(42)`，然后按 train → validation → test 顺序连续生成独立 movie；MNIST exemplar
范围分别为 `[0,40000)`、`[40000,50000)`、`[50000,60000)`。正式配置为 96×96、24 fps；40h
train 有 3,456,000 frames，validation/test 各 2,400 s，即 57,600 raw frames。

每帧用饱和加法叠加 28×28 digits。foreground speed 从 `{0,1,2,3,4,6,8}` px/frame 取值，
segment 内保持速度方向并在边缘反弹；background digit 数从 `{1,2,4,8,12}` 取值，shared
mean speed 从 `{1,2,4,6,8}` 取值，每个 background object 每帧重新采样方向。switch interval
来自 mean=1 s 的 exponential 分布；`exclusive` 模式在每次 event 以 0.5/0.5 只切 foreground
或整组 background。每帧 TSV 保存 foreground digit、中心、速度以及 fg/bg switch flags；Sector
由 foreground center 映射到 3×3 grid。

### C.2 Joint-balanced held-out test

额外 test `40h-float32-jointswitch-balanced-10digit-unique` 仍使用 held-out MNIST
`[50000,60000)`、96×96、24 fps、2,400 s，但改变了 event protocol：foreground 与 background
在每个 scheduled event **joint switch**；90 个 Digit×Sector 条件各重复 27 次，共 2,430 个
events。event frames 从 frame 2 以后无放回均匀采样并排序，因此这里是 fixed event count 下的
随机间隔，不是 standard generator 的 exponential process。

`10digit-unique` 令 foreground 加 9 个 backgrounds 在每帧恰好覆盖 digits 0–9。每次 clutter
onset 的十个 objects 覆盖全部九个 sectors，并有一个独立均匀抽取的 sector 多一个 object。
严格 balance 针对 switch-event onset 的 Digit×Sector cell，不保证不同时长 episode 的总 frame
occupancy 完全相等。

### C.3 三类结果、四个具体 inference protocol

| 用途 | Dataset | Rollout | 每个 window 的 `t=0` |
|---|---|---:|---|
| Canonical six-model test accuracy | standard 40h test | 32 | 剔除 |
| Main feedback-shuffle ablation | standard 40h test | 512 | 剔除 |
| Fig1 six-model target-switch recovery | joint-balanced test | 32 | 剔除 |
| Supplementary clear-feedback recovery | joint-balanced test | 512 | 剔除 |

因此不能把 ablation 与 canonical accuracy 的差异归因于 dataset；它们使用同一个 standard test，
但 rollout 从 32 改为 512。两处 recovery 使用同一个 joint-balanced test，但只有 Fig1 是
32-frame rollout，Supplementary clear-feedback recovery 是 512-frame rollout。joint-balanced
test 还用于需要 switch 降到近 chance level 的相关 continuous switch dynamics/PCA。

## Appendix D — Training、selection 与 compute

| 项目 | 正式实现 |
|---|---|
| Objective | 每个 frame 的 Digit `CrossEntropyLoss` 与 Sector `CrossEntropyLoss` 等权相加；两项都在 batch×time 上取 mean。 |
| Optimizer | `AdamW`；GaWF factor 参数按 Appendix B 中的独立 param group 处理。无额外 LR scheduler。 |
| Temporal protocol | truncated BPTT / rollout length 32；每个 window 重置 hidden 与 feedback。standard long run 使用 mmap uint8、device-side float32 cast、effective-batch-sized block shuffle。 |
| Duration | Formal runs 为 150 epochs、patience 0；一个 epoch 遍历该 scale 经 block shuffle 后可组成 full batches 的训练 windows，末尾不足一个 batch 的 windows 因 `drop_last=True` 不参与该 epoch。 |
| Batch / precision | formal launcher 开启 acceleration；未覆盖 `AIM3_BATCH_SIZE`，故使用默认 batch size **256**。gradient accumulation 默认关闭（steps=1），effective batch 同为 **256 windows**；CUDA AMP 与 GradScaler 开启。train loader `drop_last=True`，validation/test `drop_last=False`。 |
| Gradient control | value clipping 1 与 global norm clipping 1。 |
| Model selection | checkpoint / reported `best` 以 validation **Digit accuracy 最大**为准，不写成 minimum validation loss；Sector 曲线可独立记录其最佳 epoch，但不改变主 checkpoint selection。 |
| Resume | 每 5 个 completed epochs 原子 checkpoint，自动 resume；partial run 不写 final result artifact。 |
| Compute disclosure | **one Ada Lovelace GPU, 16 CPUs, 64 GB memory**。正文/Appendix 不再展开节点型号、driver、wall time 等细节。 |

`MovieDataset.__len__ = floor((N_raw−2)/32)`；formal train loader 再丢弃最后一个不足 256
windows 的 batch。因此各 scale 的 nominal windows、optimizer steps/epoch 与实际用于一次 train
pass 的 output frames 为：

| Scale | Raw frames | Dataset windows | Steps/epoch | Used output frames/epoch |
|---|---:|---:|---:|---:|
| 4h | 345,600 | 10,799 | 42 | 344,064 |
| 10h | 864,000 | 26,999 | 105 | 860,160 |
| 20h | 1,728,000 | 53,999 | 210 | 1,720,320 |
| 40h | 3,456,000 | 107,999 | 421 | 3,448,832 |

四个 scale 均用同一固定 40h validation split 做 selection；scale 只改变 training movie 的长度。

## Appendix E / 正文 §3.1 — 行为表现

accuracy 先在每个 seed 内按 retained frames 计算：两个 head 分别取 per-frame `argmax`，
`correct / retained frame count`，再跨 10 seeds 报 mean ± SEM。switch recovery 的 switch frame
定义为 `post1`；横轴不含 offset 0。若相邻 switches 距离小于 `2 × radius`，两者之间可能受另一
switch 污染的 frames 会被屏蔽，post assignment 优先于 pre assignment；各 offset 使用实际可用
events，`frame_counts` 随 offset 可不同，不强制 equal-n。因此 recovery 曲线没有隐含的 pooled
event-level error bar。Appendix 只介绍 recovery curves 的构造与 mean ± SEM 绘制，不定义
scalar recovery threshold、连续帧判据或 baseline reference；“earlier”等只作定性描述。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 1.2 | GaWF 的 test accuracy：digit、sector 两个读出 | **每个 32-frame rollout 剔除 t=0 后**：Digit **86.6169% ± 0.1480%**；Sector **93.2852% ± 0.1238%**；各 n=10。 |
| 1.3 | 五个 baseline 各自的 test accuracy，两个读出 | **每个 32-frame rollout 剔除 t=0 后**：RNN：Digit **80.8603% ± 0.1732%**，Sector **91.3516% ± 0.0858%**。LSTM：**80.4103% ± 0.2266%**，**90.9369% ± 0.0637%**。GRU：**79.4637% ± 0.1705%**，**90.9471% ± 0.0524%**。S5：**75.1772% ± 0.3548%**，**89.3149% ± 0.1764%**。Mamba：**83.0981% ± 0.1669%**，**92.5997% ± 0.0749%**。每项 n=10，顺序均为 Digit、Sector。 |
| 1.4 | 全文统一的 seed 数 | 本记录所有已交付统计统一为 **n=10 training seeds（seeds 1–10）**；不满足 n=10 的现有结果不报数字。 |
| 1.6 | 目标切换处准确率变化：GaWF 与各 baseline，两个读出 | 两张 recovery 图现均为 **n=10 training seeds**；Fig1 的每个 32-frame window 与 Supplementary 1 的每个 512-frame window 均排除 `t=0`，每个 offset 显示 seed mean ± SEM。只作曲线级定性分析，不压缩成单一 drop scalar。 |
| 1.7 | recovery 的定性快慢 | 不定义 threshold、连续帧数或 baseline reference，也不报告 recovery-frame scalar；“earlier”等仅描述曲线形态。 |
| 1.8 | switch window 横轴范围；绝对值或归一化 | Fig1 与 Supplementary 1 均为完整 **pre10–pre1、post1–post10**；纵轴为**绝对 accuracy (%)**，不是归一化值；每个点为 n=10 seed mean，阴影为 SEM。 |

## Appendix E / 正文 §3.1 — 反馈 shuffle 消融

正式的 baseline、shuffle-sector 与 shuffle-digit 均采用 §4.12 的相同 512-frame recurrent
rollout protocol；每个 condition、每个 seed 均排除每个 window 的 `t=0`，保留 57,232 frames。
shuffle 实现为：先按原始顺序计算真实 feedback，再在每个 512-frame window 内、对每个 sample
独立 permute 时间索引，仅替换指定的 feedback slice。这样三根柱子可作严格的同 protocol 对照。
该 ablation **只对 GaWF** 运行；没有对无 feedback 的五个 baselines 构造伪 shuffle condition。
未被 shuffle 的 feedback slice 保留 live-rollout 值，所有 feedback 仍按正式 forward path detached。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 2.2 | Digit 读出：baseline / shuffle-sector / shuffle-digit | Baseline **89.7893% ± 0.1450%**；shuffle-sector **54.3366% ± 0.9315%**；shuffle-digit **73.5005% ± 0.5073%**；各 n=10。 |
| 2.3 | Sector 读出：同三个条件 | Baseline **94.2476% ± 0.1197%**；shuffle-sector **63.1848% ± 0.7705%**；shuffle-digit **91.7661% ± 0.1708%**；各 n=10。 |
| 2.4 | 可选：切换后恢复曲线形状量化 | **尚未分析。** 每个 seed 的曲线已保存，但没有已定义并保存的 shape scalar；按要求不新增量化。 |

## Appendix F / 正文 §3.2 — 门值分布

每个 seed 的 gate histogram 使用 400 个等宽 bins（范围 [0, 1]）。median 来自每个 seed
metadata 中保存的精确值；[0.1, 0.9] 比例由已保存的 per-seed histogram bins 直接汇总。
每个 seed 使用全部 55,769 个 reset-excluded frames，不做 frame subsampling：input gate 每帧为
`256×1152`，recurrent gate 每帧为 `256×256`。effective-weight histogram 对 input/recurrent
分别在每个 seed 内取对称范围 `[-max|W|,+max|W|]` 的 500 bins，同时累计静态 $W$ 和全部
frame×connection 的 $G\odot W$；跨 seed 汇总时按 bin overlap 重投影到共同对称范围，纵轴为
每 bin probability (%)。gate PDF 展示时可将 400 bins 合并为 100 个宽度 0.01 的 bins，但不改变
已保存精确 median 与区间计数。
§3.5 则在 sjc-remote 直接读取十个 seed 的原始 Figure 3 feedback trajectories 和 `U/V`，按
原始 eager float32 公式重建 gate 并流式计数，没有把 trajectory 或新增数值文件同步到本地。
0.5 点质量判据为 $|g-0.5|<10^{-6}$；不接收反馈的目标单元判据为
$\max_r|U[j,r]|<10^{-6}$。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 3.1 | 输入门中位数 | seed median 的均值 **0.0088 ± 0.0012**，n=10。 |
| 3.2 | 循环门中位数 | seed median 的均值 **0.5108 ± 0.0094**，n=10。 |
| 3.5a | gate 值恰好为 0.5 的占比 | 按 $|g-0.5|<10^{-6}$：输入门 **3.125026% ± 0.000001%**；循环门 **3.125063% ± 0.000002%**；各 n=10。每个 seed 恰有 1,799/57,568 = **3.125%** 的 reset frames。reset 外对应占比仅为输入门 **0.0000257% ± 0.0000007%**、循环门 **0.0000633% ± 0.0000021%**。 |
| 3.5b | 0.5 点质量的跨 context 方差 | 对 reset 点质量按全部 90 个 observed digit-sector labels 分组，input 与 recurrent 的 synapse-level context-mean variance 分位数均为 **min/q25/median/q75/max = 0/0/0/0/0**；十个 seed 中每个分位数的均值均为 **0 ± 0**。因为 reset feedback 为零，所有 gate 均严格为 `sigmoid(0)=0.5`。这不是一批跨完整 trajectory 恒为 0.5 的固定 synapses；$10^{-6}$ 判据包含的极少量 reset 外近 0.5 值不属于该零方差点质量。 |
| 3.5c | 剔除 0.5 点质量后，剩余 gate 落在 [0.1, 0.9] 的比例 | 输入门 **14.8352% ± 0.3929%**；循环门 **33.3198% ± 0.9836%**；各 n=10。计算式为 `(middle_count-half_count)/(total_count-half_count)`。 |
| 3.5d | 0.5 点质量是否按行成块；不接收反馈的 hidden units | **否。** 点质量覆盖 reset frame 的全部行，而不是固定 destination rows。按 $\max_r|U[j,r]|<10^{-6}$，输入门与循环门共享的 256 个 destination units 中不接收反馈者为 **0.0 ± 0.0 / 256**，n=10；十个 seed 均为 0。各 seed 的最小 row-wise $\max_r|U[j,r]|$ 为 0.1399–0.2125，远离阈值。 |
| 3.5e | 两端 gate 值占比 | **剔除每条 sequence 的 t=0 reset frame 后**，由每个 seed 的 400-bin histogram 汇总，区间为 $[0,0.1)$ / $[0.9,1]$。输入门分别为 **64.0288% ± 0.5675%** / **21.2005% ± 0.5741%**；循环门分别为 **32.0196% ± 0.8129%** / **34.8294% ± 0.7393%**；各 n=10。 |
| 3.6 | 门值落在 [0.1, 0.9] 的比例 | **剔除每条 sequence 的 t=0 reset frame 后**：输入门 **14.8352% ± 0.3929%**；循环门 **33.3199% ± 0.9836%**；各 n=10。该口径等同于 3.5c 所报告的非 reset gate 中间区间比例。 |

**§3.5 解释：** 原先提出的两种解释并不完备。0.5 spike 的主因是每条 sequence 首帧的
zero-feedback reset，而不是 `U[j]≈0` 的死通路；剔除该 spike 后，循环门仍有约三分之一的值
位于中间区间。因此现有结果不支持把 recurrent gate 概括为普遍 binary；至少需要把输入门与
循环门的表述分开，并把 recurrent gate 描述为保留显著 graded mass。

## Appendix G / 正文 §3.3 — 方差分解

4.1–4.3 使用 sjc-remote 已完成的 60 个 model-seed units Fig4 activation ANOVA。每个
training seed 先对
20 个 fixed balanced draws 取均值，再跨 10 seeds 计算 SEM；分母是 balanced condition means
的总方差，Sector、Digit、Interaction 合计 100%，不含 trial-level residual。

每个 seed 先从 55,769 个 reset-excluded frames 中按 90 个 Sector×Digit cells equalize：每 cell
抽 **211** frames，故每 draw 为 **18,990** frames；使用 20 个 fixed draws。每个 draw 对 90 个
cell means 做 two-way decomposition：Sector 与 Digit 是相应 marginal means 相对 grand mean 的
平方和，Interaction 是 cell mean 减去两个 main effects 与 grand mean 后的平方和；三项再除以
三者总和。这里既没有 trial residual，也没有把 20 draws 当作独立 inference units。

4.8–4.11 使用 Fig5 已保存的 10-seed `unit_gate_context_variance_multiseed.json`。分母口径是
**balanced 9-sector × 10-digit condition means 的总方差**；Sector、Digit、Interaction 三项
归一化后合计 100%，**不含 trial-level residual**。GaWF 门先对每个 destination unit 的所有
incoming synapse raw sigmoid gates 做 arithmetic mean，再在 unit level 分解；LSTM/GRU 本身
为 unit gate。该口径不能替代 GaWF synapse-level 口径。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 4.1 | 六模型 encoder activation：sector / digit / interaction | **剔除 t=0 reset frame 后**。GaWF：**51.2719% ± 0.5870% / 6.7013% ± 0.0720% / 42.0267% ± 0.5170%**。RNN：**63.5789% ± 0.6671% / 5.2921% ± 0.1024% / 31.1290% ± 0.5652%**。LSTM：**65.8523% ± 0.6737% / 4.9673% ± 0.1094% / 29.1804% ± 0.5655%**。GRU：**55.1824% ± 0.7133% / 6.2990% ± 0.0937% / 38.5186% ± 0.6202%**。S5：**59.7009% ± 0.3988% / 5.9241% ± 0.0664% / 34.3749% ± 0.3410%**。Mamba：**64.2181% ± 0.4190% / 5.0755% ± 0.0564% / 30.7064% ± 0.3649%**。每项 n=10，顺序均为 Sector / Digit / Interaction。 |
| 4.2 | 六模型 hidden activation：sector / digit / interaction | **剔除 t=0 reset frame 后**。GaWF：**31.3420% ± 0.3851% / 52.4565% ± 0.4553% / 16.2015% ± 0.2271%**。RNN：**39.4247% ± 0.2474% / 41.9165% ± 0.2957% / 18.6588% ± 0.1605%**。LSTM：**41.6517% ± 0.2713% / 36.9830% ± 0.2365% / 21.3653% ± 0.2851%**。GRU：**39.0142% ± 0.3533% / 46.2347% ± 0.3884% / 14.7511% ± 0.2568%**。S5：**47.0554% ± 0.6575% / 28.7683% ± 0.7687% / 24.1763% ± 0.2371%**。Mamba：**40.1968% ± 0.2300% / 43.5827% ± 0.2141% / 16.2205% ± 0.1384%**。每项 n=10，顺序均为 Sector / Digit / Interaction。 |
| 4.3 | GaWF encoder 与其余模型的 sector、interaction | **剔除 t=0 reset frame 后**。Sector / Interaction：GaWF **51.2719% ± 0.5870% / 42.0267% ± 0.5170%**；RNN **63.5789% ± 0.6671% / 31.1290% ± 0.5652%**；LSTM **65.8523% ± 0.6737% / 29.1804% ± 0.5655%**；GRU **55.1824% ± 0.7133% / 38.5186% ± 0.6202%**；S5 **59.7009% ± 0.3988% / 34.3749% ± 0.3410%**；Mamba **64.2181% ± 0.4190% / 30.7064% ± 0.3649%**；各 n=10。 |
| 4.4 | 输入门（突触级）sector / digit / interaction | **剔除 t=0 reset frame 后**：Sector **76.7069% ± 0.5348%**；Digit **15.9906% ± 0.4413%**；Interaction **7.3025% ± 0.1266%**。各 seed 在 55,769 frames 上做相同的 20-draw balanced ANOVA，n=10。 |
| 4.5 | 循环门（突触级）sector / digit / interaction | **剔除 t=0 reset frame 后**：Sector **23.2512% ± 0.5253%**；Digit **71.7333% ± 0.4868%**；Interaction **5.0155% ± 0.1135%**。同 4.4 的 raw-synapse、20-draw、n=10 口径。 |
| 4.8 | GaWF 两个门 destination-unit 投影后的三分量 | **剔除 t=0 reset frame 后**：Input gate：Sector **88.2487% ± 3.1062%**，Digit **9.7954% ± 3.0684%**，Interaction **1.9558% ± 0.0739%**。Recurrent gate：**11.3422% ± 1.6160%**，**86.5034% ± 1.6762%**，**2.1543% ± 0.0777%**。各 n=10。 |
| 4.9 | GRU reset / update 门（unit 级）三分量 | **剔除 t=0 reset frame 后**：Reset：Sector **56.9358% ± 0.3889%**，Digit **22.7363% ± 0.5218%**，Interaction **20.3279% ± 0.3687%**。Update：**51.7696% ± 0.8600%**，**37.5716% ± 0.8743%**，**10.6587% ± 0.2951%**。各 n=10。 |
| 4.10 | LSTM input / forget / output 门（unit 级）三分量 | **剔除 t=0 reset frame 后**：Input：Sector **36.5091% ± 0.7504%**，Digit **52.3285% ± 0.7399%**，Interaction **11.1624% ± 0.2600%**。Forget：**62.6161% ± 0.8204%**，Digit **24.2261% ± 0.7064%**，Interaction **13.1578% ± 0.2224%**。Output：**65.8904% ± 0.7740%**，**23.8193% ± 0.6304%**，**10.2903% ± 0.2858%**。各 n=10。 |
| 4.11 | 七个门的 interaction 项，同一投影层级 | **剔除 t=0 reset frame 后**：GaWF input **1.9558% ± 0.0739%**；GaWF recurrent **2.1543% ± 0.0777%**；GRU reset **20.3279% ± 0.3687%**；GRU update **10.6587% ± 0.2951%**；LSTM input **11.1624% ± 0.2600%**；forget **13.1578% ± 0.2224%**；output **10.2903% ± 0.2858%**；各 n=10。 |

### 4.12 — §2 shuffle 条件下的 activation / gate ANOVA

复用 §2 的 `40h-uint8`、512-frame rollout 和逐 sample feedback permutation 顺序，在未
shuffle 的 ground-truth label 上做分解；每个条件、每个 seed 排除首个 reset frame 后有
57,232 frames，以相同的 20 个 fixed balanced draws 汇总。以下均为 **10-seed mean ± SEM**。
`between_condition_var` 是未归一化的 balanced condition-mean total variance；三项百分比以该
variance 为分母并合计 100%，`trial-level residual` 则以 trial-level total variance 为分母。

| feedback 条件 | Digit accuracy | Sector accuracy |
|---|---:|---:|
| Baseline | 89.7893% ± 0.1450% | 94.2476% ± 0.1197% |
| Shuffle digit | 73.5005% ± 0.5073% | 91.7661% ± 0.1708% |
| Shuffle sector | 54.3366% ± 0.9315% | 63.1848% ± 0.7705% |

| 对象 | feedback 条件 | Sector / Digit / Interaction | 未归一化 `between_condition_var` | trial-level residual |
|---|---|---|---:|---:|
| Encoder activation | Baseline | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Encoder activation | Shuffle digit | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Encoder activation | Shuffle sector | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Hidden activation | Baseline | 30.2743% ± 0.3828% / 53.7094% ± 0.4698% / 16.0163% ± 0.2339% | 2.9942 ± 0.0844 | 38.3718% ± 0.3466% |
| Hidden activation | Shuffle digit | 50.3564% ± 1.0297% / 32.7871% ± 1.0481% / 16.8565% ± 0.1359% | 1.3303 ± 0.0635 | 64.0272% ± 0.8468% |
| Hidden activation | Shuffle sector | 40.1362% ± 0.5727% / 41.4261% ± 0.4246% / 18.4377% ± 0.2677% | 0.4635 ± 0.0134 | 81.8317% ± 0.3688% |
| Input gate | Baseline | 75.5101% ± 0.5488% / 16.8494% ± 0.4603% / 7.6406% ± 0.1240% | 15,923.7783 ± 97.6968 | 30.3288% ± 0.1325% |
| Input gate | Shuffle digit | 96.5238% ± 0.0701% / 1.3060% ± 0.0393% / 2.1702% ± 0.0337% | 10,441.2873 ± 139.1854 | 55.1829% ± 0.5231% |
| Input gate | Shuffle sector | 25.2028% ± 0.9480% / 54.9499% ± 1.3343% / 19.8473% ± 0.4096% | 881.2302 ± 32.7129 | 95.9226% ± 0.1254% |
| Recurrent gate | Baseline | 22.1828% ± 0.4987% / 72.7907% ± 0.4667% / 5.0266% ± 0.1147% | 4,264.4144 ± 70.5233 | 30.8503% ± 0.1737% |
| Recurrent gate | Shuffle digit | 81.2008% ± 0.3816% / 12.0795% ± 0.2815% / 6.7197% ± 0.1143% | 923.2748 ± 21.8691 | 85.2064% ± 0.3056% |
| Recurrent gate | Shuffle sector | 4.2203% ± 0.2240% / 87.7793% ± 0.4019% / 8.0004% ± 0.2118% | 794.2656 ± 27.2721 | 83.2358% ± 0.4186% |

一致性检查显示 encoder 在三种 condition 下逐 seed 完全相同，符合 shuffle 只改 feedback 的
实现。Hidden 的未归一化 condition-mean variance 从 baseline 的 2.9942 降至 shuffle-digit 的
1.3303 和 shuffle-sector 的 0.4635，同时 residual 升至 64.0272% 和 81.8317%；因此三项
归一化百分比的变化不能单独解释为 factor signal 的重新分配，而是伴随 substantial total signal
collapse。

### 4.13 — 带 residual 图的统一 trial-level 四分量

本小节对应 `Fig4_core_objects_aggregate_1x4_10seed_with_residual`、
`Fig4_shuffle_activation_anova_1x3_10seed`、
`Fig4_activation_anova_1x2_6model_10seed_with_residual` 与
`Fig5_unit_gate_marginalization_1x3_with_residual`。**不替代**上文任何以
condition-mean total variance 为分母的 Sector / Digit / Interaction 结果。为使每组四柱可加和
为 100%，每个 training seed 先按
\(\eta^2_f(\mathrm{trial})=\eta^2_f(\mathrm{condition\ mean})
\,[1-\eta^2_{\mathrm{residual}}(\mathrm{trial})]\) 转换三个 factor；Residual 保持
\(SS_{\mathrm{residual}}/SS_{\mathrm{total,trial}}\)。下列均为转换后再跨 10 seeds 的 mean ±
SEM，顺序均为 Sector / Digit / Interaction / Residual。

计算的逐 seed 输入是
`Fig4_shuffle_activation_anova_long_10seed.csv` 中同一 `object × condition × seed` 的
`sector_pct`、`digit_pct`、`interaction_pct` 与 `residual_frac`（均已先对该 seed 的 20 draws
取均值），而不是跨 seed 的列均值。对每个 factor 使用
(e_{f,s}=eta^2_{f,s}(1-r_s))，再报告 (\mathrm{mean}_s(e_{f,s})\pm\mathrm{SEM}_s(e_{f,s}))。

**Fig4 core（标准 32-frame protocol；reset-excluded）**

| 对象 | trial-level 四分量（%） |
|---|---|
| Input gate | **51.0500 ± 0.4009 / 10.6400 ± 0.2883 / 4.8593 ± 0.0814 / 33.4508 ± 0.1309** |
| Recurrent gate | **14.9963 ± 0.3500 / 46.2569 ± 0.2971 / 3.2340 ± 0.0715 / 35.5128 ± 0.1373** |
| Encoder activation | **7.3495 ± 0.1306 / 0.9597 ± 0.0092 / 6.0181 ± 0.0640 / 85.6727 ± 0.1279** |
| Hidden activation | **18.0744 ± 0.2189 / 30.2601 ± 0.3587 / 9.3449 ± 0.1430 / 42.3206 ± 0.3556** |

**Fig4 shuffle hidden activation（reset-excluded）**

| feedback 条件 | trial-level 四分量（%） |
|---|---|
| Baseline | **18.6537 ± 0.2258 / 33.1041 ± 0.3847 / 9.8704 ± 0.1531 / 38.3718 ± 0.3466** |
| Shuffle digit | **18.0564 ± 0.3000 / 11.8569 ± 0.6721 / 6.0594 ± 0.1282 / 64.0272 ± 0.8468** |
| Shuffle sector | **7.2949 ± 0.1901 / 7.5272 ± 0.1754 / 3.3462 ± 0.0671 / 81.8317 ± 0.3688** |

**Fig4 six-model activation（reset-excluded）**

| 对象 | 模型 | trial-level 四分量（%） |
|---|---|---|
| Input activation | GaWF | **7.3495 ± 0.1306 / 0.9597 ± 0.0092 / 6.0181 ± 0.0640 / 85.6727 ± 0.1279** |
| Input activation | RNN | **8.6564 ± 0.1992 / 0.7183 ± 0.0078 / 4.2258 ± 0.0416 / 86.3995 ± 0.1797** |
| Input activation | LSTM | **8.9365 ± 0.2598 / 0.6709 ± 0.0089 / 3.9423 ± 0.0445 / 86.4504 ± 0.2677** |
| Input activation | GRU | **6.2625 ± 0.2098 / 0.7120 ± 0.0120 / 4.3528 ± 0.0716 / 88.6727 ± 0.2661** |
| Input activation | S5 | **7.2094 ± 0.1849 / 0.7138 ± 0.0107 / 4.1431 ± 0.0695 / 87.9337 ± 0.2511** |
| Input activation | Mamba | **9.3009 ± 0.1466 / 0.7341 ± 0.0042 / 4.4412 ± 0.0297 / 85.5239 ± 0.1452** |
| Hidden activation | GaWF | **18.0744 ± 0.2189 / 30.2601 ± 0.3587 / 9.3449 ± 0.1430 / 42.3206 ± 0.3556** |
| Hidden activation | RNN | **19.7689 ± 0.1459 / 21.0204 ± 0.1960 / 9.3569 ± 0.0974 / 49.8538 ± 0.2666** |
| Hidden activation | LSTM | **26.5929 ± 0.1979 / 23.6142 ± 0.2020 / 13.6374 ± 0.1581 / 36.1555 ± 0.1831** |
| Hidden activation | GRU | **22.4459 ± 0.1548 / 26.6082 ± 0.2710 / 8.4930 ± 0.1737 / 42.4529 ± 0.2610** |
| Hidden activation | S5 | **16.9490 ± 0.2118 / 10.3688 ± 0.2956 / 8.7117 ± 0.1056 / 63.9705 ± 0.1813** |
| Hidden activation | Mamba | **23.9547 ± 0.1310 / 25.9744 ± 0.1607 / 9.6666 ± 0.0849 / 40.4043 ± 0.1410** |

**Fig5 unit gates（reset-excluded；GaWF 为 destination-unit projection）**

| 模型 | Gate | trial-level 四分量（%） |
|---|---|---|
| GaWF | Input | **54.7013 ± 1.9496 / 6.1058 ± 1.9180 / 1.2101 ± 0.0392 / 37.9827 ± 0.7102** |
| GaWF | Recurrent | **7.6242 ± 1.0641 / 58.3621 ± 1.2990 / 1.4509 ± 0.0465 / 32.5628 ± 0.3322** |
| LSTM | Input | **16.7857 ± 0.5232 / 24.0071 ± 0.4078 / 5.1136 ± 0.0905 / 54.0935 ± 0.6888** |
| LSTM | Forget | **19.1374 ± 0.3950 / 7.3878 ± 0.1853 / 4.0148 ± 0.0524 / 69.4600 ± 0.2825** |
| LSTM | Output | **36.6962 ± 0.5795 / 13.2659 ± 0.3764 / 5.7204 ± 0.1302 / 44.3175 ± 0.4563** |
| GRU | Reset | **20.3446 ± 0.3185 / 8.1163 ± 0.1856 / 7.2584 ± 0.1389 / 64.2807 ± 0.3945** |
| GRU | Update | **20.1538 ± 0.4549 / 14.6287 ± 0.4201 / 4.1393 ± 0.0908 / 61.0782 ± 0.5526** |

## Appendix H / 正文 §3.4 — 输入门的空间组织与符号盲性

本节只做定性分析，不增加过多定量统计检验。仅保留检验 sign-blindness 所必需的 5.9
seed-level descriptive overlap gaps、slopes 与 levels；不报告 pooled connection-level p values。

空间映射直接按 encoder flatten order `(channel=32, row=6, col=6)` reshape。每个 3×3 Sector
对应每个 channel 上严格不重叠的 `2×2` spatial block，即 **128 matching sources**；其补集为
**1,024 other sources**。当前实现不使用 receptive-field overlap threshold；历史“441/1152”
mask 不是当前 formal protocol，不应写入 Appendix。每个 sector 的 gate map 先在 equal-n frames
上求每条 synapse 的 condition mean，再 reshape 为 `(destination, channel, 6, 6)` 并对 destination
与 channel 平均。$\Delta g$ 从该 sector mean 减去九个 sector condition means 的等权 grand mean。
sign/magnitude curves 使用 9 个 $|W|$ quantile bins；W+ 与 W− slope 在每个 seed 的 shared
overlap band 内分别拟合，最后以 training seed 为 inference unit。表中的 sign means 先在
每个 Sector condition 内对 connection rows 求均值，再在 seed 内对九个 condition means
等权平均；all-connections level 采用相同的 condition-first 等权汇总，但不限制 sign 或
shared overlap band。slopes 与九分箱曲线仍在 observation level 计算，不受此次均值汇总修正影响。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 5.9a | matching sources overlap gap，九个 sectors 等权汇总 | **剔除 reset frame 后**：W+ mean Δg **+0.27583 ± 0.00524**；W− **+0.28900 ± 0.00527**；per-seed overlap gap W+−W− **−0.01317 ± 0.00181**；n=10。 |
| 5.9b | other sources overlap gap，九个 sectors 等权汇总 | **剔除 reset frame 后**：W+ mean Δg **−0.03446 ± 0.00066**；W− **−0.03614 ± 0.00065**；per-seed overlap gap **+0.00169 ± 0.00023**；n=10。 |
| 5.9c | 分符号 slope 与 SEM，matching / other 分开 | 在各 seed 自己的 shared-\|W\| band 内拟合 OLS `Δg ~ \|W\|`，**剔除 reset frame 后**。Matching：W+ slope **−0.0429 ± 0.0060**，W− **+0.0073 ± 0.0027**。Other：W+ **+0.0054 ± 0.0008**，W− **−0.0009 ± 0.0003**；各 n=10。 |
| 5.9d | matching / other 的 all-connections Δg level | **剔除 reset frame 后**，先在每个 sector 内跨全部 nonzero connections 平均、再对 9 sectors 等权平均：Matching **+0.28251 ± 0.00518**；Other **−0.03531 ± 0.00065**；各 n=10。图中的曲线仍是 observation-level binned mean ± SEM，仅作定性展示。 |

Table 14 若按五位小数显示 sign means 并由显示后的两行相减，则 matching gap 为
`0.27583 - 0.28900 = -0.01317`，other gap 为
`-0.03446 - (-0.03614) = +0.00168`；后者与未取整的 per-seed gap mean 四舍五入值
`+0.00169` 相差 $10^{-5}$，统计推断使用未取整的 per-seed differences。

## Appendix I / 正文 §3.5 — 循环门的符号依赖调制

T/R masks 来自 **validation split** 的 hidden activation selectivity，而不是 test labels。对每个
unit 做 1,000 次 stratified permutations：Sector label 在 Digit strata 内 shuffle，Digit label 在
Sector strata 内 shuffle；两个 factor 分别对 per-unit p values 做 Benjamini–Hochberg FDR，
$q\leq0.05$ 为 passed。若 $\eta^2_{interaction}$ 同时大于 $\eta^2_{sector}$ 与
$\eta^2_{digit}$，该 unit 标为 interaction-dominant。factor-specific eligible pool 为
`passed factor & not interaction-dominant`；每个 context 的 T 是该 pool 内 tuning 最高的
`ceil(0.1 × |eligible|)` units，R 为其余 hidden units。T 的实际大小见 6.12，不把 24 固定写死。

6.3–6.6 使用 formal reset-excluded 10-seed compact arrays。每个 seed 先在该 seed 自己的
positive/negative shared-|W| overlap band 内，分别为每个 Digit 或 Sector condition 计算
connection mean，再对 10 个 Digit 或 9 个 Sector condition means 等权平均。gap 定义为
$\Delta g_{W>0}-\Delta g_{W<0}$，再跨 training seeds 计算 SEM。下表沿用 efferent
`src→dst` TT/TR/RT/RR 口径，不使用 afferent companion。

6.13 在每个 seed 内对 45 个 digit pairs 计算 normalized overlap
$|T_a\cap T_b|/\sqrt{|T_a||T_b|}$，然后先对 pairs 取均值、再以 10 个 training seeds 为
独立单位计算 SEM。其 independent-set chance baseline 在每个 seed 内固定 observed $|T_a|$、
$|T_b|$，从同一 FDR-eligible 且非 interaction-dominant pool $E$ 独立均匀抽取，故为
$\sqrt{|T_a||T_b|}/|E|$。6.15 直接使用 Figure 7 已保存的 group-specific shared-|W| overlap-band
seed cell means；在 sign-gap 上做 `group × variable` repeated-measures ANOVA，等价于原始
cell means 上的 `group × sign(W) × variable` interaction。6.11 在每个 seed、variable 和
TT/TR/RT/RR group 自己的 positive/negative shared-|W| overlap band 内分别拟合 W+、W−
的 OLS `Δg ~ |W|`，再跨 10 个 training seeds 计算 SEM；overall level 不限制 overlap band。

ICLR Fig7 顶行第三根 gate bar 的代码变量名仍为 `balanced_gate`，但其数值口径实际是
**all connections**：每个 seed 先在每个 condition 内对该 group 的全部 nonzero connections
求均值，再对 Digit 或 Sector conditions 等权平均。condition 内部自然按实际 positive/negative
connection counts 加权，而不是将两个 sign means 等权平均。图中单独显示的 W+ / W− bars
使用 shared-|W| overlap band，因此第三根 all-connections bar 不应由这两根经过筛选的 sign bars 反推。

6.17 对每个 seed、Digit 或 Sector context 逐帧计算实际循环电流
$I_{group}(c)=\sum_{(i,j)\in group}\langle g_{ij}(t)W_{ij}h_j(t-1)\rangle_{t\in c}$，并以
$\Delta I(c)=I(c)-\operatorname{mean}_{c'}I(c')$ 定义 context delta。门的瞬时贡献为
$\Delta I^{gate}(c)=\sum\langle(g_{ij}(t)-\bar g_{ij})W_{ij}h_j(t-1)\rangle_{t\in c}$；其中
$\bar g_{ij}$ 是同一分析内全部条件均值的等权平均（Digit 为 10、Sector 为 9），实际
$h_j(t-1)$ 保持不变。TT/TR/RT/RR 都保留
diagonal、按 `sign(W)` 拆为 E/I，并除以该组目的地单元数。它是**瞬时分解**，不是冻结 gate 后的
完整反事实；每个 32-frame window 的 `t=0` 在 equal-n selection 前剔除。

6.18 使用同一逐帧定义，但改为 **per nonzero recurrent connection**：`W > 0` 与 `W < 0`
各自除以该 sign 的连接数；total 则以两个 sign 的连接数加权，
$\Delta I^{gate}_{\mathrm{total}}=(N_+\Delta I^{gate}_+ + N_-\Delta I^{gate}_-)/(N_+ + N_-)$，
**不是**两个 sign-specific 均值的直接相加。每个 seed 先跨 Digit（10）或 Sector（9）条件平均，
再跨 10 个 training seeds 计算 mean ± SEM。合并后的 ICLR Fig7 C 使用这一 seed-level
作图口径：每个灰点是一个 training seed 的跨条件等权均值，因此每个 bar 均有 10 个点；
bar height、SEM 与灰点来自同一组 10 个 seed values，图上不绘制 significance stars。
caption stats 的 `bar_summary[].seed_values` 保存这 10 个值，不再将跨 seed 的 condition
means 记录为灰点；其 p-value 字段只作结构化诊断记录。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 6.3–6.5 | 四组 × 两变量符号缺口完整表，delta 版 | **剔除 reset frame 后；condition-first 等权汇总。** Digit：TT **+0.09916 ± 0.00556**；TR **−0.07210 ± 0.00370**；RT **−0.01477 ± 0.00252**；RR **+0.00715 ± 0.00055**。Sector：TT **−0.01209 ± 0.00336**；TR **−0.01180 ± 0.00169**；RT **−0.00964 ± 0.00053**；RR **+0.00350 ± 0.00017**。各 n=10；这里的 inference mean 来自未取整的 per-seed differences。 |
| 6.6 | 同表 W>0、W<0 各自 Δg 水平 | **剔除 reset frame 后；condition-first 等权汇总。** Digit：TT W+ **−0.25677 ± 0.00756**，W− **−0.35593 ± 0.00850**；TR **−0.14127 ± 0.00708 / −0.06917 ± 0.00415**；RT **−0.01457 ± 0.00201 / +0.00021 ± 0.00273**；RR **+0.01804 ± 0.00085 / +0.01089 ± 0.00042**。Sector：TT **−0.10145 ± 0.00348 / −0.08936 ± 0.00368**；TR **−0.08634 ± 0.00256 / −0.07453 ± 0.00306**；RT **+0.00630 ± 0.00062 / +0.01594 ± 0.00092**；RR **+0.01014 ± 0.00030 / +0.00664 ± 0.00033**。每对均为 W+ / W−，各 n=10。 |
| 6.11a | 分符号 slope 与 SEM，Digit/Sector × TT/TR/RT/RR 分开 | 在各 seed 自己的 shared-\|W\| overlap band 内拟合、**剔除 reset frame 后**。Digit：TT W+ **+0.1285 ± 0.0161**，W− **−0.0210 ± 0.0144**；TR **+0.0386 ± 0.0043 / +0.0634 ± 0.0031**；RT **+0.0352 ± 0.0032 / +0.0211 ± 0.0017**；RR **−0.0082 ± 0.0008 / −0.0099 ± 0.0006**。Sector：TT **+0.0042 ± 0.0042 / +0.0032 ± 0.0020**；TR **+0.0169 ± 0.0022 / +0.0097 ± 0.0007**；RT **−0.0076 ± 0.0020 / +0.0089 ± 0.0010**；RR **+0.0004 ± 0.0003 / −0.0022 ± 0.0002**。每对均为 W+ / W−，各 n=10。 |
| 6.11b | 八组的 all-connections Δg level，对齐 5.9d | **剔除 reset frame 后**，每 seed 先在各 condition 内跨该组全部 nonzero connections 平均、再跨 conditions 等权平均（不限制 overlap band）：Digit TT **−0.3098 ± 0.0078**；TR **−0.0941 ± 0.0050**；RT **−0.0046 ± 0.0022**；RR **+0.0136 ± 0.0006**。Sector TT **−0.0952 ± 0.0032**；TR **−0.0790 ± 0.0028**；RT **+0.0124 ± 0.0008**；RR **+0.0079 ± 0.0003**。各 n=10。 |
| 6.12 | hidden size、实际 \|T\|、diagonal 处理 | **H=256**。T 是每个 seed 中 FDR-eligible 且非 interaction-dominant units 的 top 10%，用 `ceil(0.1 × eligible)`；seeds 1–10 的实际 \|T\| 为 **[25, 24, 24, 24, 24, 24, 24, 24, 23, 25]**，同一 seed 的所有 Digit/Sector contexts 数量相同。formal 分组**保留 diagonal `i=j`**，所以 TT 候选规模为 \|T\|²（分别为 625、576 或 529，之后仍应用 `weight != 0`）。 |
| 6.13 | 不同 digit 的 T 集合重叠度 | **剔除 reset frame 后**。每个 seed 的 10 个 digit `T` masks 来自同一 reset-excluded compact cache；45 个 digit pairs 的 normalized overlap 先在 seed 内取均值，再跨 seeds 汇总为 **12.14% ± 0.20%**（SEM，n=10）。固定各 seed 的 observed $|T|$、从其 FDR-eligible non-interaction pool 独立抽取的 chance baseline 为 **10.22% ± 0.04%**，故 observed − chance 为 **+1.93 ± 0.20 percentage points**。45 个 digit-pair 的跨-seed 均值范围为 **3.35%–21.51%**（最低 digit 0–1；最高 digit 4–6），保留了明显的 pair heterogeneity；所有 seed 内的 $|T|$ 分别固定为 23、24 或 25。 |
| 6.15 | `group × sign(W) × variable` interaction test | **剔除 reset frame 后**，在 condition-first 的 10-seed shared-\|W\| sign-gap cells 上做 `group × variable` repeated-measures ANOVA（等价原始 cell means 的三因素 interaction）：**F(3, 27) = 359.9072，p = 7.13 × 10⁻²²**。 |
| 6.17a | 净循环贡献，Digit：`I`、`ΔI` 与 `ΔI^gate` | **剔除 reset frame 后**；下列 `I / ΔI^gate` 为每 seed 先跨十个 digits 平均、再作 10-seed mean ± SEM，顺序均为 E / I / total（每目的单元）。TT：`I` **+0.2270 ± 0.0150 / −0.1384 ± 0.0092 / +0.0885 ± 0.0114**；`ΔI^gate` **−0.1477 ± 0.0072 / +0.7385 ± 0.0330 / +0.5908 ± 0.0336**。TR：**+0.1418 ± 0.0051 / −1.1770 ± 0.0586 / −1.0352 ± 0.0543**；**−0.0739 ± 0.0033 / −0.0017 ± 0.0094 / −0.0756 ± 0.0094**。RT（R→T）：**+0.4781 ± 0.0213 / −0.9206 ± 0.0417 / −0.4424 ± 0.0273**；**−0.2545 ± 0.0121 / +1.7394 ± 0.0617 / +1.4850 ± 0.0572**。RR：**+0.5565 ± 0.0151 / −2.4798 ± 0.0699 / −1.9232 ± 0.0646**；**−0.1961 ± 0.0067 / +0.5801 ± 0.0172 / +0.3841 ± 0.0154**。`ΔI` 的跨 digit 平均按定义为 0；其十个 digit 的 10-seed mean 范围（E / I / total）是 TT **−0.0848–+0.2739 / −0.1376–+0.0610 / −0.0388–+0.1363**，TR **−0.0160–+0.0295 / −0.2591–+0.4644 / −0.2587–+0.4933**，RT **−0.1355–+0.4630 / −0.7879–+0.2758 / −0.3248–+0.1695**，RR **−0.0666–+0.0476 / −0.3636–+0.0993 / −0.4303–+0.1278**；各 digit 的 mean ± SEM 及所有 seed-level 值见同一 long CSV 与 Supple4。T→T 的 I 正贡献（减少负电流）大于 E 负贡献，且十个 digits 的总 `ΔI^gate` 均为正（**+0.0769–+0.8819**，最小 `mean−SEM=+0.0439`），故支持瞬时 **disinhibition**。R→T 亦不可忽略：总 `ΔI^gate` 十个 digits 均为正（**+1.0523–+1.8893**，最小 `mean−SEM=+0.9391`）。 |
| 6.17b | 净循环贡献，Sector：`I`、`ΔI` 与 `ΔI^gate` | **剔除 reset frame 后**；每 seed 先跨九个 sectors 平均、再作 10-seed mean ± SEM，顺序均为 E / I / total（每目的单元）。TT：`I` **+0.2570 ± 0.0205 / −0.2478 ± 0.0158 / +0.0092 ± 0.0269**；`ΔI^gate` **−0.1183 ± 0.0056 / +0.1675 ± 0.0055 / +0.0491 ± 0.0078**。TR：**+0.1204 ± 0.0042 / −0.4924 ± 0.0211 / −0.3720 ± 0.0193**；**−0.0654 ± 0.0024 / +0.1944 ± 0.0067 / +0.1290 ± 0.0052**。RT（R→T）：**+0.5769 ± 0.0209 / −2.4396 ± 0.0986 / −1.8627 ± 0.0950**；**−0.2290 ± 0.0106 / +0.5085 ± 0.0299 / +0.2794 ± 0.0213**。RR：**+0.5603 ± 0.0148 / −3.0418 ± 0.0980 / −2.4815 ± 0.0919**；**−0.1958 ± 0.0064 / +0.5430 ± 0.0187 / +0.3472 ± 0.0165**。`ΔI` 的跨 sector 平均按定义为 0；九个 sector 的 10-seed mean 范围（E / I / total）是 TT **−0.1096–+0.3676 / −0.0934–+0.0621 / −0.1846–+0.3999**，TR **−0.0169–+0.0445 / −0.1190–+0.1131 / −0.1283–+0.1576**，RT **−0.0781–+0.1978 / −0.5872–+0.3352 / −0.3894–+0.2625**，RR **−0.0805–+0.0221 / −0.0941–+0.1772 / −0.0964–+0.1395**。Sector TT 的 E 项始终为负、I 项始终为正，但总 `ΔI^gate` 依 sector 变号（**−0.0350–+0.0912**；最小 `mean−SEM=−0.0554`），故其跨 sector 平均的正净贡献不能写成“每一 sector 均 disinhibition”。相反，RT 总 `ΔI^gate` 九个 sectors 均为正（**+0.0868–+0.3800**，最小 `mean−SEM=+0.0248`）。所有 condition-level mean ± SEM 与 seed-level 值见 sector long CSV 与 sector Supple4。 |
| 6.18 | Fig8 的 per-connection `ΔI^gate` 四个正文填空 | **剔除 reset frame 后**；每 seed 先跨条件平均、再做 10-seed mean ± SEM，单位均为 per nonzero recurrent connection。Digit TT：`W > 0` **−0.01339 ± 0.00063**（`[VAL-8a]`）；`W < 0` **+0.05628 ± 0.00260**（`[VAL-8b]`）；total **+0.02461 ± 0.00149**（`[VAL-8c]`）。Sector TT total **+0.00200 ± 0.00032**（`[VAL-8d]`）。方向与正文预期一致：两种条件下 W>0 均为负、W<0 均为正，total 为正；sector total 约为 digit total 的 **8.1%**。各 n=10。 |
| 6.16 | §6 10-seed 完成状态 | **reset-excluded 10-seed 数值完成：6.3–6.6、6.11、6.13、6.15、6.17 与 6.18。** 所需 recurrent-current panels 已并入 ICLR Fig7；standalone Fig8/Supplementary 4 不再列为 retained figures。6.9 已按用户要求删除，不计入完成状态。 |

Table 15 按五位小数显示 W+ / W− means，并按显示后的两行相减生成 gap。因此正式表内 gap 为：
Digit TT **+0.09916**、TR **−0.07210**、RT **−0.01478**、RR **+0.00715**；Sector TT
**−0.01209**、TR **−0.01181**、RT **−0.00964**、RR **+0.00350**。其中 Digit RT 与
Sector TR 分别和未取整的 per-seed gap mean 四舍五入值相差 $10^{-5}$；统计检验始终使用未取整值。

### 6.18 完整 per-connection current 表

下表补齐合并 ICLR Fig7 所需的全部 24 个 bar 值。单位、聚合与 6.18 相同；`Total` 是按该组
W+/W− 实际连接数加权的 per-connection mean，不是两列相加。

| Variable | Group | W>0 | W<0 | Total |
|---|---|---:|---:|---:|
| Digit | TT | −0.01339 ± 0.00063 | +0.05628 ± 0.00260 | +0.02461 ± 0.00149 |
| Digit | TR | −0.00876 ± 0.00043 | −0.00005 ± 0.00063 | −0.00313 ± 0.00041 |
| Digit | RT | −0.00317 ± 0.00013 | +0.01147 ± 0.00040 | +0.00641 ± 0.00024 |
| Digit | RR | −0.00221 ± 0.00007 | +0.00404 ± 0.00011 | +0.00166 ± 0.00007 |
| Sector | TT | −0.00983 ± 0.00044 | +0.01372 ± 0.00057 | +0.00200 ± 0.00032 |
| Sector | TR | −0.00704 ± 0.00026 | +0.01316 ± 0.00047 | +0.00534 ± 0.00021 |
| Sector | RT | −0.00255 ± 0.00011 | +0.00362 ± 0.00022 | +0.00121 ± 0.00009 |
| Sector | RR | −0.00225 ± 0.00007 | +0.00374 ± 0.00013 | +0.00150 ± 0.00007 |

相应的 nonzero recurrent connection counts 也不固定为旧的 `576/5,568/5,568/53,824`，
因为实际 $|T|$ 为 23–25。下表按每 seed 先对 contexts 等权平均、再跨 10 seeds 报 mean ± SEM；
E/I 分别对应 `W>0`/`W<0`。

| Variable | Group | E connections | I connections | I share（仅 TT） |
|---|---|---:|---:|---:|
| Digit | TT | 269.47 ± 5.05 | 311.63 ± 5.62 | 53.6273% ± 0.5321% |
| Digit | TR | 1,975.87 ± 17.63 | 3,612.63 ± 29.06 | — |
| Digit | RT | 1,945.01 ± 20.46 | 3,643.49 ± 34.58 | — |
| Digit | RR | 20,543.15 ± 136.70 | 33,234.75 ± 129.55 | — |
| Sector | TT | 288.92 ± 5.56 | 292.18 ± 5.05 | 50.2880% ± 0.5348% |
| Sector | TR | 2,160.41 ± 24.93 | 3,428.09 ± 28.47 | — |
| Sector | RT | 2,172.77 ± 28.02 | 3,415.73 ± 37.66 | — |
| Sector | RR | 20,111.40 ± 128.12 | 33,666.50 ± 115.38 | — |

## 附 — 已删除或降级主张的当前处理

| 位置 | 原主张 | request 中的现状 | 本记录处理 |
|---|---|---|---|
| §3 末 / 转场 T3 | context 改变哪些突触打开，但不改变打开多少 | 已删除 | 不恢复；本记录只报告 3.1、3.2、3.6。 |
| §3 第 2 段 | 27% 通过 / 73% 清零、norm ratio 0.55 | 已删除 | 不恢复；这些不是本次请求的保留统计。 |
| §5 第 4 段 | 两条曲线跨 \|W\| 平行 | 附条件保留，5.9c 已完成 | matching 与 other 中正负曲线 slopes 均不同方向，因此不支持“平行”这一无条件表述。 |
| §6 第 1 段 | 四组具体 delta 值 | 已完成 6.3–6.5 | 已由 sjc-remote formal 10-seed summary 补齐，见 6.3–6.5。 |

## 已使用的数据源

- Local：
  `results/data/analysis/G_behaviour/data_scale_comparison/data_scale_summary_10seed.json`
  与 `data_scale_mean_sem_10seed.csv`
  （Appendix B 跨 4h/10h/20h/40h、六模型、10-seed train/validation/gap 完整表）
- Current implementation：`utils/training/recurrent_cores/gawf.py`、
  `utils/training/clutter/clutter_task_models.py`、`source/clutter/generate_movies.py` 与
  `source/GenerateMovies_joint_balanced.py`（Appendix A、C 的架构、feedback 与数据生成口径）

- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final/reset_excluded_test_accuracy_10seed.csv`
  （§1 formal test accuracy；每个 32-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig1_target_switch_recovery_resetexcluded_6model_10seed_v4/`
  （Fig1 formal recovery；独立 joint-balanced test dataset、每个 32-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/supple1_feedback_ablation_resetexcluded_10seed_v5/`
  （Supplementary 1 formal recovery；每个 512-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_shuffle_activation_anova_10seed/final/Fig4_shuffle_activation_anova_long_10seed.csv`
  （§2 formal baseline / shuffle-digit / shuffle-sector accuracy；§4.12、512-frame、排除 `t=0`）
- `results/save_data/fig3/seed*/gawf_gate_distribution_meta.json`
- `results/save_data/fig3/seed*/gawf_gate_distribution_stats.npz`
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig3_gate_half_mass_10seed/fig3_gate_half_mass.json`
  （§3.5 十个 seed 的 0.5 点质量、reset 来源、点质量剔除后的中间占比与 `U` row audit）
- `results/save_data/fig5/unit_gate_context_variance_multiseed.json`
- `results/save_data/supple1/feedback_ablation/` 是旧 single-seed、未排除 window-initial frame
  的 historical leaf（`n_frames=57,568`，且缺少 exclusion provenance），formal summary loader
  会拒绝它；不得用于正文、Appendix 或当前 Supplementary recovery。
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_activation_anova_6model_10seed_residual_resetexcluded/gawf-seed*/activation_anova.npz`
  （§4.1–4.3、Fig4 core activation；32-frame、排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_gate_synapse_anova_resetexcluded_10seed/seed*/gate_synapse_anova.npz`
  （§4.4–4.5 raw-synapse gate；32-frame、排除 `t=0`、20-draw balanced ANOVA）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_encoder_patterns_resetexcluded_gawf_10seed/`
  （Fig6 encoder Sector/Digit patterns；32-frame、每 seed 排除 1,799 个 `t=0`，随后做 equal-n selection）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_reset_excluded_10seed_r3/final/fig7_seed_level_summary.npz`
  （§6.3–6.6 与 §6.15 的 reset-excluded seed-level sign gaps）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_resetexcluded_stats_r3/supple3_seed_level_sign_magnitude_stats.json`
  （§6.11 的 10-seed 分符号 slopes、SEM 与 overall delta levels）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_reset_excluded_10seed_r3/seed*/compact/recurrent_gate_condition_means.npz`
  （只读计算 tuned-mask counts、hidden size 与 6.13 observed digit-pair overlap）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_10seed/seed*/selectivity/part1_selectivity.npz`
  （6.13 的每 seed FDR-eligible、non-interaction-dominant pool，用于 conditional independent-set chance baseline）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_net_recurrent_current_resetexcluded_10seed_r2/seed*/net_recurrent_current.npz`
  与 `final/net_recurrent_current_10seed_long.csv`、`final/net_recurrent_current_10seed_summary.npz`
  （§6.17 的 10-seed × digit × group × sign `I`、`ΔI`、`ΔI^gate`；逐帧实际
  `g·W·h(t−1)`，reset-excluded、按目的地单元数归一）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_net_recurrent_current_sector_resetexcluded_10seed/seed*/net_recurrent_current.npz`
  与 `final/net_recurrent_current_10seed_long.csv`、`final/net_recurrent_current_10seed_summary.npz`
  （§6.17 的 10-seed × sector × group × sign `I`、`ΔI`、`ΔI^gate`；逐帧实际
  `g·W·h(t−1)`，reset-excluded、按目的地单元数归一）
- sjc-remote：
  `results/save_data/fig8/recurrent_current/connection/{digit,sector}/`
  `net_recurrent_current_connection_10seed_long.csv` 与
  `Fig8_recurrent_current_connection_caption_stats.json`
  （§6.18：从 §6.17 retained current means 转为 per nonzero recurrent connection；`W > 0` /
  `W < 0` 各按自身连接数平均，total 按全部非零连接数加权；long CSV 保留 condition-level
  记录，caption stats 的 `bar_summary[].seed_values` 保留作图使用的 10 个 seed-level 均值）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/supple2_input_gate_sign_magnitude_9sector_10seed/`
  （10-seed × 9-sector §5.9 compact results 与 seed-level stats）

上述 remote 数值是此前核验并记录的固定结果，不代表本轮重新访问成功。本轮已在本地重新核验
Fig1 recovery v4 的 60 个 metadata 均有 `exclude_window_initial_frame=true`，且 loader 产生
20 个 offsets（`pre10`–`pre1`、`post1`–`post10`）；当前本地没有 Supplementary recovery v5
数值叶。SSH control socket 当前不存在，因此 Supplementary recovery v5 只保留此前记录，
不宣称完成了本轮 live re-audit。Mamba/S5 已按当前 dependency parameter shapes 重新计算，
但 Amarel 正在维护，训练节点 metrics 与 cross-scale scheduler GPU-hours 交叉核验均暂缓；
Appendix B 的 2×3 合并图已从当前本地 mean/SEM CSV 生成并完成 PDF render 检查。
Supplementary figure 的最终编号暂不冻结，本文均以 analysis content 与当前代码 entry point 标识。

## 历史对照：未排除每个 window 的 t=0 的 accuracy（不作为正式结果）

以下数值保留以便和旧图/旧文字比对，**不得作为正文正式 accuracy 引用**。它们没有剔除每个
recurrent window 的 `t=0`；而同一帧的 gate 是 zero-feedback artifact 的 0 值，故它们与 gate
统计口径不一致。loss curves 未受这一替换影响。

| 原统计 | 未排除 t=0 的历史值（10-seed mean ± SEM） |
|---|---|
| 旧 §1.2 GaWF canonical test | Digit **85.8765% ± 0.1466%**；Sector **92.7826% ± 0.1199%**。 |
| 旧 §1.3 RNN / LSTM / GRU / S5 / Mamba canonical test | RNN：**80.1815% ± 0.1721% / 90.8102% ± 0.0870%**；LSTM：**79.7103% ± 0.2268% / 90.3591% ± 0.0637%**；GRU：**78.7389% ± 0.1708% / 90.3415% ± 0.0568%**；S5：**74.6588% ± 0.3558% / 88.7969% ± 0.1787%**；Mamba：**82.4159% ± 0.1635% / 92.0645% ± 0.0718%**；顺序均为 Digit / Sector。 |
| 旧 §2.2 shuffle Digit | Baseline **85.8765% ± 0.1466%**；shuffle-sector **54.1572% ± 0.8730%**；shuffle-digit **73.3829% ± 0.4910%**。 |
| 旧 §2.3 shuffle Sector | Baseline **92.7826% ± 0.1199%**；shuffle-sector **62.9858% ± 0.7528%**；shuffle-digit **91.7324% ± 0.1464%**。 |
