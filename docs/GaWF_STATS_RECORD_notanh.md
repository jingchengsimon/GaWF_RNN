# GaWF no-tanh 正文与 Appendix 统计记录

更新日期：2026-09-24

## 口径与范围

本文件是当前 ICLR reported model 的数值事实源。`GaWF` 在本文中特指
`gawf_legacy_notanh` checkpoint 所对应的当前公开实现：

$$
z_t=(G_t^{ih}\odot W_{ih})x_t+(G_t^{hh}\odot W_{hh})h_{t-1}+b_{ih}+b_{hh},
\qquad
h_t=\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{LayerNorm}(z_t))).
$$

这里没有 bounded inner activation；$h_t$ 同时是 recurrent/state-space layer output 与
下一时刻 recurrence input。行为汇总图按 panel 区分 protocol：A 为 standard movie、32-frame
rollout，C 为 joint-balanced movie、32-frame rollout，D 为 standard movie、512-frame rollout。
其余统计除非另行标注，均来自 standard test movie、32-frame rollouts；所有统计都剔除每个
rollout 的 `t=0` reset frame。
所有 `mean ± SEM` 均先在每个 training seed 内完成 condition/connection aggregation，再以
seeds 1–10 为独立重复计算 $s/\sqrt{10}$。`Digit` 等同 manuscript 的 target identity，
`Sector` 等同 target location。

本记录不混入以下结果：含 `tanh(z_t)` 的 earlier GaWF、data-scale control、width-128
control、旧 wrapped baselines，以及未完成十 seeds 的开发结果。历史实现与历史数值只保留在
`docs/GaWF_STATS_RECORD.md`，不得用于当前 manuscript 填数。

### 正式 protocol 总表

| 分析 | Movie | Rollout | Reset | Seeds |
|---|---|---:|---|---:|
| test accuracy、activation ANOVA、gate ANOVA | standard | 32 | 每个 window 排除 `t=0` | 10 |
| target-switch recovery | joint-balanced | 32 | 每个 window 排除 `t=0` | 10 |
| feedback shuffle | standard | 512 | 每个 window 排除 `t=0` | 10 |
| gate distribution、sign/magnitude、recurrent current | standard | 32 | 每个 window 排除 `t=0` | 10 |

## Appendix A — 当前六模型定义与配置

### A.1 从 encoder 到 readout 的 data flow

Clutter encoder 的输出固定为 `(32, 6, 6)`，按 channel-major order flatten 为 1152 features。
当前六模型的 recurrent/state-space layer 为：

- **GaWF**：上述 feedback-gated affine preactivation；`LayerNorm → ReLU → Dropout(0.5)`
  位于 recurrence 内，且不含 `tanh`。
- **RNN**：$z_t=W_{ih}x_t+W_{hh}h_{t-1}+b_{ih}+b_{hh}$，随后使用与 GaWF 完全相同的
  in-loop `LayerNorm → ReLU → Dropout(0.5)`。
- **GRU/LSTM**：直接使用 native `nn.GRU` / `nn.LSTM` layer output，无额外
  `LayerNorm → ReLU → Dropout` wrap。
- **Mamba/S5**：使用各自 imported block 的 projected residual stack output，无额外 readout
  wrap。

GaWF 的 detached feedback 是上一时刻 Digit 与 Sector 两个 linear heads 的 raw logits 拼接，
维数为 19。令 $f_{t-1}$ 为 clamp 到 $[-10,10]$ 的 feedback，

$$
M_t=(U\odot f_{t-1})V,\qquad G_t=\sigma(M_t/0.5),
$$

再沿 source dimension 拆成 $G_t^{ih}$ 与 $G_t^{hh}$。当前 single-layer model 未使用 projector；
公开 CLI 仍保留 `--dz`。

### A.2 Parameter-matched formal configuration

Total 是完整 trainable model（encoder + recurrent/state-space core + 两个 heads）。

| Model | Width/state | Total | LR | WD | 特殊 optimizer rule |
|---|---:|---:|---:|---:|---|
| RNN | 275 | 586,865 | 0.001 | 0.00001 | — |
| LSTM | 80 | 584,515 | 0.001 | 0.001 | — |
| GRU | 105 | 586,695 | 0.005 | 0.001 | — |
| GaWF | 256 | 586,067 | 0.005 | 0.001 | `U/V/projector` 无 WD，LR=`1×` base LR |
| Mamba | 170 | 586,935 | 0.001 | 0.001 | — |
| S5 | 256（state 128） | 586,963 | 0.001 | 0 | S5 core LR=`0.1×` base LR |

以上 LR/WD 与本批正式 checkpoints 的训练配置一致。六模型均训练 150 epochs、patience 0；
checkpoint selection 以 validation Digit accuracy 最大为准。训练 objective 是逐 frame Digit 与
Sector `CrossEntropyLoss` 的等权和。

## Appendix B — Hyperparameter provenance

当前 no-tanh GaWF、in-loop ReLU RNN 与四个 no-wrap baselines 没有分别重做独立
hyperparameter search。本批正式训练冻结并复用 earlier architecture search 选出的 LR/WD，
再以更新后的 recurrent semantics 运行 seeds 1–10；width/state 则重新按完整 trainable model
parameter count 对齐，结果列于 Appendix A.2。因而本记录只报告本批 checkpoints 的实际正式
配置，不把旧 `tanh` / wrapped architectures 的 search validation score 当作当前模型的
hyperparameter-search 结果。

`docs/GaWF_STATS_RECORD.md` Appendix B.2 的 4h/10h/20h/40h cross-scale 数值来自 earlier
architectures；当前 no-tanh/no-wrap models 尚无对应完整 cross-scale rerun，故该表和
`data_scale_performance_2x3_10seed.pdf` 不迁入本记录。

## Appendix C — CM-MNIST evaluation datasets

standard train/validation/test 使用同一生成算法和分布、不同随机 realization：96×96、24 fps，
40h train 有 3,456,000 raw frames，validation/test 各 57,600 raw frames。MNIST exemplar
indices 分别来自 `[0,40000)`、`[40000,50000)`、`[50000,60000)`。每个 frame 的监督标签为
foreground Digit（10 classes）与 coarse 3×3 Sector（9 classes）。

joint-balanced movie 也是独立 held-out test realization。它在每帧令 foreground 加九个
backgrounds 恰好覆盖 digits 0–9，并显式平衡 identity/location switches。它仅用于
target-switch recovery；feedback-shuffle 使用 standard movie，不与 joint-balanced recovery
混报。

## Appendix D — Training、selection 与 compute

| 项目 | 当前 formal protocol |
|---|---|
| Objective | 每 frame 的 Digit CE 与 Sector CE 等权相加；均在 batch×time 上取 mean。 |
| Optimizer | `AdamW`；GaWF feedback factors 使用上表独立 parameter group；无 LR scheduler。 |
| Temporal protocol | truncated BPTT，rollout=32；每个 window 重置 recurrent state 与 feedback。 |
| Batch | batch size 256，gradient accumulation 1，train `drop_last=True`。 |
| Duration | 150 epochs，patience 0；每 5 个 completed epochs 原子 checkpoint。 |
| Acceleration | CUDA AMP/GradScaler；value clipping 1 与 global norm clipping 1。 |
| Selection | validation Digit accuracy 最大的 checkpoint；不是 minimum validation loss。 |

40h train 的 `MovieDataset` 有 107,999 nominal windows；每 epoch 为 421 optimizer steps，
实际参与一次 train pass 的 output frames 为 3,448,832。正式运行使用一张 GPU；checkpoint、
metrics 与 150-epoch history 三者匹配后才计为完成。

## Appendix E / 正文 §3.1 — 行为结果

### E.1 Standard movie test accuracy

每个 seed 保留 55,769 个 reset-excluded frames；以下 rollout=32。

| Model | Digit / identity (%) | Sector / location (%) |
|---|---:|---:|
| GaWF | **86.3340 ± 0.1600** | **93.0908 ± 0.1035** |
| RNN | 83.5504 ± 0.1598 | 91.8240 ± 0.1231 |
| LSTM | 80.6513 ± 0.2154 | 90.8200 ± 0.0869 |
| GRU | 79.8141 ± 0.1987 | 90.6735 ± 0.0836 |
| Mamba | 82.2826 ± 0.1849 | 92.4363 ± 0.1455 |
| S5 | 69.5307 ± 0.2879 | 86.3227 ± 0.2354 |

在这两个指标上，GaWF 的 10-seed mean 均高于其余五个当前 baselines。该句只描述跨 seed
mean 的排序，不等价于“每一个 seed 都胜过所有 baseline seeds”。

target-switch recovery 使用独立 joint-balanced test movie、rollout=32，显示完整
`pre10…pre1` 与 `post1…post10` 轨迹。曲线是 seed mean ± SEM，不压缩为未经预定义的单一
recovery scalar。

### E.2 Feedback-shuffle ablation

该分析只对 GaWF 运行。movie 为 standard `40h-uint8`，rollout=512，每个 seed 保留 57,232
frames。
shuffle 在每个 rollout 内、对每个 sample 独立 permute 指定 feedback slice；未被 shuffle 的
slice 保持 live-rollout 值。

| Condition | Digit / identity (%) | Sector / location (%) |
|---|---:|---:|
| Baseline | 89.7162 ± 0.1399 | 94.0846 ± 0.1056 |
| Shuffle digit feedback | 83.5019 ± 0.3513 | 92.5196 ± 0.0780 |
| Shuffle sector feedback | 52.2896 ± 0.4713 | 60.5925 ± 0.5282 |
| Shuffle both | 50.4968 ± 0.4500 | 61.5556 ± 0.6041 |

相对同一 512-frame baseline，shuffle digit / shuffle sector 导致的 percentage-point drop 为：
Digit **6.2144 / 37.4266**，Sector **1.5650 / 33.4921**。A 与 D 虽然都使用 standard movie，
仍分别是独立的 32-frame 与 512-frame recurrence rollouts；不得把两个 baseline 当作同一
protocol。

## Appendix F / 正文 §3.2 — Gate distribution

每个 seed 的 trajectory 有 57,568 frames，其中每个 32-frame window 的 `t=0` 共 1,799
frames；正式 gate distribution 使用其余 55,769 frames。median 是先对每个 seed 的完整
gate population 求 exact median，再跨 seeds 汇总。端点比例直接以 eager float32 公式重建 gate，
严格使用 $g<0.1$ 与 $g>0.9$；恰好等于 0.1 或 0.9 的值单独计数，未混入端点。

| Gate | Exact median | $g<0.1$ (%) | $g>0.9$ (%) |
|---|---:|---:|---:|
| Input | 0.000710 ± 0.000154 | 67.5857 ± 0.3790 | 20.6975 ± 0.4582 |
| Recurrent | 0.996497 ± 0.001657 | 18.9434 ± 0.2991 | 66.0896 ± 0.9847 |

阈值相等项极小：input 的 $g=0.1$ / $g=0.9$ 为
0.000000769% ± 0.000000041% / 0.000002532% ± 0.000000081%；recurrent 为
0.000000646% ± 0.000000053% / 0.000004881% ± 0.000000330%。因此 closed/open 的方向不依赖
边界约定：input gate 明显偏 closed，recurrent gate 明显偏 open。

以 $|g-0.5|<10^{-6}$ 定义 point mass。包含 reset frames 时，input / recurrent gate 的占比为
3.125020447% ± 0.000000688% / 3.125026789% ± 0.000002057%；每个 seed 的 reset-frame count
严格为 1,799/57,568 = 3.125%。reset 外的同一判据仅占全部 gate samples 的
0.000020447% ± 0.000000688% / 0.000026789% ± 0.000002057%，且 256 个 destination units 中
不存在 $max_r|U[j,r]|<10^{-6}$ 的 feedback-dead row。因此 0.5 spike 来自 sequence reset，
不是固定 dead synapses。

## Appendix G / 正文 §3.3 — Variance decomposition

condition-mean variance 先在每个 seed 内对 90 个 Sector×Digit cells equalize，并对 20 个
fixed draws 取均值；Sector、Digit、Interaction 归一化后合计 100%，不含 trial-level residual。
以下顺序均为 **Sector / Digit / Interaction**。

### G.1 六模型 encoder 与 recurrent/state-space layer output

| Model | Encoder output (%) | Recurrent/state-space output (%) |
|---|---|---|
| GaWF | 56.2242 ± 0.6221 / 6.0980 ± 0.0893 / 37.6778 ± 0.5338 | 21.9414 ± 0.4466 / 62.5528 ± 0.5325 / 15.5058 ± 0.1127 |
| RNN | 59.1280 ± 0.2824 / 5.9141 ± 0.0483 / 34.9579 ± 0.2505 | 37.3275 ± 0.2242 / 41.6590 ± 0.1945 / 21.0135 ± 0.1180 |
| LSTM | 64.9236 ± 0.2605 / 5.1626 ± 0.0475 / 29.9137 ± 0.2138 | 57.8012 ± 0.3348 / 33.5392 ± 0.2886 / 8.6596 ± 0.2021 |
| GRU | 58.2688 ± 0.2395 / 5.9239 ± 0.0317 / 35.8073 ± 0.2094 | 65.9400 ± 0.2075 / 24.2104 ± 0.1169 / 9.8496 ± 0.1259 |
| Mamba | 64.1030 ± 0.2618 / 5.1180 ± 0.0455 / 30.7790 ± 0.2185 | 38.9705 ± 0.5126 / 56.0096 ± 0.4611 / 5.0198 ± 0.1033 |
| S5 | 62.8372 ± 0.4449 / 5.3838 ± 0.0698 / 31.7790 ± 0.3787 | 47.7020 ± 0.7518 / 47.2816 ± 0.7262 / 5.0164 ± 0.0978 |

### G.2 GaWF synapse gates

| Gate | Sector / Digit / Interaction (%) |
|---|---|
| Input gate | **73.6913 ± 0.5181 / 19.0233 ± 0.4494 / 7.2854 ± 0.0924** |
| Recurrent gate | **20.0625 ± 0.4709 / 72.6035 ± 0.4406 / 7.3340 ± 0.1040** |

这里的 unit axis 是 raw synapse，不是 destination neuron。每个 seed 使用全部 55,769 个
reset-excluded frames 与相同的 20 draws。

### G.3 Destination-unit gate projection

GaWF 先对每个 destination unit 的所有 incoming synapse gates 取 arithmetic mean；GRU/LSTM
使用 native unit gate。各项仍以 balanced condition-mean total variance 为分母。

| Model/gate | Sector / Digit / Interaction (%) |
|---|---|
| GaWF input | 82.7304 ± 1.6420 / 14.9773 ± 1.6259 / 2.2923 ± 0.0665 |
| GaWF recurrent | 46.1261 ± 1.8287 / 49.8073 ± 1.8945 / 4.0666 ± 0.1224 |
| GRU reset | 62.3329 ± 0.7352 / 11.1280 ± 0.6895 / 26.5390 ± 0.4545 |
| GRU update | 58.5840 ± 0.7111 / 20.0758 ± 0.4811 / 21.3402 ± 0.4454 |
| LSTM forget | 64.3299 ± 0.5861 / 17.1541 ± 0.5462 / 18.5160 ± 0.4627 |
| LSTM input | 54.2853 ± 0.5045 / 21.0917 ± 0.7965 / 24.6230 ± 0.4358 |
| LSTM output | 55.4781 ± 0.6217 / 17.7065 ± 0.7169 / 26.8154 ± 0.5377 |

## Appendix H / 正文 §3.4 — Input-gate spatial/sign organization

encoder flatten order 为 `(channel=32, row=6, col=6)`。每个 3×3 Sector 在每个 channel 上
对应不重叠的 2×2 block，故有 128 个 matched sources 与 1,024 个 other sources。每个 seed
先在各 sector 内汇总 connection，再对九个 sectors 等权平均；W+ / W− 使用该 seed 的共同
$|W|$ support。

| Source group | W+ mean $\Delta g^{in}$ | W− mean $\Delta g^{in}$ | W+−W− | All connections |
|---|---:|---:|---:|---:|
| Matched | 0.35304 ± 0.00483 | 0.36579 ± 0.00496 | −0.01274 ± 0.00098 | 0.35950 ± 0.00488 |
| Other | −0.04413 ± 0.00061 | −0.04572 ± 0.00061 | 0.00159 ± 0.00013 | −0.04494 ± 0.00061 |

逐 seed OLS `Δg ~ |W|` slopes 为：matched W+ **−0.00821 ± 0.00465**、W−
**0.01749 ± 0.00228**；other W+ **0.00103 ± 0.00058**、W−
**−0.00219 ± 0.00029**。这些是 descriptive seed-level summaries，不使用 pooled
connection-level p values。

## Appendix I / 正文 §3.5 — Recurrent-gate modulation and current

T/R masks 来自 validation split 的 hidden-activation selectivity。每个 factor 对 per-unit
permutation p values 做 Benjamini–Hochberg FDR；interaction-dominant units 排除后，每个
context 的 T 取 eligible pool 中 tuning 最高的 `ceil(0.1×|eligible|)`，R 是其余 hidden units。
H=256；seeds 1–10 的 |T| 为 25–26，|R| 为 230–231，digit eligible pool 为 249–254。

十个 digit T sets 的 45 个 pair normalized overlap
$|T_a\cap T_b|/\sqrt{|T_a||T_b|}$ 先在 seed 内平均，再跨 seeds 为
**9.7730% ± 0.2072%**；固定 observed |T| 并从同一 eligible pool 独立抽取的 chance baseline
为 **10.2981% ± 0.0316%**，observed−chance = **−0.5251 ± 0.2102 percentage points**。
45 个 pairs 的跨-seed mean 范围为 **3.08%–17.75%**（digit 6–7 至 digit 1–7）。

下表是 shared-$|W|$ support 上的 $\Delta g^{rec}$；每个 seed 先对 Digit 或 Sector conditions
等权平均，再跨十 seeds 汇总。

| Variable/sign | T→T | T→R | R→T | R→R |
|---|---:|---:|---:|---:|
| Digit W+ | −0.01765 ± 0.00501 | −0.02816 ± 0.00231 | −0.00393 ± 0.00094 | 0.00376 ± 0.00028 |
| Digit W− | −0.15749 ± 0.01209 | −0.02233 ± 0.00290 | −0.00500 ± 0.00167 | 0.00441 ± 0.00039 |
| Digit gap | 0.13984 ± 0.00833 | −0.00583 ± 0.00291 | 0.00107 ± 0.00162 | −0.00065 ± 0.00040 |
| Sector W+ | −0.06505 ± 0.00289 | −0.04923 ± 0.00219 | −0.00526 ± 0.00074 | 0.00702 ± 0.00032 |
| Sector W− | −0.06014 ± 0.00298 | −0.04999 ± 0.00171 | 0.00545 ± 0.00075 | 0.00567 ± 0.00019 |
| Sector gap | −0.00491 ± 0.00215 | 0.00076 ± 0.00102 | −0.01072 ± 0.00066 | 0.00135 ± 0.00020 |

all-connections $\Delta g^{rec}$ levels（不限制 overlap band）为：Digit T→T / T→R / R→T /
R→R = **−0.06135 ± 0.00831 / −0.02526 ± 0.00217 / −0.00427 ± 0.00110 /
0.00410 ± 0.00027**；Sector = **−0.06273 ± 0.00267 / −0.04954 ± 0.00189 /
0.00047 ± 0.00062 / 0.00632 ± 0.00024**。Digit T→T 的两个 weight signs 均 close，
但主要来自 W− subgroup。对应 `Δg ~ |W|` slope 为 W+ **0.03710 ± 0.00701**、W−
**−0.14743 ± 0.01923**；negative-weight slope 保持负号。

per-nonzero-connection $\Delta I^{gate}$ 为：

| Variable/sign | T→T | T→R | R→T | R→R |
|---|---:|---:|---:|---:|
| Digit W+ | −0.00888 ± 0.00111 | −0.01232 ± 0.00098 | −0.00524 ± 0.00031 | −0.00722 ± 0.00032 |
| Digit W− | 0.08621 ± 0.00461 | 0.01704 ± 0.00229 | 0.01917 ± 0.00103 | 0.01174 ± 0.00052 |
| Digit All | 0.02137 ± 0.00192 | 0.00317 ± 0.00145 | 0.00810 ± 0.00062 | 0.00243 ± 0.00039 |
| Sector W+ | −0.01972 ± 0.00117 | −0.01608 ± 0.00093 | −0.00679 ± 0.00032 | −0.00624 ± 0.00032 |
| Sector W− | 0.02602 ± 0.00130 | 0.02912 ± 0.00157 | 0.01211 ± 0.00066 | 0.01139 ± 0.00062 |
| Sector All | 0.00019 ± 0.00085 | 0.00720 ± 0.00079 | 0.00310 ± 0.00051 | 0.00279 ± 0.00047 |

这里 `All` 以 W+/W− 的实际 nonzero connection counts 加权，不是两个 sign rows 的算术平均。
按 destination unit 求和的 $\Delta I^{gate}$（W+ / W− / total）为：Digit T→T
**−0.1512 ± 0.0177 / 0.7053 ± 0.0612 / 0.5542 ± 0.0506**，T→R
**−0.1526 ± 0.0140 / 0.2346 ± 0.0337 / 0.0821 ± 0.0377**，R→T
**−0.5439 ± 0.0296 / 2.4085 ± 0.1525 / 1.8646 ± 0.1417**，R→R
**−0.8187 ± 0.0462 / 1.3790 ± 0.0755 / 0.5603 ± 0.0905**。Sector 对应为 T→T
**−0.2908 ± 0.0192 / 0.2957 ± 0.0166 / 0.0049 ± 0.0222**，T→R
**−0.2021 ± 0.0120 / 0.3887 ± 0.0241 / 0.1866 ± 0.0206**，R→T
**−0.7497 ± 0.0433 / 1.4630 ± 0.0942 / 0.7133 ± 0.1177**，R→R
**−0.7050 ± 0.0456 / 1.3466 ± 0.0895 / 0.6415 ± 0.1089**。

total per-destination ordering 为 Digit **R→T > R→R > T→T > T→R**、Sector
**R→T > R→R > T→R > T→T**。该 current 是保持实际 $h_{t-1}$ 的瞬时 gate contribution
decomposition，不是冻结 gate 后重跑完整动力学的 counterfactual。

## 正式可视化

`results/save/Figures_notanh/` 中当前模型范围的正式 PDF 为：

- `best6_multiseed_shuffle_2x4_seq512.pdf`
- `activation_anova_1x2_6model_10seed.pdf`
- `gate_task_variable_specialization_2x3_10seed.pdf`
- `gate_and_weight_distributions_1x4_10seed.pdf`
- `overall_sector_input_gate_1x3_10seed.pdf`
- `input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.pdf`
- `rec_gate_disinhibit_and_current_2x3_10seed.pdf`
- `rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf`
- `recurrent_current_unit.pdf`
- `gawf_and_cm_mnist.pdf`（static schematic）

`best8_multiseed_noshuffle_training_test_2x4_seq32*` 混合 old/new recurrent semantics，已由当前
六模型行为图取代；`data_scale_performance_2x3_10seed.pdf` 与
`clutter_4h_h128_multiseed_2x4.pdf` 属于本记录明确排除的 data-scale / width controls，因没有
当前 no-tanh checkpoints 而不复制或改名冒充当前结果。

## 结构化事实源与完整性

当前六模型 merge root 为
`results/data/analysis/current6_notanh_baselines_refresh_20260924_v1/`：60 个 test-accuracy rows、
60 个 activation units、60 个 recovery units、10 个 GaWF gate-synapse units、10 个 GaWF
trajectories、60 个 training histories，以及 10 个历史 joint-balanced/512 shuffle units；后者
保留作 compatibility control，不用于当前 Figure D。Mamba 与 S5 no-wrap 均为 seeds 1–10
完整覆盖。其主要事实源 SHA-256 为：

| File | SHA-256 |
|---|---|
| `manifest.json` | `b97ec35801a60150d2f387bacad4f8007febdaefa8a4f32b76547737cee55e85` |
| `current6_statistics.json` | `b08c612c0a5e314c69a41f5e44093c17985f5e85f063ef333015e6021ce66e7d` |
| `test_accuracy_current6_10seed.csv` | `92846f74bb2f8819226d6d172a7a89d55dc97102d3058cb8e73e559fddcdea52` |
| `unit_gate/unit_gate_context_variance_multiseed.json` | `4a690c91bc1f609b00ddeaa31df34d7e3dde9a572b169d8d4417da71631851cf` |

当前 Figure D 的独立事实源为
`results/data/analysis/gawf_legacy_notanh_feedback_shuffle_standard_resetexcluded_seq512_10seed_v1/`。
其中 10 个 `gawf-seed*/ablation_metrics.json` 均记录 `data_suffix=40h-uint8`、
`sequence_length=512`、`exclude_window_initial_frame=true` 与四个完整 conditions；汇总 manifest
`source_manifest.json` 的 SHA-256 为
`97286eafb6c4b48fad6b0eca9b8de530578412f809d7c9d458ab5eedb560d829`。

GaWF-only 严格 refresh root 为
`results/data/analysis/gawf_legacy_notanh_refresh_10seed_v4/`；十个 `seedNN.complete.json`、
`status/all.done` 与 final `.complete` 均存在。主要事实源为：

| File（均相对 v4 root） | SHA-256 |
|---|---|
| `final/fig3/fig3_gate_half_mass_notanh.json` | `0b9303c5f1c17d17ea8a35378d540d6a531e07c782fd67d162d834cd441136e9` |
| `final/fig3/gate_and_weight_distributions_1x4_10seed_notanh.json` | `cbb11d3eb15846e430f173f440b3e52442920dc0d2d22278b024e563fdbdf82c` |
| `final/fig3/fig3_gate_endpoint_fractions_strict.json` | `885f259663ba4c254f9b305356d64e40dae6cb3827aa5fb2d0a56d7afac3351f` |
| `final/supple2/Supple2_input_gate_sign_vs_mag_9sector_10seed_stats.json` | `9f70c159feeb066362cec7a0da27347613d9bde57c3ddf9d214ed426a2416278` |
| `final/fig7/supple3_seed_level_sign_magnitude_stats.json` | `6cf0fb1c3f81a7d77d7fc8122f2c6ff21ad9757f2d6c7835b07a1259550a533c` |
| `final/fig7/fig7_seed_level_summary.npz` | `039bba087c3c8dff3b7779b2862d44b0b043ded2ba761f405fcf08af9a99139a` |
| `final/current_records/Supple4_recurrent_current_unit_caption_stats.json` | `40d22ba85b520c6ca1ace807bf2e41b8a61bd3c226c7d24b88c468d03ed21a86` |
| `final/current_records/Fig8_recurrent_current_connection_caption_stats.json` | `d6b9b47ea323bc252c66af9fbae7bd74dcd5e8402eb4bbcb95352475c6b16f1f` |

所有表中数值均来自结构化 CSV/JSON/NPZ，而不是从 PDF 读数。GaWF-only refresh 与六模型
baseline merge 使用 source commit `0c614e7`；GaWF-only aggregate 使用 `5de4abd`，严格端点审计
脚本来自 `be32ca0`。旧
`gawf_legacy_notanh_refresh_10seed_v1` 的 trajectory 曾错误地在 evaluation 中硬编码
`tanh(preactivation)`，仅保留为失败 provenance，不用于本记录或正式图。
