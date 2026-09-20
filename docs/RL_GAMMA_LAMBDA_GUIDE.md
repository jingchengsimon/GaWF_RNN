# 理解 DQN 中的 gamma、lambda 与多步收益

整理日期：2026-09-04。

本文面向人类读者，整理本次关于长期收益、one-step/multi-step、lambda-return、
replay 和 episode 边界的讨论。它是概念说明，不是训练提交指令，也不代表新协议已经实施。
“当前实现”指整理时检查的本地 Atari DQN/DRQN 代码；历史运行应以各自 snapshot 为准。

## 1. 先记住四句话

- **gamma（γ）决定远期奖励有多重要**：越大，远期奖励折扣越少。
- **lambda（λ）决定如何组合不同长度的收益估计**：越大，越偏重先看更多实际奖励、
  再接上剩余预测的估计。
- **one-step 也估计长期收益**，并不是只考虑一个时间步的收益。
- **seq_len 是 RNN 的训练展开长度；return horizon 是 target 的实际奖励读取上限**。
  二者不是同一个参数。

提高 γ 是“让远期成功更值得追求”；提高 λ 是“让较长实际经历更直接地参与学习”。
两者都不保证性能必然提高。

## 2. 符号与时间编号

在状态 $s_t$ 执行动作 $a_t$，环境返回奖励 $r_t$，然后到达 $s_{t+1}$。
有些教材把这个奖励写作 $r_{t+1}$；本文统一使用 $r_t$，避免编号混用。

| 符号 | 含义 |
|---|---|
| $r_t$ | 第 $t$ 次交互实际得到的奖励 |
| $Q(s_t,a_t)$ | 模型对从当前状态执行该动作后的长期折扣收益的预测 |
| $V_i$ | 简写：$\max_a Q_{\mathrm{target}}(s_{t+i},a)$ |
| $y_t$ | 训练当前 $Q(s_t,a_t)$ 所用的 target |
| $G_t^{(n)}$ | 从同一个时间点 $t$ 出发的 n-step return |
| $H$ | 构造 return 时最多使用多少步实际奖励 |
| $L$ | 连续参与 recurrent 训练的时间位置数，即这里的 seq_len |

$G$ 是 return 的惯用符号，不是一个英文缩写。这里的 n-step return 是一种长期收益估计，
包含 n 个实际奖励，以及在允许时接上的末端价值预测。

本文为便于说明，用状态 $s$ 表示模型的输入信息；对于 DRQN，价值预测也依赖 recurrent
state 所概括的历史。当前代码在 TD target 中把环境奖励 clip 到 $[-1,1]$，而报告的
episodic return 使用 raw reward。因此下文公式中的“实际奖励”在项目中是按训练协议处理
后的奖励，不应与未裁剪的游戏分数混为一谈。

## 3. gamma：改变长期收益的时间权重

长期折扣收益的形式是：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
$$

有限 episode 在 terminal 处停止；持续任务可概念性地延伸到无限远。
通常使用 $0\leq\gamma<1$；某些适当的有限 episodic 问题也可使用 γ=1。

当前 Atari 配置为 γ=0.99。距离当前 k steps 的奖励，其权重为 $\gamma^k$：

| γ | 100 steps 后 | 300 steps 后 | 500 steps 后 |
|---|---:|---:|---:|
| 0.99 | 0.366 | 0.049 | 0.0066 |
| 0.995 | 0.606 | 0.222 | 0.082 |
| 0.999 | 0.905 | 0.741 | 0.606 |

因此，“已经是 0.99”不意味着它对长动作链足够大。提高到 0.995 仍可能明显改变远期奖励
的权重，但也可能增大 Q-value 尺度和学习难度。这里不存在 γ 所定义的硬性未来截止步数。

fs4 下一个 agent step 通常推进四个 ALE frames。按 60 frames/s 粗略换算，300 agent
steps 约为 20 秒游戏时间；这不是训练程序的实际运行时间。

## 4. one-step 与固定 n-step：同一收益的不同估计

### 4.1 One-step DQN

$$
y_t^{(1)}=r_t+\gamma(1-d_t)\max_aQ_{\mathrm{target}}(s_{t+1},a).
$$

其中 $d_t=1$ 表示该 transition 应停止 bootstrap。自然 terminal 是这种情况；
不能不加区分地把所有环境 reset 都视为 $d_t=1$。

直觉是：“当前奖励已知，剩余未来全部交给模型估算。”
模型不是内部模拟未来 N 步，而是直接输出一个长期价值预测。

### 4.2 固定 2-step 与 3-step

先忽略 terminal，写成：

$$
\begin{aligned}
G_t^{(1)}&=r_t+\gamma V_1,\\
G_t^{(2)}&=r_t+\gamma r_{t+1}+\gamma^2V_2,\\
G_t^{(3)}&=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\gamma^3V_3.
\end{aligned}
$$

固定 n-step 使用 **n 个实际奖励加一个末端预测**。相对于 one-step，它额外用 n−1 步
实际奖励替换了部分原本由模型预测的未来。固定 n-step 本身不需要 λ。

“替换”是一种估计方式的变化，不是严格的代数展开：一般不能认为
$V_1=r_{t+1}+\gamma V_2$。两者可能因预测误差、环境随机性以及实际动作偏离目标策略而不同。

例如：

$$
G_t^{(2)}-G_t^{(1)}
=\gamma\left[r_{t+1}+\gamma V_2-V_1\right].
$$

下一步实际经历比原预测更好，2-step target 可能更高；反之更低。
只有在价值预测准确、后续行为符合相同目标策略等条件下，两种估计的条件期望才一致，
单次样本仍不一定相等。

## 5. lambda：混合同一起点的不同长度 return

### 5.1 不是把三个不同时间点的 one-step targets 相加

λ 的通常范围是 $0\leq\lambda\leq1$。
以下是忽略 off-policy 修正、最多使用 H 步的截断 lambda-return：

$$
G_t^{\lambda,H}
=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}G_t^{(n)}
+\lambda^{H-1}G_t^{(H)}.
$$

H=1 时它就是 one-step。λ=0 时选择 one-step；λ=1 时选择当前可用的最长 H-step return。
中间取值组合不同长度的估计。

H=3 时：

$$
y_t^\lambda
=(1-\lambda)G_t^{(1)}
+\lambda(1-\lambda)G_t^{(2)}
+\lambda^2G_t^{(3)}.
$$

这是从同一个 t 出发的 1-step、2-step、3-step estimates 的加权平均，不是
“t、t+1、t+2 各自的长期收益相加”。

| λ | 1-step 权重 | 2-step 权重 | 3-step 权重 |
|---|---:|---:|---:|
| 0 | 1 | 0 | 0 |
| 0.5 | 0.5 | 0.25 | 0.25 |
| 0.9 | 0.1 | 0.09 | 0.81 |
| 1 | 0 | 0 | 1 |

权重之和为 1；target 不会因为混合了三份估计就自动变成三倍。

### 5.2 完全展开后，实际奖励和预测都有 λ 权重

$$
\begin{aligned}
y_t^\lambda={}&r_t
+\gamma\lambda r_{t+1}
+\gamma^2\lambda^2r_{t+2}\\
&+\gamma(1-\lambda)V_1
+\gamma^2\lambda(1-\lambda)V_2
+\gamma^3\lambda^2V_3.
\end{aligned}
$$

λ 加权的是整份 return，而不只是实际奖励。$r_t$ 的总系数仍为 1。
$V_1,V_2,V_3$ 对应不同状态，系数又都非负，一般不能相互抵消。

同一个公式也可以改写成：

$$
\begin{aligned}
y_t^\lambda={}&r_t+\gamma V_1\\
&+\gamma\lambda(r_{t+1}+\gamma V_2-V_1)\\
&+\gamma^2\lambda^2(r_{t+2}+\gamma V_3-V_2).
\end{aligned}
$$

直觉是：在 one-step target 上，加上后续实际经历揭示的预测偏差。
如果两个括号都恰好为零，混合 target 才退回相同的 one-step target。

### 5.3 是不是要求三个 V 都尽可能大？

不是。训练让当前 $Q(s_t,a_t)$ 接近 target，而不是直接最大化 target。
当前项目在 `no_grad` 下计算 target，里面的 V 不接收这次 loss 的梯度。
如果后续表现低于预期，multi-step 修正反而会降低当前 target。

动作选择偏向高 Q-value，但价值学习首先要求预测准确。增加 λ 不会把模型输出改成
“整条动作链的分数”；模型仍输出当前状态下各个动作的 Q-values。

### 5.4 能否完全不预测未来？

如果已经获得到自然 terminal 为止的完整轨迹，并且 horizon 允许覆盖它，
标准 episodic λ-return 在 λ=1 时可以使用完整 Monte Carlo return，不再接末端预测。

但 horizon=32、λ=1 并不意味着无限长实际收益；若 32 步后未终止，仍需 bootstrap。
Watkins 的 trace 截断也可能使实际长度更短。
此外，完整旧轨迹反映的是行为策略的实际经历，不能无条件当成当前 greedy policy 的收益。

即使训练 target 完全来自实际奖励，模型在下一次决策时仍要预测未来，因为那时未来尚未发生。

## 6. 实际奖励从哪里来？当前训练如何交替进行？

实际 reward 由环境交互返回，不需要人为设计一个“实际收益策略”。
如果环境才走到 k，就无法知道 k+1、k+2 的实际奖励；需要等后续交互完成，
或从 replay 中读取已经发生的历史连续轨迹。

例如环境已走到第 100,000 步，可以使用 replay 中第 1,000 步及之后的记录训练模型，
前提是这些记录仍保留在 buffer 中。环境当前进度和本次训练样本的时间位置是两回事。

整理时的单环境 Atari recurrent 配置如下：

| 项目 | 当前设置/含义 |
|---|---|
| 单任务 warm-up | 达到 20k steps 前不更新参数，继续交互和写 replay |
| Five-task warm-up | 每 task 至少 20k valid steps，并满足 global 门槛，约 100k global 或更晚 |
| 更新频率 | 满足门槛后，每 4 global steps 做 1 次 optimizer update |
| Recurrent batch | 8 sequences × 16 个训练位置，聚合有效位置的 loss 后统一更新 |
| Target network | 每 1,000 global steps 同步 online network |
| Current target | one-step DQN，γ=0.99，无 λ-return 开关 |

warm-up 期间参数固定，但 epsilon 会衰减，RNN hidden state 也会随交互演化。
“每 4 步更新一次”不是只用四条记录训练；一次更新会读取多条历史序列，数据可以被重复使用。
无效 autoreset 行由 loss mask 排除，128 是每次更新的最大训练位置数，不一定全都有效。

buffer 中的 action 不是“最优动作 GT”，而是当时实际执行的动作。
训练让该动作对应的 Q-value 接近 TD target，不是模仿这些 action 标签。

## 7. seq_len 与 return horizon 为什么不冲突？

### 7.1 seq_len=16 的含义

RNN 训练时按顺序处理一个短窗口，让 hidden state 在窗口中传递，计算各位置的 loss，
并在这段展开上反向传播。当前实现从窗口起点的零 recurrent state 开始，没有额外 burn-in。

这不是 episode 长度，不是 value 的预测截止点，也不意味着交互时每 16 步清空记忆。
交互中的 recurrent state 可以延续到需要 reset 的边界。

### 7.2 horizon=32 的含义

每个训练位置构造 target 时，最多读取后面 32 个实际奖励，再接上剩余价值预测。
对于 λ-return，这也是参与混合的最长 n-step 上限，不代表每个样本必定完整用到 32 步。

讨论中的 3-step 是公式示例；“λ=0.9、horizon=32”是实验候选，不是已验证最优值。
fs4 下，3 steps 约 0.2 秒游戏时间，32 steps 约 2.1 秒。Watkins 截断和 episode 边界
会缩短实际范围，λ 本身也会降低长 backup 的混合权重。

若保持 L=16 个训练位置，同时让每个位置最多向后看 H=32 步，在不中断、不含 burn-in
的例子中，需要覆盖 L+H−1=47 个 transitions，即 48 个 observations。
额外轨迹可以用于不反向传播的 target 计算，不必全部变成 loss 位置。

因此两者概念上独立，但现有仅采样 `seq_len + 1` 行的实现不能直接提供这个组合。
未来若实现，需要明确采样长度、loss 区间、target 尾部、recurrent state 与各类 mask。
不能只增加一个 horizon 参数而不调整数据读取。

## 8. Bootstrap 与三种边界

**Bootstrap 是用模型的价值预测补足未直接累加的未来收益。**
V 是预测值；使用 V 构造 target 的做法叫 bootstrap，不是重新启动环境。

| 边界 | 语义 | Target 原则 |
|---|---|---|
| 自然 terminal | 任务真的结束，没有剩余收益 | 累计到 terminal，末端不接 V |
| 外部 time-limit truncation | 停止交互，但限制未必是任务本身的终点 | 若目标允许继续，从 final observation bootstrap |
| Replay sample/window 到头 | 本次只读取到这里，环境并未因此终止 | 读取更多数据，或使用较短、带 bootstrap 的 return |

### 8.1 不能把 reset observation 接到上一局

对允许 bootstrap 的 truncation，需要截断前的 final observation。
reset observation 已经属于新一局；不能把下一局的价值或奖励接到上一局。
如果时间限制本来就是任务定义中的真正截止点，应按该任务语义处理，不能一律保留 V。

Replay window 边界不会导致环境 reset；后续交互可能早已发生，只是没被本次读取。
因此它不同于一次实际的 episode 截断。

### 8.2 为什么讨论中包括 Skiing？当前代码究竟怎么做？

修改后的 Skiing 在连续 450 agent steps 没有赛程进展时，由 wrapper 人为触发
`truncated=True`、`end_reason="stalled"`；这时 ALE 游戏未必自然结束。
当前协议对该边界重置 episode/recurrent state，但保留 bootstrap。

整理时 `_replay_boundary_flags` 的实际行为是：

- 自然 `terminated`：停止 bootstrap。
- Skiing `stalled` truncation：保留 bootstrap。
- 其他 `truncated`：目前仍停止 bootstrap。

因此，不能把“理论上外部 time-limit 可以保留 bootstrap”描述成当前代码已经普遍支持。
Skiing 的处理是一项协议选择，而不是 `truncated=True` 自动证明该选择正确。
若把 stall 定义为最终任务失败，停止 bootstrap 也是另一种任务定义，需要独立记录和验证。

### 8.3 具体例子：训练窗口 10～20，horizon=10

若 episode 继续到状态 $s_{100}$，计算第 20 步的 target 可以额外读取
$r_{20},\ldots,r_{29}$，最长项接上 $\gamma^{10}V(s_{30})$。
训练窗口在 20 结束，不等于可读取的数据也必须在 20 结束。

### 8.4 具体例子：第 95 步距离边界只剩五次交互

若记录在状态 $s_{100}$ 处结束，只有 $r_{95},\ldots,r_{99}$ 五个奖励，
则可以把该位置的有效 horizon 从 10 缩短为 5。

忽略 Watkins 更早截断时：

$$
y_{95}^{\lambda}
=(1-\lambda)\sum_{n=1}^{4}\lambda^{n-1}G_{95}^{(n)}
+\lambda^4G_{95}^{(5)}.
$$

最长项根据边界语义决定：

$$
G_{95}^{(5)}
=\sum_{j=0}^{4}\gamma^j r_{95+j}
+\begin{cases}
0, & \text{自然 terminal},\\
\gamma^5V(s_{100}), & \text{允许 bootstrap 的边界}.
\end{cases}
$$

不能简单删掉 6～10-step 项并保留原来的全部短项权重，否则总权重小于 1。
最后可用的 5-step return 应承接剩余权重。也不能拿下一 episode 的奖励凑满十步。

如果只是数据尚未收集齐而 episode 仍在进行，可以暂不抽该位置，
或明确使用较短 horizon 并 bootstrap；不能假装它已经 terminal。

## 9. Q(lambda) 在当前项目中的定位

经典 Q-learning 和原始 DQN 使用 one-step target，没有 λ。
TD(λ)/Q(λ) 早于 DQN；它们不是 DQN 之后才出现，但也不是经典 DQN 默认包含的参数。
当前代码的 one-step 形式可理解为 λ=0 的端点，不能据此说已有一个可开启的 λ 开关。

项目使用自写 PyTorch DQN/DRQN loop。引入 λ 需要实现新的 target 构造与采样逻辑，
不是重新设计环境 reward，也不是 PyTorch 会自动完成的行为。

### 9.1 为什么还要指定 Watkins-style？

Replay 由旧参数下的 epsilon-greedy 行为产生，可能不同于当前 greedy target policy。
因此，普通 lambda-return 混合公式不等于完整的 off-policy Q(λ) 算法。

Watkins-style 的核心是：若后续实际动作不是所选 target policy 的 greedy 动作，
停止沿这次偏离继续累计 trace，在该处 bootstrap。需要明确 greedy 的网络来源、
recurrent state 和并列最大值规则，而不是只检查当时是否进入随机选动作分支。
随机分支也可能刚好选中 greedy 动作；旧策略当时的 greedy 动作也可能不再是现在的 greedy。

因此，λ=0.9、H=32 也可能得到很短的实际 backup。应记录有效 horizon 和截断率，
而不是假设每个样本都强化了 32 步长链。Retrace(λ) 等方法采用不同的 off-policy 修正。
相关理论保证不能直接当作当前神经网络实现必然稳定的保证。

### 9.2 如果未来实施，应验证什么？

以下仅为检查清单，不构成实施授权：

- λ=0 或 H=1 退回原 one-step target。
- 小型手算轨迹与公式一致，缩短 horizon 后权重仍和为 1。
- 自然 terminal 不 bootstrap，不跨 episode/task 累计奖励。
- 允许 bootstrap 的 truncation 使用 final observation，而不是 reset observation。
- 非 greedy 动作按选定 Watkins 规则截断，记录实际有效 horizon。
- 额外 target 尾部不意外改变 loss 区间、梯度范围或每次更新的有效样本数。
- 保持 reward clipping、更新频率等无关因素不变，记录所有协议差异。

## 10. 最常见的误解

| 误解 | 正确理解 |
|---|---|
| One-step 只看一步收益 | 它用一个实际奖励加上整个剩余未来的预测 |
| Multi-step 才能学动作链 | One-step 也能；multi-step 让实际奖励更直接参与向前传播 |
| λ 提高远期奖励的重要性 | γ 决定时间折扣；λ 改变不同长度估计的混合方式 |
| 混合三个 return，价值变三倍 | 它们估计同一个量，混合权重和为 1 |
| λ 只乘实际奖励，不影响 V | λ 对整份 return 加权，末端 V 也受影响 |
| 增大 λ 等于所有 V 都往大推 | 目标是预测准确，较差的实际经历也会降低 target |
| H=32 要求每个 episode 至少剩 32 步 | 可以在边界缩短，但必须正确处理末端 V 与剩余权重 |
| seq_len=16 限制 Q 只能预测未来 16 步 | 它控制训练展开，不是 Q 的长期价值预测范围 |
| λ=1 永远不需要模型预测 | 只有覆盖到 terminal 等条件满足时，target 才可不用末端 V |

## 11. 阅读依据与代码入口

本文公式是为解释讨论而整理的简化形式；普通 lambda-return 与完整 off-policy 算法
的区别参见 [Munos et al., Safe and Efficient Off-Policy Reinforcement Learning (2016)](
https://arxiv.org/abs/1606.02647)。其引言、Notation、Off-Policy Algorithms 和 Watkins
讨论提供了算法背景，并列出了更早的 Q(λ)/TD 文献。

实现事实来自整理时的本地代码：

- [Atari DQN/DRQN 训练入口](../utils/training/train_scripts/atari_dqn.py)：
  参数默认值、`_learning_ready`、`_drqn_sequence_loss`、`_replay_boundary_flags` 和训练循环。
- [Replay 采样实现](../utils/training/atari/atari_replay.py)：
  `sample_sequences`、连续窗口、episode/reset metadata 与 loss mask。
- [项目 Atari 协议说明](../experiments/rl/atari/README.md)：
  five-task 协议和 Skiing stall/actionfix 的版本化约定。

本文不替代上述文件的协议定义，也不宣称已完成 λ-return 的实现或实验验证。
