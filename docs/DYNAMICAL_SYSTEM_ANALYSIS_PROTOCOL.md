# RNN/GaWF dynamical-system 分析 protocol

更新：2026-09-07。范围：从完整 recurrent state、PCA 轨迹到 fixed/slow points、局部动力学和
motif 验证。以下八步是四篇文献方法的工作流综合，不是任何单篇论文原有的八步编号，也不是
所有任务必须呈现某一种低维几何结构的假设。CM-MNIST 的选择与文献方法分开记录。

## 文献与证据边界

1. [Structured experience shapes strategy learning and neural dynamics in the medial entorhinal cortex](https://www.nature.com/articles/s41593-026-02409-7)
   （Bowler 等，2026）：按 Cue On/Cue Off 分析，结合 fixed/slow points、长时运行与行为解释
   动力学。方法中的 input/internal noise 设为零，不能被概括为所有 cue/input 都被移除。
   此处纠正此前对话中“Bowler 将 input 全设为0”的过度概括；不据此推断其 bias 消融方式。
2. [Flexible multitask computation in recurrent networks utilizes shared dynamical motifs](https://pmc.ncbi.nlm.nih.gov/articles/11239504/)
   （Driscoll 等，2024）：研究 task period/input 条件下的动力学，并用 input interpolation
   追踪 fixed-point 结构；部分 PCA 基底来自指定 task period，不是所有条件混合的全局 PCA。
3. [Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks](https://web.stanford.edu/class/cs379c/archive/2016/calendar_invited_talks/articles/SussilloandBarakNC-13.pdf)
   （Sussillo & Barak，2013）：从完整 state 搜索 fixed/slow points，局部线性化并检验其解释
   任务计算的作用。slow point 不必是严格 fixed point。
4. [FixedPointFinder: a Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks](https://joss.theoj.org/papers/10.21105/joss.01003)
   （Golub & Sussillo，2018）：使用自动微分进行候选点优化和 Jacobian 分析。
   [官方代码](https://github.com/mattgolub/fixed-point-finder) 是实现参考；当前仓库的分析实现
   不是对其软件运行结果的直接复现。

不要将某一论文的输入设置、residual threshold、trial 结构或图形外观当成通用规定。
实现参数、脚本与输出约定由 [DEVELOPMENT_WORKFLOWS.md](DEVELOPMENT_WORKFLOWS.md) 管理。

## 第1步：定义完整 state，并记录连续 activation

明确离散更新 `s[t+1] = F(s[t], u[t])`，列出所有跨步传递的 state、输入、输出与时间索引。
state 必须足以决定下一步；readout activation 不一定就是 recurrent state。

- RNN：当前单层模型使用 raw recurrent hidden；LayerNorm/ReLU 后的 readout activation
  不能作为下一步传入的 hidden。
- GaWF：记录 hidden 与 feedback，并明确 feedback 更新。若 feedback 可由 hidden 唯一重建，
  可构建闭合的 reduced state 更新，但 Jacobian 必须包含整个 feedback 闭环的导数。
- 冻结权重，使用 eval/inference；记录 reset、输入噪声和 state 噪声设置。
- 独立 movie 各自初始化；连续 stream 的计算分块不应无意引入 reset 或输入 stacking 断裂。

**产物/验收：** full-stream state、readout、input/frame IDs、labels、checkpoint 身份；
分块和不分块 rollout 一致，以及一步重放与记录的索引一致。此处是 inference，不是训练。

## 第2步：构造 PCA 输入矩阵并保留索引

将事件/条件、时间、神经元组织成 `H[event, time, unit]`，再展开为 `X[sample, unit]`。
不同条件可以有自己的 H，但为了比较投影，需要说明是否共用同一 PCA 基底。
条件数量和重复数无需相等；不平衡时必须说明加权/抽样方式，不能静默补齐样本。

中心化后 `Xc = X - mean`，用 SVD/PCA 求基底 V，轨迹坐标为 `Z = Xc V`。
是否 unit-wise 标准化、是否先做条件平均、拟合哪些时间段必须明确。
条件平均可突出可重复成分，也会消除 trial variation；应与单事件轨迹分开保存和解释。

**产物/验收：** X 的索引映射、mean、basis、scores、explained variance；图可由数值重建。
不同训练 seeds 各自拟合并保存，不能直接平均未经对齐的 PC 坐标。

## 第3步：检验低维投影保留的信息

报告累计解释方差、达到选定方差比例所需 PC 数，以及独立 events 的投影检验。
在训练部分拟合 mean/basis 后对测试部分投影；相邻/重叠窗口要防止共享原始帧造成泄漏，
还应认识到没有共享帧不等于消除了同一 stream 内的时间相关。

当前 held-out capture 的精确定义为：

`|| (Xtest - mean_train) Vtrain ||_F^2 / ||Xtest - mean_train||_F^2`。

它是围绕训练均值的投影能量比例，不能与测试集自身重新中心化的 explained variance 混称。
PCA 最大化整体方差，不直接优化 switch 分离、分类或动力学闭合。

**产物/验收：** seed-wise 诊断与跨 seed mean ± sample SD；3D 只作为展示。
高解释方差不证明 attractor，低3D解释方差也不否定高维中存在有用结构。

## 第4步：定义 dynamical system 及固定输入条件

实际电影满足 `s[t+1] = F(s[t], u[t])`，是时变输入驱动的系统。
fixed-point 问题需先指定 `u_bar` 和固定参数，研究 `F_u_bar(s)`。
输入条件必须保存到结果中；digit/sector 相同不代表完整图像或 CNN feature 相同。

用真实轨迹 state 验证一步更新，再定义固定输入或明确干预。
对实际轨迹的相邻差分可以画速度，但不能将不同输入下的速度混成一个自治 flow field。
将3D坐标抬升为 `mean + V z` 会丢失其余维度；该平面上的矢量场也不自动构成闭合动力学。

**当前 CM-MNIST 已授权两组（项目适配）：**

- 自治干预：`F_auto(h) = tanh(W_hh h + b_hh)`，移除整个 `W_ih u + b_ih`。
- 真实输入：`F_real(h; u_bar) = tanh(W_hh h + b_hh + W_ih u_bar + b_ih)`；
  固定每个 clean switch 前一帧和 switch 帧的实际两帧 CNN feature。

不包含“CNN feature 置零但仍保留 b_ih”的第三组。自治组改变了原模型的 feedforward 项，
其 fixed points 首先解释这个干预后的系统，不能直接视为真实电影轨迹的吸引子。

## 第5步：在完整 state 中搜索 fixed/slow points

离散系统优化 `q(s; u_bar) = 0.5 ||F(s, u_bar) - s||^2`。
连续系统 `ds/dt = f(s,u)` 则使用 `q = 0.5 ||f(s,u_bar)||^2`；不能混用这两个 F/f 定义。

从实际访问的 states 和其扰动生成多个初始化，保留初始化来源。优化只改变候选 state，
不训练模型。记录 residual、收敛过程、失败候选和搜索覆盖；在同一输入条件下按明确距离
阈值去重。跨输入的位置接近本身不意味着是同一个 fixed point。

**产物/验收：** 每个输入下的候选点和 residual；只有满足数值标准的点标为 numerical root。
低但非零 residual 可作为 slow-point 候选，需要结合任务时间尺度；优化停止不等于找到解。
多次初始化没有发现其他点，不是穷尽性证明。

## 第6步：Jacobian、稳定性与局部动力学

在 fixed point 计算 state Jacobian `J = dF/ds`；需要时计算输入 Jacobian `B = dF/du`。
离散时间中 `|lambda| < 1` 对应局部渐近稳定，`|lambda| > 1` 表示不稳定方向，接近单位圆
需要数值与非线性验证。连续时间判据是 `Re(lambda) < 0`，不能混淆。

记录 eigenvalues/eigenvectors、稳定/不稳定维数、慢方向和复特征值对应模式。
对 slow points，局部展开须保留非零常数漂移项；不能仅凭特征值赋予它严格平衡点稳定性。
非正规系统可能出现短期放大，即使谱半径小于1也需检查 perturbation dynamics。

**产物/验收：** 高维 Jacobian 与谱、数值精度敏感性；有限差分/小扰动检验局部预测。
目前一步 perturbation 检查只验证局部线性近似，不能代替长时间吸引域验证。

## 第7步：将固定点与真实轨迹、输入变化联系起来

使用第2步同一 seed 的 mean/basis 投影 fixed points 和局部模式，不重新挑选“更好看”的基底。
同时计算高维距离、局部线性预测误差、沿稳定/不稳定方向的运动以及固定输入长时 rollout。
检验真实轨迹是否进入相关局部区域；只在投影上接近可能是假接近。

对 CM-MNIST，冻结一帧得到的是 input-conditioned reference system；原电影输入继续变化，
轨迹未必会到达或追随该输入的 fixed point。需要检验 state 松弛时间与输入变化时间尺度。
比较 switch 前后固定输入是起点；input interpolation 可进一步追踪结构，但插值 feature
未必对应真实图像，应标为分析干预。

**产物/验收：** 轨迹—fixed-point 对应证据、扰动恢复/发散、真实输出预测关系。
同一真实输入序列下的小扰动轨迹收缩可支持 conditional stability，而不要求三维聚成一点。

## 第8步：识别、验证并比较 dynamical motifs

motif 应以计算功能和动力学证据定义，例如保持、积累、切换、旋转；不以点云形状命名。
候选解释需结合固定输入长时行为、局部模式、输入插值/扰动、readout 与行为变化验证。
环形图不等于 limit cycle；其存在需要相应长期轨迹/周期性/横向稳定性证据。

逐 seed 报告并汇总可比统计量。跨 seed 比较几何时须先定义对齐，避免 PC 符号、旋转及
神经元置换差异；对齐拟合与评价样本分开。不可把不同条件的 fixed points 数量简单相加后
称为网络拥有的 attractor 数量。

**产物/验收：** 可复现的 motif 定义、跨 seed 支持程度、与任务功能的关系及反证控制。
没有获得足够证据时，结论停留在几何描述或 input-conditioned 局部动力学。

## 当前 CM-MNIST 第1–3步应如何解释

### 已知事实

- 记录来自同一 held-out movie 的连续 B=1 rollout，32步仅是计算分块，switch 不 reset。
- 筛选出30个 clean joint-switch events；每窗100帧，相对索引 -50…49。没有按标签细分。
- -50 与 +49 是分析者截取的边界，不是 episode 初始化和任务终止。0是switch帧输入处理后
  的 hidden 记录，不是网络必然完成切换或决策的时刻；两帧 stacking 还包含切换前一帧。
- 30个events不是每一种条件的30次匹配重复，也不是90种digit×sector组合的均匀覆盖。
- 不发生第二次 switch 不意味着画面静止、CNN输入恒定，或运动中的 sector label 不变。
- 当前展示 RNN raw hidden 和 GaWF canonical hidden；绝对坐标范围、幅度不可直接跨模型比较。

数字来自 [pca_summary.csv](../results/data/analysis/F_timing/continuous_switch_pca_20260906/pca_summary.csv)，
均为10个seeds的统计平均：

| 指标 | RNN | GaWF |
|---|---:|---:|
| PC1–3 累计解释方差 | 20.94% | 21.82% |
| Held-out PC1–3 capture | 13.09% | 12.69% |
| 达到90%方差所需 PCs | 115.0 | 87.9 |

这些数字说明目前混合条件的 raw-state ensemble 没有被一个三维线性子空间充分概括；
不是对局部/非线性 intrinsic dimension 的估计，也不是对所有任务变量可解码性的结论。

### 图上能说什么、不能说什么

三类时间标记在当前投影中交叠，没有明显分离；轨迹在可视区域内形成非均匀占据的点云。
仅凭此图不能确认统计聚类、低维 invariant manifold、attractor、limit cycle、混沌，
也不能把缺乏视觉分离归结为“没有学习”。曲线相交可能只是高维投影重叠。

“所有switch应落在同一点”需要额外假设：这些events具有相同/近似输入、历史状态或共同
计算初态。当前条件不满足该假设。任务可能要求随时表示digit/位置，而非统一的event clock；
这是待检验解释，不是图形已经证明的机制。即使网络成功响应switch，也可能是每条轨迹从
各自旧表征转向各自新表征，从而出现可重复的相对变化而没有跨event的绝对位置聚类。

全局PCA按方差选轴。图片内容、位置、背景与历史可能贡献主要方差；switch相关方向可能
不在前三PC中。但目前没有方差归因结果，不能直接宣称是哪一个因素主导。

### fixed-point 稳定性之前的建议诊断（尚未执行）

1. 先看逐event、同一基底上的轨迹和同步输入/输出正确率，避免30条轨迹叠加遮挡。
2. 检查高维 `||h[t]-h[t-1]||`、`||h[t]-h[-1]||` 的event-aligned变化，以及输出恢复时间。
   与同stream、匹配上下文的伪switch窗口比较，并用时间block处理自相关。速度峰只支持
   event相关瞬态，不足以证明状态转换由某个attractor控制。
3. 使用已保存的digit、sector与transition标签解释颜色/读出；样本不足的组合不作稳定估计。
   优先检查高维cross-validated读出或已完成dPCA的对应结果，而非强求90类逐类成图。
4. 若展示基线对齐的位移轨迹 `delta h[t] = h[t]-h[-1]`，须单独标为relative geometry，
   同时保留原始轨迹。人为减去基线造成的起点聚集不能算网络自行收敛的证据。
5. 连续传播不同于原训练的32-step reset分布。利用原窗口reset评估作为对照，比较同帧输出
   与hidden统计，判断可视化反映的是任务表征还是state-carry协议改变后的行为。不能直接
   用当前形状判定两者，也不要求为此重新训练。

因此当前最稳妥的结论是：**在混合条件的全局前三PC里，未观察到统一的event-phase几何分离；
输入驱动的高维表征和event相关瞬态是否存在，仍需针对变量与时间变化的定量检验。**
无需等到图呈现漂亮的线/环才允许继续fixed-point分析；后者必须独立证明其任务相关性。
