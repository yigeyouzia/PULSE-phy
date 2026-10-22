# PULSE：论文、模型架构与创新点学习笔记

> 来源：本地论文 [ICML-2026-PULSE.pdf](../paper/ICML-2026-PULSE.pdf)、本地源码与项目 README。先读论文，再核对实现。
> 状态：已阅读论文与核心代码；已通过原模型 CPU 随机数据前向、反向检查；尚未完成真实数据集复现。
> 更新时间：2026-10-22
> 同目录：[组合到其他模型的通用方法](PULSE-组合到其他预测模型的通用方法.md) · [Gje 使用与 HMSFormer 对比计划](PULSE-Gje大桥数据集使用与HMSFormer对比计划.md)
> 事实标记：“论文”指作者的论述与报告结果；“代码”指本地版本行为；“理解/判断”是本文解释，不代表额外实验结论。

## 0. 先给结论与来源

**PULSE 的核心，是给时序预测建立一个会演化的参考系：先构造历史相位锚点，把残差单独归一化；再生成未来锚点，用它辅助恢复预测；训练时对输入和残差统计量共同做增强。**

因此，学习重点是“分解—演化—恢复—增强”的完整机制。默认预测 Backbone 只是一个两层 MLP，注意力主要用于未来锚点的生成。

| 项目 | 核实内容 |
| --- | --- |
| 论文 | **PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting** |
| 作者 | Yangyou Liu、Zezhi Shao、Xinyu Chen、Hu Chen、Fei Wang、Yuankai Wu |
| 本地 PDF | 26 页；标注 arXiv:2605.16793v2，2026-05-20；论文标注 ICML 2026 |
| 缩写 | Phased Unfolding & Latent Stochastic Evolution |
| 论文列出的代码地址 | <https://github.com/Gemost/PULSE> |
| 本地仓库 origin | <https://github.com/yigeyouzia/PULSE-phy.git>，属于个人 fork 地址 |
| 本次阅读代码版本 | `b26449764b72bfc6aa0d5b961d702f0b59da3848` |
| 核验范围 | 本地 PDF、源码、脚本；未进行远端版本比对，也未把仓库现有日志视为本次复现结果 |

这是 Gemost/PULSE 对应的非平稳时序预测工作。不要与 `yenhochen/PULSE` 的同名工作混用方法描述和引用。

### 0.1 它到底是不是“物理融合”？

**作者称其为 physics-informed；从实际约束看，它采用的是受统计分解、动力系统与异方差思想启发的建模先验。代码没有显式物理控制方程或物理残差损失。**

| 论文的出发点 | 落地方式 | 阅读边界 |
| --- | --- | --- |
| Wold 分解启发：结构成分与随机波动分开处理 | 可学习锚点与残差相减，残差单独归一化 | 代码没有强制两部分正交、独立的损失；不能把学习到的锚点直接当作可识别物理量 |
| 动态相位演化 | Router 根据历史锚点、初步未来表示生成未来锚点 | 没有预先给定 ODE/PDE；“相位”也不是代码计算的 Hilbert 瞬时相位 |
| 异方差残差的统计分布变化 | Statistic-Aware Mixup，插值残差均值和标准差 | 模拟统计扰动，不等于仿真物理过程 |

**与你的研究目标的关系：**它适合用来学习如何把先验转化为网络结构、归一化和增强策略；若你要研究大桥动力学、传热或明确守恒约束，仍需另外建立可检验的物理方程与观测量之间的联系。

## 1. 论文要解决什么问题

![PULSE 动机与效率概览](../assets/fig1_motivation_efficiency.png)

图源：原论文 Figure 1。左侧对应结构演化与残差统计变化的动机，右侧为作者在基准设置下的效率比较。

### 1.1 任务定义

任务是多变量时间序列的确定性预测。历史输入为 $X\in\mathbb{R}^{B\times T\times C}$，输出为 $\widehat Y\in\mathbb{R}^{B\times H\times C}$。

| 符号 | 含义 | ETTh1 的一个实际配置 |
| --- | --- | --- |
| $B$ | 批大小 | 训练脚本为 256；本次最小检查为 4 |
| $T$ | 历史长度 `seq_len` | 96 |
| $H$ | 预测长度 `pred_len` | 96、192、336、720，分别训练 |
| $C$ | 变量数 `enc_in` | 7 |
| $F$ | 时间特征数 `time_dim` | 1 |
| $W$ | 数据加载器使用的全局周期 | 24 |
| $L$ | 可学习码本长度 `inv_len` | H=96 时为 96；不必等于 $W$ |
| $P$ | Router 的 `patch_size` | 24 |
| $d$ | Router 的 `d_model` | 16；Backbone 隐藏宽度另外固定为 512 |

时间输入包括历史与未来时间戳的特征。未来时刻的时、星期等可以预先得到，**不需要未来真实观测值作为模型输入**。默认 `label_len=0`；虽然实验框架创建了 `dec_inp`，PULSE 的 forward 并不使用它。

“Generative”在这里主要指未来锚点的生成。当前实现不是扩散模型，也没有输出预测分布或区间；关闭训练模式后，对同一输入重复推理得到相同结果。

### 1.2 非平稳性带来的矛盾

论文第 1–4 页把问题概括为 **Phase Amnesia**，即处理非平稳性时丢失了演化中的全局时间背景。可按下面两种场景理解：

| 场景 | 作者指出的局限 | PULSE 的处理 |
| --- | --- | --- |
| 归一化历史输入，预测后使用历史统计量恢复 | 未来的结构、均值与波动未必与历史一致 | 先分离锚点，仅对残差做归一化，并单独生成未来锚点 |
| 从历史周期中提取模板并延伸 | 未来的相位和幅度可能变化 | 用可学习 Router 生成未来结构，不只复制历史模板 |

另一个动机是 **Variance Masking**：如果强周期等结构的方差远大于微小波动的方差，整体标准化可能压小残差信号。论文据此提出先扣除锚点，再按残差自身尺度归一化。

这是带条件的建模论证。它依赖“锚点捕获了主要结构”等假设，不能据此认定所有数据都存在可准确分离的两类成分，或训练一定更容易。

## 2. 方法论：模型结构

### 2.1 总览：一条预测主路与一条锚点生成支路

![PULSE 原论文架构图](../assets/fig2_framework.png)

图源：本地仓库提供的论文 Figure 2。重点看：锚点在归一化前被减去、归一化后被加回；未来恢复使用生成的 $A_y$。

```text
历史时间特征 ── TimeEncoder_x ──┐
周期索引 ── 可学习 Codebook ────┴── 历史锚点 Ax
                                      │
历史观测 X ── 减 Ax ── 残差 Rx ── RevIN ── 加 Ax ── X̃
                              │                       │
                            μR、σR                 Backbone
                              │                       │
                              │                       Y0
                              │                       │
                  Ax ───────────── Phase Router ◀─────┘
                                              │
未来时间特征 ── TimeEncoder_y ── 加到 Router 输出 ── Ay
                                              │
                  Y0 − Ay ── 用 μR、σR 逆归一化 ── 加 Ay ── Ŷ
```

训练时在 $\widetilde X$ 进入 Backbone 前做 Mixup，同时插值残差统计量并混合监督目标；推理时关闭增强。

下文先解释不含增强的推理路径，再解释训练。

### 2.2 历史锚点：周期码本＋时间特征编码

对应论文第 4 页式 (1)–(2)，代码 [models/PULSE.py](../models/PULSE.py) 的 `codebook`、`time_enc_x`、`norm()`。

可学习码本为 $M\in\mathbb{R}^{L\times C}$。加载器给出周期索引 $p$，模型按下式取出历史锚点原型：

$$
\operatorname{idx}(h)=(p-h)\bmod L,\qquad h=0,\ldots,T-1,
$$

$$
A_x=M[\operatorname{idx}]+\operatorname{TimeEncoder}_x(X_{\mathrm{timestamp}}).
$$

短代码对应关系：

```python
idx = (cycle_index[:, None] - torch.arange(T, device=x.device)[None, :]) % L
anchor_x = codebook[idx] + time_enc_x(x_mark)  # [B, T, C]
```

码本是所有窗口共享、可训练的参数。它提供重复结构的先验；时间编码根据当前窗口的时间位置补充条件。**码本并非直接存放一段已观测历史，锚点也不是手工移动平均。**

需要区分 $W$ 与 $L$：ETTh 的加载器先对全局样本位置取 `% 24`，模型再按 `inv_len` 取模。源码没有在运行时自动做 ACF 搜索；当前周期已经写在各加载器里。

另一个值得记录的细节：论文用历史终点 $t_{end}$ 描述 $p$，但加载器实际取 `self.cycle_index[s_end]`，其中历史切片是 `[s_begin:s_end]`。它对应第一步未来的位置，随后模型按 `p-h` 递减取码本。复现应先保持这套约定；若改成 `s_end-1` 或改取码本方向，应作为独立对齐实验。

TimeEncoder 的实际顺序为：

```text
[B, T, F]
 → Linear(F, C//dsa) → LayerNorm → GELU
 → Linear(C//dsa, C//dsb) → LayerNorm → GELU
 → Conv1d(in_channels=T, out_channels=T, kernel_size=ksize)
 → Linear(C//dsb, C)
 → [B, T, C]
```

未来分支结构相同，但独立参数，时间长度为 $H$。

**代码轴向提醒：**`Conv1d` 前没有交换维度。PyTorch 把第二维当通道，因此这里时间位置对应卷积通道，卷积核滑动在最后的特征轴上。不能照着论文的“局部细化”描述，将当前实现解释成普通的沿时间轴滑动的一维卷积。

### 2.3 残差单独归一化：分开尺度处理，但重新合并输入

对应论文第 4 页式 (3)，代码 `ReVIN.forward()` 与 `Model.norm()`。

$$
R_x=X-A_x,\qquad
\widetilde X=\operatorname{RevIN}(R_x)+A_x.
$$

对每个样本、每个变量，沿历史时间维计算残差均值和方差：

$$
\mu_R=\operatorname{mean}_t(R_x),\qquad
\sigma_R=\sqrt{\operatorname{var}_t(R_x)+\epsilon},
$$

$$
\operatorname{RevIN}(R_x)
=\gamma\odot\frac{R_x-\mu_R}{\sigma_R}+\beta.
$$

代码 `eps=1e-5`，均值、方差形状为 `[B,1,C]`，仿射参数按变量学习；`torch.var` 使用默认修正方式。当前类没有像一些 RevIN 实现那样对统计量显式 `.detach()`。

**最重要的理解：Backbone 看到的是“标准化残差＋原锚点”。**残差与锚点在尺度处理上被分开，但模型没有把二者送进两个完全独立的预测器。论文称其为 Residual Stream，不应据此将输入或 $Y_0$ 当成纯残差。

### 2.4 Backbone：共享于各变量的两层 MLP

对应 `Model.__init__()` 与 `forward()`：

```python
backbone = nn.Sequential(
    nn.Linear(T, 512),
    nn.GELU(),
    nn.Linear(512, H),
)
y0 = backbone(x_in.permute(0, 2, 1)).permute(0, 2, 1)
# [B,T,C] → [B,C,T] → [B,C,H] → [B,H,C]
```

这里沿时间维进行映射，各变量共享同一套 MLP 权重。它没有显式跨变量注意力、图结构或多层 Transformer 编码器。CLI 中的 `d_model` 在本模型主要控制 Router，**不会把 Backbone 中硬编码的 512 自动改掉**。

$Y_0$ 是供后续锚点生成与恢复使用的未来潜在表示，还不是最终观测尺度上的预测。

### 2.5 Generative Phase Router：生成未来锚点

对应论文第 4 页式 (4)–(5)，代码 `Model.phase_router()`。

Router 接收历史锚点 $A_x$ 和初步未来表示 $Y_0$。它先把每个变量独立展开，再将两个时间长度投影到共同的小维度 $d$。

#### 2.5.1 Patch 的方向容易读反

历史输入的真实变换为：

```text
Ax: [B,T,C]
 → [B,C,T]
 → [B*C,T/P,P]
 → [B*C,P,T/P]
 → Linear(T/P,d)
 → Tx: [B*C,P,d]
```

未来分支同理得到 `Ty: [B*C,P,d]`，投影层输入宽度为 $H/P$。

因此注意力 token 数是 **$P$**，每个 token 聚合多个 patch 中相同偏移位置的信息。它不是常见的“$T/P$ 个 token、每个 token 编码一整块”的 PatchTST 式组织。

以 $T=H=96,P=24,d=16,B=4,C=7$ 为例，两个 token 张量均为 `[28,24,16]`。本次用原代码 hook 已观察到这一形状。

当前实现要求 $T$ 和 $H$ 都能被 $P$ 整除，没有通用的末尾补齐逻辑。

#### 2.5.2 两步交叉注意力

$$
Z=\operatorname{Attn}(T_x,T_y,T_y),\qquad
E=\operatorname{Attn}(T_y,Z,Z).
$$

| 阶段 | Query | Key / Value | 理解 |
| --- | --- | --- | --- |
| sender | 历史锚点 token $T_x$ | 未来表示 token $T_y$ | 形成结合未来预测上下文的中间路由表示 |
| receiver | 未来表示 token $T_y$ | 中间表示 $Z$ | 将中间信息读回未来表示坐标 |

底层是标准投影后的缩放点积注意力：

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{Softmax}\left(\frac{(QW_Q)(KW_K)^\top}{\sqrt d}\right)(VW_V)W_O.
$$

代码为单头、无因果 mask 的 FullAttention，注意力权重有 0.1 Dropout。Router 没有堆叠完整 Transformer encoder 的残差连接、LayerNorm、FFN 等结构；末端另外有投影 MLP。

随后通过 `Linear(d,2d) → GELU → Linear(2d,H/P)` 并恢复时间顺序，得到 `[B,H,C]`，再加入未来时间特征编码：

$$
A_y=\operatorname{Unpatch}(\operatorname{MLP}(E))
+\operatorname{TimeEncoder}_y(Y_{\mathrm{timestamp}}).
$$

**与复制历史周期的区别：**$A_y$ 依赖 $A_x$、当前预测 $Y_0$ 和未来时间特征，可以随输入改变；它不是直接从同一个码本取出未来序列。

![作者展示的历史与未来锚点](../assets/fig3_phase_anchor_visualization.png)

图源：原论文 Figure 3。重点看相似历史参考结构如何对应不同未来演化；这是作者的可视化证据，不代表本次已在 Gje 上得到相同结果。

### 2.6 Generative DeNorm：在未来锚点参考系中恢复

对应论文第 4 页式 (6)，代码 `Model.denorm()` 与 `ReVIN.reverse()`。

$$
\widehat Y=\operatorname{RevIN}^{-1}(Y_0-A_y;\mu_R,\sigma_R)+A_y.
$$

展开得到：

$$
\widehat Y
=\left(\frac{Y_0-A_y-\beta}{\gamma}\right)\odot\sigma_R+\mu_R+A_y.
$$

可以把它读成三步：减去未来锚点，恢复残差尺度，再放回未来锚点。

需要明确一个边界：**推理时残差的均值和标准差仍来自历史窗口。**模型通过生成 $A_y$ 调整未来参考结构，并未显式增加一个独立网络去预测未来残差方差。这一点有助于理解它对非平稳性的处理范围。

### 2.7 Statistic-Aware Mixup：增强输入，也同步增强恢复统计量

对应论文第 5 页式 (7)–(8)，代码 `Model.forward()` 第 168–186 行、`Exp_Main.train()` 第 206–209 行。

训练时采样一组 batch 置换 $\pi$ 与一个 batch 共享系数 $\lambda\sim\operatorname{Beta}(0.15,0.15)$：

$$
\widetilde X_{mix}=\lambda\widetilde X+(1-\lambda)\widetilde X_\pi,
$$

$$
\mu_{mix}=\lambda\mu_R+(1-\lambda)\mu_{R,\pi},\qquad
\sigma_{mix}=\lambda\sigma_R+(1-\lambda)\sigma_{R,\pi},
$$

$$
Y_{mix}=\lambda Y+(1-\lambda)Y_\pi.
$$

Backbone 使用混合输入，恢复使用混合统计量；目标配对必须用模型返回的同一个 `lam,index`。

这里混合的是已经保留锚点的完整 $\widetilde X$，不是只混合原始 $R_x$；插值的是**标准差**，不是方差，也不是从混合后的波形重新估计尺度。代码为了复用逆归一化接口，先计算 `sigma_mixed`，再存成 `var_mixed = sigma_mixed**2 - 1e-5`。

为什么这样做？对相关系数为 $\rho$ 的两个残差，直接混合波形的方差为：

$$
\operatorname{Var}(\lambda R_i+(1-\lambda)R_j)
=\lambda^2\sigma_i^2+(1-\lambda)^2\sigma_j^2
+2\lambda(1-\lambda)\rho\sigma_i\sigma_j.
$$

反相时可能发生抵消。例如 $R_j=-R_i$、$\lambda=0.5$ 时，混合波形为零；线性插值标准差仍为正。更严格地说，在 $\rho=-1$ 时完全抵消还要求 $\lambda\sigma_i=(1-\lambda)\sigma_j$，不能把论文主文的简化表述理解成任意反相关、任意混合比例都会尺度归零。

**实现细节：**虽然 `x_in` 与统计量被混合，传给 `denorm()` 的 `phase_anchor_x` 和未来 `timestamp_y` 仍来自原 batch 次序，没有显式用相同系数混合。未来锚点因此由“混合后的 $Y_0$＋原样本锚点/时间条件”共同生成。不要自行补写一个“所有锚点和时间戳都同步 Mixup”的流程。

### 2.8 损失、训练与推理

对应论文第 5 页式 (9)、`Exp_Main.train()` 第 214–237 行。

默认脚本设置 `rec_lambda=0, auxi_lambda=1`，配合默认 `auxi_mode=fft, auxi_loss=MAE, module_first=1`：

$$
\mathcal L=\operatorname{mean}\left|
\operatorname{FFT}_t(\widehat Y_{mix})-
\operatorname{FFT}_t(Y_{mix})\right|.
$$

```python
loss = (torch.fft.fft(pred, dim=1) - torch.fft.fft(target_mix, dim=1)).abs().mean()
```

它计算复数 FFT 结果之差的模，包含相位差的影响；不是先分别取幅值再计算幅度谱之差。频域损失来自论文所引用的已有方法，应与 PULSE 的三项主要设计区别看待。

| 环节 | 当前默认路径 |
| --- | --- |
| 优化器 | Adam |
| 训练输入 | SAM 混合后的表示、残差统计量与目标 |
| 主训练目标 | 频域 MAE |
| 验证与早停 | 无 Mixup，使用时域 MSE |
| 最终测试 | MSE、MAE 等指标；不应把训练频域损失与测试 MSE 数值直接比较 |
| forward 返回 | train：`(prediction, mixup_params)`；eval：仅 `prediction` |

`--loss mse` 这个参数名不能说明默认训练就使用 MSE；必须查看实际训练分支和两项 loss 权重。

## 3. 论文的创新点与实验支持

### 3.1 按“问题—设计—证据”理解贡献

| 问题 | 作者的设计 | 相对贡献在哪里 | 证据与边界 |
| --- | --- | --- | --- |
| 整体归一化可能弱化残差并丢失时间结构 | Phase-Anchored Disentanglement | 用学习到的参考结构定义残差，只归一化残差，再把锚点加回 | 表 3 去掉锚点后平均误差上升；不证明锚点一定是物理真值 |
| 历史周期不一定能直接外推 | Generative Phase Router＋Generative DeNorm | 将未来参考结构生成放入预测与恢复闭环 | 表 3 去掉 Router 后平均误差上升；效果依赖数据与锚点质量 |
| 残差统计量漂移、朴素波形 Mixup 可能发生尺度抵消 | Statistic-Aware Mixup | 输入增强时显式插值恢复用的均值与标准差 | 表 3 的 SAM / Statistic-Aware 消融提供支持；不是对所有 OOD 扰动的保证 |
| 复杂 Backbone 成本高 | 小 MLP＋固定分辨率的 Router | 通过合适先验和训练策略提高小模型能力 | 主表、替换 Backbone 与效率实验；不是“加到任何模型都提升” |

理解上的重点是这些组件如何一起工作。RevIN、MLP、Attention、时间特征、Mixup、频域损失都不是本文首次提出的基础工具；创新主要体现在它们围绕未来参考结构和残差尺度的组织方式。

### 3.2 做了哪些实验，结果说明什么

论文第 6–8 页及附录：12 个多变量数据集，默认历史长度 $T=96$。

| 数据组 | 数据集 | 预测长度 |
| --- | --- | --- |
| 长期预测 | ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Solar、Traffic、Weather | 96、192、336、720 |
| 短期预测 | PEMS03、PEMS04、PEMS07、PEMS08 | 12、24、48、96 |

作者在主表报告：按各数据集预测长度平均，12 组中取得 10 组 MSE 第一、8 组 MAE 第一。它说明该实验设置下具有竞争力；不能外推为任意预测长度、任意 Backbone 或桥梁数据上都最优。

![原论文主结果表](../assets/table1_main_results.png)

图源：原论文 Table 1。阅读时按数据集和指标比较，表中数值属于作者实验。

表 3 的消融均值如下，覆盖 8 个长期预测数据集：

| 设置 | 平均 MSE | 平均 MAE | 可支持的判断 |
| --- | --- | --- | --- |
| 完整 PULSE | 0.306 | 0.321 | 完整配置的平均表现最好 |
| 去掉 Phase Anchor | 0.326 | 0.334 | 锚点相关处理对平均表现贡献较大 |
| 去掉 SAM | 0.314 | 0.323 | 数据增强对平均表现有帮助 |
| 去掉 Statistic-Aware | 0.313 | 0.325 | 显式处理恢复尺度有贡献 |
| 去掉 Phase Router | 0.315 | 0.326 | 学习未来参考结构有贡献 |

![原论文组件消融表](../assets/table3_ablation.png)

图源：原论文 Table 3。表中的平均提升支持组件贡献，但不能代替逐数据集判断。

这是均值结论，不代表每个数据集、每个指标都严格优于所有变体。例如表中 Traffic 的去掉 SAM 版本 MAE 为 0.282，低于完整版本 0.286。

即插即用实验也有明确反例：表 2 中 PatchTST 在 Traffic 上的平均 MSE 从 **0.481 上升到 0.565**。作者给出局部 patch 归纳偏置与处理后信号不匹配的解释，但该原因仍应看作解释性假说。

![原论文替换 Backbone 的组合结果](../assets/table2_plug_and_play.png)

图源：原论文 Table 2。尤其关注 PatchTST＋PULSE 在 Traffic 的 MSE 退化。通用组合方法另见同目录文档。

另外，论文采用固定随机种子 2024。表格不能直接提供多随机种子方差或统计显著性结论。

### 3.3 复杂度与效率怎么理解

Router 的 token 数固定为 $P$，其注意力矩阵为 $P\times P$。在 $P,d$ 固定时，**注意力部分**不会随 $T$ 增长为 $T\times T$，这是源码可核实的优势。

论文给出整体 $O(T)+O(P^2)$ 的表述。对本地实现需要补充限定：TimeEncoder 的 `Conv1d(T,T,k)` 权重数随 $T^2$ 增长，未来分支对应 $H^2$；其卷积计算也不能不加条件地当作关于长度的线性项。故“固定大小的路由注意力”不能自动推出“当前完整源码对任意长度严格线性”。

效率实验也存在实际开销：表 4 中 ETTh1 的 DLinear 推理延迟从 0.438 ms 增到 3.828 ms。该表是在同一 batch 上重复推理 100 次、报告中位延迟与峰值 GPU 显存，不能当作总训练时间或最低显卡需求。

![组合后的精度与效率展示](../assets/fig4_plug_and_play_efficiency.png)

图源：原论文 Figure 4。横向比较时需要同时看精度与开销。

![原论文实际推理开销](../assets/table4_efficiency_tradeoff.png)

图源：原论文 Table 4。此表是作者的指定批次推理测量，不能当作 Gje 的耗时估算。

## 4. 后面值得继续学习的地方

### 4.1 建议的学习顺序

| 顺序 | 要掌握的内容 | 完成标准 |
| --- | --- | --- |
| 1 | RevIN、残差归一化与逆变换 | 不看代码写出 `X−Ax → norm → +Ax` 与 `Y0−Ay → reverse → +Ay` |
| 2 | 周期码本与时间编码 | 解释清楚 $W,L,P$ 的不同含义；用实际窗口检查索引和时间特征 |
| 3 | Router 的 tensor reshape 与 Q/K/V | 手算并打印 `[B*C,P,d]`，说明两次注意力的数据来源 |
| 4 | SAM 与频域损失 | 说明为什么混标准差；证明朴素 Mixup 的反相抵消例子；明确训练与验证目标 |
| 5 | 真实实验复现 | 先完成 ETTh1 单配置，核对数据切分、脚本、损失与指标，再扩到多个数据集和随机种子 |

后续比较可从论文自身引用的 RevIN、Dish-TS / SAN、CycleNet、PatchTST / DLinear 与频域损失工作入手，分别对照归一化、统计量预测、周期结构、Backbone 和优化目标。这里列的是学习方向，尚未对这些原论文逐篇重新核验。

### 4.2 对你做论文的启发与边界

**工程判断：PULSE 是一个值得先跑通的非平稳预测基线，但已经覆盖“周期先验＋残差归一化＋未来结构生成＋统计增强”这条组合路线。**再简单增加一个物理命名模块，难以仅靠命名形成有说服力的研究贡献。

| 可继续提出的问题 | 能做的验证 | 当前状态 |
| --- | --- | --- |
| 锚点是否具有稳定可解释性？ | 多种子锚点比较、已知结构的合成数据、锚点置乱对照 | 研究设想，未运行 |
| SAM 改善的是哪类分布变化？ | 分开设计均值、尺度、周期、相位扰动，比较 SAM / 普通 Mixup / 不增强 | 研究设想，未运行 |
| 时间特征、周期码本各贡献多少？ | 独立消融时间编码、码本与未来时间条件，控制参数规模 | 研究设想，未运行 |
| 换 Backbone 后是否稳定获益？ | 保持训练目标与调参预算一致，对多个 Backbone 复测 | 研究设想，未运行 |
| 桥梁上能否得到真正的物理约束？ | 先确定预测对象、外部荷载/环境量、采样率，再判断动力学方程、模态或准静态关系是否可观测 | 需要桥梁数据核查，未验证 |

这些是可检验问题，不是已通过查新确认的创新点。尤其不能把桥梁监测数据的所有频率成分都直接解释成结构模态：环境趋势、采样方式、传感器位置与激励条件都需要先查清。

## 5. 本地源码对齐与最小测试

### 5.1 阅读入口与代码映射

| 论文概念 / 工程步骤 | 本地源码定位 | 要看什么 |
| --- | --- | --- |
| 模型总入口 | [models/PULSE.py](../models/PULSE.py)，`Model.forward`，第 161 行起 | 时间编码、norm、SAM、Backbone、denorm、返回值 |
| 历史锚点与残差输入 | 同文件 `Model.norm`，第 107 行起 | 码本索引、相减、残差归一化、加回锚点 |
| 归一化 / 恢复 | 同文件 `ReVIN`，第 6 行起 | 时间维统计量、仿射参数、逆变换 |
| Backbone 与时间编码器 | 同文件第 57–104 行 | MLP 宽度、TimeEncoder 的真实卷积轴 |
| Phase Router | 同文件 `phase_router`，第 120 行起 | patch 轴、两次 cross-attention、恢复顺序 |
| 未来锚点与输出 | 同文件 `denorm`，第 147 行起 | `Ay=router+timestamp`，减锚点、恢复、加锚点 |
| Attention | [layers/SelfAttention_Family.py](../layers/SelfAttention_Family.py)，`FullAttention / AttentionLayer` | 投影 Q/K/V、Softmax、mask 与 Dropout |
| 混合监督与损失 | [exp/exp_main.py](../exp/exp_main.py)，`train` 第 191–237 行 | 同配对目标 Mixup、时域和频域损失权重 |
| 验证与早停 | 同文件 `vali / train` | 验证 MSE、早停与 checkpoint |
| 时间特征与周期 | [data_provider/data_loader.py](../data_provider/data_loader.py)、[utils/timefeatures.py](../utils/timefeatures.py) | 真实 `freq`、特征截取、`cycle_index[s_end]` |
| 数据装载设置 | [data_provider/data_factory.py](../data_provider/data_factory.py) | `timeenc`、batch、shuffle、drop_last |
| 参数与随机种子 | [run.py](../run.py) | 实际默认参数、seed、训练/测试入口 |
| 推荐起跑配置 | [scripts/ETTh1.sh](../scripts/ETTh1.sh) | 数据路径与四个独立 horizon 配置 |
| 学习率变化 | [utils/tools.py](../utils/tools.py)，`adjust_learning_rate` | `type3` 的 epoch 衰减行为 |

表中行号对应本次记录的 commit，更新后可能变化。

### 5.2 论文叙述与当前代码需要分清的地方

| 项目 | 论文 / 直觉 | 当前本地代码 |
| --- | --- | --- |
| 学习率 | 附录 B.2 写 fixed learning rate 0.005 | 脚本参数为 0.005；训练器总是构造 OneCycleLR，会先把优化器 LR 设为约 0.0002（本次用同参数实测）；默认 `type3` 又在 epoch 后调整、衰减，因此需要记录实际 LR 轨迹 |
| Dropout | 附录写各层一致 0.1 | PULSE 内显式 0.1 在注意力权重；Backbone 和 TimeEncoder 未添加 Dropout |
| 卷积含义 | TimeEncoder 的“局部细化” | 输入未 permute，卷积通道轴是时间位置；见 2.2 |
| 消融开关 | 论文比较锚点、Router、SAM 等变体 | 当前 PULSE 文件没有完整的 CLI 消融开关；`--use_revin` 也没有被该 Model 读取 |
| 换 Backbone | 论文报告四种替换实验 | 当前 `Model` 内 Backbone 硬编码为 MLP；`--model DLinear` 走独立 DLinear，不会自动得到 DLinear＋PULSE |
| 时间特征 | 附录列出多种候选特征 | Loader 按 `freq` 构造，再截取前 `time_dim` 列；ETTh1 当前为 HourOfDay，不能默认所有配置都输入全部日历特征 |
| ETT 切分 | 附录概括为 6:2:2 | 代码按固定 12/4/4 月长度切分，验证/测试前保留历史上下文；不能改成对任意 CSV 全长直接按比例切分后当作同一设置 |
| AMP | 可能被当作纯提速开关 | 当前 AMP 训练分支使用未做目标 Mixup 的 MSE；AMP 测试调用还缺少时间特征参数。与默认非 AMP 路径不等价 |

这张表用于理解和后续复现记录，并未在本次修改实现。若后续要修复或标准化入口，应单独记录每项改动，保留与原论文设置的对应关系。

### 5.3 本次真正运行了什么

使用原始 `models/PULSE.py` 及其 Attention 实现，未用自己重写的小模型代替。环境为 Python 3.10.20、PyTorch 2.12.0、NumPy 2.2.6，CPU，随机种子 2024。

两组配置均为 `B=4,T=96,C=7,P=24,d=16,time_dim=1,dsa=dsb=1,ksize=5`，分别使用 ETTh1 脚本中对应的预测长度和码本长度：

| 检查 | H=96，L=96 | H=192，L=24 |
| --- | --- | --- |
| 输出形状 | `[4,96,7]` | `[4,192,7]` |
| sender / receiver 的 Q、K、V | 均为 `[28,24,16]` | 均为 `[28,24,16]` |
| 参数量 | 195,270 | 382,546 |
| norm 与 reverse 往返最大绝对误差 | 约 `4.77e-7` | 约 `4.77e-7` |
| eval 同输入重复结果 | 完全相同 | 完全相同 |
| train＋SAM＋混合目标 FFT-MAE backward | 通过 | 通过 |
| 关键梯度 | 码本、MLP、两级注意力 Q 投影、两个时间编码器和 RevIN 仿射权重均有有限非零梯度 | 同左 |
| train 下 `do_mixup=False` | 通过，元数据为 `None` | 通过，元数据为 `None` |

**这些检查证明当前两组配置的核心计算与梯度路径可执行，不证明真实数据精度、收敛性或论文结果已复现。**本地根目录尚无脚本要求的 `all_datasets/`；本次没有启动完整数据训练器。仓库已有 `logs/` 与 `result.txt` 不能算作本次实验。

运行位置为项目根目录，解释器为 `/Users/cyt/miniconda3/envs/pinn_ac/bin/python`。下面给出可重复的精简检查，覆盖上述最关键的前向、增强和反向路径；表中的路由形状与更细梯度由额外 hook / 断言检查得到：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/cyt/miniconda3/envs/pinn_ac/bin/python - <<'PY'
from types import SimpleNamespace
import numpy as np
import torch
from models.PULSE import Model

np.random.seed(2024)
torch.manual_seed(2024)
torch.set_num_threads(2)
for H, L in [(96, 96), (192, 24)]:
    cfg = SimpleNamespace(seq_len=96, pred_len=H, enc_in=7,
        d_model=16, inv_len=L, patch_size=24, time_dim=1,
        dsa=1, dsb=1, ksize=5)
    model = Model(cfg).cpu()
    x, xm, ym = torch.randn(4,96,7), torch.randn(4,96,1), torch.randn(4,H,1)
    cycle = torch.tensor([0,1,12,23])
    model.eval()
    with torch.no_grad():
        pred = model(x, cycle, xm, ym)
        assert pred.shape == (4,H,7) and torch.isfinite(pred).all()
        assert torch.equal(pred, model(x, cycle, xm, ym))
    model.train()
    pred, (lam, idx) = model(x, cycle, xm, ym)
    y = torch.randn_like(pred)
    y_mix = lam*y + (1-lam)*y[idx]
    loss = (torch.fft.fft(pred,dim=1)-torch.fft.fft(y_mix,dim=1)).abs().mean()
    loss.backward()
    grad = model.codebook.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    print('PASS', H, tuple(pred.shape), sum(p.numel() for p in model.parameters()))
PY
```

原始依赖文件没有固定版本，不能由本次核心模型通过推断整个训练项目兼容该环境。例如 `utils/tools.py` 仍使用 `np.Inf`，在 NumPy 2 环境构造 EarlyStopping 时需要单独处理；完整训练还涉及其他模型的导入依赖及 GPU checkpoint 加载路径。

正式复现的推荐入口仍是项目根目录执行 `bash scripts/ETTh1.sh`，但它会依次运行四个 horizon，并把 stdout 写入各自的 `logs/ETTh1/*.log`；需要先准备数据与匹配环境。若先做单配置，应从脚本复制第一组参数，而非只依赖 `run.py` 默认值。

## 6. 我的理解：最值得带走的几句话

1. **PULSE 先建立可学习参考结构，再处理相对于参考结构的波动。**这种参考系思路比单记某一个模块名更有价值。
2. **历史锚点参与输入，未来锚点参与恢复。**未来锚点由模型生成，预测过程才形成闭环。
3. **SAM 的关键不只在混样本，也在同步插值恢复统计量。**它控制解码尺度，而不是从可能抵消的混合波形重新估计尺度。
4. **默认主预测器很小。**论文研究的是先验、表示和训练策略如何帮助预测，不是更深的 Transformer。
5. **它提供物理启发的通用时序建模经验。**面向具体桥梁系统的物理约束研究，还需要可观测物理量、方程、参数与实验验证之间的证据链。

### 掌握清单与后续状态

- [ ] 能手画完整前向路径，并区分 $Y_0$、$A_y$ 与最终 $\widehat Y$。
- [ ] 能解释为什么 Backbone 输入仍含 $A_x$，以及推理时哪些统计量来自历史。
- [ ] 能区分 $W$、`inv_len` 与 `patch_size`，手算 Router token 形状。
- [ ] 能区分完整表示 Mixup、标准差插值和目标 Mixup。
- [ ] 在真实数据上完成单配置训练，保留环境、commit、参数、LR 轨迹与指标。
- [ ] 补充多随机种子、可控分布漂移与组件消融，再判断是否适合作为自己论文的基线。

本次已完成论文与代码阅读、两组原模型最小检查；清单描述的是你后续需要掌握和完成的研究任务，未替你标记为已掌握或已复现。
