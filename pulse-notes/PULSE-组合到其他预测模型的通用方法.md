# PULSE 组合到其他预测模型的通用方法

> 来源：本地 [PULSE 论文](../paper/ICML-2026-PULSE.pdf)第 3–8 页；[PULSE 模型](../models/PULSE.py)、本仓库 DLinear / PatchTST / iTransformer / TimesNet；本地 HMSFormer 源码。
> 状态：已核对接口；PULSE＋本地 DLinear 已完成 CPU 前向与反向检查；其他组合为迁移方案，尚未完成真实数据实验。
> 更新时间：2026-10-22
> 关联：[论文与架构笔记](PULSE-论文模型架构与创新点学习笔记.md) · [Gje 使用与 HMSFormer 对比计划](PULSE-Gje大桥数据集使用与HMSFormer对比计划.md)

## 1. “即插即用”具体是什么意思

**把其他预测模型放进 PULSE 原有 Backbone 的位置，接收锚点保留后的历史表示，输出供 Router 和恢复模块使用的未来表示。**

它不是把 PULSE 的最终预测与另一个模型的预测求平均，也不是只把某个 Attention 层插进另一个模型。完整组合还包含输入侧归一化、输出侧恢复、训练增强和监督目标配对。

```text
原模型：                  X ── Backbone ── Ŷ

组合后： X ── PULSE.norm ── X̃ ── 目标 Backbone ── Y0
                     │                              │
                  Ax、μR、σR ── Router＋PULSE.denorm ── Ŷ

训练额外处理：X̃、μR、σR 与目标 Y 按同一 lam / index 混合
```

论文的通用性是框架设计与实验层面的主张；**当前本地 `models/PULSE.py` 将 MLP 硬编码在类内，没有现成的 `--backbone XXX` 开关。**`--model DLinear` 只会创建独立 DLinear，不能自动创建 DLinear＋PULSE。

![论文中的 Backbone 组合实验](../assets/table2_plug_and_play.png)

图源：论文 Table 2。作者测试了四种 Backbone，但 PatchTST 在 Traffic 上出现 MSE 退化，所以组合后仍需验证，不能保证提升。

## 2. 先固定五个接口契约

### 2.1 输入输出形状与坐标

设目标 Backbone 的适配后函数为 $f_\theta$：

$$
f_\theta:\mathbb R^{B\times T\times C}\rightarrow\mathbb R^{B\times H\times C}.
$$

| 契约 | 必须保证的行为 | 常见错误 |
| --- | --- | --- |
| 输入 | 接收 $\widetilde X=\operatorname{RevIN}(X-A_x)+A_x$ | 仍把原始 $X$ 直接送入目标模型 |
| 输出 | 返回 `[B,H,C]` 的未来表示 $Y_0$ | 把隐藏特征 `[B,N,d]` 直接传给 Router |
| 通道 | 输出变量数和顺序与输入一致 | 输出单目标后，通过 broadcast 与多通道锚点相减 |
| 信息 | 推理只使用历史观测、已知时间特征及允许的外生输入 | Encoder-decoder 输入了未来真实目标或未来实测风信号 |
| 状态 | `train/eval`、梯度、SAM 元数据传播正确 | 用 `no_grad()` 或 `.detach()` 切断 Backbone 梯度 |

原版 PULSE 在内部令 `c_out=enc_in`。如果任务是“多变量输入、只预测少量目标”，最省改动的方案是先预测全部同维变量，再统一选择目标计算损失和指标；若只生成目标通道，就需要同时改锚点、统计量和通道映射，已经超出简单替换。

### 2.2 PULSE 内部的特殊接口

当前 PULSE 的调用是：

```python
y0 = self.backbone(x_in_mixed.permute(0, 2, 1)).permute(0, 2, 1)
```

也就是说，直接替换 `self.backbone` 时，新模块实际应满足 **`[B,C,T] → [B,C,H]`**。很多目标模型使用 `[B,T,C]`，需要两次转置。

统一工程设计可以在未来把这一行重构成 BTC 接口，但那是另一种实现方案。本文件的最小迁移示例保留原版调用方式，只添加 Adapter。

### 2.3 时间特征与周期索引

| 输入 | 形状 | 要求 |
| --- | --- | --- |
| `x` | `[B,T,C]` | 浮点，预处理方式与基线一致 |
| `cycle_index` | `[B]` | 整数；明确周期、参考起点及窗口定位约定 |
| `x_mark` | `[B,T,F]` | 与历史数值时间对齐 |
| `y_mark` | `[B,H,F]` | 只包含推理时可得的未来时间信息 |
| `target` | `[B,H,C]` | 仅供损失与评估，不进入推理 forward |

还要满足 `T % patch_size == 0`、`H % patch_size == 0`，且 `C//dsa`、`C//dsb` 非零。码本长度 `inv_len`、周期 $W$、Router `patch_size` 是三个独立概念。

### 2.4 归一化的归属

应区分三层操作：训练集拟合的全局标准化、PULSE 的残差归一化、目标 Backbone 自带的归一化。

1. **全局 StandardScaler** 可以保留，但只能在训练数据上拟合，所有模型共享同一个结果。
2. **PULSE 残差 norm/denorm** 是框架主机制，普通 RevIN Callback 不能替代它。
3. **Backbone 内部 norm/denorm** 需要检查是否成对，最后回到输入 $\widetilde X$ 的坐标尺度。

完整成对的内部归一化并非必然错误。可以先保留目标模型原有行为，命名为“PULSE＋原生 Backbone”；另设旁路内部 norm/denorm 的版本，用消融判断是否更合适。**不能只删内部归一化而留下逆变换，也不能宣称“关掉所有归一化才正确”。**

外置 Callback 则尤其要检查顺序：若把普通 `RevInCB` 套在整个 PULSE 外面，它处理的就不是原版预期的残差参考系。HMSFormer＋PULSE 组合应由 PULSE 接管外层 norm/denorm，保留原 HMSFormer 单独运行时的 RevIN 作为基线自身设计。

### 2.5 配置与初始化

Router 和 Backbone 的隐藏宽度分别命名为 `pulse_d_model`、`backbone_d_model` 更清楚。当前几种模型构造函数会原地修改 `configs.d_model`、`dropout` 等字段，**为两者传同一个 Namespace 有串改风险**。

迁移时分别 `deepcopy` 配置，记录最终生效值；创建最终模型之后再创建优化器，确保新增模块都进入参数组。换 Backbone 后不能直接把默认 MLP 的 checkpoint 当作完整组合权重使用。

## 3. 通用迁移步骤

| 步骤 | 做什么 | 完成证据 |
| --- | --- | --- |
| 1 | 跑清目标模型单独预测路径 | 输入、输出、norm、时间条件的记录 |
| 2 | 确认 PULSE 两侧保持不变 | `norm → SAM → Backbone → denorm` 流程可追踪 |
| 3 | 实现 Backbone Adapter | 明确形状与返回值，不在 Adapter 中再计算最终监督目标 |
| 4 | 处理目标模型输入形式 | patching、encoder-decoder 或额外时间输入都放在合适位置 |
| 5 | 接通训练元数据 | 目标用模型返回的同一 `lam,index` 混合 |
| 6 | 固定验证、测试路径 | 关闭 Mixup，无未来目标泄漏，输出同一评估尺度 |
| 7 | 小张量前向和反向 | Backbone、码本、Router 的梯度可达，输出有限 |
| 8 | 与单独 Backbone 成对对比 | 相同数据、样本清单、预算和调参规则，多种子报告 |

### 3.1 最小可运行示例：本地 DLinear＋PULSE

下面的 Adapter 用目标模型的标准 BTC 接口，外侧适配原版 PULSE 的 BCT 调用。**它只替换 Backbone，保留原版锚点、Router、SAM 与恢复代码。**

在项目根目录、已有 PyTorch 环境中运行：

```python
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from models.PULSE import Model as PULSE
from models.DLinear import Model as DLinear

class BTCBackboneAdapter(nn.Module):
    def __init__(self, core, horizon):
        super().__init__()
        self.core, self.horizon = core, horizon

    def forward(self, x_bct):
        x_btc = x_bct.transpose(1, 2).contiguous()
        y_btc = self.core(x_btc)
        expected = (x_btc.shape[0], self.horizon, x_btc.shape[2])
        assert y_btc.shape == expected, (y_btc.shape, expected)
        return y_btc.transpose(1, 2).contiguous()

np.random.seed(2024)
torch.manual_seed(2024)
cfg = SimpleNamespace(seq_len=96, pred_len=96, enc_in=7,
    d_model=16, inv_len=96, patch_size=24, time_dim=1,
    dsa=1, dsb=1, ksize=5, individual=False)
model = PULSE(deepcopy(cfg))
model.backbone = BTCBackboneAdapter(DLinear(deepcopy(cfg)), cfg.pred_len)

x = torch.randn(4,96,7)
xm, ym = torch.randn(4,96,1), torch.randn(4,96,1)
cycle = torch.tensor([0,1,12,23])
model.train()
pred, (lam, idx) = model(x, cycle, xm, ym)
y = torch.randn_like(pred)
y_mix = lam*y + (1-lam)*y[idx]
loss = (torch.fft.fft(pred,dim=1)-torch.fft.fft(y_mix,dim=1)).abs().mean()
loss.backward()
assert model.backbone.core.Linear_Seasonal.weight.grad is not None
assert model.codebook.grad is not None
model.eval()
with torch.no_grad():
    pred = model(x, cycle, xm, ym)
assert pred.shape == (4,96,7) and torch.isfinite(pred).all()
print(tuple(pred.shape), sum(p.numel() for p in model.parameters()))
```

本次检查结果：`(4,96,7)`，总参数量 **114,982**；使用随机数据的训练 forward、混合目标 FFT-MAE backward 和 eval 均通过。环境为 Python 3.10.20、PyTorch 2.12.0、CPU。测试脚本在临时目录执行，未把它写入模型源码或声称得到论文 DLinear＋PULSE 的精度。

正式代码可以把 Adapter 和模型工厂独立存放，新增 `--pulse_backbone` 之类参数，并统一分派训练返回值。本次仅记录方案，尚未新增这个命令行开关。

### 3.2 各类模型分别怎么接

| 目标模型 | 本地实际接口 / 行为 | 组合处理 |
| --- | --- | --- |
| MLP / DLinear | BTC 输入输出；DLinear 内部分解后按时间映射 | 用上述 Adapter；DLinear 本地实现无独立 RevIN 开关 |
| PatchTST | `Model.forward(x_enc)`；内部含成对标准化/恢复 | Adapter 转轴；先保留原生配对，再做去内部 norm 的消融 |
| iTransformer | `forward(x_enc,x_mark_enc)`；可能返回 `(pred,attn)` | 显式接时间参数并取预测张量；原模型 `use_norm=1` 是硬编码成员，不能靠无关 CLI 自动关闭 |
| TimesNet | `forward(x_enc,x_mark_enc)`；本地输出覆盖 `T+H` | 明确只取最后 `H` 步作为 $Y_0$，检查配对 norm/denorm |
| HMSFormer | core 输入是 `[B,Npatch,C,Q]` | 在 Adapter 内先 patch，再调用 prediction head；移除组合外层的普通 RevIN Callback |
| Encoder-decoder 模型 | 通常还要求 decoder 输入和时间特征 | decoder 未来数值位置填零或使用模型合法推理机制；不能塞未来真实 $Y$ |
| 概率 / 扩散模型 | 输出可能是分布、样本或多个参数 | 先定义作为 $Y_0$ 的确定性表示；原版恢复不能直接套在任意概率参数上，需要额外方法设计 |

对需要 `x_mark` 的模型，应设计带显式 `forward(x_btc,x_mark,y_mark)` 的新核心接口并修改调用处；**不要用全局变量或跨 batch 缓存来偷偷传时间条件**。这是比 DLinear 示例多一步的适配，不能把上面的单参数 Adapter 原样套到所有模型。

### 3.3 HMSFormer 的专门接口

本地实现位于 [HMSFomer.py](/Users/cyt/projects/i/bridge/PatchTST_self_supervised/src/models/HMSFomer.py)，类名为 `HMSFromer`，预测头设置 `head_type='prediction'`。

```text
PULSE 的 X̃: [B,720,C]
 → create_patch(patch_len=36,stride=36)
 → [B,20,C,36]
 → HMSFromer 的多尺度编码与 prediction heads
 → Y0: [B,H,C]
 → PULSE Router / denorm
```

HMSFormer 的 `patch_sizes=[36,12,6]` 属于其内部多尺度设计；PULSE 的 `patch_size=24` 属于 Router，二者没有必要相等。当前组合尺寸 `720/36=20`，也能被 12、6 整除，避免尾部裁切造成有效输入长度变化。

普通 HMSFormer 的顺序为 `StandardScaler → RevInCB → PatchCB → core → RevIN 恢复`。组合版应为 `StandardScaler → PULSE → Adapter 内 patch → core → PULSE 恢复`，不能在 PULSE 输入前由外部 PatchCB 先切成四维。

**状态：**这里已静态核对 HMSFormer 的真实接口，但尚未运行 HMSFormer＋PULSE，更未验证迁移预训练权重的效果。

## 4. 训练端必须一起迁移的行为

### 4.1 Mixup 的数据流

```python
pred, mix = model(x, cycle, x_mark, y_mark)
if mix is not None:
    lam, idx = mix
    target = lam*target + (1-lam)*target[idx]
loss = (torch.fft.fft(pred,dim=1)-torch.fft.fft(target,dim=1)).abs().mean()
```

仅替换模型而沿用“不混目标的 MSE 训练器”会改变方法。原版 `phase_anchor_x` 与时间条件没有额外同步混合；若想改变这一行为，需要单列变体，而不是称为忠实迁移。

关闭 SAM 时模型可能返回 `(pred,None)`；基线可能直接返回 tensor。正式训练器应先统一输出协议，防止变量未初始化。当前 `exp_main.py` 的非 PULSE 路径没有统一初始化 `mixup_params`，不适合不经检查就批量运行新组合或基线。

### 4.2 保留哪些训练设置

| 设置 | 忠实迁移 | 用于研究组件贡献的对照 |
| --- | --- | --- |
| SAM | 保留 | 关闭 / 普通 Mixup / Statistic-Aware |
| 损失 | 默认复数 FFT 差的 MAE | 两模型都用 MSE、都用 FFT-MAE；分辨结构与目标函数收益 |
| 验证 | 无 Mixup，统一原评估坐标的 MSE | 相同验证集和 checkpoint 规则 |
| 优化器 | 对全部可训练参数优化 | 相同调参预算，允许模型各自选择验证最优 LR |
| 预训练 | 不能假定替换后零成本生效 | 从头训练与预训练方案分行报告 |

当前 AMP 分支与非 AMP 分支训练语义不一致，正式接入前应统一损失与目标 Mixup 后再启用。学习率也应记录实测轨迹，不能只记录 CLI 值；原版构造 OneCycleLR 后又使用 `type3` 调整，见论文笔记。

## 5. 一个组合什么时候才算完成

- [ ] Adapter 的形状、时间长度和变量顺序全部明确。
- [ ] train / eval 路径都运行；至少一次 loss.backward，新增模块与目标 Backbone 均有梯度。
- [ ] 历史和未来时间特征正确，forward 不依赖未来观测目标。
- [ ] 归一化成对，最终输出与目标处在相同评估尺度。
- [ ] SAM 的输入、统计量与监督配对一致，验证/测试关闭增强。
- [ ] 对照模型使用相同 split manifest、scaler 和样本清单。
- [ ] 训练预算、预训练来源和调参次数有记录，至少多种子重复。
- [ ] 允许没有提升或局部退化，不只挑有收益的数据与预测长度。

## 6. 建议的推进顺序

先完成原版 PULSE 在 Gje 的数据适配，并重跑统一协议下的 HMSFormer；之后再做 HMSFormer＋PULSE。这样可以分别回答“PULSE 本身是否适合 Gje”和“PULSE 是否能改善 HMSFormer”，避免把数据修正、Backbone 替换和预训练变化一次混在一起。

所有组合都应重新训练并验证。框架可迁移不代表现成权重可直接复用，也不代表组合本身就构成新的论文贡献。
