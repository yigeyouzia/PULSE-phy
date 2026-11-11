# Gjemnessund 物理信息、模态先验与稳定振荡模型

更新：2026-11-10。遵守 [工作区开发约束](../../AGENTS.md)。本次是两个独立的物理先验原型；既有模型、脚本、数据处理和实验产物保留。

## 1. 起点和本次交付计划

代码起点 `a9f8e287c5dffc4b5d5440d711225ab58e713b1c`。工作区已有日志删除、笔记修改及其他未跟踪文件，均不是本次更改，不覆盖或清理。起点快照在 `/tmp/pulse_gje_physics_start_20261110.json`，交付时核对源码哈希及已有产物的大小和修改时间。

| 文件 | 计划职责 |
| --- | --- |
| `models/PULSE_FACT_MODAL.py` | 完整展开 PULSE_FACT 数据流；在恢复预测尺度后，增加模态谐波外推的软先验，仅作用于六路加速度 |
| `models/PULSE_FACT_OSC.py` | 完整展开 PULSE_FACT 数据流；用稳定受迫振荡层生成六路加速度，四路风通道保留原预测 |
| `layers/gje_physics/modal_prior.py` | 根据真实 0.5 秒采样间隔建立模态投影和外推基，不把 MLP 的 512 维隐藏轴当成时间轴 |
| `layers/gje_physics/stable_oscillator.py` | 有界频率、正阻尼、精确零阶保持离散化、低维平滑受迫项和模态观测映射 |
| `layers/gje_physics/units.py` | 训练集 StandardScaler 绑定、g 与 SI 单位转换和协议校验 |
| `utils/gje_model_support.py` | 只追加两个新模型的注册及来源记录 |
| `exp/exp_main.py` | 最小模型导入/注册；为新模型绑定已有训练集 scaler 并补充物理元数据；旧模型分支不变 |
| `scripts/PULSE_FACT_MODAL/Gjemnessund.sh` | 模态软先验大桥脚本，720 → 96/192/336/720 |
| `scripts/PULSE_FACT_OSC/Gjemnessund.sh` | 稳定受迫振荡层大桥脚本，同一预算、日志和 checkpoint 规则 |
| `tests/test_gje_physics.py` | 单位、无未来标签输入、模态外推、稳定性、梯度、保存恢复及原入口合成数据检查 |
| `notes/tools/audit_gje_physics_metadata.py` | 只读核对本地 H5 与官方 H5 的属性及样本，不修改原始数据 |
| `notes/data/gje_physics_20261110/` | 来源哈希、官方物理属性、本地单位核对和验证记录 |

`run.py`、旧模型文件、旧数据集和旧运行脚本不修改；不增加 CLI 参数、不另建训练入口。沿用允许的最小模型注册。所有新模型参数采用专属名称，并在日志、checkpoint extra state 和 `gje_run.json` 记录。

## 2. 官方信息与可用范围

官方数据：[Zenodo v1](https://zenodo.org/records/5979695)，DOI `10.5281/zenodo.5979695`。本次对应 `data_2Hz.h5`，官方发布 MD5 为 `9f9e25d26998afc023e77ca9980c2fab`；读取部分内容不等于完成这个全文件 MD5 校验。原始加速度 200 Hz，当前数据为 2 Hz，`Δt=0.5 s`，Nyquist 频率 1 Hz。

官方 H5 的 `acceleration/G5 east/{x,y,z}` 和 `acceleration/G5 west/{x,y,z}` 均标明 **unit = g**。换算：`a[m/s²] = a[g] × 9.80665`。标准化张量本身没有 g 的数值尺度，必须先反 StandardScaler，不能直接把网络输入乘 9.80665。

本地文件：`/Users/cyt/projects/i/bridge/PatchTST_self_supervised/data/datasets/public/BRIDGE/gjemnessund.h5`。这是 pandas fixed 格式的重新打包文件，不保留全部官方传感器属性。官方单位与本地数值的对应核对见下一节；服务器建议继续使用既有 `GJE_ROOT_PATH` 指向包含 `gjemnessund.h5` 的目录。

| 信息 | 官方属性/公开证据 | 本次使用边界 |
| --- | --- | --- |
| 坐标 | 桥中跨桥面原点；X 沿桥、正北；Y 横桥、正西；Z 向上 | 保存原文；不强行从 east/west 名称推导符号 |
| 两组三轴加速度 | G5 east / G5 west；各自 x/y/z；g；属性转换矩阵为单位阵 | 使用既有六通道顺序，均作为加速度观测 |
| 风 T/U/angle/w | 单位分别 degC、m/s、deg、m/s；原采样 32 Hz | 不臆定 angle 的零方向/旋向、U 的精确定义或风荷载系数 |
| 加速度位置属性 | east `[-269,4.5,0]`；west `[-269,-4.5,0]` | 属性位置与命名/公开示意需要交叉核实；暂不用其推导硬几何关系 |
| 结构参数 | 当前没有可靠质量、刚度、阻尼矩阵、完整振型及边界条件 | 不写 Euler–Bernoulli 梁 PDE 残差，不声称精确恢复真实结构方程 |
| 预处理 | 官方说明做了最低限度处理和降采样 | 不假设未经确认的滤波器、去偏置、抗混叠参数或角度定义 |

模态频率来自同桥研究 [Fenerci 等，2023，Table 3](https://doi.org/10.1016/j.jweia.2023.105583)，[作者公开全文](https://www.researchgate.net/publication/375175945_Full-scale_monitored_wind_and_response_characteristics_of_a_suspension_bridge_compared_with_wind_tunnel_investigations_at_the_design_stage)。采用 **Full-scale** 列，而不是 Prototype 或 Aeroelastic 列：

| 模态 | 频率 Hz | 当前 2 Hz 数据是否采用 |
| --- | ---: | --- |
| 侧向对称 | 0.090 | 是 |
| 竖向反对称 | 0.176 | 是 |
| 竖向对称 | 0.220 | 是 |
| 侧向反对称 | 0.250 | 是 |
| 扭转对称 | 0.720 | 是，先验可信度仍需训练段谱检查 |
| 扭转反对称 | 1.400 | 否，超过 Nyquist |

这些是跨工况的文献先验，不是每个窗口的精确真值。旧 PULSE 的 `inv_len=96` 是技术配置，不能解释为桥梁固有周期。后续仅使用训练段识别谱峰/调整先验，不能根据测试集选频。

## 3. 本地单位核对

**结论：当前本地文件的原始六路加速度按 g 使用；不能再把它误当作 m/s²。** 已按记录 ID 对照官方对应的六个 `unit=g` 数组，核对三个完整记录，共 65,160 个加速度数值。全部通过绝对误差 `1e-12` 的逐点比较，最大绝对差 `1.3877787807814457e-17`，属于浮点重新打包误差，不宣称二进制数组哈希完全一致。

| 本地记录 ID | 每通道行数 | 比较通道数 | 官方单位 | 结论 |
| --- | ---: | ---: | --- | --- |
| `NTNU_134_2013_11_27_10_06_12` | 3620 | 6 | g | 数值一致，浮点容差内 |
| `NTNU_134_2013_11_28_00_10_26` | 3620 | 6 | g | 数值一致，浮点容差内 |
| `NTNU_134_2013_11_30_21_06_58` | 3620 | 6 | g | 数值一致，浮点容差内 |

完整证据：[unit_audit.json](data/gje_physics_20261110/unit_audit.json)，包含官方传感器、通道和坐标属性，以及本地/官方每个数组的哈希、差值、原始文件身份与核对范围。这里确认的是这份本地重打包文件的单位来源和抽样数值对应；没有遍历全部记录，也没有完成官方 542.9 MB 文件的全量 MD5。服务器若使用不同来源或另行缩放的数据，需重新核对，不能重复乘重力加速度。

可复查命令（仅读原 H5；缓存和下载块放 `/tmp/`）：

```bash
python notes/tools/audit_gje_physics_metadata.py \
  --local-h5 /实际路径/gjemnessund.h5 \
  --output /tmp/gje-unit-audit.json
```

审计工具额外使用 `requests`；模型运行仅使用现有 `requirements-gje.txt` 的依赖，本次不修改依赖文件。官方元数据与原始数组是单位的证据，网络预测误差不能用于推断单位。

## 4. 两个独立原型的设计

### 4.1 PULSE_FACT_MODAL：模态频率软先验

保持 PULSE_FACT 原有归一化、MLP、FACT、Router、恢复和训练 SAM 混合。以最后一个历史采样点为 `t=0`，仅用历史加速度估计各频率的 sin/cos 系数，再外推到未来 `t=0.5,1.0,…` 秒。加入有界可学习增益的模态残差。它是软频率归纳偏置，不要求预测严格落在有限模态子空间，不施加 PDE。

具体流程：`[B,L,10] → 原 PULSE_FACT → [B,H,10]`；并行取混合历史的前六通道，反 StandardScaler 后由 g 转为 SI，去历史均值，使用中心化谐波基做 ridge 投影。未来基乘历史系数得到模态外推，乘有界逐通道增益，再按 `scale_g × 9.80665` 转回标准化残差，添加到基模型六通道输出。四个风通道不经过物理分支。

设历史基为 `Φx`、未来基为 `Φy`、中心化历史为 `ac`，则系数为 `(ΦxᵀΦx + λL I)⁻¹Φxᵀac`。基的中心只由历史决定。这里没有从未来真实信号估计相位，也不把幅值相近的左右传感器强制相等。

### 4.2 PULSE_FACT_OSC：稳定受迫振荡层

仅用历史及 PULSE_FACT 自己预测的未来作为条件，估计模态初态和低维平滑受迫项。每个模态满足名义受迫振荡方程 `q¨ + 2ζω q˙ + ω²q = f`；频率在文献值附近有界调整，阻尼严格为正，采用零阶保持精确离散化而非可能不稳定的显式 Euler。六路动态加速度必须经过模态观测映射输出，之后不再添加自由的时变加速度残差；四路风通道使用基模型预测。

历史加速度均值作为传感器/缓变偏置独立保存，动态部分在 SI 单位计算，输出恢复为现有 StandardScaler 空间。这个名义层的稳定极点是结构保证；受迫系统的能量可以上升，不能宣称所有预测单调衰减，也不能把学习的模态、外力、观测矩阵解释成已辨识真实桥梁参数。没有未来真实风或真实加速度进入 forward。

具体状态采用 `p=ω²q/a_ref`、`r=ωq˙/a_ref`，其中 `a_ref` 是仅从历史六通道计算的动态加速度 RMS。状态微分矩阵为 `M=ω[[0,1],[-1,-2ζ]]`，无量纲输入为 `u=f/a_ref`。每步 `A=exp(MΔt)`、`B=(I−A)[1,0]ᵀ`，递推 `z[t]=A z[t−1]+B u[t]`。模态加速度为 `a_ref × (−p−2ζr+u)`，经学习的六传感器观测映射输出，并加回历史 DC 偏置。这是质量归一化的有效振荡器；`f` 不是已知牛顿力，潜在 `q` 也不是经过测量验证的桥面位移。

实现用补零 FFT 计算与递推相同的**线性因果卷积**，不进行未来序列循环卷积。输入受迫轨迹由最多 8 个有界控制点线性插值，避免每个未来采样点一个自由外力。初态来自最后最多 64 步历史；受迫控制来自基模型自己的完整预测，包含预测风信息。所有条件最终都来自历史或已知日历，不能用未来真实风。离散极点模长为 `exp(−ζωΔt)<1`；FFT 数值计算存在浮点误差，不声称零数值残差。

5 个潜在模态和学习的观测矩阵不能代替真实振型；尤其纵向模态信息不足，六通道的低维观测假设需实验检验。固定历史 DC 偏置也可能限制缓慢漂移预测。8 点平滑受迫项在长预测时距可能过度抑制宽带响应，这些属于明确待验证的结构假设。

两个新模型均沿用既有 SAM 和 FFT 训练目标；物理分支也使用同一 `lam/index` 混合后的历史，确保输入、条件和监督一致。不更改旧模型的损失、增强或训练流程。

### 4.3 参数、可调范围与保存方式

| 模型/参数 | 初始值 | 说明 |
| --- | --- | --- |
| 两者的 FACT | hidden 512、d_ff 256、dilation 2、kernels 2、dropout 0.1 | 继承原 PULSE_FACT 实际默认设置 |
| 两者的模态频率 | `[0.090,0.176,0.220,0.250,0.720]` Hz | 公开先验；不含 1.4 Hz |
| `pulse_modal_ridge` | `0.001` | ridge 计算实际加 `λL I` |
| `pulse_modal_gain_limit` | `0.25` | 学习增益为 `limit × tanh(raw_gain)`，可正可负 |
| `pulse_modal_initial_gain` | `0.05` | 每个加速度通道从弱先验开始 |
| `pulse_osc_frequency_band` | `0.1` | `f=f0(1+0.1 tanh δ)`，相对偏移受限，整体仍在 Nyquist 以下 |
| `pulse_osc_damping_min/max` | `0.002 / 0.1` | 正阻尼和欠阻尼范围，是设计假设而非官方实测值 |
| `pulse_osc_initial_damping` | `0.02` | 通过 sigmoid 参数化学习 |
| `pulse_osc_force_controls` | `8` | 实际 `min(8,H)` 个控制点，无自由 H 点外力 |
| `pulse_osc_force_limit` | `2.0` | 无量纲受迫控制幅值上界 |
| OSC 初态限制 | `2 × tanh(head)` | 有界潜在初态，由历史估计 |
| 采样间隔/重力换算 | `0.5 s / 9.80665 m/s²` | 固定物理尺度，不作为可学习参数 |

模型主文件完整展开原有计算，无薄继承或别名隐藏数据流。来源与复制差异保存在 [sources.json](data/gje_physics_20261110/sources.json)，FACT 的原始副本和许可证继续位于 `layers/pulse_fact/vendor/`。

上述 `pulse_modal_*` / `pulse_osc_*` 是构造配置字段，**本次没有新增同名 CLI 参数**，不要直接将它们写成 `run.py --pulse_osc_...`。两个基础脚本使用主文件中明确记录的默认配置。通过 Python 直接实例化可做机制检查；正式调参若需 CLI，按约束另行确认入口接入。模型配置存于 `_extra_state`，训练集标定存于 `acceleration_units` 的 buffer/extra state，加载不同配置或不同数据协议会拒绝。

## 5. 首轮脚本、输出与后续实验

首轮沿用既有 PULSE_FACT 大桥脚本的设置：`seq_len=720`、预测 `96/192/336/720`、10 通道、6 维日历标记、batch 16、30 epoch、patience 5、seed 2024、原 FFT 损失。数据切分继续逐记录 50/25/25、仅训练行拟合 scaler、不跨记录窗口。

从 `PULSE-phy/` 根目录运行：

```bash
bash scripts/PULSE_FACT_MODAL/Gjemnessund.sh
bash scripts/PULSE_FACT_OSC/Gjemnessund.sh
```

数据不在 `./all_datasets/` 时，在每条命令前加 `GJE_ROOT_PATH=/实际/H5目录`，文件名保持 `gjemnessund.h5`。脚本只包含顶部变量、路径提示和四次直接 `python -u run.py`；没有 shell 函数或菜单，失败由 `set -e` 立即传递。每个模型仍独立运行，不默认并发。

```text
PULSE-phy/
├── logs/
│   ├── PULSE_FACT_MODAL/Gjemnessund/
│   │   ├── PULSE_FACT_MODAL_720_96.log
│   │   ├── PULSE_FACT_MODAL_720_192.log
│   │   ├── PULSE_FACT_MODAL_720_336.log
│   │   └── PULSE_FACT_MODAL_720_720.log
│   └── PULSE_FACT_OSC/Gjemnessund/
│       ├── PULSE_FACT_OSC_720_96.log
│       ├── PULSE_FACT_OSC_720_192.log
│       ├── PULSE_FACT_OSC_720_336.log
│       └── PULSE_FACT_OSC_720_720.log
├── checkpoints/
│   ├── PULSE_FACT_MODAL/<setting>/checkpoint.pth
│   └── PULSE_FACT_OSC/<setting>/checkpoint.pth
└── results/<setting>/gje_metrics.json
```

每个 checkpoint 目录还保存原有数据协议及本次模型/单位/先验来源。固定日志再次运行会覆盖，先归档旧日志；checkpoint 继续保留独立实验标识。正式实验结果全部 **待运行**。

后续顺序：先比较 Acc6 及逐通道 MSE/MAE，兼看全 10 通道和 Wind4；再做增益为零、移除/错置频率、无受迫项、固定/学习阻尼等消融。运行 3–5 个随机种子，按记录统计误差，检查短时预测收益、长时误差、频谱和突发风响应。频率/阻尼调整只在训练及验证部分完成。有限模态可能压制宽带或局部响应，若预测下降应如实记录；硬输出原型失败不等于已证明数据没有物理规律。

下一轮建议按下表推进；以下是后续计划，不是本次已交付功能：

| 阶段 | 如何做 | 后续文件范围 |
| --- | --- | --- |
| 先跑统一比较 | 保持旧 PULSE_FACT 和两个新模型的切分、SAM、FFT、预算一致；首先看 Acc6，不只看十通道平均值 | 使用本次两份脚本和旧脚本；运行结果填回本 MD，新增结果汇总工具时放 `notes/tools/` |
| 训练段频谱审计 | 对每条记录只取 `[0,train_end)`，检查公开模态附近谱峰、带宽及东西侧相干性；不能由测试谱选频 | 拟新增 `notes/tools/audit_gje_training_spectra.py` 和 `notes/data/` 下审计输出；不改当前 loader |
| 模态先验消融 | 增益零控制已做数值等价检查；正式训练再比较无先验、正确/错置频率、不同弱增益 | 保留当前模型；正式 CLI 调参需单独确认，或者新增明确命名的独立消融模型与脚本 |
| 振荡层消融 | 比较固定/学习频率与阻尼、零受迫/8 控制点受迫、不同观测维数；避免把参数量变化当作物理贡献 | 新的专属类/模型/脚本按机制命名；本次 `PULSE_FACT_OSC.py` 和结果保留 |
| 强化论文证据 | 增加多随机种子、逐记录误差分布、物理频带误差和极端风记录案例；额外整记录留出用于跨工况验证 | 统计与绘图工具放 `notes/tools/`；如新增数据协议，使用独立 adapter/数据名并另行确认入口，不改旧切分 |

继续研究的判断依据应包括：Acc6 提升是否跨种子稳定、是否只提升某个通道/时距、是否以明显损伤宽带响应为代价。名义层的数学稳定性是实现属性，不能单独作为桥梁物理正确性和预测优势的证据。

## 6. 实现与验证记录

已实现第 1 节的两个独立模型、三个专属层、单位模块、两份大桥脚本和检查；没有新增其他数据集脚本。共享接入仅为 `exp/exp_main.py` 的两个模型导入/字典注册、新模型专用 scaler 绑定和元数据范围，以及 `utils/gje_model_support.py` 的注册列表/来源记录。旧桥梁接口检查仅更新允许模型列表的断言。没有更改 `run.py`、数据划分、优化器、损失或训练调用顺序。

已完成验证：

| 检查 | 结果 |
| --- | --- |
| 新物理模块与两个模型 | 9 项检查，8 通过、1 项 CUDA 因无设备跳过；四个时距均完成严格确定性 CPU FFT 梯度、优化更新、checkpoint 保存恢复 |
| 原 PULSE/SCF/FACT 大桥接口回归 | 5 项通过，包括原入口合成 H5 的训练、恢复和独立测试 |
| 原 PULSE_FACT 回归 | 15 项检查，14 通过、1 项 CUDA 因无设备跳过 |
| 原大桥数据处理回归 | 8 项通过，包括训练段 scaler、边界和时间特征 |
| 合计 | 37 项，35 通过、2 项因没有 CUDA 跳过 |
| Python 3.8 兼容 | Python 3.8.20 / PyTorch 2.0.0 CPU：两个新模型的小输入前向和 FFT 反向均通过 |
| 真实记录检查 | 一个真实记录、两个模型、四个时距，共 8 组前向/FFT 梯度/推理均有限；只拟合该记录训练半段 scaler，缓存放临时目录并删除 |
| 旧实现保护 | 起点已有源码仅修改上述 3 个允许的接入/检查文件；`run.py` 和所有原模型、原数据处理、原脚本、用户已有笔记哈希保持原样 |
| 训练流程保护 | 去掉新增模型注册/绑定/元数据条目后，`Exp_Main` 的 AST 与起点完全一致 |
| 既有产物保护 | 原 `logs/`、`checkpoints/`、`results/` 共 271 个文件的路径、大小、修改时间保持原样；本地原始 H5 大小及修改时间不变 |

记录文件：[validation.json](data/gje_physics_20261110/validation.json)、[真实记录检查](data/gje_physics_20261110/real_record_checks.json)。原入口完整检查在隔离的临时工程进行，未在正式输出目录留下随机权重或合成指标。

默认 `seq_len=720` 的可训练参数量如下，帮助后续判断增加参数的影响：

| 预测长度 | PULSE_FACT | PULSE_FACT_MODAL | PULSE_FACT_OSC |
| --- | ---: | ---: | ---: |
| 96 | 3,340,636 | 3,340,642 | 3,385,526 |
| 192 | 3,528,416 | 3,528,422 | 3,611,706 |
| 336 | 3,982,886 | 3,982,892 | 4,123,776 |
| 720 | 6,208,566 | 6,208,572 | 6,503,056 |

复查命令：

```bash
python -B -m unittest discover -s tests -p test_gje_physics.py -v
python -B -m unittest discover -s tests -p test_gje_extended_models.py -v
python -B -m unittest discover -s tests -p test_pulse_fact.py -v
python -B -m unittest discover -s tests -p test_gjemnessund.py -v
```

本地完整检查使用临时 h5py 环境 `/tmp/gje-metadata-read-20261110/bin/python`，没有安装或改写用户已有 conda 环境。**正式大桥训练、CUDA 检查及论文贡献验证均尚未完成**；合成数据和真实单记录检查只验证接口、数值和训练路径，不能作为论文指标。当前没有因物理先验而宣称的性能提升。
