# PULSE_FACT_PC：第一版相位条件化交互开发计划与记录

日期：2026-11-04。代码起点：`c6cfcf625e1280144e3af41f090c8980099aabed`。本文件先于模型、层和脚本的新增写入；先记录授权范围和方案，实施后再更新实际验证结果。

## 1. 开发依据与保护范围

- 约束入口：[AGENTS.md](../../AGENTS.md)，遵守第 2 节开闭原则、第 3 节训练入口保护、第 4 节来源记录、第 8—9 节脚本／输出规则、第 10 节完整主模型风格。
- 研究依据：[论文编写与研究思路](PULSE_FACT论文编写与研究思路.md)、[论文创新与调参](PULSE_FACT论文创新与调参.md)、[原 PULSE_FACT 开发记录](PULSE-FACT开发计划与记录.md)。
- 开始时已有 `.DS_Store`、`.gitignore`、`scripts/Gjemnessund.sh` 修改、历史日志删除、`.vscode/`、论文笔记及图表／分析工具等未跟踪内容。全部保留，不覆盖、清理或提交。
- 已在临时快照记录既有源码／脚本／约束／参考来源的哈希。原 `models/`、`layers/`、旧脚本、数据处理和 `run.py` 保持原样；旧 `exp/exp_main.py` 只增加两个模型的导入与注册，交付前比较其余字节及源码行为。
- FACT-main、SOFTS-main 继续只读；新模型运行时仅依赖 PULSE-phy 内部模块。不新增替代训练入口、不修改参数解析、损失或训练流程。
- 本轮实现模型与三组诊断脚本，完成相关最小检查；不启动正式数据集训练，不推送 GitHub。正式训练与论文贡献验证分别记录。

## 2. 名称、问题与贡献边界

**机制工作名：PCSM — Phase-Conditioned Scale Mixing（相位条件化尺度混合）。** 这是可供论文讨论的描述性名称，尚未完成名称查重、近邻查新或贡献验证，不据此宣称首创。

- 主模型标识：`PULSE_FACT_PC`，在 PULSE_FACT 的单个交互块内加入 PCSM。
- 对照模型标识：`PULSE_FACT_STATIC`，仅学习所有样本共用的分支权重。
- 当前 `PULSE_FACT` 为固定权重 1／1 的直接组合基准，保持实现、脚本和已归档结果。

要检验的问题：**相比固定或全局可学习的交互尺度比例，历史相位参照条件化的尺度选择是否提供额外收益？** 本轮不同时增加深度、时滞、频域分支或损失。

当前设计是样本级膨胀分支权重，不是变量对动态图，也没有新增物理方程或真实拓扑。相位参照包含学习 codebook 与历史时间编码，其物理含义仍受原论文编写文档中的证据边界约束。

## 3. 修改前后与接口

| 位置 | 修改前：PULSE_FACT | 新主模型：PULSE_FACT_PC | 静态对照：PULSE_FACT_STATIC |
| --- | --- | --- | --- |
| 主模型组织 | 完整 ReVIN／Model | 新文件完整展开，保留原数据流 | 另一新文件完整展开 |
| 相位参照、归一化、Router／恢复 | 原 PULSE 语义 | 保留 | 保留 |
| FACT 输入与输出 | `[B,C,512]` | 不变 | 不变 |
| 两个膨胀分支 | `U+f1(U)+f2(U)` | `U+w1(q)f1(U)+w2(q)f2(U)` | `U+w1*f1(U)+w2*f2(U)` |
| 条件来源 | 无显式条件控制 | 仅历史参照 `A_x` | 无样本条件，两个可学习 logits |
| 分支后 FFN | 原内部残差 | 保留，不再额外加 identity | 保留 |
| SAM | 混合输入、均值／标准差、监督 | 继承，并对新增条件特征用相同 lambda／index 混合 | 继承 |
| 外层深度 | 1 | 1 | 1 |
| 训练目标与验证 | 原 FFT 目标／MSE 验证 | 不变 | 不变 |

主模型接口仍为 `forward(x, cycle_index, batch_x_mark, batch_y_mark, do_mixup=True)`。输入 `x:[B,L,C]`、历史／未来时间标记 `[B,L,time_dim]`／`[B,H,time_dim]`；训练返回 `(prediction, mixup_params)`，推理只返回 `[B,H,C]`。当前继续拒绝未验证的 AMP／多 GPU SAM 路径。

## 4. PCSM 计算定义与初始化

按每个变量的历史参照序列提取有序特征，再沿变量轴平均：

\[
q_b=\frac{1}{C}\sum_{c=1}^{C}\operatorname{GELU}\left(W_a A_{x,b,:,c}+b_a\right),\quad W_a:L\to16.
\]

不先沿时间直接取均值，避免整周期平均抹掉相位差异。该条件编码器沿变量共享；这不使原 FACT 索引网格整体具有置换不变性。

\[
w_b=m\operatorname{softmax}(g(q_b)),\quad
V=U+\sum_{j=1}^{m}w_{b,j}f_j(U),\quad Y=V+\operatorname{FFN}(V).
\]

默认 `m=2` 是膨胀分支数量，对应 dilation 1／2，不是多核数量。权重和固定为 m，初始 logits=0 得到权重全 1，保持原求和尺度。条件网络为 `Linear(16,16)→GELU→Linear(16,m)`，最后一层权重与偏置置零。静态对照的 m 个 logits 初始也为零。

原 PULSE 模块和 FACT 算子先构造，新增编码器／控制器最后构造，使相同种子下的原有参数初始化可对齐。新增条件网络在局部 CPU 随机状态中初始化并恢复构造后的原状态，避免额外初始化移位数据加载、SAM 和 dropout 的随机序列。需要比较初始化随机状态、前向、共享参数梯度与 BN 状态，不仅检查输出形状。

SAM 训练时，先从各自原始历史参照编码 q，再执行 `q_mix=lambda*q+(1−lambda)*q[index]`；不线性混合整数周期索引，不加入未来真实观测。原 Router 仍使用原锚点约定，不声称所有路径都已获得物理一致的混合解释。

## 5. 文件计划与来源

| 文件 | 计划职责 |
| --- | --- |
| `models/PULSE_FACT_PC.py` | 显式完整主模型，展开 norm／SAM／条件混合／主干／Router／恢复 |
| `models/PULSE_FACT_STATIC.py` | 显式完整静态对照，不用别名或 variant 隐藏计算 |
| `layers/pulse_fact_pc/phase_condition.py` | 历史参照编码、动态／静态分支权重 |
| `layers/pulse_fact_pc/interaction.py` | 加权来源卷积，独立形状适配和周期补齐 |
| `layers/pulse_fact_pc/sources.json` | 源文件哈希、版本、许可证位置与实际修改说明 |
| `tests/test_pulse_fact_pc.py` | 等价、条件响应、SAM、梯度、恢复、注册、脚本及保护检查 |
| `scripts/PULSE_FACT_PC/{PEMS03,PEMS04,ETTh1}.sh` | 主模型单时距诊断脚本 |
| `scripts/PULSE_FACT_STATIC/{PEMS03,PEMS04,ETTh1}.sh` | 对照模型单时距诊断脚本 |
| `exp/exp_main.py` | 只增加两个导入／注册条目 |

新主模型从工程内 `models/PULSE_FACT.py` 复制后独立扩展；新交互适配器参考 `layers/pulse_fact/fact_block.py`，使用已有 `layers/pulse_fact/vendor/fact_conv.py` 中的来源算子。旧文件及许可证不改动。保留 PULSE／FACT 来源署名及原许可证链，不将复制部分算作自主机制。

专属设置分别采用 `pulse_fact_pc_*` 与 `pulse_fact_static_*` 命名，模型构建时打印并写入 checkpoint `_extra_state`。旧 `d_model` 继续表示 Router 宽度，MLP 隐藏宽度固定 512；旧 `e_layers` 不控制本轮深度。这些专属字段暂不增加 CLI，脚本使用明确记录的默认值，不能写入未经入口支持的参数。

## 6. 首轮配置与脚本／输出

| 数据集 | L→H | C | Router d | inv_len | patch | dsa/dsb | ksize | batch | LR |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| PEMS04 | 96→96 | 307 | 16 | 48 | 12 | 4/1 | 5 | 32 | 0.005 |
| PEMS03 | 96→96 | 358 | 32 | 12 | 12 | 4/2 | 5 | 32 | 0.005 |
| ETTh1 | 96→96 | 7 | 16 | 96 | 24 | 1/1 | 5 | 256 | 0.005 |

各配置复制对应 `scripts/PULSE_FACT/` 中 H=96 的命令块；历史长度、数据切分、预处理、时间标记、训练预算与 loss 参数保持对应设置。轮数 30，patience 5，种子 2024，itr 1，`rec_lambda=0, auxi_lambda=1`，共 6 个新训练命令。已有基准结果需核验版本与配置才可复用。

FACT 默认设置：D=512、FFN=256、膨胀分支 m=2、多核数=2、dropout=0.1。PC 条件宽度与控制器宽度均为 16；STATIC 仅有两个可学习 logits。二者参数量不同，实际统计后报告，不称为严格等参数对照。

脚本从 PULSE-phy 根目录运行，保持顶部变量、直接 `python -u run.py`、`set -e`、无 shell 函数和选择器；每份仅含 H=96 的命令，并打印日志绝对路径。使用现有参数，不新增一键训练入口。

```bash
bash scripts/PULSE_FACT_PC/PEMS04.sh
bash scripts/PULSE_FACT_PC/PEMS03.sh
bash scripts/PULSE_FACT_PC/ETTh1.sh
bash scripts/PULSE_FACT_STATIC/PEMS04.sh
bash scripts/PULSE_FACT_STATIC/PEMS03.sh
bash scripts/PULSE_FACT_STATIC/ETTh1.sh
```

```text
PULSE-phy/
├── logs/
│   ├── PULSE_FACT_PC/
│   │   ├── PEMS04/PULSE_FACT_PC_96_96.log
│   │   ├── PEMS03/PULSE_FACT_PC_96_96.log
│   │   └── ETTh1/PULSE_FACT_PC_96_96.log
│   └── PULSE_FACT_STATIC/
│       ├── PEMS04/PULSE_FACT_STATIC_96_96.log
│       ├── PEMS03/PULSE_FACT_STATIC_96_96.log
│       └── ETTh1/PULSE_FACT_STATIC_96_96.log
└── checkpoints/
    ├── PULSE_FACT_PC/<setting>/checkpoint.pth
    └── PULSE_FACT_STATIC/<setting>/checkpoint.pth
```

同名日志重跑按既有重定向规则覆盖，需要多种子或复跑时先归档日志；checkpoint 使用独立运行标识。常规训练并自动测试沿用已有路径；独立 `is_training=0` 的旧加载分支问题仍未在本轮适配。

## 7. 验收与保护检查

1. 全 1 权重的交互前向、梯度、BN 状态与原块一致；初始化新模型与 PULSE_FACT 的 eval／非混合训练／SAM 输出对齐。
2. 权重非负、每样本求和为 m；静态权重不随样本变，动态条件头解除零初始化后能响应参照差异，时间顺序编码不等同时间均值。
3. 验证条件特征与 SAM 使用同一 lambda／index，原目标接口保持兼容；不引入标签依赖或未来观测。
4. 检查小 C 周期补齐、三组真实配置前向、严格确定性 CPU FFT 反向与优化步；CUDA 无设备则明确跳过并给出服务器命令。
5. 检查保存重载、模型／配置不匹配保护、有效配置打印、参数量及原实验类注册／验证兼容。
6. 检查六份脚本语法、入口参数解析、基准 H=96 配置对齐、日志位置与失败状态；只用临时 stub 验证 shell，不启动训练。
7. 复核旧模型／层／脚本／参考目录／run.py 哈希，确认 exp_main 增量仅为导入和注册。必要时运行相关旧测试，检查新增导入不会破坏既有路径。

## 8. 当前实施状态与待补实验

当前：两个独立完整主模型、新命名空间的条件／交互模块、来源记录、六份诊断脚本和检查已实现；正式训练、CUDA 检查及论文贡献验证待进行。第 5 节所列文件均已交付，原模型／层／脚本和 run.py 未改变。

### 8.1 参数量与有效配置

| 数据集，96→96 | 原 PULSE_FACT | PULSE_FACT_STATIC | PULSE_FACT_PC |
| --- | ---: | ---: | ---: |
| PEMS04 | 726,384 | 726,386 | 728,242 |
| PEMS03 | 648,732 | 648,734 | 650,590 |
| ETTh1 | 473,542 | 473,544 | 475,400 |

STATIC 增加 2 个 logits；PC 在 L=96 时增加 1,858 个参数，其中历史编码器 1,552，控制器 306。这里的参数量不是 GPU 速度或显存实测，也不是等参数对照。

PC 专属默认字段为 `pulse_fact_pc_d_ff=256`、`pulse_fact_pc_dilation=2`、`pulse_fact_pc_num_kernels=2`、`pulse_fact_pc_dropout=0.1`、`pulse_fact_pc_context_dim=16`、`pulse_fact_pc_controller_hidden=16`；STATIC 对应前四个字段使用 `pulse_fact_static_` 前缀。均通过直接配置属性读取，现有 CLI 不增加这些参数。有效值会打印到日志并保存进 `_extra_state`，配置或模型类型不符时拒绝静默恢复。

### 8.2 实际检查结果

| 环境／检查 | 结果 |
| --- | --- |
| Python 3.13.5／PyTorch 2.9.0：新增检查 | 16 项 CPU 通过，1 项 CUDA 跳过 |
| Python 3.8.20／PyTorch 2.0.0：兼容检查 | 16 项 CPU 通过，1 项 CUDA 跳过 |
| 原 `test_pulse_fact.py` 回归 | 14 项 CPU 通过，1 项 CUDA 跳过 |
| 六份脚本 | shell 语法、原 argparse 支持、对应原 H=96 配置、固定日志／分组 checkpoint、失败状态均通过 |
| 三任务 × 两模型实际配置形状 | 随机输入 B=1 的 eval 输出 `[1,96,C]` 均通过；未使用正式训练数据 |
| 初始化等价性 | 原有参数、CPU RNG、eval／非混合训练／SAM 预测，以及共享参数梯度对齐 |
| SAM 条件配对 | hook 核对 q_mix 与同一 lambda／index 一致；新增条件不依赖观测残差或未来时间标记 |
| 交互与条件响应 | 全 1 权重时与原块的前向／梯度／BN 状态一致；小网格、非均匀权重与时间顺序编码通过 |
| FFT 目标与保存恢复 | 两个严格确定性 CPU 优化步、控制器更新、有限梯度、checkpoint 回读与配置保护通过 |
| 旧实现保护 | 193 个既有文件的哈希核查中，仅 exp_main.py 有允许的修改；移除新增的 1 行导入与 2 行注册后，与原内容完全一致 |

零初始化的动态输出头使首步前级条件编码器梯度为零，这是预期的初始等价设计；输出头更新后第二步已检查到前级编码器的非零梯度，不能把首步零梯度误认为断图。`fact_block.last_branch_weights` 提供最近一次 forward 的 detached 权重用于诊断，既不额外改变训练返回值，也不作为 checkpoint 缓冲保存；它不是全数据集统计。

检查快照、参数量和旧文件哈希见 [开发检查 JSON](data/pulse_fact_pc_development_checks.json)，来源链见 [sources.json](../layers/pulse_fact_pc/sources.json)。当前本机无 CUDA，CPU 通过不代替目标服务器的 CUDA 确定性／显存验证。

在 PULSE-phy 根目录、已配置依赖的环境中运行：

```bash
python -B -m unittest discover -s tests -p test_pulse_fact_pc.py -v
```

在有 CUDA 的服务器执行同一命令会额外运行 CUDA 分支。验证后再执行第 6 节对应数据集脚本；本轮没有启动这六个正式训练命令。

### 8.3 待运行结果

| 模型 | 数据集 | L→H | seed | MSE | MAE | 验证集分支权重统计 |
| --- | --- | --- | ---: | --- | --- | --- |
| PULSE_FACT_PC | PEMS04 | 96→96 | 2024 | 待运行 | 待运行 | 待分析 |
| PULSE_FACT_PC | PEMS03 | 96→96 | 2024 | 待运行 | 待运行 | 待分析 |
| PULSE_FACT_PC | ETTh1 | 96→96 | 2024 | 待运行 | 待运行 | 待分析 |
| PULSE_FACT_STATIC | PEMS04 | 96→96 | 2024 | 待运行 | 待运行 | 所有样本共用，待分析实际比例 |
| PULSE_FACT_STATIC | PEMS03 | 96→96 | 2024 | 待运行 | 待运行 | 所有样本共用，待分析实际比例 |
| PULSE_FACT_STATIC | ETTh1 | 96→96 | 2024 | 待运行 | 待运行 | 所有样本共用，待分析实际比例 |

正式训练后先看验证集选模、测试 MSE／MAE与分支权重变化；单种子三任务仅是探索诊断。若有依据，再补种子与时距，随后开展 STATIC／PC × N=1／2 的深度对照；共享、嵌套和损失改造留给后续独立方案。
