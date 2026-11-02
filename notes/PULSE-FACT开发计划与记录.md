# PULSE＋FACT 开发计划与记录

日期：2026-11-02。代码起点：`a1d555d50ca4bae27d1053b5fe31cfe925af2266`。本轮先保存风格与方案，再开发代码；当前独立模型、来源复制、注册、十二份脚本和一键脚本已实现，CPU 检查通过，正式训练与目标 CUDA 验证待进行。

## 1. 开发范围与风格

开发目录为 `PULSE-phy/`，遵守工作区 [AGENTS.md](../../AGENTS.md)，特别是第 3 节训练入口保护及第 10 节主模型可读性要求。FACT-main、SOFTS-main 只读。

新建 `models/PULSE_FACT.py`，显式定义 `ReVIN`、`Model(nn.Module)`，完整展开初始化、norm、phase_router、denorm、forward，不继承旧模型或通过 variant 隐藏计算。原 `PULSE.py` 及之前的三个新模型、模块、脚本不改写。

开始时已有 `.DS_Store`、`.gitignore`、历史日志删除、`scripts/Gjemnessund.sh`、前轮结果分析文档等未提交变化；全部保留。开始时将 184 个既有源码、脚本及参考目录文件的哈希记录到本次临时快照，交付时只允许 `exp/exp_main.py` 有导入／注册变化。

## 2. 结构与接口

```text
输入 x [B,L,C]
  → 原 PULSE 时间编码与 norm
  → 原 SAM 输入及统计量混合
  → Linear(L,512) → GELU
  → PULSEFACTBlock [B,C,512] → [B,C,512]
  → Linear(512,H)
  → 原 phase_router / denorm
  → 预测 [B,H,C]
```

保留原 MLP 的两层映射和 512 隐藏维度。`self.backbone` 仍声明为原三层 Sequential，forward 明确逐层调用，在 GELU 与输出线性层之间插入独立的 `self.fact_block`。FACT 在原 PULSE 模块全部初始化后创建，使同一种子下的原有参数初始化可直接对齐。

FACT 只处理真实变量，不追加时间标记 token；网格尺寸沿用来源 `floor(sqrt(C))+1`。适配器先沿变量轴用切片、repeat、cat 周期补齐，再调用原样复制的 Conv_block2d，最后裁剪并恢复轴。保留来源内部残差，不再额外相加。不引入 STAR、共享核心偏差或状态门控。

训练返回 `(prediction, mixup_params)`，推理只返回 prediction；SAM 与原 PULSE 使用相同的输入混合、mean/sigma 配对及原 Router 锚点约定。不支持既有实验流程尚不完整的 AMP 或多 GPU SAM 分支。

这里的 PULSE＋FACT 指 PULSE 的 MLP 中加入 FACT 变量交互块，不是完整 FACT 时频双分支网络的复现。

## 3. 首版配置

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| MLP 隐藏维度 | 512，固定 | 原 PULSE 的隐藏宽度 |
| pulse_fact_d_ff | 256 | FACT 点卷积 FFN 宽度 |
| pulse_fact_dilation | 2 | 并行膨胀层数量，实际 dilation 为 1、2 |
| pulse_fact_num_kernels | 2 | 每个 Inception 的核数 |
| pulse_fact_dropout | 0.1 | FACT FFN 的 dropout |

专属字段用于直接实例化，原 run.py 尚无这些 CLI 参数；脚本使用固定默认值，不修改旧 `d_model` 或 `d_ff` 的含义。`d_model` 继续控制 PULSE Router。有效设置写入权重 `_extra_state` 并在构建时打印，避免日志只含旧参数而无法核验实际 FACT 设置。

## 4. 文件计划

| 文件 | 本轮职责 |
| --- | --- |
| models/PULSE_FACT.py | 独立完整主模型 |
| layers/pulse_fact/fact_block.py | 形状、补齐、来源模块接入 |
| layers/pulse_fact/vendor/fact_conv.py | 原样摘取 Inception2d 与 Conv_block2d |
| layers/pulse_fact/vendor/FACT_LICENSE | 来源许可证 |
| layers/pulse_fact/vendor/sources.json | 来源、原文哈希及适配说明 |
| exp/exp_main.py | 仅增加 PULSE_FACT 导入和注册 |
| scripts/PULSE_FACT/*.sh | 十二个数据集，原配置与原命令风格 |
| scripts/run_PULSE_FACT.sh | 逐行 bash 执行十二份数据集脚本 |
| tests/test_pulse_fact.py | 来源语义、梯度、基线等价、保存恢复、接入及脚本验证 |

来源从只读 FACT-main/models/FACT.py 复制，许可证原样保留；运行时不导入参考目录，也不依赖旧 pulse_scf 模块。

## 5. 脚本与输出计划

脚本名：ETTh1.sh、ETTh2.sh、ETTm1.sh、ETTm2.sh、electricity.sh、traffic.sh、Weather.sh、Solar.sh、PEMS03.sh、PEMS04.sh、PEMS07.sh、PEMS08.sh。

从对应 `scripts/PULSE/` 脚本复制配置，顶部固定 model_name=PULSE_FACT，逐个时距直接调用 `python -u run.py`，不定义函数、模型选择器或时距选择器。保留 set -e，标准输出与错误均重定向至日志。八个长预测数据集运行 96/192/336/720，四个 PEMS 运行 12/24/48/96，均为 L=96，共 48 条训练命令。

```text
logs/PULSE_FACT/ETTh1/PULSE_FACT_96_96.log
logs/PULSE_FACT/ETTh1/PULSE_FACT_96_192.log
logs/PULSE_FACT/PEMS03/PULSE_FACT_96_12.log
checkpoints/PULSE_FACT/<setting>/checkpoint.pth
```

日志不含运行标识目录，重跑同配置覆盖同名日志；checkpoint 继续使用独立实验标识。旧脚本、旧一键入口和历史产物不迁移或覆盖。

本次仅准备脚本和直接实例化检查，不启动正式训练。十二份脚本配置保持不变，不为显存推测而擅自降低 batch size。服务器先验证 CUDA 反向与实际显存，再启动全量训练。

## 6. 验收顺序

1. 对齐来源块的前向、梯度及 BatchNorm 状态，确认无重复残差。
2. 将 FACT 换为 Identity，对齐原有参数及随机状态，验证 eval、SAM 与不混合训练行为和原 PULSE 一致。
3. 检查 C=1/2/4/7/9/21/321/883 等变量数的补齐、梯度和有限性。
4. 完成模型 FFT 目标反向、优化步、严格确定性 CPU 检查；CUDA 检查保留服务器命令，无 GPU 时如实跳过。
5. 检查权重保存、重载预测一致及有效配置不匹配保护。
6. 通过原实验类最小注册，检查现有验证流程；run.py 不改。
7. 检查十三份 shell 语法、48 条参数与来源等价、日志位置、重复运行标识和一键失败停止。
8. 核对既有文件哈希及参考目录，更新实测记录与待办。

## 7. 实验结果与分析

| 数据集 | 时距 | 种子 | MSE | MAE | 相对 PULSE | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 十二个标准数据集 | 各四时距 | 2024（脚本默认） | 待运行 | 待运行 | 待比较 | 本轮不启动正式训练 |

该对照首先比较 PULSE 与 PULSE_FACT。它与 SOFTS/SCF 的隐藏宽度和结构不同，记录参数量及耗时，不将五模型表当作严格的等容量机制拆分实验。多种子与偏差／门控消融仍按[首轮实验分析计划](四模型首轮实验分析与下一步计划_20261102.md)后续推进。

## 8. 实现与检查记录

先更新根 AGENTS.md 第 10 节的后续开发风格，再创建本文件，随后开发代码。三个既有新模型及其模块和脚本未改写。原样保留 PULSE 的 ReVIN、norm、phase_router、denorm 和 SAM 运算，在完整 forward 中显式加入 FACT。

`exp/exp_main.py` 本轮仅增加两行：一行 PULSE_FACT 导入、一项模型注册。原 run.py、训练、验证、测试、损失和数据处理逻辑均未修改。没有新增运行依赖，沿用已有 `requirements-scf.txt` 环境；新模块只使用现有 PyTorch 和标准库。

| 检查 | 真实结果 |
| --- | --- |
| 新增 PULSE_FACT 专属检查 | 15 项：14 通过、1 CUDA 项因无 NVIDIA GPU 跳过 |
| 旧 SCF 专属回归 | 21 项：20 通过、1 CUDA 项跳过；旧测试文件原样保留 |
| Python 3.8.20 / PyTorch 2.0.0 CPU 兼容 | 同一组 15 项：14 通过、1 CUDA 项跳过 |
| FACT 来源语义 | train/eval 前向、输入与参数梯度、BatchNorm 缓冲状态与来源一致 |
| 原 PULSE 等价边界 | 同种子原有权重逐项相同；FACT 换为 Identity 后，eval、SAM 开启／关闭的预测与原 PULSE 精确一致 |
| 小变量及网格补齐 | C=1/2/4/7/9/21/321/883 的输出和梯度通过 |
| FFT 主目标与优化 | 所有训练参数具有有限梯度，严格确定性 CPU 优化步通过 |
| 保存与恢复 | 预测精确恢复；有效配置不匹配会明确报错 |
| 原实验类接入 | 注册及真实 vali 方法的小批量 CPU 调用通过 |
| 13 份脚本 | Bash 语法通过，没有函数、选择器或日志运行标识目录 |
| 48 条训练命令 | 参数逐项与来源一致，只有模型、实验标识、checkpoint 根目录改变 |
| 日志与一键执行 | stdout/stderr 写入、固定日志覆盖、独立 checkpoint 标识、十二数据集顺序及失败退出通过 |
| 48 个实际脚本配置 | B=1 随机输入 CPU eval 全部输出 [1,H,C]，数值有限；未读取真实数据或启动训练 |
| 184 个既有源码／脚本及参考文件 | 仅允许的 exp_main.py 导入／注册发生变化，其余哈希不变 |

主检查环境为 macOS arm64、Python 3.13.5、PyTorch 2.9.0 CPU。CUDA 仍需在训练服务器核验，CPU 通过不代表 GTX 1080 Ti 或目标旧版 PyTorch 的 CUDA 已验证。脚本保留来源 batch size，尚未测量正式训练峰值显存。

新增 FACT 参数固定为 **278,272**。以下为当前代码的参数量，不是已训练结果或效率基准：

| 配置 | 原 PULSE 参数量 | PULSE_FACT 参数量 |
| --- | ---: | ---: |
| ETTh1 96→96 | 195,270 | 473,542 |
| ETTh1 96→720 | 3,062,696 | 3,340,968 |
| PEMS04 96→96 | 448,112 | 726,384 |
| traffic 96→192 | 1,998,182 | 2,276,454 |

## 9. 运行与服务器核验

在工程根目录执行单数据集或全部数据集：

```bash
bash scripts/PULSE_FACT/ETTh1.sh
bash scripts/run_PULSE_FACT.sh
```

首个日志为 `logs/PULSE_FACT/ETTh1/PULSE_FACT_96_96.log`。训练日志会先打印原入口参数，再打印 `PULSE_FACT effective settings`，包含本模型实际使用的 FACT 宽度、膨胀层数、核数和 dropout。

另开终端查看：

```bash
tail -f logs/PULSE_FACT/ETTh1/PULSE_FACT_96_96.log
```

服务器先运行专属检查，其中 CUDA 可用时会执行严格确定性优化步：

```bash
python -B -m unittest discover -s tests -p test_pulse_fact.py -v
```

旧实验类独立 `--is_training 0` 的加载分支仍硬编码旧 checkpoint 根路径，不将本次分组保存视为修复了该旧分支。当前支持复用训练完成后自动恢复最佳权重并测试的原流程。
