# PEMS 交通流消融实验 · 种子 2024

从仓库根目录运行。环境和数据沿用完整模型的 PEMS 实验，数据文件为 `all_datasets/PEMS04.npz` 和 `all_datasets/PEMS08.npz`。在训练环境中执行

```bash
bash scripts/run_PULSE_FACT_ablation_2024.sh
```

启动脚本按下表顺序执行 6 个脚本，每个脚本先跑 H=12，再跑 H=96，合计 12 次独立训练。任一命令失败即停止。需要单独运行某项时，在仓库根目录用 `bash` 执行对应的数据集脚本。

| 模型 | PEMS04 脚本及日志目录 | PEMS08 脚本及日志目录 |
|---|---|---|
| A00 | [PEMS04.sh](../scripts/PULSE_FACT_A00/PEMS04.sh) · `logs/PULSE_FACT_A00/PEMS04/` | [PEMS08.sh](../scripts/PULSE_FACT_A00/PEMS08.sh) · `logs/PULSE_FACT_A00/PEMS08/` |
| A10 | [PEMS04.sh](../scripts/PULSE_FACT_A10/PEMS04.sh) · `logs/PULSE_FACT_A10/PEMS04/` | [PEMS08.sh](../scripts/PULSE_FACT_A10/PEMS08.sh) · `logs/PULSE_FACT_A10/PEMS08/` |
| A01 | [PEMS04.sh](../scripts/PULSE_FACT_A01/PEMS04.sh) · `logs/PULSE_FACT_A01/PEMS04/` | [PEMS08.sh](../scripts/PULSE_FACT_A01/PEMS08.sh) · `logs/PULSE_FACT_A01/PEMS08/` |

每次训练前终端打印日志绝对路径。文件名为 `<ModelName>_96_12.log` 或 `<ModelName>_96_96.log`。使用验证 MSE 选择的权重保存在 `checkpoints/<ModelName>/<setting>/checkpoint.pth`。程序自动恢复最佳权重并测试，在日志末尾读取 `mse` 和 `mae`，两者均为整个预测窗口的指标。重复启动相同脚本会覆盖固定名称的日志，每次启动使用独立的 checkpoint 标识。

## 模型和协议

| 组别 | 模型文件 | 输入归一化与重构 | 隐藏层 |
|---|---|---|---|
| A00 | [PULSE_FACT_A00.py](../models/PULSE_FACT_A00.py) | 对观测做 RevIN，使用配对统计量恢复预测 | 逐传感器共享残差 MLP |
| A10 | [PULSE_FACT_A10.py](../models/PULSE_FACT_A10.py) | 完整模型的历史与未来锚点、残差 RevIN、Anchor update 及重构 | 与 A00 相同的 MLP |
| A01 | [PULSE_FACT_A01.py](../models/PULSE_FACT_A01.py) | 与 A00 相同 | 完整模型的 Cross-sensor mixer |

三组都接受历史观测 `[B,96,C]`，以 `[B,C,512]` 隐藏特征做预测。推理返回 `[B,H,C]`。训练另外返回当前 mixup 的系数和样本排列，供原训练器混合监督目标。输入、均值和标准差共用该系数与排列。A00、A01 的模型内没有时间编码、周期表或 Anchor update。A10 保留原始历史锚点和未来时间编码用于重构。逐传感器 MLP 为两层带偏置的 `512 → 271 → 512`，使用 GELU、0.1 dropout 和残差连接。

以下为本地实例化后实测的整网可训练参数量。MLP 有 278,287 个参数，mixer 有 278,272 个。

| 数据集 | H | A00 | A10 | A01 |
|---|---:|---:|---:|---:|
| PEMS04 | 12 | 334,721 | 661,010 | 334,706 |
| PEMS04 | 96 | 377,813 | 726,399 | 377,798 |
| PEMS08 | 12 | 334,447 | 412,540 | 334,432 |
| PEMS08 | 96 | 377,539 | 594,563 | 377,524 |

启动命令保留完整模型对应数据集和时距的超参数，使用原 `run.py` 与训练器。种子为 2024，batch size 为 32，最多 30 轮，早停耐心为 5，配置学习率为 0.005。原训练器按同一时序切分、标准化方式、频域损失与验证规则执行。新增的训练器接入点为三个模型的注册项，以及训练结束后按本次训练设备恢复最佳权重。[实验类](../exp/exp_main.py) · [基准 PEMS04 脚本](../scripts/PULSE_FACT/PEMS04.sh) · [基准 PEMS08 脚本](../scripts/PULSE_FACT/PEMS08.sh)

## 本地验证

- 使用 macOS CPU、PyTorch 2.0.0，通过原 `run.py` 和一键脚本顺序跑通 12 组。验证在临时目录进行，保留传感器数和全部模型尺寸。采用原始 PEMS 数据的前 202 个时刻（H=12）或 495 个时刻（H=96），batch size 4、1 轮、0 个加载进程。每组均完成反向传播、验证、保存与恢复最佳权重和终测，得到有限的 MSE、MAE。12 份 checkpoint 均再次通过严格加载。
- 在全量 PEMS 数据的原训练划分上，使用正式 batch size 32 为每一组做前向、FFT 损失反向和一次 Adam 更新，所有可训练参数均有有限梯度。上述检查用于验证运行流程。正式结果以服务器完整训练的终测日志为准。
- 核对了 12 条服务器命令，除模型名、checkpoint 路径和实验标识外，参数与原 PEMS 脚本相同。7 个脚本均通过 Bash 语法检查，并验证了一键脚本传递非零退出码。
- 专项测试覆盖锚点关闭时的原始窗口统计量、逐传感器独立性、有效参数量和梯度。将完整模型的 mixer 替换为相同 MLP 后，A10 在推理、训练及启用 mixup 三种情况下与参考路径逐元素一致。可运行 `python -m unittest discover -s tests -p test_pems_ablations.py -v`。[专项测试](../tests/test_pems_ablations.py)

## 来源记录

开发起点为 `23d17b9ff93aef02c0d1da3da87573a07739bf95`。以下来源文件在开始工作时与该提交一致。主模型原文件保留为参照，既有共享算子直接在工程内导入。

| 本地来源 | SHA-256 | 本次使用 |
|---|---|---|
| [models/PULSE_FACT.py](../models/PULSE_FACT.py) | `4f31942eb91380f1f473952bb8c50152f0d041f9ba9ab542bafa4acfd55e9857` | 三组复用 ReVIN。A10 在独立模型文件中保留锚点构造、路由与重构的数据流。统计感知混合提取到消融专属模块。 |
| [layers/pulse_fact/fact_block.py](../layers/pulse_fact/fact_block.py) | `114e27d8bcb1da583c2cea1f1de6955e7b031cb888cb3904db2df1159b375de1` | A01 直接使用原网格模块及其残差结构。上游卷积与许可证见[来源清单](../layers/pulse_fact/vendor/sources.json)。 |
| [scripts/PULSE_FACT/PEMS04.sh](../scripts/PULSE_FACT/PEMS04.sh) | `d7ac9b86c7a55fcb43d0ba95d8798cdc33fb4f3312f038a3e7359b39ec7207d5` | 各模型保留 H=12、96 两个命令块，使用独立的模型名、日志和权重目录。 |
| [scripts/PULSE_FACT/PEMS08.sh](../scripts/PULSE_FACT/PEMS08.sh) | `43f0532f5c28a4550f4749c068662f5b16f009cb0f15b47e7832da0b3f8c82a3` | 同上，保留 PEMS08 对应时距的原始配置。 |
