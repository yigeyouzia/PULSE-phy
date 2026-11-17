# PULSE_FACT 接入四个 FACT 交通数据集

2026-11-17，代码起点 `a03db52966081a4e558f189f4124f7ed59c2f2f5`。开工时已有本地笔记、`scripts/Gjemnessund.sh`、历史日志、论文文件与 `.gitignore` 等工作区变化。本次新增数据适配、运行脚本和本记录，仅在现有数据工厂登记新类型并为它保留末批样本。复用 `run.py` 和 `Exp_Main`。遵循[工作区开发约束](../../AGENTS.md)。

## 数据与来源

参考目录 `../../FACT-main/` 为解压源码。按该源码的 CSV 和 NPY 分区规则实现新加载器。两类文件按原始观测行滑窗，步长为 1。NPY 原盘为 `[变量, 时间]`，读取后转置。CSV 保留原传感器顺序和日期。PEMS-BAY 的一次时钟跳变保留为相邻的原始观测行。标准化仅使用训练段拟合，验证和测试可借用切分起点前的历史，目标仍位于各自分区。三种切分保留最后一个不足整批的样本批次。

| 来源文件 | SHA-256 |
| --- | --- |
| `FACT-main/FACT.sh` | `6794bda6b2633e3e155d4a1bd3b80589896c9ab016800700636c0b232b15b9ab` |
| `FACT-main/data_provider/data_loader.py` | `4e2ef3416f1d3187262b36fe7ae1d2c3a0b99a3d52d175c7ed66443da536ee9c` |
| `FACT-main/data_provider/data_factory.py` | `f02d62397f9608b28626b4fbc2d2f70a808848466a47953dde5b5a154008183c` |

PEMSD7、PEMS-BAY、METR-LA 与 FACT 作者数据包对应文件一致。本地 Traffic2 的形状、采样粒度及预期窗口数与 FACT 报告相符。FACT 压缩包未包含 `traffic.npy`，因此无法比对该实验所用原文件的校验和。

| 文件（`all_datasets/`） | 时间点 × 变量数 | SHA-256 |
| --- | --- | --- |
| `PEMSD7.npy` | 12672 × 228 | `0b3f8e4e9446adcca8bdb76af489def063250896e5e6d69a16ae089632d2dac1` |
| `PEMS-BAY.csv` | 52116 × 325 | `013452b13da35b5a22e5762fbd8b28b61154f17c4a5ce5f463638d8df43f91b6` |
| `traffic.npy` | 10560 × 963 | `c9dae7c57e13080365e18bc22806730626d628e7b34c0b08fb701bea74ed0ada` |
| `METR-LA.csv` | 34272 × 207 | `f715d4ef988a68bf542a31d8687b10c68e0ffb90238f903e497cb1daa407e17a` |

## 实验设置

`FACTTraffic` 通过[专用加载器](../data_provider/data_loader_fact_traffic.py)返回 `[96,C]` 历史、`[H,C]` 目标、`[96,time_dim]` 和 `[H,time_dim]` 时间特征、预测开始行的全局索引。两份 CSV 用 5 个真实日历特征，两份 NPY 用 1 个零时间特征。`PULSE_FACT` 按观测行索引和 `inv_len` 计算周期相位，三份 5 分钟数据取 288，小时 Traffic2 取 24。

脚本沿用作者 `FACT.sh` 中对应的数据集与四档预测长度。首轮单种子 2024、`itr=1`，每组输入 96 步、预测 12/24/48/96 步，共 16 组。参考作者脚本设 batch size 32、`learning_rate=0.001`、最多 15 个 epoch、`lradj=type1`，PEMS-BAY 耐心值为 5，其余为 3。在现有 PULSE_FACT 训练器中使用 Adam、SAM 增强和 FFT MAE 辅助损失（`rec_lambda=0`、`auxi_lambda=1`），按验证 MSE 保存权重。学习率实际调度沿用 PULSE 训练器。模型统一使用 `label_len=0`、`patch_size=12`、`dsa=4`、`dsb=1`、`ksize=5`。三个 5 分钟数据集的 `d_model=32` 参考现有 PEMS04 12 步脚本，Traffic2 的 `d_model=16` 参考现有 Traffic 96 步脚本。

从项目根目录执行 `bash scripts/run_PULSE_FACT_FACT_Traffic.sh`，顺序与下表一致。每个数据集顺序运行 12、24、48、96 步，任一命令失败即退出。日志名中的 `<H>` 为实际预测长度。权重放在 `checkpoints/PULSE_FACT/<setting>/checkpoint.pth`，每次启动生成独立实验标识。

| 数据集 | 脚本 | 固定日志路径 | 12/24/48/96 步测试窗口数 |
| --- | --- | --- | --- |
| PEMSD7 | [PEMSD7.sh](../scripts/PULSE_FACT/PEMSD7.sh) | `logs/PULSE_FACT/PEMSD7/PULSE_FACT_96_<H>.log` | 2523 / 2511 / 2487 / 2439 |
| PEMS-BAY | [PEMS-BAY.sh](../scripts/PULSE_FACT/PEMS-BAY.sh) | `logs/PULSE_FACT/PEMS-BAY/PULSE_FACT_96_<H>.log` | 10412 / 10400 / 10376 / 10328 |
| Traffic2 | [Traffic2.sh](../scripts/PULSE_FACT/Traffic2.sh) | `logs/PULSE_FACT/Traffic2/PULSE_FACT_96_<H>.log` | 2101 / 2089 / 2065 / 2017 |
| METR-LA | [METR-LA.sh](../scripts/PULSE_FACT/METR-LA.sh) | `logs/PULSE_FACT/METR-LA/PULSE_FACT_96_<H>.log` | 6843 / 6831 / 6807 / 6759 |

## 验证与结果

五份 Bash 脚本通过语法检查。用模拟命令验证启动顺序、16 个实验标识、绝对日志路径、stdout/stderr 合并和失败退出。四个实际数据集的 48 个训练／验证／测试组合通过窗口数、首尾目标索引、训练段标准化和时间特征检查。16 组实际配置均通过 CPU 小批次前向、FFT 反向、Adam 更新及权重保存恢复。四个数据集各做了一次 `96→12` 小批次整窗测试，确认现有测试流程能够写出指标。

旧 PULSE_FACT 测试套件 14 项通过，CUDA 项在本机跳过。额外比较了旧 ETTh1、PEMS03 和 Traffic 的加载窗口、采样器及末批处理，结果与变更前一致。现有训练器构造 `OneCycleLR` 时会初始化学习率，随后按 `type1` 更新，因此学习率参数与作者脚本一致，首轮实际调度沿用 PULSE 训练器。正式脚本应在 CUDA 训练机运行，本机未验证完整训练及自动恢复最佳权重流程。

正式训练待运行。每档保存完整预测窗口的 MSE 和 MAE，按数据集分别计算四档等权均值。
