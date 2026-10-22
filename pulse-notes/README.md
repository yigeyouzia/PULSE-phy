# PULSE 学习、组合迁移与 Gje 实验计划

> 来源：本地 PULSE 论文与代码、Gjemnessund 探索 notebook、HMSFormer 当前本地源码。
> 状态：论文与接口核查已完成；Gje 接入及 CPU 短流程已完成；正式 GPU 训练与 HMSFormer 对比待实施。
> 更新时间：2026-10-22

| 阅读顺序 | 文档 | 内容与状态 |
| --- | --- | --- |
| 1 | [PULSE 论文、模型架构与创新点](PULSE-论文模型架构与创新点学习笔记.md) | 完整方法、公式、原论文图片、消融、代码映射；原模型 CPU 检查已完成 |
| 2 | [组合到其他预测模型的通用方法](PULSE-组合到其他预测模型的通用方法.md) | Backbone 接口、归一化、SAM、损失、DLinear 示例与 HMSFormer 适配；DLinear 组合 CPU 检查已完成 |
| 3 | [Gje 使用与 HMSFormer 对比计划](PULSE-Gje大桥数据集使用与HMSFormer对比计划.md) | 数据与窗口对齐、scaler 修正、预训练预算、实验矩阵、阶段验收；全量数据预处理与原入口 CPU 短训练通过；第 9 节提供 Linux 运行方法 |

原项目入口：[根目录 README](../README.md)；原论文：[ICML-2026-PULSE.pdf](../paper/ICML-2026-PULSE.pdf)。

本目录的图片引用两类本地来源：

- PULSE 论文图表通过 `../assets/` 引用，保留项目原有图片。
- Gje 图片存于 [images/](images/)，原样复制 / 提取自已存在的 notebook 和附件，详细出处见 [图片来源](images/README.md)。

原版 PULSE 已接入 Gje，入口仍为 `run.py`。将文件放到 `all_datasets/gjemnessund.h5` 后，执行 `bash scripts/Gjemnessund.sh`。详见计划文档第 9 节；HMSFormer 对齐与正式指标比较尚未运行。

Gje 的四个预测长度已合并到 [Gjemnessund.sh](../scripts/Gjemnessund.sh)。从项目根目录执行：

```bash
bash scripts/Gjemnessund.sh
```

脚本固定输入 720 点，按 `96 → 192 → 336 → 720` 的顺序逐个训练和测试，每组使用独立的 `model_id` 和日志文件，统一每 100 个 batch 打印一次。任一组报错会停止后续实验。只需部分长度时，修改脚本中的 `for pred_len in 96 192 336 720` 列表。

训练与数据参数沿用 96 起跑配置，尚未针对各 horizon 选优。日志位于 `logs/Gjemnessund/Gjemnessund_720_<H>.log`；再次运行会覆盖同名文本日志，重要结果请先归档。
