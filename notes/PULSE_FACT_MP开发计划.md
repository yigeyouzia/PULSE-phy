# PULSE_FACT_MP 开发计划：预测均值保持的交互修正

创建：2026-11-10；开发更新：2026-11-11。状态：模型、最小注册已实现，三份脚本均已补齐对应原数据集的四个时距，共十二组配置；两个 CPU 环境检查通过，正式实验与 CUDA 检查待运行。暂定机制名 **MPIC（Mean-Preserving Interaction Correction）**，中文“预测均值保持的交互修正”。名称仅作工作标识，未验证文献唯一性或创新优先权。

## 1. 依据、目标与开发约束

依据：[ETTh1真实权重诊断](ETTh1权重诊断与模型改进计划_20261110.md)。ETTh1 96→720 的融合 MSE 退化约 12.49%，其中约 80.10% 来自逐样本、逐变量未来窗口的均值预测误差；96→96 的退化则主要来自波动，不能预设单一机制解决二者。

遵守 [AGENTS.md](../../AGENTS.md) 第 2、3、5、8、9、10 节。采用新增模型、完整主文件的方式；保留 PULSE、PULSE_FACT、SOFTS／SCF、STATIC／PC 和全部旧模块、脚本、日志及权重。FACT-main／SOFTS-main 只读。初版先记录计划，2026-11-11 经用户明确授权后完成代码与脚本；正式效果仍待实验检验。

本候选首轮只改变最终预测的时间均值分配，不同时改深度、BN、损失、SAM、网格布局或变量级控制器。

## 2. 核心设计

设相同混合输入经 PULSE 时间映射得到隐藏表示 h。原 PULSE 结构分支与融合分支分别为：

$$
Y_P=\mathcal{D}(W_o h; A_x,\mu,\sigma,\text{timestamp}_y),\qquad
Y_F=\mathcal{D}(W_o F(h); A_x,\mu,\sigma,\text{timestamp}_y),
$$

其中 F 是原 FACT 交互块，D 包含原 Router、相位参照与配对反归一化。两个分支共享时间映射、输出映射、Router、时间编码、codebook 和 ReVIN 参数；分别按自身中间预测计算 Router，保持原算法清楚可核对。

最终输出：

$$
\boxed{Y_{MP}=\operatorname{Mean}_t(Y_P)+
\left[Y_F-\operatorname{Mean}_t(Y_F)\right]}
$$

等价于对修正量 `Delta=Y_F-Y_P` 做零均值投影，再加到 `Y_P`。投影只沿预测时间轴，均值形状为 `[B,1,C]`，绝不跨变量、跨样本取平均。

设计性质：

- `mean_t(Y_MP)=mean_t(Y_P)`：最终预测窗口均值交给 PULSE 结构分支。
- `center_t(Y_MP)=center_t(Y_F)`：FACT 分支提供去均值的预测波动。
- 两个均值都来自预测，不读取真实未来均值或测试标签；已知未来时间标记的使用沿用原模型。
- 在最终恢复后的同一输出坐标中投影，避免在隐空间去均值却被后续非线性 Router 重新改变均值。外部 scaler 是固定逐变量仿射变换，因此均值保持关系也对应到原量纲。
- 原主干和 FACT 共享参数，预计不新增可训练参数，但增加一次输出／Router 恢复计算；开发时实际统计参数和运行开销。

### 必须说清的边界

`Y_P` 是新模型内部的 PULSE 结构分支，并非加载并冻结的独立 PULSE checkpoint。共同训练后其均值未必与原 PULSE 一样准确，不能声称自动保留原模型指标。

两分支共享参数，预测成分的职责分配不等于优化过程完全解耦。所谓“波动”仅指未来窗口去均值成分，不是完整多尺度趋势分解，也不是原输入的相位残差。均值保持是统计结构约束，不等于能量守恒、温度方程或 PINN 物理残差。

## 3. 与当前实现相比修改哪里

| 环节 | 当前 PULSE_FACT | PULSE_FACT_MP 计划 |
| --- | --- | --- |
| 时间映射前缀 | 归一化／相位参照／SAM，Linear→GELU | 保持相同，只计算一次 |
| FACT | 一层，两膨胀分支，FFN、BN、dropout | 保持相同，只计算一次 |
| 输出映射 | 对 FACT 隐藏表示映射 | 同一输出头分别映射 h、F(h) |
| Router／恢复 | 仅恢复融合预测 | 用同一参数分别恢复两个预测 |
| 最终输出 | Y_F | Y_P 的窗口均值＋Y_F 的去均值波动 |
| 损失／增强 | FFT 复数 MAE，SAM | 保持相同，不增均值辅助损失 |
| 配置 | 原数据集脚本配置 | 原配置，专属机制元数据另存 |

这种双预测计算是同一个新模型内部明确展开的数据流，不另建训练器、不通过包装旧模型导出别名隐藏主逻辑。

## 4. 新增与最小修改文件

下列路径相对于 PULSE-phy；核心代码、测试和三份脚本已经交付，正式结果尚未产生。

| 操作 | 文件 | 职责 |
| --- | --- | --- |
| 新增 | `models/PULSE_FACT_MP.py` | 完整 class Model、ReVIN、init、norm、Router、denorm、forward；显式展开两预测与时间均值投影 |
| 最小修改 | `exp/exp_main.py` | 仅新增导入和注册，不改训练、验证、损失、测试分支 |
| 新增 | `tests/test_pulse_fact_mp.py` | 均值／波动性质、接口、SAM、梯度、保存恢复和旧模型保护 |
| 新增并补齐 | `scripts/PULSE_FACT_MP/ETTh1.sh` | 直接顺序执行 H=96、192、336、720 |
| 新增并补齐 | `scripts/PULSE_FACT_MP/ETTh2.sh` | 直接顺序执行 H=96、192、336、720 |
| 新增并补齐 | `scripts/PULSE_FACT_MP/PEMS04.sh` | 直接顺序执行 H=12、24、48、96 |
| 维护 | 本计划及独立来源记录 | 复制来源、源文件 SHA256、配置、检查和真实结果 |

来源与检查记录：[来源 manifest](data/pulse_fact_mp_sources.json)、[开发检查 JSON](data/pulse_fact_mp_development_checks.json)。

不需要新层目录来封装一个时间均值投影；主文件使用明确方法名说明该步骤。继续调用内部 `layers/pulse_fact/`，不导入兄弟仓库。若后续需要更改任何来源计算，另建专属模块，不能直接改 vendor 或原适配层。

`run.py` 无计划修改，不新增 CLI 参数或替代训练入口。模型导入和注册按现有允许范围完成。专属结构配置写入 `_extra_state` 并检查恢复兼容，不把旧参数改成新含义。

新配置统一使用 `pulse_fact_mp_*` 前缀，默认值与对应原 FACT 块相同；均值轴、恢复后投影、参数共享等结构语义写入独立 MP schema。用于受控检查时可显式复制同形状参数，但不将原 FACT 的配置元数据冒充 MP checkpoint。

## 5. 接口与 SAM

- 输入：观测 x `[B,L,C]`、cycle_index `[B]`、历史标记 `[B,L,time_dim]`、已知未来标记 `[B,H,time_dim]`。
- 输出：训练 `(prediction, mixup_params)`，推理 prediction `[B,H,C]`。
- SAM 只抽取一次 lambda／index；两个预测使用同一混合输入、mean／sigma。标签混合继续由原实验流程处理，原历史参照使用方式不单独改变。
- 先做两个分支各自的原配对恢复，再投影。不得把未恢复预测与已恢复预测相加。
- 额外 Router／dropout 调用会改变训练时随机数消耗，必须记录；不把相同 seed 说成逐步相同的训练轨迹。

## 6. 开发与验收顺序

1. 再次记录实际开发时的 Git 起点和用户既有修改；记录所复制 PULSE／PULSE_FACT 文件及其哈希，不假定今天的 HEAD 仍是后续开发起点。
2. 在独立主文件复制原清楚的数据流；只增加共享参数的第二条预测和最终均值投影。旧文件保持原样。
3. 先做最小 CPU 形状／梯度／恢复检查；再做原实验类兼容和真实脚本配置的轻量前向。
4. 经最小注册接入，检查每份 shell 语法、参数支持、失败状态、日志路径和 checkpoint 路径；不改 run.py。
5. 按当前三数据集、各四个时距的范围执行；记录逐变量、预测距离、窗口水平／波动误差以及完整原始指标，不只报总 MSE。
6. 用一致协议重复种子；根据真实验证结果决策，预定测试结果完整报告，不挑有利种子、变量或时距。

必须通过的检查：

- 随机张量及真实小批次验证 mean(Y_MP)=mean(Y_P)、center(Y_MP)=center(Y_F)，误差限考虑 float32 舍入。
- 同一权重下，工具展开得到的 Y_F 与原融合模型 eval 输出一致；关闭交互的受控检查对应同权重 PULSE 数据流，不能等同独立训练 PULSE 指标。
- 梯度确实流向均值分支和交互分支，FFT 目标下可有限优化；输入／标签不泄漏。
- SAM 的 lambda、index、统计量和两个分支严格一致。
- 保存／严格恢复配置与预测一致；原有模型、来源层、入口、旧脚本和输出哈希不变。

## 7. 当前完整脚本与预计输出

| 模型—脚本 | 对应参考 | 输入→预测时距 | 预计日志 |
| --- | --- | --- | --- |
| `PULSE_FACT_MP/ETTh1.sh` | `PULSE_FACT/ETTh1.sh` 全部四段命令 | 96→96／192／336／720 | `logs/PULSE_FACT_MP/ETTh1/PULSE_FACT_MP_96_<H>.log` |
| `PULSE_FACT_MP/ETTh2.sh` | `PULSE_FACT/ETTh2.sh` 全部四段命令 | 96→96／192／336／720 | `logs/PULSE_FACT_MP/ETTh2/PULSE_FACT_MP_96_<H>.log` |
| `PULSE_FACT_MP/PEMS04.sh` | `PULSE_FACT/PEMS04.sh` 全部四段命令 | 96→12／24／48／96 | `logs/PULSE_FACT_MP/PEMS04/PULSE_FACT_MP_96_<H>.log` |

数据路径、batch、Router 参数、inv_len、patch、学习率、30 轮预算、patience=5、FFT／SAM 均复制对应原命令，首轮 seed=2024。新模型不能借用源模型日志路径。脚本沿用顶部变量＋直接续行 `python -u run.py`，无 shell 函数、菜单或选择器；`set -e`，`> log 2>&1`，先打印日志绝对路径。本轮未要求启动清单，计划不额外引入一键入口。

```text
PULSE-phy/
├── scripts/PULSE_FACT_MP/
│   ├── ETTh1.sh
│   ├── ETTh2.sh
│   └── PEMS04.sh
├── logs/PULSE_FACT_MP/
│   ├── ETTh1/
│   │   ├── PULSE_FACT_MP_96_96.log
│   │   ├── PULSE_FACT_MP_96_192.log
│   │   ├── PULSE_FACT_MP_96_336.log
│   │   └── PULSE_FACT_MP_96_720.log
│   ├── ETTh2/
│   │   ├── PULSE_FACT_MP_96_96.log
│   │   ├── PULSE_FACT_MP_96_192.log
│   │   ├── PULSE_FACT_MP_96_336.log
│   │   └── PULSE_FACT_MP_96_720.log
│   └── PEMS04/
│       ├── PULSE_FACT_MP_96_12.log
│       ├── PULSE_FACT_MP_96_24.log
│       ├── PULSE_FACT_MP_96_48.log
│       └── PULSE_FACT_MP_96_96.log
└── checkpoints/PULSE_FACT_MP/<setting>/checkpoint.pth
```

重复种子前归档固定路径日志，checkpoint 保持独立 setting；不在日志目录或文件名里加入 seed／时间戳。后续独立 `is_training=0` 的旧路径限制仍须单独适配，不能把训练后自动测试可用说成独立加载分支已修复。

## 8. 实验判断与后续分支

初版四个诊断配置仍是机制分析重点；按用户补充要求，当前脚本默认覆盖三份数据集的全部十二组配置，不能只报告其中有利的结果。主要假设：720 步窗口水平误差下降，并转化为整体 MSE 改善；96 步波动误差不进一步明显恶化；ETTh2、PEMS04 原有交互收益不被普遍破坏。上述都是待检验目标，不填写推测数值。

筛选后，ETTh1 两个时距比较 PULSE／PULSE_FACT／MP，补 seed=2025、2026：三模型×两配置×两种子，共十二次新增训练，已有对应 seed=2024 可在协议不变时合并。若先完成了原两模型的八次重复，可复用其中结果，仅再补 MP 四次。三种子主要用于初步波动评估，不预设显著性。

原验证仍随机 shuffle／drop_last。若统一改为完整固定验证，需要单独记录协议修订，并让三模型全部在同一新协议下重跑 seed=2024／2025／2026；不能只调整新模型，或把旧 seed=2024 混入新协议。

结果分支：

- 水平改善、波动未恶化：进一步验证均值保持机制，扩展 ETT 其余时距与外部基线。
- 水平改善但短时距波动仍差：单独做 `PULSE_FACT_POINT` 空间邻域对照，判断七变量网格是否引入负迁移；不直接叠加新条件器。
- 均值分支也学不好：检查共享参数的干扰，保留失败；固定／独立均值分支是下一项新设计，不能悄悄冻结原 PULSE 或增加辅助损失后仍归因于同一改动。
- PEMS 收益被均值约束破坏：说明跨变量信息也可能帮助预测水平；考虑后续条件化的水平修正，先建立无条件约束对照，不立刻将多项变化合并。
- 没有可重复收益：停止把 MPIC 作为主创新；保留原模型与负结果，回到空间交互、损失或选模协议的单因素研究。

## 9. 论文表述边界

可以表述为“预测窗口水平与跨变量波动修正的职责分配”，或“在保留周期参照结构的预测分支上施加均值保持约束”。必须说明继承的 PULSE、FACT 与来源层，并用实验检验该约束。

不能称其为已知真实均值、能量守恒、温度演化方程，不能仅因有频域／均值约束就称为 Physics-Informed。更强物理叙事需要另外引入可核实的领域关系与相应证据，本计划不同时添加这些内容。

## 10. 状态记录

已完成：真实 checkpoint 的诊断、独立主模型、exp_main 的两行导入／注册、检查代码、三份脚本和来源记录。未完成：正式 benchmark 训练、多种子验证和本机不可执行的 CUDA 检查。代码可通过原入口接入，不把开发检查当作正式精度结果。

## 11. 2026-11-11 初版实现与验收记录（e116695）

主模型：[models/PULSE_FACT_MP.py](../models/PULSE_FACT_MP.py)。保留完整 ReVIN、norm、Router、denorm；原算法方法的 AST 与 PULSE_FACT 一致。两个输出分支使用同一个预测头和同一组 Router 参数，FACT 只计算一次、BN 每次训练 forward 只更新一次。

恢复顺序明确为 FACT 分支在先、PULSE 分支在后。因此，同种子、同权重的 FACT 分支在 eval、train 无 SAM、train 有 SAM 三种受控检查中，都与原 PULSE_FACT 的对应输出逐元素一致；完整 MP 的训练随机数消耗仍不同，不声称训练轨迹相同。

MP 的投影输入是两个完成 Router 与 ReVIN 恢复后的预测，不是隐藏特征；没有 detach，没有第二个预测头或第二组 Router 参数。与对应 PULSE_FACT 相比，可训练参数数目相同。

| 初版四配置的参数示例 | PULSE_FACT 参数 | PULSE_FACT_MP 参数 |
| --- | ---: | ---: |
| ETTh1 96→96 | 473542 | 473542 |
| ETTh1 96→720 | 3340968 | 3340968 |
| ETTh2 96→720 | 3341004 | 3341004 |
| PEMS04 96→96 | 726384 | 726384 |

| 检查 | 结果 |
| --- | --- |
| MP 专项，Python 3.13.5／torch 2.9.0 CPU | 16 项中 14 项通过，2 项 CUDA 检查跳过 |
| MP 专项，Python 3.8.20／torch 2.0.0 CPU | 同样 14 项通过、2 项跳过 |
| 原 PULSE_FACT 回归，Python 3.13／torch 2.9 CPU | 15 项中 14 项通过，1 项 CUDA 检查跳过 |
| 机制性质 | 按样本／变量保持预测时间均值与 FACT 去均值波动；均值／波动方向的梯度路由检查通过 |
| 原算法兼容 | 同权重 FACT 分支、SAM 一次混合和两次一致统计恢复、单步预测边界、小变量网格均通过 |
| 严格确定性 | CPU FFT 复数 MAE 反向与两次优化更新通过，未关闭 deterministic algorithms |
| 保存恢复 | MP 独立 schema 严格恢复，旧 FACT schema 和不匹配配置被拒绝 |
| 脚本 | 三份 shell 语法、四组原配置对照、实际形状、固定日志、独立 checkpoint ID、失败即停通过 |

初次 CPU 原入口冒烟已完成合成数据的一轮训练并保存最佳权重，但到旧 Exp_Main 的最佳恢复时失败：该普通数据分支强制 `storage.cuda(args.gpu)`。这是既有代码限制，不是 MP 的反向或保存失败；本轮保留原实现，没有修改 run.py、恢复方法或添加 CPU 绕过。

因此，专项检查将原 run.py 的完整“训练→恢复→自动测试”设为 CUDA 检查，本机跳过；不能宣称 Mac CPU 入口全流程已经通过。直接模型 CPU 前向、反向、优化、严格保存恢复和原验证接口已经通过。正式 `.sh` 沿用单 GPU 服务器使用方式。

从工程根目录运行专项检查：

```bash
python -B -m unittest discover -s tests -p test_pulse_fact_mp.py -v
```

在已准备数据、原依赖与 CUDA 环境的服务器执行：

```bash
bash scripts/PULSE_FACT_MP/ETTh1.sh
bash scripts/PULSE_FACT_MP/ETTh2.sh
bash scripts/PULSE_FACT_MP/PEMS04.sh
```

初版脚本依次是两次、一次、一次训练；本节检查结果对应 e116695 的历史状态。补齐后，上述三条命令分别执行四次训练，合计十二次。没有自动启动任何正式训练。服务器测试会自动启用本机跳过的 CUDA 反向与原入口冒烟检查，建议先执行专项检查，再执行这三份脚本。没有额外新增一键启动清单或大桥配置。

Git 提交只纳入 MP 代码、脚本、检查和相关诊断笔记／产物。工作区同时存在的大桥开发和其他用户修改保留；exp_main 本轮增量仅新增 MP 导入与注册，run.py 未改动。推送状态以实际 Git 结果为准，正式指标保留待运行。

## 12. 三数据集四时距脚本补齐

2026-11-11 按用户补充要求，三个现有脚本分别补齐原仓库对应的四段命令，合计十二组配置；不新增十二份单时距文件，也不增加函数、菜单或训练入口。每个时距的 inv_len、dsa、dsb、ksize、d_model、patch_size 等均复制对应 PULSE_FACT 原脚本的实际值，不套用某一个时距的配置。

本次仅维护三份 MP 脚本、脚本范围检查、来源范围与本记录，主模型和训练代码保持原样。既有初版开发审计保留；新的脚本检查单独记录在 [四时距脚本检查 JSON](data/pulse_fact_mp_script_checks.json)。

已在 Python 3.13／torch 2.9 和 Python 3.8／torch 2.0 两个 CPU 环境执行脚本专项检查：各两项通过、一项 CUDA 原入口检查跳过。检查覆盖三份 shell 语法、十二组完整参考参数、十二组实际模型前向形状、日志路径、覆盖规则、checkpoint 标识与失败即停；没有启动正式数据集训练。

运行方式仍是上一节的三条 bash 命令。日志继续使用固定 `logs/PULSE_FACT_MP/<dataset>/PULSE_FACT_MP_96_<H>.log`，checkpoint 继续使用 `checkpoints/PULSE_FACT_MP/<setting>/checkpoint.pth`；没有迁移或改写历史实验产物，正式结果待运行。
