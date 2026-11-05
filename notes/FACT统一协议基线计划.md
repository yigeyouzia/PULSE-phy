# FACT 统一协议基线计划与记录

日期：2026-11-05。代码起点：`d299819`。本轮按用户最新要求，只复制完整 FACT 预测网络，复用 PULSE 工程的数据与训练预算，完成模型、实验适配、十二个通用数据集和大桥脚本并推送 GitHub。这不是论文原指标的 1∶1 复现。

## 1. 开发边界

- 遵守 [AGENTS.md](../../AGENTS.md)。FACT-main 为 ZIP 解压的只读来源，不推测提交号；记录原文件及复制文件 SHA256、许可证。
- `models/FACT.py` 原样复制，完整保留时域、幅值、相位、融合、内部归一化和输入输出映射；不借用 PULSE_FACT 或拆成只含 DConvBlock 的融合模型。
- 原模型、exp_main、loader、工具函数、脚本、结果与用户已有工作区修改保留；开始快照记录在临时目录，交付前逐项核对。
- 模型专属适配放入 `utils/fact_original/`，新增 `exp/exp_fact.py` 复用现有实验类的稳定能力，不另建训练入口。
- 用户已在 2026-11-05 对第 6 节的具体入口接入回复“允许这些 FACT 入口改动并继续验证推送”；本轮仅按此范围修改 run.py。其他模型、其他参数默认值及旧训练流程保留。

## 2. 协议和模型配置

| 项目 | 本轮设置／来源 |
| --- | --- |
| 模型 | 完整 FACT；源码 `../FACT-main/models/FACT.py` |
| 数据层 | 当前 data_provider，不复制原 FACT 的切分或缺失处理 |
| 数据／预算 | 十二份 `scripts/PULSE/*.sh` 对应的路径、通道、L/H、batch、lr、epochs、patience、调度；大桥参考当前 Gjemnessund 脚本 |
| 损失 | MSE，rec_lambda=1、auxi_lambda=0，无 SAM；不把旧模型的 FFT/SAM 结果说成同目标比较 |
| 调度 | 复用当前 OneCycleLR 构造与 lradj 行为；type3 并非全程固定 lr，记录实际 epoch lr |
| FACT 内部维度 | fact_d_model=512、fact_d_ff=2048，不复用 Router 的 d_model=16/32 |
| FACT 分支 | fact_core=0.5，fact_dilation=[1,2,3,2,1]，fact_num_kernels=4 |
| FACT 归一化 | fact_use_norm=1，fact_dropout=0.1 |
| 时间特征 | loader 沿用各数据集配置；模型内部 freq=n，只预测真实变量，不额外追加日历通道 |
| 重复 | 首轮沿用当前脚本 itr=1、seed=2024；不声称完成论文五种子复现 |
| 选模 | validation MSE；测试仅在训练和最佳权重恢复后执行 |

已知源码问题：forecast 重复写 `elif alpha == 1.0`，core=0 会访问未构造的 conv_time。保留原文件，适配层明确拒绝 core=0，不静默修复。源码单次前缀 padding 对少量变量数也有限制，适配层校验而不更改网格算法。首轮均使用合法通道和 core=0.5。

若后续比较纯网络能力，需要所有参比模型按相同 MSE／增强协议重新运行，单独归档；本轮不改写或覆盖旧结果。

## 3. 文件计划

| 文件 | 职责 |
| --- | --- |
| models/FACT.py | 原样完整模型 |
| utils/fact_original/ | 配置映射、参数校验、CLI 注册函数、来源记录、许可证与 manifest |
| exp/exp_fact.py | MSE 训练、原生四参数调用、恢复、标准／大桥指标；复用已有数据、优化器、验证等能力 |
| scripts/FACT/*.sh | 十三份简约脚本，每份直接写四个时距命令，无函数和选择器 |
| tests/test_fact_baseline.py | 原源码与复制模型同权重预测／梯度、脚本预算、旧入口兼容、前向及恢复检查 |
| run.py | 已获额外确认的 FACT 专属参数、验证、实验类选择和独立配置标识；原模型分支及默认值保留 |

新增可选 FACT CLI 参数使用 default=argparse.SUPPRESS，非 FACT 旧命令不增加字段；gje_setting 原函数保持不变，避免旧 checkpoint 指纹改变。FACT 配置在支持函数中解析，明确打印和保存实际值。普通数据集与大桥均通过原 run.py，不伪装模型名称。

## 4. 脚本与输出

脚本清单：ETTh1.sh、ETTh2.sh、ETTm1.sh、ETTm2.sh、electricity.sh、traffic.sh、Weather.sh、Solar.sh、PEMS03.sh、PEMS04.sh、PEMS07.sh、PEMS08.sh、Gjemnessund.sh。

长预测 L=96，H=96/192/336/720；PEMS L=96，H=12/24/48/96；大桥 L=720，H=96/192/336/720。Weather 日志标识沿用 weather，electricity.csv 仍为 321 通道 ECL，不使用原 FACT 的 370 通道 electricity.npy 代替。

| 运行对象 | 命令（从工程根目录） | 预计日志 |
| --- | --- | --- |
| FACT + ETTh1 | bash scripts/FACT/ETTh1.sh | logs/FACT/ETTh1/FACT_96_96.log 等 |
| FACT + ETTh2 | bash scripts/FACT/ETTh2.sh | logs/FACT/ETTh2/FACT_96_96.log 等 |
| FACT + ETTm1 | bash scripts/FACT/ETTm1.sh | logs/FACT/ETTm1/FACT_96_96.log 等 |
| FACT + ETTm2 | bash scripts/FACT/ETTm2.sh | logs/FACT/ETTm2/FACT_96_96.log 等 |
| FACT + ECL321 | bash scripts/FACT/electricity.sh | logs/FACT/Electricity/FACT_96_96.log 等 |
| FACT + Traffic | bash scripts/FACT/traffic.sh | logs/FACT/traffic/FACT_96_96.log 等 |
| FACT + Weather | bash scripts/FACT/Weather.sh | logs/FACT/weather/FACT_96_96.log 等 |
| FACT + Solar | bash scripts/FACT/Solar.sh | logs/FACT/Solar/FACT_96_96.log 等 |
| FACT + PEMS03 | bash scripts/FACT/PEMS03.sh | logs/FACT/PEMS03/FACT_96_12.log 等 |
| FACT + PEMS04 | bash scripts/FACT/PEMS04.sh | logs/FACT/PEMS04/FACT_96_12.log 等 |
| FACT + PEMS07 | bash scripts/FACT/PEMS07.sh | logs/FACT/PEMS07/FACT_96_12.log 等 |
| FACT + PEMS08 | bash scripts/FACT/PEMS08.sh | logs/FACT/PEMS08/FACT_96_12.log 等 |
| FACT + 大桥 | bash scripts/FACT/Gjemnessund.sh | logs/FACT/Gjemnessund/FACT_720_96.log 等 |

```text
PULSE-phy/
├── scripts/FACT/<dataset>.sh
├── logs/FACT/<dataset>/FACT_<L>_<H>.log
├── checkpoints/FACT/<setting>/
│   ├── checkpoint.pth
│   ├── fact_run.json
│   └── <data protocol files>
└── results/<setting>/
    ├── fact_metrics.json
    └── gje_metrics.json  # 仅大桥，另附原数据协议
```

固定日志重跑覆盖，模型／数据集目录不含 run_tag；checkpoint 的独立 model_id 含运行标识，FACT 专属 setting 记录有效结构配置。脚本 set -e，stdout/stderr 重定向至日志，每个时距先打印日志绝对路径。

大桥默认路径为 all_datasets/gjemnessund.h5，可通过 GJE_ROOT_PATH 指定实际 H5 所在目录。沿用既有十通道、0.5 秒、逐记录 50/25/25、训练行 scaler、不跨记录拼窗；time_dim=6、label_len=0。FACT 不使用 cycle，也不读取未来真实信号。报告十通道、Acc6、风通道、逐通道及逐记录宏平均指标。

## 5. 验收与后续状态

依次执行：源码／许可证哈希和同权重数值梯度核对；全部脚本参数与预算对照；通用及大桥前向／梯度／恢复；临时合成数据上的少量 batch 训练与独立测试；入口确认后的旧模型参数／指纹回归。真实数据完整训练及 CUDA 检查必须另行标明，不能用 CPU 冒烟指标冒充正式结果。

本轮已实现，检查记录见第 7 节。尚未进行正式训练，不填写推测指标。

## 6. 已确认并实施的限定入口接入

模型、实验类、十三份脚本及独立检查代码已完成。用户单独确认后，入口补丁只涉及以下四处：

1. parse_args 前调用 add_fact_arguments，新增 fact_d_model、fact_d_ff、fact_core、fact_dilation、fact_num_kernels、fact_use_norm、fact_dropout；全部 SUPPRESS，旧命令参数字典不变。
2. 仅 model=FACT 时调用 validate_fact_arguments，并将实验类选为 Exp_FACT；原 Exp_Main 仍为其他模型默认值。
3. 大桥原校验保留在非 FACT 分支；FACT 使用自己的十通道、时间维度和数据参数校验，不施加 PULSE patch_size 整除条件。
4. 仅 FACT 的训练／独立测试 setting 增加有效配置和预算指纹；原 gje_setting 函数、旧模型命名、种子设置与 train/test 调用顺序不变。

必要性：原入口不接受 fact_* 参数，未选择新的原生 MSE 实验类，并在大桥处拒绝 FACT 名称。已通过原 run.py 运行临时标准 CSV 和合成大桥 H5 的训练、最佳恢复、自动测试和独立测试，并运行旧大桥／脚本回归；非 FACT 参数默认值和 gje_setting AST 与起点完全相同。

当前已通过：原模型／许可证哈希、同权重预测和参数梯度一致、52 个配置的轻量形状检查、真实默认宽度 512 的大桥形状梯度／恢复、十三份脚本预算／路径／失败传递、临时合成 H5 直接实例化实验类的训练／独立测试。52 项形状检查使用较小模型维度验证接口，不作为默认宽度运行或正式训练；512 宽度另有独立检查。

## 7. 最终检查、兼容性与交付

| 检查 | 实际结果 |
| --- | --- |
| 新增 FACT 检查 | 7 项通过，含 52 个脚本配置的接口检查、十三份 shell 语法／预算／固定日志／失败即停检查 |
| 原大桥接入 | test_gje_extended_models.py 的 5 项通过，原 PULSE、PULSE_SCF、PULSE_FACT 的训练、恢复及独立测试仍可用 |
| 原大桥数据层 | test_gjemnessund.py 的 8 项通过；合计 20 个不同检查项通过 |
| 环境 | Python 3.13.5／PyTorch 2.9.0 CPU，使用现有临时环境与既有依赖，无须增加依赖包 |
| Python 3.8 兼容 | 新增 Python 源码通过 3.8 语法检查；另在 Python 3.8／PyTorch 2.0 CPU 核对原文件预测／参数梯度与 512 宽度的大桥形状训练恢复 |
| 文件保护 | 起点 186 个已有代码、脚本、参考文件中仅 run.py 有获授权修改；267 个既有日志、权重和结果文件大小与修改时间未变 |
| 模型保持 | models/FACT.py 与 ZIP 原文件逐字相同，SHA256 为 43e433220a204461b6bb579ef6a4d802be3ac0b8c4eea15c0880cdc393dba831；原文件格式包括空白保留 |

独立测试使用相同配置、model_id、数据路径、debug 参数及 --checkpoints 根目录，--is_training 改为 0。FACT 分支尊重指定根目录并核对数据指纹、scaler 和模型配置；不同数据不能静默加载旧实验。标准数据和大桥的两条测试路径均已核验自动／独立测试指标一致。

源码还有一个边界：内部 in-place 归一化不支持对原始输入张量求梯度。本轮使用普通数据输入（requires_grad=False），原模型参数梯度正常；没有为输入梯度用途修复源码。

未完成：服务器 CUDA、真实十二数据集和大桥的完整训练；本机缺少 PEMS07 原始文件及大桥 H5，不用合成结果代替。当前复制原模型支持的 core=0 问题和特殊小变量网格限制仍明确拒绝。正式实验结果待填写，提交／推送仅包含本轮增量。
