# PULSE SCF 首版实现与开发检查

日期：2026-10-29。范围：ETTh1 与 Weather 的模型和运行脚本；暂不安排大桥数据适配或实验。

记录位置：`PULSE-phy/notes/`。目录已由用户移入主开发仓库，根目录开发约束见 [AGENTS.md](../../AGENTS.md)，工程内链接已按新位置修正。

当前状态：三个模型版本、模块复制、最小注册和脚本已实现。首版新增 15 项检查通过，连同 8 项既有回归检查共 23 项通过。后续用户服务器截图显示完整模型的 ETTh1 96→96 已完成单次训练与测试，完整日志及同协议基线比较仍待补齐，不能据此宣称性能提升。

## 1. 开发起点与保留内容

PULSE 工程起始提交为 `8e4e3226be5685170a048a664fb359b807b742a3`。开发开始时已有以下工作区变化，本次均保留：

```text
 M .DS_Store
 M scripts/Gjemnessund.sh
?? logs/.DS_Store
?? logs_back/.DS_Store
?? pulse-notes/.DS_Store
?? pulse-notes/experiments/
?? scripts/diagnose_prior_reliability.py
```

对开发开始时的 125 个既有代码、脚本及参考目录文件进行哈希对比，只有 `PULSE-phy/exp/exp_main.py` 因本次最小注册发生变化：新增 1 行导入和 3 个模型注册项。原 `run.py`、`models/PULSE.py`、数据加载、原脚本及两个来源目录保持原样。

## 2. 文件与职责

| 文件或目录 | 实现内容 |
| --- | --- |
| [PULSE_SCF.py](../models/PULSE_SCF.py) | `PULSESCFModel`：复用 PULSE 参考处理、Router 与恢复；实现历史条件与 SAM 配对 |
| [PULSE_SOFTS.py](../models/PULSE_SOFTS.py) | 仅替换 SOFTS 式骨干的对照模型 |
| [PULSE_SCF_SIMPLE.py](../models/PULSE_SCF_SIMPLE.py) | 在原变量表示上直接接入 FACT 增量的对照模型 |
| [config.py](../layers/pulse_scf/config.py) | SCF 专属配置及合法性校验，与 Router 参数分开 |
| [backbone.py](../layers/pulse_scf/backbone.py) | SOFTS 式嵌入、编码器与预测头 |
| [interaction.py](../layers/pulse_scf/interaction.py) | STAR 核心访问、FACT 增量适配、历史状态门控与连接 |
| [vendor 来源清单](../layers/pulse_scf/vendor/sources.json) | 复制 4 个来源模块，记录文件哈希、选择的类与改动说明，保留两个来源许可证 |
| [ETTh1.sh](../scripts/PULSE_SCF/ETTh1.sh) | 从原 ETTh1 脚本复制配置与命令结构，增加明确的模型和长度选择 |
| [Weather.sh](../scripts/PULSE_SCF/Weather.sh) | 从原 Weather 脚本复制配置与命令结构，增加明确的模型和长度选择 |
| [test_pulse_scf.py](../tests/test_pulse_scf.py) | 模块、模型、原验证流程与脚本的 15 项检查 |

`exp_main.py` 仅注册 `PULSE_SOFTS`、`PULSE_SCF_SIMPLE`、`PULSE_SCF`。原实验类中的训练、损失、验证、测试和保存逻辑未改写，未新增 Python 训练入口。

## 3. 实际配置与行为

| SCF 专属设置 | 默认值 |
| --- | --- |
| `scf_d_model` | 128 |
| `scf_d_core` | 32 |
| `scf_d_ff` | 256 |
| `scf_e_layers` | 1 |
| `scf_dropout` | 0.1 |
| `scf_fact_dilation` | 2，对应原块中 dilation 1、2 的并行卷积分支 |
| `scf_num_kernels` | 2，每个分支使用 1×1 和 3×3 核 |
| `scf_gate_hidden` | 32 |
| `scf_gate_init` | 0.1，完整模型门控初始化在该值附近 |

这些字段可供直接实例化模型时使用；现有 `run.py` 尚未增加对应命令行参数。脚本使用上述独立默认值，Router 仍由原 `d_model` 等参数控制。有效设置和模型变体写入检查点的 `_extra_state`，加载时校验一致性。

变量嵌入不额外添加时间标记 token；PULSE 已将时间编码加入参考结构。STAR 使用原训练随机聚合与推理加权聚合，完整模型仅暴露并复用当次聚合得到的核心。

FACT 适配层沿用原网格和并行 dilation 语义，返回 `F(D)−D`；骨干外层残差只加一次。来源块对部分小变量数的单次切片补齐不足，SCF 适配层使用循环前缀补齐，原来源副本保持不变。

SAM 使用原抽样的同一个系数和样本索引配对。新增门控条件中的锚点、归一化残差及统计量描述混合后的骨干输入；Router 与未来时间编码仍按原 PULSE 的未混合参考规则处理，监督由原实验类配对。推理不进行混合。

第一版显式支持非 AMP、单设备路径；不支持的 AMP、多设备配置会报明确信息，避免落入原流程中不匹配的增强监督分支。正式 CUDA 训练尚待实际环境验证。

## 4. 脚本交付与使用

脚本保持原数据集文件名、顶部配置变量和 `python -u run.py` 续行格式。四个预测长度的原始参数已逐项对比，除模型名和独立模型标识外保持一致。

从 `PULSE-phy/` 根目录运行：

```bash
bash scripts/PULSE_SCF/ETTh1.sh
bash scripts/PULSE_SCF/Weather.sh
MODEL_NAME=PULSE bash scripts/PULSE_SCF/ETTh1.sh
MODEL_NAME=PULSE_SOFTS bash scripts/PULSE_SCF/ETTh1.sh
MODEL_NAME=PULSE_SCF_SIMPLE bash scripts/PULSE_SCF/ETTh1.sh
PRED_LENS="96 192 336 720" bash scripts/PULSE_SCF/ETTh1.sh
```

原 `PULSE_SCF/` 共享脚本默认预测长度为 96；新增专用脚本的默认范围见第 4.1 节。输入长度仍为 96，数据位置沿用原脚本的 `all_datasets/`，需要实际数据和 CUDA 环境。未确认的数据文件可用性不视为已满足。

`MODEL_NAME` 和 `PRED_LENS` 是 shell 层的模型、长度选择，最终仍调用原入口的既有参数，不提供新的 SCF Python CLI 参数。`RUN_TAG` 默认使用 UTC 时间和进程号，用于独立日志和模型标识；如手动指定，应使用新的标识。已有日志受到禁止覆盖保护。

日志位于 `logs/<ModelName>/<dataset>/<run_tag>/`；检查点和结果沿用原入口的目录规则，模型标识包含新模型名和运行标识。脚本会传递失败状态并保存标准错误，任何一次训练失败会停止后续长度。

### 4.1 已开发模型、运行脚本与日志目录

当前提供共享脚本与模型专用脚本两种方式。新增 `PULSE_SOFTS/`、`PULSE_SCF_SIMPLE/` 专用脚本固定模型名，默认跑齐四个预测长度；原 `PULSE_SCF/` 共享脚本仍默认预测 96，并保留 `MODEL_NAME` 选择。下表从工程根目录运行：

| 模型 | 作用 | 运行命令 | 预计日志目录 |
| --- | --- | --- | --- |
| `PULSE_SOFTS` | PULSE＋SOFTS 骨干对照 | `bash scripts/PULSE_SOFTS/ETTh1.sh` | `logs/PULSE_SOFTS/ETTh1/<run_tag>/` |
| `PULSE_SCF_SIMPLE` | PULSE＋SOFTS＋直接 FACT 增量对照 | `bash scripts/PULSE_SCF_SIMPLE/ETTh1.sh` | `logs/PULSE_SCF_SIMPLE/ETTh1/<run_tag>/` |
| `PULSE_SCF` | 共享核心偏离＋历史状态条件交互 | `bash scripts/PULSE_SCF/ETTh1.sh` | `logs/PULSE_SCF/ETTh1/<run_tag>/` |
| `PULSE` | 原模型参照 | `MODEL_NAME=PULSE bash scripts/PULSE_SCF/ETTh1.sh` | `logs/PULSE/ETTh1/<run_tag>/` |

Weather 使用对应模型目录下的 `Weather.sh`。其日志子目录使用原脚本变量 `weather`，为小写；ETTh1 子目录保留 `ETTh1` 大小写。新增专用脚本可通过 `PRED_LENS=96` 限定一次运行，模型名保持固定。

每个预测长度单独生成一个文件，文件名规则为：

```text
<ModelName>_<dataset>_96_<pred_len>_<run_tag>.log
```

本机完整模型的预计日志路径为：

```text
/Users/cyt/projects/i/pinn-code/PULSE-phy/logs/PULSE_SCF/ETTh1/<run_tag>/PULSE_SCF_ETTh1_96_96_<run_tag>.log
/Users/cyt/projects/i/pinn-code/PULSE-phy/logs/PULSE_SCF/weather/<run_tag>/PULSE_SCF_weather_96_96_<run_tag>.log
```

`<run_tag>` 默认由 UTC 时间与进程号生成，实际运行时不保留尖括号。每次执行脚本，终端会先打印模型、数据集、输入与预测长度、日志绝对路径，以及可直接复制的 `tail -f` 命令；训练输出与报错仍通过 `> "$log_file" 2>&1` 写入该文件。

### 4.2 指定标识与实时查看

例如，在工程根目录用一个新的固定标识运行：

```bash
RUN_TAG=etth1_scf_001 bash scripts/PULSE_SCF/ETTh1.sh
```

可在另一个终端查看：

```bash
tail -f /Users/cyt/projects/i/pinn-code/PULSE-phy/logs/PULSE_SCF/ETTh1/etth1_scf_001/PULSE_SCF_ETTh1_96_96_etth1_scf_001.log
```

该标识是用法示例，不表示已经执行或生成结果。再次运行时换用新标识，避免触发已有日志的防覆盖保护。选择多个预测长度时，同一运行目录内分别保存 96、192、336、720 的日志。

日志预期包含启动参数、设备、数据窗口数量、训练轮次和验证损失，训练及测试成功后包含最终 MSE／MAE；若启动失败，则在同一文件中保留 traceback。创建日志文件不等于已经开始训练。此前本机启动尝试因缺少 `all_datasets/ETTh1.csv` 在读取数据时退出，没有产生训练指标。

## 5. 开发检查记录

本次环境：macOS 15.3 arm64、Python 3.13.5、PyTorch 2.9.0、NumPy 2.3.4；检查使用 CPU，CUDA 不可用。完整既有回归所需的可选依赖安装在临时目录，未修改用户 Python 环境或项目依赖文件。

| 检查 | 结果 |
| --- | --- |
| 新增检查 | 15 项通过 |
| 连同既有回归的完整检查 | 23 项通过 |
| STAR 训练／推理与来源行为等价 | 通过，固定权重与随机状态后输出一致 |
| FACT 来源等价与小变量数适配 | 通过；补充 1、2、4、9、321 变量检查 |
| 三种模型前向和 FFT 目标反向 | 通过；所有可训练参数获得有限梯度 |
| SAM 条件、输入、统计量和原 Router 配对 | 通过 |
| 门控仅由历史状态决定 | 通过；改变未来时间标记不会改变门控值 |
| 检查点保存、重载和配置不匹配保护 | 通过 |
| 原实验类注册及验证流程 | 原 PULSE 与三个新模型均通过 |
| 两个脚本的 Bash 语法 | 通过 |
| 两个脚本四个长度的原始参数对比 | 通过；使用命令捕获检查，不启动训练 |
| 脚本默认长度、显式对照、失败传递及防覆盖 | 通过 |
| 原 `run.py --help` | 通过 |
| 来源模块原文与副本哈希 | 通过 |
| 开始时既有文件哈希对比 | 125 个文件中仅允许的实验类注册发生变化 |

新增检查命令（从工程根目录运行）：

```bash
python -B -m unittest discover -s tests -p test_pulse_scf.py -v
bash -n scripts/PULSE_SCF/ETTh1.sh
bash -n scripts/PULSE_SCF/Weather.sh
```

完整回归在测试所需依赖可用时使用 `python -B -m unittest discover -s tests -v`。这些检查使用随机张量、临时数据与命令捕获，不能替代真实数据集训练或预测精度比较。

## 6. 当前待办

独立的 PULSE＋FACT 还未实现；现有 `PULSE_SCF_SIMPLE` 同时包含 STAR 与 FACT，不代表这一基础对照。该组合后续单独开发，本次仅增加已实现模型的脚本。

1. 归档服务器已完成的 PULSE_SCF、ETTh1 96→96 完整日志，继续运行两个对照及其他预测长度；Weather 另行完成首轮验证。
2. 需要从 CLI 调节 SCF 专属配置时，先提出具体 `run.py` 改动并取得额外确认。
3. 完成同协议的原 PULSE、骨干替换、简单连接与完整方法比较，将真实指标填入[开发计划的结果与分析部分](PULSE-SOFTS-FACT开发计划.md)。
4. 基于结果决定继续、简化或修改机制，再开展必要的连接对照和相关工作区别核查。

大桥数据不属于当前待办，不新增相关适配、脚本或实验。

## 7. 依赖清单核对

2026-10-29 按当前代码实际导入核对：原 `requirements.txt` 列出 `numpy`、`matplotlib`、`pandas`、`scikit-learn`、`torch`，但未列出 `einops`。原训练入口会导入全部已注册模型，其中 `PatchTST.py` 在模块顶层导入 `einops`；即便选择 PULSE 或 SCF 模型，干净环境也需要该包。

新增 [requirements-scf.txt](../requirements-scf.txt)，内容为引用原基础清单并补充 `einops`。来源 SOFTS、FACT 的整套版本清单不直接合并到主工程，避免引入未使用的依赖或强制降级 PyTorch。当前 SCF 模块本身使用 PyTorch、NumPy 和标准库。

SciPy 由 `scikit-learn` 的依赖关系安装，当前 SCF 模块未直接使用它，无需重复作为新依赖加入。`h5py` 不属于当前 ETTh1／Weather 路线；可选的 wavelet、FLOPs 工具也未纳入本次执行路径。

此前完成检查所用的 Conda base 环境已经具有当前所需的包：

| 包 | 本机已安装版本 |
| --- | --- |
| torch | 2.9.0 |
| numpy | 2.3.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| scikit-learn | 1.7.2 |
| scipy | 1.16.3 |
| einops | 0.8.1 |

以上为本机 CPU 检查环境记录，不是 GPU 版本锁定文件。依赖清单沿用原仓库未固定版本的方式；正式 GPU 训练需另行记录实际 Python、PyTorch、CUDA 与驱动环境。

本机 `python` 指向 `/Users/cyt/miniconda3/bin/python`，而默认 `python3` 指向另一套 Framework Python。安装与运行应使用同一解释器；本机复用已检查环境时使用 `python -m pip`。

本次执行 `python -m pip install --dry-run --no-index -r requirements-scf.txt`，声明的包及传递依赖均已满足，无需安装；原实验入口与三个 SCF 模型的导入检查通过。原依赖文件、训练入口及 Python 环境均未改动。

全局 `pip check` 发现已有的 Streamlit 1.51.0 与 Protobuf 7.36.2 约束冲突。当前训练入口不导入 Streamlit，该问题不计入本项目依赖清单，也未对用户全局环境作修复或降级。

## 8. Linux CUDA 严格确定性兼容修复

2026-10-29，服务器在 `PULSE_SCF`、ETTh1、96→96、batch 256 的首次反向传播报错：`scatter_add_cuda_kernel does not have a deterministic implementation`。日志已成功读取 train 8449、val 2785、test 2785 个窗口，但未完成训练轮次，没有新的预测指标。服务器精确的 PyTorch／CUDA 版本尚未提供。

根因定位到 STAR 的训练硬采样使用 `torch.gather`。其反向调用 `scatter_add_`，与服务器当前环境中的 `torch.use_deterministic_algorithms(True)` 冲突。该调用关系经本机算子跟踪及 [PyTorch 1.10 的官方实现](https://github.com/pytorch/pytorch/blob/v1.10.0/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L1067-L1072)核对。FACT 补齐使用的 `index_select` 反向还会调用 `index_add_`，也一并在 SCF 适配范围内规避。

新增 [deterministic.py](../layers/pulse_scf/deterministic.py)，分别实现：

- `scf_select_core_deterministic`：将相同采样索引转成布尔选择掩码，再求和得到同一个核心，保持硬采样输出和梯度；不改变 multinomial 采样规则。
- `scf_pad_channels_deterministic`：使用前缀切片或重复后切片进行循环补齐，保持原 SCF 布局及梯度，替代 `index_select`。

完整模型与简单连接模型在现有 SCF 适配类中使用新函数；纯 SOFTS 对照也使用参数名兼容的 `SCFSTAR`。来源副本、原 PULSE、`run.py` 与其严格确定性设置保持原样；不增加可训练参数或新超参数。这是执行兼容修复，不作为新增方法贡献。

新增 [test_scf_deterministic.py](../tests/test_scf_deterministic.py)，包含前向／梯度与原索引算子的等价检查、算子跟踪，以及三个模型在严格确定性下的 FFT 目标、反向和 Adam 更新。CUDA 项在 CUDA 可用时自动执行。

| 环境 | 结果 |
| --- | --- |
| 本机 Python 3.13／PyTorch 2.9 CPU | SCF 相关 20 项检查中 19 项通过、1 项 CUDA 检查跳过 |
| 本机 Python 3.8／PyTorch 2.0 CPU | 专属兼容检查 5 项中 4 项通过、1 项 CUDA 检查跳过 |
| 服务器目标 GPU／PyTorch 环境 | 待服务器运行新增 CUDA 检查和真实训练验证 |

服务器在项目根目录更新后，先运行专属兼容检查，再重跑脚本：

```bash
git pull --ff-only origin main
python -B -m unittest discover -s tests -p test_scf_deterministic.py -v
bash scripts/PULSE_SCF/ETTh1.sh
```

服务器 CUDA 可用时，`test_strict_deterministic_training_step_cuda` 应实际执行并显示 `ok`，不能用跳过替代 GPU 验证。正式结果仍待该环境重新训练后填写；现有其他服务器任务无需因本次代码修复被中止。

## 9. 基础组合核对与专用运行脚本

用户明确的基础组合为 PULSE＋SOFTS、PULSE＋FACT、PULSE＋SOFTS＋FACT。当前独立 PULSE＋FACT 尚未实现；`PULSE_SCF_SIMPLE` 属于含 STAR 的三者简单连接，不能代替这一缺失对照。

本次不改变模型结构、代码组织或训练入口，只新增四个专用 shell 脚本：

| 脚本 | 固定模型 | 默认预测长度 | 日志目录 |
| --- | --- | --- | --- |
| [PULSE_SOFTS/ETTh1.sh](../scripts/PULSE_SOFTS/ETTh1.sh) | PULSE_SOFTS | 96、192、336、720 | `logs/PULSE_SOFTS/ETTh1/<run_tag>/` |
| [PULSE_SOFTS/Weather.sh](../scripts/PULSE_SOFTS/Weather.sh) | PULSE_SOFTS | 96、192、336、720 | `logs/PULSE_SOFTS/weather/<run_tag>/` |
| [PULSE_SCF_SIMPLE/ETTh1.sh](../scripts/PULSE_SCF_SIMPLE/ETTh1.sh) | PULSE_SCF_SIMPLE | 96、192、336、720 | `logs/PULSE_SCF_SIMPLE/ETTh1/<run_tag>/` |
| [PULSE_SCF_SIMPLE/Weather.sh](../scripts/PULSE_SCF_SIMPLE/Weather.sh) | PULSE_SCF_SIMPLE | 96、192、336、720 | `logs/PULSE_SCF_SIMPLE/weather/<run_tag>/` |

专用脚本不读取 `MODEL_NAME`，以避免与其他实验的环境变量混淆。设置 `PRED_LENS=96` 可只跑一次。原 `scripts/PULSE_SCF/` 共享脚本保持不变，仍默认单个预测长度 96。

四个新脚本均通过 Bash 语法、默认四时距、单时距选择、参数等价和实际日志路径捕获检查；这些检查使用临时命令捕获，不启动训练。逐项对比了全部 51 个既有 Python 文件的哈希，本次均未改变。

另据用户提供的服务器截图，PULSE_SCF 在 ETTh1 96→96 上已运行至第 20 轮，连续 5 轮验证未改善后早停，随后完成测试：MSE 约 0.3709、MAE 约 0.3985。此处记录截图观察值，完整日志、实际环境和种子仍待归档；两个对照及独立 FACT 组合没有被这次运行验证。
