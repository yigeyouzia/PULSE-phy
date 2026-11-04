# Gjemnessund 扩展模型接入计划与通用方法

日期：2026-11-04。当前用户要求：接入 PULSE_SCF、PULSE_FACT，保持模型不变，保证原实验继续可用。先记录计划再开发；入口最小修改已另获用户明确确认，最终检查记录见第 8 节。

## 1. 范围、授权与保护边界

- 主工程为 `PULSE-phy/`，遵守 [AGENTS.md](../../AGENTS.md)。本轮只接入两个已实现模型，不增加模型机制或修改模型参数默认值。
- `run.py` 受 AGENTS.md 第 3 节保护。用户已于 2026-11-04 对具体白名单修改回复“允许这项最小入口修改”；按此授权仅允许 PULSE、PULSE_SCF、PULSE_FACT 使用 Gje，不新增 CLI 参数，不改变原参数默认值、实验命名、调用顺序或非 Gje 分支。其他入口修改仍须另行确认。
- `models/`、`layers/`、Gje loader、时间特征、优化器、SAM／FFT 损失、指标及所有既有脚本保持原样。原版 Gje 的来源记录格式继续兼容；扩展模型单独补充真实来源和有效配置。
- 不改动 FACT-main、SOFTS-main 或其他项目。不复制、移动或改写服务器 H5、旧日志、缓存、权重及指标。
- 不启动正式训练、不连接服务器。本地检查只在临时目录使用合成 H5 和临时权重，不能作为真实桥梁实验精度。用户随后已明确要求将本次接入增量提交并推送 GitHub，Git 交付按此后续授权执行，不包含用户原有的无关工作区修改。
- 开始时已有 `.DS_Store`、`.gitignore`、`scripts/Gjemnessund.sh` 修改，历史日志删除，以及 `.vscode/` 等用户内容；全部保留。代码起点与保护文件哈希、既有输出状态已另存临时快照，交付前核对。

起点提交：`6082898567d9692e6c8d976af09668e7c352f1db`。新脚本参考的是用户当前 `scripts/Gjemnessund.sh` 工作区版本，SHA256 为 `37ac49b1ec835afa83a28347e61009938620a7c91bf545e51383db6bfc857149`；与提交版本的差异仅为 `time_dim` 和 `dsa` 两行参数顺序调整，参数值没有变化。本轮保留该用户修改，不将工作区版本误记为干净提交版本。

## 2. 数据名称、位置与事实边界

| 项目 | 当前确定内容 |
| --- | --- |
| 数据集参数 | `--data Gjemnessund` |
| H5 文件名 | `gjemnessund.h5`，pandas fixed 结构，不是普通 CSV |
| 默认脚本位置 | 工程根目录下 `all_datasets/gjemnessund.h5` |
| Linux 访问路径 | 若工程仍为 `/home/simplexity/cyt/PULSE-phy`，则为该目录下 `all_datasets/gjemnessund.h5`；尚未远程核查实际文件或软链接位置 |
| Mac 当前状态 | 工程内有历史日志与缓存，没有该位置的 H5 本体；本轮不据此声称已核实 Linux 数据内容 |
| 数据协议 | 10 通道：前六个加速度、后四个风通道；采样间隔 0.5 秒 |
| 既有切分 | 每记录裁剪重叠时间前缀后按 50%／25%／25%；仅训练行拟合公共 StandardScaler，不跨记录拼窗 |
| 旧实验依据 | [原脚本](../scripts/Gjemnessund.sh)、[已有接入说明](../pulse-notes/PULSE-Gje大桥数据集使用与HMSFormer对比计划.md)、`logs/Pulse/Gjemnessund/` 四份日志 |

当前缓存标记为 `per_record_50_25_25_train_scaler_trim_overlap_v1`。新模型复用同一数据协议，不将其称为跨月份或跨桥独立泛化。后续若研究跨记录／跨月份切分，需另建协议，并让全部对照按新协议重跑。

## 3. 本轮接口与首轮参数

统一输入 `(X, cycle_index, x_mark, y_mark)`：`X=[B,720,10]`，时间标记为 `[B,720,6]` 与 `[B,H,6]`。训练返回 `(prediction,mixup_params)`，推理返回 `[B,H,10]`；当前两个模型已有这一接口，不改写 forward。

| 参数 | 首轮取值／含义 |
| --- | --- |
| `seq_len` | 720 点，即 6 分钟历史 |
| `pred_len` | 96／192／336／720 点，即 48／96／168／360 秒 |
| `features / enc_in / time_dim / label_len` | M／10／6／0 |
| `d_model / patch_size` | 16／24，d_model 仍控制 Router |
| `dsa / dsb / ksize` | 1／1／5 |
| `inv_len / gje_cycle_len` | 96／96，技术参照索引，不是辨识出的物理周期 |
| `gje_window_stride` | 16 |
| `batch_size / learning_rate` | 16／0.005；保留原调度行为，不声称学习率全程固定 |
| `train_epochs / patience / itr / seed` | 30／5／1／2024 |
| 损失 | rec_lambda=0，auxi_lambda=1，fft／complex／MAE，module_first=1 |
| 设备 | 单设备、非 AMP；不支持未观测未来预测接口 |
| 正式数据范围 | gje_record_limit=0、gje_max_batches=0；调试值必须明确标记 |

训练仍监督和预测全部 10 通道，另外报告 Acc6、风通道及逐通道误差。不能只把 `c_out` 改成 6 就声称已完成纯加速度任务。未来实际加速度、实测风信号均不得作为已知未来输入；六维时间标记只是已知日历特征。

## 4. 计划改动的文件

| 文件 | 计划改动 |
| --- | --- |
| `run.py` | 仅将 Gje 模型条件改成明确的已验证白名单；原 PULSE 和其他数据集行为不变 |
| `utils/gje_model_support.py` | 新增支持模型清单和扩展模型来源／配置记录函数，不包含训练或模型计算 |
| `exp/exp_main.py` | 只在新模型的 Gje 来源记录中调用新增函数，记录实际主模型及依赖文件哈希和 get_extra_state；不改训练／验证／测试计算 |
| `scripts/PULSE_SCF/Gjemnessund.sh` | 新增简约脚本，固定模型，四个时距逐个直接调用原 run.py |
| `scripts/PULSE_FACT/Gjemnessund.sh` | 同上，固定为 PULSE_FACT |
| `tests/test_gje_extended_models.py` | 补入口保护、配置和来源记录、Gje 接口、临时原入口训练与重载检查 |
| 本文及脚本输出索引 | 记录方法、路径、用法、检查结果与未完成事项 |

不把大桥任务自动加入既有全数据集一键脚本，避免旧命令突然增加训练。当前未要求新增一键批量入口。新脚本不定义函数、菜单、模型选择器或预测长度选择器。

## 5. 路径参数、执行方式与输出

从工程根目录运行。新脚本顶部默认 `root_path_name=./all_datasets/`，允许用 `GJE_ROOT_PATH` 环境变量指定 H5 所在目录；`data_path_name=gjemnessund.h5`。文件名不同时只调整顶部变量，不修改 loader。

```bash
# 默认：PULSE-phy/all_datasets/gjemnessund.h5
bash scripts/PULSE_SCF/Gjemnessund.sh
bash scripts/PULSE_FACT/Gjemnessund.sh

# H5 在另一目录时：环境变量仅改变数据根目录，不另建入口
GJE_ROOT_PATH=/实际/H5所在目录 bash scripts/PULSE_FACT/Gjemnessund.sh
```

用户已明确确认入口白名单修改，该项补丁已落入原入口。上述新脚本使用原 `run.py`，服务器准备好真实 H5 和依赖后即可运行。`requirements-gje.txt` 已包含所需 h5py、einops、scipy 及原依赖，无需为本轮接入新增依赖。

| 参数 | 路径规则 |
| --- | --- |
| `--root_path` | H5 所在目录；可相对当前工程根目录，也可为 Linux 绝对目录 |
| `--data_path` | H5 文件名；与 root_path 组合读取 |
| `--gje_cache_dir` | 现有可选参数；默认在数据文件旁的 `.gje_cache/`，需可写；旧缓存缺失会重建 |
| `--checkpoints` | 新脚本显式设置为 `./checkpoints/<ModelName>/` |
| `--model_id` | 独立模型名、数据集、长度及运行标识，避免旧权重冲突 |

```text
PULSE-phy/
├── all_datasets/
│   ├── gjemnessund.h5                 # Linux 上的真实数据，Mac 本轮不补造
│   └── .gje_cache/<identity>/         # 复用现有预处理协议
├── logs/
│   ├── PULSE_SCF/Gjemnessund/PULSE_SCF_720_96.log
│   └── PULSE_FACT/Gjemnessund/PULSE_FACT_720_96.log
├── checkpoints/
│   ├── PULSE_SCF/<setting>/checkpoint.pth
│   └── PULSE_FACT/<setting>/checkpoint.pth
└── results/<setting>/
    ├── gje_metrics.json
    └── gje_data_protocol.json
```

其他三个时距同样到长度对为止命名，不添加日志运行标识层。固定日志重跑会覆盖同名新日志，需保留时由运行者提前归档；权重使用独立实验标识。原 `scripts/Gjemnessund.sh`、旧日志路径和旧权重不迁移。

Gje 独立测试分支已使用 `args.checkpoints`，不同于普通数据集的旧固定根路径分支；本轮不改后者。独立测试需要沿用相同 model_id、数据配置、debug 设置及 checkpoint 根目录，以便生成相同 setting 和校验数据协议。

## 6. 普通模型接入同一桥梁数据的通用流程

1. **先固定数据任务**：明确十通道全预测、只预测 Acc6，还是历史风条件输入；记录真实采样率、时间段、通道及指标尺度。改变任务时全部基线重新训练，不混用旧结果。
2. **复用数据层**：Gje loader 负责解析、标准化、时间与 cycle、窗口及协议输出；不用为每个模型复制或修改切分逻辑。
3. **检查原模型接口**：原生 `forward(x)`、四参数 encoder-decoder、或依赖图／外生变量的模型，分别定义必要适配。保持原模型计算，在独立命名适配器中处理形状与调用约定，不能伪装成 PULSE 名称绕过分支。
4. **明确训练协议**：本轮两个 PULSE 系模型原有 SAM／FFT 协议直接复用。其他模型应分别决定忠实原生训练或受控统一训练；不能无说明地强加 SAM／FFT，再宣称复现其原方法。
5. **最小接入实验类**：注册适配模型，并在明确命名的方法或子类中处理必要差异；不为单个接入重写整套训练器。Gje 当前测试函数直接调用四个输入并期待张量，普通模型不能仅注册名字就宣称跑通。
6. **按验证结果开放入口**：只对白名单中已检查的接口放行，不自动允许所有注册模型。新增参数或入口行为按 AGENTS.md 单独说明、确认。
7. **逐模型检查**：前向／梯度、训练与推理返回、保存重载、协议一致、真正只用历史信息，再进行原入口的少量 batch 试运行。
8. **正式比较与归档**：同一窗口、scaler、监督通道、时距和选模规则，记录模型有效设置、源码、种子和计算预算；报告10通道及 Acc6，而非只选择有利指标。

本轮只实现 PULSE_SCF／PULSE_FACT，以上流程不等于已经支持 iTransformer、CycleNet、DLinear 等普通模型。后续接入也不应通过另一个训练入口绕过当前约束。

## 7. 验收计划

- 对比开始快照，确认 models、layers、loader、参考项目及所有旧脚本字节不变，旧输出未写入。
- 检查 run.py 增量只涉及 Gje 模型条件；原参数默认值、gje_setting 与训练调用原样。
- 新脚本通过 bash 语法和参数支持检查，路径含空格时正常传递，失败即停；不执行真实训练脚本。
- 检查 C=10、time_dim=6、L=720、四个 H 的前向与 FFT 梯度；模型设置继续沿用各自既有默认值。
- 用临时合成 H5 和隔离工程运行原 run.py，检查新模型训练、最佳权重恢复及独立测试；同时保留原版 Gje 回归检查。
- 核对新模型 gje_run.json 的实际主模型哈希、依赖哈希、有效设置；原 PULSE 来源记录保留兼容。
- 跑既有相关 Gje、SCF、FACT 检查，CPU 与 GPU 验证分别记录，不用 CPU 通过替代 Linux CUDA 实测。

## 8. 实施与结果记录

已完成用户明确授权的入口白名单修改、两个独立脚本、新模型来源记录函数、实验类的来源记录接入、脚本索引和检查代码；未改模型、层、loader、损失与训练计算。`gje_run.json` 为新模型记录实际主文件及实现依赖的 SHA256、有效模型配置；旧 PULSE 的记录字段和值保持兼容。

本机原有 Python 环境缺少 h5py，检查使用 `/tmp/pulse-gje-check-20261104/` 临时环境，复用本机 Python 3.13／PyTorch 2.9 CPU，额外安装既有 requirements-gje.txt 已列明的 h5py；没有修改用户环境或依赖清单。

### 已执行检查

| 检查 | 结果与边界 |
| --- | --- |
| 目标桥梁形状（随机张量） | 两模型 × 四时距，C=10、time_dim=6、L=720；严格确定性 CPU FFT 反向、优化器更新及权重保存重载通过 |
| 临时合成 H5 原入口 | 原 PULSE、PULSE_SCF、PULSE_FACT 的一批训练、最佳权重恢复、自动测试及独立 `is_training=0` 重载通过；独立测试指标与自动测试一致。L=24、H=12，仅验证流程，不代表正式桥梁精度 |
| 新脚本 | bash 语法、全部参数受现有解析器支持、四时距与原桥梁脚本配置一致、路径含空格、stdout／stderr 日志、失败即停通过；未启动正式脚本训练 |
| 原相关检查 | Gje 8 项、SCF 16 项、确定性 5 项、FACT 15 项，加本轮 5 项共 49 项；47 项通过，2 项 CUDA 检查因无 GPU 跳过 |
| 文件保护 | 比对 201 个既有代码／脚本／参考文件，仅 `run.py` 白名单及 `exp/exp_main.py` 来源记录有预期改动；模型、层、数据处理和旧脚本字节不变。259 个既有日志、权重和结果文件大小与修改时间未变，用户原有删除／修改保留 |

2026-11-04 用户明确回复“允许这项最小入口修改”后，已落入下方补丁。随后重跑新增接入检查 5 项与原 Gje 回归 8 项，13 项全部通过；上述总计 49 项为不同检查项的合计，不重复计算重跑。另逐字核对 `run.py` 与起点的差异，确认只有所列白名单修改，参数默认值、其余校验、实验命名和非 Gje 代码均保持原样。两个新脚本已能使用原入口，真实 H5 和 CUDA 实验仍需服务器运行。

### 已确认并实施的唯一入口补丁

位置：`run.py` 的 `if args.data == 'Gjemnessund'` 首个条件。必要性：原条件在模型实例化之前拒绝任何非 PULSE 名称。影响：仅增加两个已经核对四参数接口的允许名称；原 PULSE、非大桥分支、其余校验、参数及实验命名保持原样。验证：补丁落入后的端到端检查、原 Gje 回归和最终差异核对均通过。

```diff
 if args.data == 'Gjemnessund':
-    if args.model != 'PULSE' or args.features != 'M' or args.enc_in != 10:
-        parser.error('Initial Gje adapter requires --model PULSE --features M --enc_in 10')
+    from utils.gje_model_support import GJE_SUPPORTED_MODELS
+    if args.model not in GJE_SUPPORTED_MODELS or args.features != 'M' or args.enc_in != 10:
+        parser.error('Gje requires --model PULSE/PULSE_SCF/PULSE_FACT --features M --enc_in 10')
```

未完成：Linux 真实 H5 路径核查与 CUDA 正式训练待服务器验证。本轮没有新增真实桥梁实验结果；代码、脚本、检查与本文档按用户后续指令交付至 GitHub。
