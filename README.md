# Guangning Wind Power Weather Comparison

本项目统一比较 PatchMLP、DLinear、StockEcho 与 MultiTurbine。StockEcho 以同步的 16 台风机
作为一个多对象面板样本，学习风机之间随时间变化的可靠关系，再完成多步功率预测。

Dataset、任务接线和可视化按开发边界放在 `tasks/single/` 和 `tasks/multi_turbine/`。
两个顶层 Model 入口分别是 `models/single.py` 和 `models/multi_turbine.py`，内部实现分别放在
`models/single_impl/` 和 `models/multi_turbine_impl/`。顶层类只持有
`self.backbone`，`forward` 原样委托给骨干；两个 `task.py` 也只导入各自的顶层 Model 入口。

## StockEcho风电模型

StockEcho 位于 `models/multi_turbine_impl/stockecho.py`，包括因果时序编码、可靠关系证据、
稠密关系图、响应传递和对齐残差融合。每台风机拥有独立的可学习身份嵌入，
该嵌入加入该风机的全部时间token；构图时再由投影后的身份嵌入学习两两关系，
并通过可学习权重与窗口内的动态轨迹关系融合。输入变量映射为16维风电关系表示，
每个样本输入
形状为 `[16台风机, 过去16步(240分钟), 19变量]`，输出为
`[16台风机, 未来1/2/3/4/8/12/16/32/48/96步功率]`
（15/30/45分钟 / 1/2/3/4/8/12/24小时）。

## MultiTurbine 风电模型

`MultiTurbine` 的实际骨干位于 `models/multi_turbine_impl/multi_turbine.py`，采用从上级 `SATRA`
项目迁入的 PSTR-Net 设计：每台风机先经过
State-Conditioned Pattern Expert（残差时序 CNN 与六个异构专家的 Top-3 路由），再在每个
历史时刻经过关注全部风机的语义 Transformer 塔，按“特征到功率、历史到预测步”输出多步功率。
训练与测试均读取每台风机原始的全部 56 个历史特征，其中包含历史功率；预测目标仍为未来功率。
语义塔默认开启；传入 `--satra-use-spatial-relation 0` 时，SPE 直接连接预测头，不构建任何
跨风机注意力。
该模型没有风机 ID 嵌入，风机身份由输入顺序保留。

`MultiTurbine` 目前仅支持 `MultiTurbine` 场景，历史中不使用未来天气。传入 `--pretrain` 时，先对随机遮蔽的
`风机 × 历史时刻` token 重建全部历史特征，再丢弃重建头并微调 SPE 与已启用的语义塔。DTW 先验
默认关闭：不构建训练段关系图，也不创建先验塔；空间关系关闭时，DTW 先验也不会构建。

```bash
env -u LD_LIBRARY_PATH uv run python run_multi.py \
  --mode train --model MultiTurbine --scenario MultiTurbine --loss MSE \
  --point-interval-seconds 900 --history-steps 16 --horizon-steps 4 \
  --window-stride-steps 1 --seed 2026 --epochs 20 --patience 3 \
  --batch-size 32 --learning-rate 0.001 --num-workers 4 --tqdm 1 \
  --dataset-name GuangningWindPower16Turbine15min \
  --result-name multi_turbine_h16_p4_s1_seed2026
```

需要恢复原始 DTW 先验关系塔时，增加 `--satra-use-dtw-prior 1`；此时
`--satra-prior-top-k`、`--satra-prior-candidate-k`、`--satra-prior-downsample` 和
`--satra-prior-band` 才生效，且 DTW 仍只使用训练段功率轨迹。

## QwenMLP

`models/single_impl/qwen.py` 使用过去4小时的20维历史变量生成连续数值Patch，送入冻结的
Qwen2.5-1.5B Base编码器，再由MLP残差头直接预测未来功率。该模型不读取预测区间
内的真实天气；官方预训练权重放在 `models/pretrained/Qwen2.5-1.5B`，且不纳入Git。

```bash
HORIZONS="15" LOSSES="MSE" BATCH_SIZE=64 \
  bash scripts/lzh/run_qwen_mlp.sh
```

处理后数据固定保留原始10秒连续点。模型相邻点跨度由
`--point-interval-seconds` 控制，例如900秒表示从10秒底层序列中每隔90个基础点
选取一个模型点。预测起点间隔由 `--window-stride-steps` 控制，默认值为1，
即每个模型点都生成一个预测窗口。

## TimerWeatherMLP

`models/single_impl/timer.py` 使用预训练的 Timer-84M 因果时序 Transformer。
过去4小时的19维历史变量与预测区间的12维未来天气均以15分钟为一个点输入。
Timer 检查点保持96点 Patch 约束，Adapter 会将较短窗口补齐到该宽度后再沿时间
方向拼接。模型不增加单独的历史—未来交叉注意力；Timer 主干完成时序交互。

官方权重放在 `models/pretrained/timer-base-84m`，且不纳入Git。默认只解冻Timer
最后2层，Adapter/MLP学习率为 `1e-4`，主干学习率为 `1e-5`。

```bash
PYTHON=/home/xuke/.conda/envs/lzh_3090/bin/python GPU=0 \
  BATCH_SIZE=64 \
  bash scripts/lzh/run_timer_weather_mlp.sh
```

## TimeMoE + A-RevIN

`models/single_impl/time_moe_arevin.py` 按 Single 任务统一接口实现
`forward(past, turbine_id)`：历史特征先经 A-RevIN 归一化，再经门控数值
嵌入和风机编号嵌入输入预训练 TimeMoE，最后直接输出多步功率并使用
功率通道的 A-RevIN 统计量恢复尺度。输入特征数、功率列位置和风机数量均
从 `Repository` 动态读取。

模型目前仅支持 `NoFutureWeather`，且不使用 PatchMLP 专用的
`--pretrain` 掩码重建阶段。TimeMoE-50M 权重默认位于
`models/pretrained/TimeMoE-50M`，也可通过 `--time-moe-path` 指定其他本地目录。

最小调用示例：

```bash
python run_single.py \
  --model TimeMoEARevIN \
  --scenario NoFutureWeather \
  --loss MSE \
  --point-interval-seconds 900 \
  --history-steps 16 \
  --horizon-steps 1 \
  --time-moe-unfreeze-layers 0
```

当前默认实验为 `NoFutureWeather`：每个样本只输入预测起点之前的19维历史特征，
不构造也不传入预测窗口内的12维未来协变量。`OracleFutureWeather` 与
两阶段天气预测代码保留在工程中，但不作为当前实验入口。

## Acc30 定向损失

`MSEAcc30` 使用固定的 `MSE + 0.3 × 平滑Acc30边界损失`，训练期间不做动态
权重或动态尺度调整。边界损失只统计真实功率
严格大于100 kW的点，并重点惩罚相对误差超过30%的预测。验证阶段按照最高
严格Acc30选择检查点，Acc30相同时选择验证MSE更低的轮次。

同一批广宁风电场01—16号风机的细粒度实验。工程结构参考 StockEcho 的职责分层，但不导入、不调用也不修改 `/home/xuke/fund/StockEcho`。

## 正式协议

- 底层数据：固定保留原始10秒连续点，不按实验粒度重建 processed 数据。
- 模型点跨度：通过 `--point-interval-seconds` 从10秒底层序列等距取点；
  15分钟粒度对应900秒和90个基础点，不做区间均值聚合。
- 预测起点：默认每1点（15分钟）一个，严格落在上述整刻网格；可通过
  `--window-stride-steps` 修改。
- 历史：默认过去240分钟 = 16步。
- 预测：15分钟（1步）、30分钟（2步）、45分钟（3步）、1小时（4步）、2小时（8步）、
  3小时（12步）、4小时（16步）、8小时（32步）、12小时（48步）、
  24小时（96步）。其中 `horizon_steps=3` 即未来45分钟。
- CLI `--history-steps`、`--horizon-steps` 和 `--window-stride-steps` 均使用模型点数，
  每点实际跨度由 `--point-interval-seconds` 决定。
- 模型：PatchMLP、DLinear、StockEcho、MultiTurbine、TimeMoEARevIN。
- 气象口径：NoFutureWeather；历史中可使用已观测的气象和SCADA特征，预测窗口内不使用任何协变量。
- 损失：MSE、DBLoss；StockEcho另包含MSEAcc30实验。
- 随机种子：2026。
- 最大20轮，验证集MSE早停，耐心值3。
- Strict ACC30仅统计真实功率严格大于100 kW的点。

## 数据划分

- 按全部时间点的先后顺序划分：前70%训练、接着10%验证、最后20%测试。
- 当前处理后数据的70%/80%原始时间边界分别为 2026-08-01 09:36
  和 2026-08-14 14:24；模型窗口仍按各实验的整刻网格对齐。
- 历史和目标窗口均不得跨越时间缺口。

## Mac 本地运行 TimeMoE + A-RevIN

本机可直接复用之前项目的 Conda 环境 `shantou-wind`。原始数据不需要复制进仓库，
下面命令假设 `guangningshuju/` 与项目目录同级：

```bash
cd /path/to/turning-point-extremum-research-main
conda activate shantou-wind

# GitHub 不提交 227 MB 预训练权重；新机器首次运行时下载一次
python scripts/lmt/download_time_moe.py

# 仅需执行一次；会生成 dataset/processed（约 7.7 GB）
python generate_data.py --raw-root ../guangningshuju --workers 8

# 先用一个真实窗口验证 TimeMoE+A-RevIN 前向、反向和优化器更新
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/lmt/smoke_time_moe_arevin.py

# 按已验证参数正式训练：15分钟点，历史16点，预测1点
PYTHON="$(which python)" MODE=train EPOCHS=3 PATIENCE=3 BATCH_SIZE=8 NUM_WORKERS=0 \
  bash scripts/lmt/run_time_moe_arevin_mac.sh

# 训练完成后评估 validation 最优检查点
PYTHON="$(which python)" MODE=test BATCH_SIZE=8 NUM_WORKERS=0 \
  bash scripts/lmt/run_time_moe_arevin_mac.sh
```

在其他机器上从 GitHub 重建环境时，建议使用本次已验证的 Python 3.11 创建 `.venv` 并执行
`python -m pip install -r requirements.txt`。

`generate_data.py` 会检查原始列、必需特征、16台风机以及全部时间轴，校验通过后才替换
`dataset/processed/`。只有 `TimeMoEARevIN` 在 Apple Silicon 上会选择 MPS；其他原有模型的
设备选择保持不变。MPS 不使用 CUDA bf16 autocast，遇到尚未支持的算子时由
`PYTORCH_ENABLE_MPS_FALLBACK=1` 回退到 CPU。

提交 GitHub 前先确认忽略规则生效：

```bash
git init
git status --ignored --short
git add .
git status --short
git commit -m "Add Guangning TimeMoE A-RevIN pipeline"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

`dataset/`、原始 Excel、检查点、运行产物和 `models/pretrained/` 均已忽略。
TimeMoE 权重与官方自定义代码由上述下载脚本按固定版本重建，不会进入 Git。

## 其他实验运行

```bash
cd /path/to/turning-point-extremum-research-main
python generate_data.py --raw-root ../guangningshuju

# DLinear训练
bash "scripts/noweather/length_prediction/15min/dlinear_correlation_ablation_train.sh"
bash "scripts/noweather/length_prediction/10min/dlinear_all_features_train.sh"
bash "scripts/noweather/length_prediction/5min/dlinear_all_features_train.sh"
bash "scripts/noweather/length_prediction/1min/dlinear_all_features_train.sh"

# DLinear测试已有checkpoint
bash "scripts/noweather/length_prediction/15min/dlinear_correlation_ablation_test.sh"
bash "scripts/noweather/length_prediction/10min/dlinear_all_features_test.sh"
bash "scripts/noweather/length_prediction/5min/dlinear_all_features_test.sh"
bash "scripts/noweather/length_prediction/1min/dlinear_all_features_test.sh"

# 1分钟PatchMLP全部特征随机时间token热身训练与测试
bash "scripts/noweather/length_prediction/1min/patchmlp_all_tokenmask_dbloss_train.sh"
bash "scripts/noweather/length_prediction/1min/patchmlp_all_tokenmask_dbloss_test.sh"

# 30秒和10秒PatchMLP全部特征随机时间token热身训练
bash "scripts/noweather/length_prediction/30s/patchmlp_all_tokenmask_dbloss_train.sh"
bash "scripts/noweather/length_prediction/10s/patchmlp_all_tokenmask_dbloss_train.sh"

# 30秒和10秒16台风机联合面板MultiTurbine训练
bash "scripts/multi_turbine/length_prediction/30s/multi_turbine_mmr_pretrain_train.sh"
bash "scripts/multi_turbine/length_prediction/10s/multi_turbine_mmr_pretrain_train.sh"

# 16台风机联合面板StockEcho长度预测
bash "scripts/multi_turbine/length_prediction/15min/stockecho_multi_turbine_train.sh"
bash "scripts/multi_turbine/length_prediction/15min/stockecho_multi_turbine_test.sh"

# 16台风机联合面板MultiTurbine长度预测（DTW先验关闭）
bash "scripts/multi_turbine/length_prediction/15min/multi_turbine_train.sh"
bash "scripts/multi_turbine/length_prediction/15min/multi_turbine_test.sh"
```

`generate_data.py` 只运行一次，之后不同粒度脚本共享同一份10秒 processed 数据。
10/30秒和1/5/10/15分钟脚本分别每隔1/3/6/30/60/90个基础点取一个模型点；短视界组统一使用
60分钟历史，更长视界组使用与预测时长相同的历史长度。PatchMLP 的10/30秒脚本固定80个历史点，
以保留最细2点嵌入分支的非零投影。默认每个模型点建立一个预测窗口。
每个任务的实验类型下按分钟划分目录；`*_train.sh`负责训练，配对的`*_test.sh`只读取正式record指向的
validation最优checkpoint；尚未训练或缺少产物时会直接报错，不会自动重训。

根目录的 `run_multi.py` 与 `run_single.py` 分别负责联合面板和常规预测，训练完成后都会立即生成 checkpoint 预测图：

- `run_multi.py` 只调用 `tasks/multi_turbine/task.py`，读取该任务 Dataset，并从
  `models/multi_turbine.py` 通过顶层选择函数进入 `MultiTurbineModel`。
- `run_single.py` 只调用 `tasks/single/task.py`，读取该任务 Dataset，并从
  `models/single.py` 通过顶层选择函数进入 `SingleModel`。
- `tasks/multi_turbine/trainer.py` 与 `tasks/single/trainer.py` 分别维护各自的训练、验证、
  测试、预训练和结果写入管道；各自的模型与场景参数也分别定义在对应 `configuration.py`。

多风机可视化位于 `tasks/multi_turbine/visualization.py`：固定随机选择一个联合面板窗口，
模型一次输入全部16台风机，图中展示该窗口的前10台风机。相同数据粒度、历史长度、预测长度、
split 和 selection seed 会选择同一窗口；样本归档同时记录完整16台输入和唯一面板索引。

```bash
env -u LD_LIBRARY_PATH uv run python run_single.py \
  --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE \
  --point-interval-seconds 900 --history-steps 4 --horizon-steps 1 \
  --epochs 20 --patience 3 --batch-size 1024 --learning-rate 0.001 \
  --seed 2026 --num-workers 4 --dataset-name GuangningWindPower15min \
  --result-name dlinear_all_features_h4_p1_s1_seed2026 \
  --visualization-split test --visualization-seed 20260913 \
  --visualization-device cuda:0
```

`--visualization-split`、`--visualization-seed` 和 `--visualization-device` 分别控制画图数据集、
固定样本随机种子和设备，默认为 `test`、`20260913` 和 `cuda:0`。训练成功后会使用
同一个 `run_dir` 里的 `best_checkpoint.pt`，将 PDF、PNG 和对应的 NPZ 样本统一写入
`output/figs/`。文件名包含 dataset、result、split、选样 seed 和 checkpoint epoch，可在多组
实验同时绘图时直接区分。

## 随机时间 token 自监督热身

`NoFutureWeather` 下的 `PatchMLP`、`PatchMLPAllFeatures` 支持先重建历史再预测功率。
掩码直接作用于 `[B,L,D]` 的时间维，每个样本随机遮住 `ceil(L × mask_ratio)` 个时间点，
同一时间点的全部D维一起置零，不按patch或连续片段采样。现有PatchMLP骨干内部的
patch embedding保留，没有引入PatchTST或新建patch切分。

热身仅使用训练历史，按缺失位置的重建MSE优化；窗口均值和标准差只从可见位置计算，
避免缺失值从归一化统计量泄露。验证历史完整位于验证段，每轮使用相同随机掩码选模。
第二阶段恢复最佳热身编码器，使用完整历史和原功率标签联合微调预测模型；监督损失、
时间切分及最终选模规则沿用原流程。重建头不进入最终功率checkpoint。

只有命令中带 `--pretrain` 才执行热身；轮数不再决定是否启用。当前项目默认使用
20轮、学习率0.001、随机token比例0.3和余弦硬重启调度。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--pretrain` | false | 带上该开关时执行第一阶段 |
| `--pretrain-epochs` | 20 | 第一阶段最大轮数 |
| `--pretrain-learning-rate` | 0.001 | 热身学习率 |
| `--pretrain-lr-scheduler` | cosine_hard_restarts | 热身学习率调度器 |
| `--mae-mask-ratio` | 0.3 | 随机时间token遮蔽比例，数量向上取整 |
| `--pretrain-patience` | 3 | 验证重建MSE早停耐心值 |
| `--print-freq` | 0 | 常规epoch摘要打印间隔；0=按epoch耗时自动，正数=固定间隔 |

调用时保证至少有一个缺失token和一个可见token，即 `1 <= ceil(L × mask_ratio) < L`。
例如4点历史、比例0.3时实际遮住2点；16点历史时遮住5点。

下面是20轮热身、最多100轮功率微调的运行示例，尚未作为正式实验执行。
使用10秒processed数据并设置15分钟模型点跨度，完整结果名与直接监督训练区分：

```bash
env -u LD_LIBRARY_PATH uv run python run_single.py \
  --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 \
  --point-interval-seconds 900 --history-steps 16 --horizon-steps 1 --window-stride-steps 1 \
  --pretrain --pretrain-epochs 20 --mae-mask-ratio 0.3 \
  --pretrain-learning-rate 0.001 --pretrain-lr-scheduler cosine_hard_restarts \
  --pretrain-patience 3 --epochs 20 --patience 3 --batch-size 32 --learning-rate 0.001 \
  --seed 2026 --num-workers 4 --dataset-name GuangningWindPower15min \
  --result-name patchmlp_tokenmask_m0p4_e20_h16_p1_s1_seed2026
```

重定向日志或后台运行时显式增加 `--tqdm 0`。
热身保存 `checkpoints/{dataset}/{result_name}/round_1_pretrain.pt`，包含可复查的重建头和
共享编码器；最终功率模型保存为同目录 `round_1.pt`。运行目录保存
`pretrain_history.json`、`pretrain_summary.json`，最终records的round中保存实际热身选模与checkpoint路径。
