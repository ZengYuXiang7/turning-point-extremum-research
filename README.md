# Guangning Wind Power Weather Comparison

本项目统一比较 PatchMLP、DLinear 与 StockEcho。StockEcho 以同步的 16 台风机
作为一个多对象面板样本，学习风机之间随时间变化的可靠关系，再完成多步功率预测。

基本任务模型位于 `models/noweather.py`，使用未来天气的功率预测位于
`models/futureweather.py`，未来天气预测位于 `models/weatherforecast.py`；DLinear、
PatchMLP、StockEcho、Qwen、Timer 和配置等模型骨干放在 `models/backbone/`。

## StockEcho风电模型

StockEcho 位于 `models/backbone/stockecho.py`，包括因果时序编码、可靠关系证据、
稠密关系图、响应传递和对齐残差融合；输入变量映射为16维风电关系表示，
每个样本输入
形状为 `[16台风机, 过去16步(240分钟), 19变量]`，输出为
`[16台风机, 未来1/2/3/4/8/12/16/32/48/96步功率]`
（15/30/45分钟 / 1/2/3/4/8/12/24小时）。

## QwenMLP

`models/backbone/qwen.py` 使用过去4小时的20维历史变量生成连续数值Patch，送入冻结的
Qwen2.5-1.5B Base编码器，再由MLP残差头直接预测未来功率。该模型不读取预测区间
内的真实天气；官方预训练权重放在 `models/pretrained/Qwen2.5-1.5B`，且不纳入Git。

```bash
HORIZONS="15" LOSSES="MSE" BATCH_SIZE=64 \
  bash scripts/lzh/run_qwen_mlp.sh
```

原始记录只保留每小时 `00/15/30/45` 分且秒为 `00` 的整刻值，形成15分钟
时间网格。预测起点间隔由 `--window-stride-steps` 控制，默认值为1，即每个
15分钟网格点都生成一个预测窗口。

## TimerWeatherMLP

`models/backbone/timer.py` 使用预训练的 Timer-84M 因果时序 Transformer。
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

- 粒度：原始数据直接取 `HH:00:00 / HH:15:00 / HH:30:00 / HH:45:00` 的记录，
  每个时间点间隔15分钟，不做区间均值聚合。
- 预测起点：默认每1点（15分钟）一个，严格落在上述整刻网格；可通过
  `--window-stride-steps` 修改。
- 历史：默认过去240分钟 = 16步。
- 预测：15分钟（1步）、30分钟（2步）、45分钟（3步）、1小时（4步）、2小时（8步）、
  3小时（12步）、4小时（16步）、8小时（32步）、12小时（48步）、
  24小时（96步）。其中 `horizon_steps=3` 即未来45分钟。
- CLI `--history-steps`、`--horizon-steps` 和 `--window-stride-steps` 均直接传15分钟采样点数。
- 模型：PatchMLP、DLinear、StockEcho。
- 气象口径：NoFutureWeather；历史中可使用已观测的气象和SCADA特征，预测窗口内不使用任何协变量。
- 损失：MSE、DBLoss；StockEcho另包含MSEAcc30实验。
- 随机种子：2026。
- 最大100轮，验证集MSE早停，耐心值10。
- Strict ACC30仅统计真实功率严格大于100 kW的点。

## 数据划分

- 按全部时间点的先后顺序划分：前70%训练、接着10%验证、最后20%测试。
- 当前处理后数据的边界分别为 2026-08-01 09:30 和 2026-08-14 14:15。
- 历史和目标窗口均不得跨越时间缺口。

## 运行

```bash
cd /path/to/ElectricityForecasting
uv run python generate_data.py
bash scripts/run_dlinear_15min.sh
```

默认脚本依次运行DLinear的15/30/45分钟和1/2/3/4/8/12/24小时预测实验，统一输入
60分钟历史（4点），每15分钟生成一个整刻对齐的预测窗口。
