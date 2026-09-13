# Guangning Wind Power Weather Comparison

本项目统一比较 PatchMLP、DLinear 与 StockEcho。StockEcho 以同步的 16 台风机
作为一个多对象面板样本，学习风机之间随时间变化的可靠关系，再完成多步功率预测。

## StockEcho风电模型

风电实现完整位于 `models/StockEcho_windpower.py`，包括因果时序编码、可靠关系
证据、稠密关系图、响应传递和对齐残差融合。输入变量映射为16维风电关系表示，
每个样本输入
形状为 `[16台风机, 过去1440步(240分钟×10s), 20变量]`，输出为
`[16台风机, 未来90/4320/8640/17280步功率]`（15分钟 / 12小时 / 1天 / 2天）。

## QwenMLP

`models/QwenMLP.py` 使用过去4小时的20维历史变量生成连续数值Patch，送入冻结的
Qwen2.5-1.5B Base编码器，再由MLP残差头直接预测未来功率。该模型不读取预测区间
内的真实天气；官方预训练权重放在 `models/pretrained/Qwen2.5-1.5B`，且不纳入Git。

```bash
HORIZONS="15" LOSSES="MSE" BATCH_SIZE=64 WINDOW_STRIDE_STEPS=360 \
  bash scripts/lzh/run_qwen_mlp.sh
```

上述命令用于1.5B路线的快速可行性验证，每小时取一个预测起点；设置
`WINDOW_STRIDE_STEPS=1` 可恢复每10秒移动一次的完整实验口径。

## TimerWeatherMLP

`models/TimerWeatherMLP.py` 使用预训练的 Timer-84M 因果时序 Transformer。
过去4小时的20维历史变量与预测区间的13维未来天气分别按96个10秒点生成
Token，再沿时间方向直接拼接。模型不增加单独的历史—未来交叉注意力；Timer
主干完成时序交互，每个未来天气Token经共享MLP输出对应的96点功率Patch。

官方权重放在 `models/pretrained/timer-base-84m`，且不纳入Git。默认只解冻Timer
最后2层，Adapter/MLP学习率为 `1e-4`，主干学习率为 `1e-5`。

```bash
PYTHON=/home/xuke/.conda/envs/lzh_3090/bin/python GPU=0 \
  BATCH_SIZE=64 WINDOW_STRIDE_STEPS=360 \
  bash scripts/lzh/run_timer_weather_mlp.sh
```

默认脚本运行15分钟 `OracleFutureWeather` 快速实验，因此结果只能解释为未来13维
变量完全已知时的理论上限。部署实验应改用 `PredictedFutureWeather` 和第一阶段
生成的天气预测文件。

## Acc30 定向损失

`MSEAcc30` 使用固定的 `MSE + 0.3 × 平滑Acc30边界损失`，训练期间不做动态
权重或动态尺度调整。边界损失只统计真实功率
严格大于100 kW的点，并重点惩罚相对误差超过30%的预测。验证阶段按照最高
严格Acc30选择检查点，Acc30相同时选择验证MSE更低的轮次。

同一批广宁风电场01—16号风机的细粒度实验。工程结构参考 StockEcho 的职责分层，但不导入、不调用也不修改 `/home/xuke/fund/StockEcho`。

## 正式协议

- 粒度：原生 10 秒采样，不做分钟平均。
- 历史：过去 240 分钟 = 1440 步。
- 预测：15分钟（90步）、12小时（4320步）、1天（8640步）、2天（17280步）。
- CLI `--horizon` 传分钟数，不限档位；正式协议为 `15 / 720 / 1440 / 2880`，内部自动换成10秒步数。
- 模型：PatchMLP、DLinear、StockEcho。
- 气象口径：NoFutureWeather、OracleFutureWeather。
- 损失：MSE、DBLoss；StockEcho另包含MSEAcc30实验。
- 随机种子：2026。
- 最大100轮，验证集MSE早停，耐心值10。
- Strict ACC30仅统计真实功率严格大于100 kW的点。

## 数据划分

- 训练目标：2026-06-07 00:00之前（连续段 6/1–6/7）。
- 验证目标：`[2026-06-08 00:00, 2026-06-11 00:00)`，覆盖6月8—10日。
- 测试目标：`[2026-06-11 00:00, 2026-06-14 00:00)`，覆盖6月11—13日。
- 历史和目标窗口均不得跨越时间缺口。

## 运行

```bash
cd /path/to/ElectricityForecasting
python scripts/dataChange/build_guangning_10s_dataset.py
# 已切到原生 10s；实验组合写在 sh 里，训练入口仍是 run.py
bash scripts/lzh/run_full.sh
bash scripts/lzh/run_stockecho.sh
bash scripts/lzh/run_acc30.sh
```

OracleFutureWeather使用预测窗口内13维天气相关特征，只表示这些未来变量完全已知时的理论上限。
