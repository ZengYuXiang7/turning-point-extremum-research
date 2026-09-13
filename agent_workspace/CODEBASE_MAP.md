# Codebase Map

Project : ElectricityForecasting
Updated : 2026-09-14 02:17:31 +0800
Commit  : 08c31cc

## Directory Roles

| Path | Responsibility | Key Files |
|---|---|---|
| `config/` | 采样、窗口、特征与指标常量 | `settings.py` |
| `data_provider/` | 16台风机数据读取、时间切分、窗口和 DataLoader | `common.py`, `no_future_weather_dataset.py` |
| `models/` | 任务包装与预测模型 | `noweather.py`, `backbone/dlinear.py`, `backbone/patchmlp.py` |
| `exp/` | 训练、验证、选模、测试与结果接线 | `trainer.py` |
| `utils/` | 损失、复现和测试指标 | `dbloss.py`, `metrics.py`, `reproducibility.py` |
| `observability/` | experiment-contract-v2 路径和 report/record writer | `exp_results_report.py` |
| `src/scripts/` | 正式实验 shell | `9.14 dlinear_all_features_{15min,10min,5min,1min}.sh`, `9.14 patchmlp_all_features_dbloss_1min.sh` |

## Entrypoints

| Task | Path | Symbol or Command | Role |
|---|---|---|---|
| Data generation | `generate_data.py` | `main` | 从原始 XLS 按 `--sample-seconds` 构建 processed NPY |
| Training and evaluation | `run.py` | `parse_args`, `train_and_test` | 设置窗口参数后进入统一训练链 |
| Full-feature campaign | `src/scripts/9.14 dlinear_all_features_15min.sh` | `bash "src/scripts/9.14 dlinear_all_features_15min.sh"` | 顺序运行10个预测尺度 |
| Multi-resolution campaigns | `src/scripts/9.14 dlinear_all_features_{10min,5min,1min}.sh` | `bash "src/scripts/9.14 dlinear_all_features_INTERVAL.sh"` | 分别重建对应粒度数据并顺序运行10个预测尺度 |
| PatchMLP power DBLoss campaign | `src/scripts/9.14 patchmlp_all_features_dbloss_1min.sh` | `bash "src/scripts/9.14 patchmlp_all_features_dbloss_1min.sh"` | 1分钟粒度下顺序运行10个59维PatchMLP设定，当前仅计划未启动 |
| Checkpoint visualization | `visualize_checkpoint_predictions.py` | `main` | 重建数据与模型并导出固定样本预测图 |

## Execution Chain

| Step | Stage | Path | Symbol | Input | Output |
|---:|---|---|---|---|---|
| 1 | Argument and config parsing | `run.py` | `parse_args` | shell 参数 | `args` 与 run directory |
| 2 | Dataset construction | `data_provider/common.py` | `Repository` | processed NPY、特征口径 | 标准化后的16台风机序列 |
| 3 | Window construction | `data_provider/no_future_weather_dataset.py` | `make_no_future_weather_loaders` | 历史点数、预测点数 | train/val/test loaders |
| 4 | Model construction | `models/noweather.py` | `NoFutureWeatherModel` | `[B,H,C]`、turbine id | `[B,P]`功率预测 |
| 5 | Training and validation | `exp/trainer.py` | `run_epoch`, `train_and_test` | train/val loaders | validation 最优 checkpoint |
| 6 | Final evaluation | `exp/trainer.py` | `train_and_test` | test loader、最优 checkpoint | kW 预测和测试指标 |
| 7 | Result persistence | `exp/trainer.py` | `write_power_result_contract` | final metrics | report、record、round checkpoint |

## Experiment Semantics

| Concern | Path | Symbol | Current Logic |
|---|---|---|---|
| Dataset source | `dataset/processed/` | `turbine_01.npy` ... `turbine_16.npy` | 当前运行脚本所重建粒度的16台风机共享缓存 |
| Sampling | `config/settings.py`, `generate_data.py`, `src/scripts/` | `SAMPLE_SECONDS`, `scan_member`, `write_member` | shell 显式传900/600/300/60秒，每次按粒度重建 processed |
| Train/valid/test split | `data_provider/common.py` | `Repository`, `belongs` | 按时间顺序70%/10%/20%，目标窗口不跨边界 |
| Full feature set | `data_provider/common.py` | `Repository(all_features=True)` | 56个无缺失字段中排除训练段恒定液压站压力，4个周期角度替换为sin/cos，共59维 |
| Normalization | `data_provider/common.py`, `models/noweather.py` | `Repository`, `RevIN` | 每台风机训练段 StandardScaler，再做窗口级 RevIN |
| Target transform | `data_provider/common.py` | `Repository` | 每台风机训练段功率均值和标准差 |
| Loss | `exp/trainer.py`, `utils/dbloss.py` | `run_epoch`, `DBLoss` | `MSE`为标准化功率MSE；`DBLoss`为功率MSE加`--dbloss-weight`倍功率DBLoss |
| Validation monitor | `exp/trainer.py` | `train_and_test` | validation MSE 最小，patience=10 |
| Final metrics | `utils/metrics.py` | `metric_bundle` | Acc30、MAE、MSE、RMSE、MAPE，反归一化到kW |
| Runs and seeds | `src/scripts/9.14 *.sh` | explicit commands | 每个粒度的每个历史/视界设定单seed 2026；PatchMLP DBLoss脚本含10个计划设定 |
| Model/result identity | `run.py` | `--model`, `--result-name` | model 选择实现，result name 唯一标识产物 |

## Artifact Integration

| Artifact | Code Path | Symbol or Save Site | Trigger | Output |
|---|---|---|---|---|
| Round record | `exp/trainer.py` | `write_power_result_contract` | final evaluation 完成 | record 中 `rounds[0]` |
| Checkpoint | `exp/trainer.py` | validation improvement | validation MSE 改善 | `checkpoints/{dataset}/{result_name}/round_1.pt` |
| Human report | `observability/exp_results_report.py` | `write_result_report` | 测试指标完成 | `results/reports/{dataset}/{result_name}.log` |
| Machine record | `observability/exp_results_report.py` | `write_result_record` | 测试指标完成 | `results/records/{dataset}/{result_name}.json` |
| Experiment memory | `agent_workspace/EXPERIMENT_MEMORY.md` | lifecycle update | plan/start/complete | 当前目标、活动状态和结果索引 |

## Edit Index

| Change | Primary Path and Symbol | Related Paths | Verification |
|---|---|---|---|
| Add or modify full-feature DLinear | `models/noweather.py:NoFutureWeatherModel` | `models/backbone/dlinear.py` | `tests.test_no_future_weather` |
| Add or modify full-feature PatchMLP | `models/noweather.py:NoFutureWeatherModel` | `models/backbone/patchmlp.py` | `tests.test_power_dbloss` |
| Change full feature processing | `data_provider/common.py:Repository` | `data_provider/no_future_weather_dataset.py` | finiteness and variance check |
| Change training behavior | `exp/trainer.py:train_and_test` | `run.py` | one-epoch smoke run |
| Change power DBLoss weight | `exp/trainer.py:run_epoch` | `run.py`, `utils/dbloss.py` | `tests.test_power_dbloss` |
| Add an experiment argument | `run.py:parse_args` | formal shell | `python -m py_compile`, `bash -n` |
| Add a sampling-resolution campaign | `src/scripts/{M.D} {experiment_tag}.sh` | `generate_data.py`, `run.py` | `bash -n` and expanded command count |
| Read or extend result output | `exp/trainer.py:write_power_result_contract` | `observability/exp_results_report.py` | contract checker |

## Verification

| Check | Command or Path | Last Status |
|---|---|---|
| Python syntax | `uv run python -m py_compile ...` | passed 2026-09-14 |
| Dataset semantics | `Repository(all_features=True)` finiteness/variance inspection | 59维，finite=True，zero_std=0 |
| Test suite | `env -u LD_LIBRARY_PATH uv run python -m unittest discover -s tests -p 'test_*.py'` | 20 tests passed |
| Training entry | one-epoch DLinearAllFeatures H4/P1 smoke | passed |
| Result contract | contract checker on all 10 report/record pairs | 10 passed 2026-09-14 |
| Multi-resolution scripts | `bash -n "src/scripts/9.14 dlinear_all_features_INTERVAL.sh"` | 10/5/1分钟脚本均通过，且各含10条训练命令 |
| PatchMLP power DBLoss | `env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='' uv run python -m unittest tests.test_power_dbloss` | 3 tests passed；59维前向、加权损失与单步梯度通过 |
| PatchMLP 1-minute script | `bash -n "src/scripts/9.14 patchmlp_all_features_dbloss_1min.sh"` | passed；10条训练命令均显式传DBLoss权重，未运行 |
