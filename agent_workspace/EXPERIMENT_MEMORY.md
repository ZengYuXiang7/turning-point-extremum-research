# Experiment Memory

Project  : ElectricityForecasting
Updated  : 2026-09-14 02:17:31 +0800
Commit   : 08c31cc
Contract : experiment-contract-v2

<!-- experiment-memory-contract:start -->
## Reading Order

1. Read `agent_workspace/CODEBASE_MAP.md` for code locations and the execution chain.
2. Read this file for the current objective, experiment status, and result paths.
3. Read `results/records/{dataset}/{result_name}.json` for exact experiment values and checkpoint paths.
4. Load `checkpoints/{dataset}/{result_name}/round_{round}.pt` when model state is needed.
5. Read `results/reports/{dataset}/{result_name}.log` for the human-facing report.
6. Read the exact summary path in Result Index for cross-setting conclusions.

## Result Contract

```text
results/reports/{dataset}/{result_name}.log
results/records/{dataset}/{result_name}.json
checkpoints/{dataset}/{result_name}/round_{round}.pt
```

## Agent Workspace

```text
agent_workspace/
├── CODEBASE_MAP.md
├── EXPERIMENT_MEMORY.md
└── summaries/{M.D} {experiment_tag}.md
src/
└── scripts/{M.D} {experiment_tag}.sh
```

`M.D` uses the local creation month and day, such as `7.23`.
`src/scripts/` is flat and contains only `.sh` files; run modes do not create subdirectories.
Experiment scripts use `src/scripts/{M.D} {experiment_tag}.sh`.
<!-- experiment-memory-contract:end -->

## Current Objective

Completed the ten 15-minute DLinearAllFeatures settings and prepared a separate one-minute PatchMLPAllFeatures power-DBLoss campaign without launching it.

## Active Experiments

| Experiment | Purpose | Dataset | Model | Result(s) | Stage | Host | PID/Job | Status | Script | Temp Log |
|---|---|---|---|---|---|---|---|---|---|---|
| dlinear_all_features_10min | 10分钟粒度与预测时长扫描 | GuangningWindPower10min | DLinearAllFeatures | 10 settings in Result Index | planned | - | - | planned | `src/scripts/9.14 dlinear_all_features_10min.sh` | - |
| dlinear_all_features_5min | 5分钟粒度与预测时长扫描 | GuangningWindPower5min | DLinearAllFeatures | 10 settings in Result Index | planned | - | - | planned | `src/scripts/9.14 dlinear_all_features_5min.sh` | - |
| dlinear_all_features_1min | 1分钟粒度与预测时长扫描 | GuangningWindPower1min | DLinearAllFeatures | 10 settings in Result Index | planned | - | - | planned | `src/scripts/9.14 dlinear_all_features_1min.sh` | - |
| patchmlp_all_features_dbloss_1min | 59维历史特征MLP交互与功率DBLoss | GuangningWindPower1min | PatchMLPAllFeatures | 10 settings in Result Index | planned | - | - | planned | `src/scripts/9.14 patchmlp_all_features_dbloss_1min.sh` | - |
| dlinear_all_features_15min | 全部可训练历史特征的多尺度功率预测 | GuangningWindPower15min | DLinearAllFeatures | 10 settings | completed | rtx4090 | 788351 | completed | `src/scripts/9.14 dlinear_all_features_15min.sh` | unified session 99575 |

## Result Index

| Dataset | Result | Model | Purpose | Rounds | Status | Report | Record | Checkpoints | Summary |
|---|---|---|---|---:|---|---|---|---|---|
| GuangningWindPower15min | dlinear_all_features_h4_p1_s1_seed2026 | DLinearAllFeatures | 1h→15min | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h4_p1_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h4_p1_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h4_p1_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h4_p2_s1_seed2026 | DLinearAllFeatures | 1h→30min | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h4_p2_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h4_p2_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h4_p2_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h4_p3_s1_seed2026 | DLinearAllFeatures | 1h→45min | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h4_p3_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h4_p3_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h4_p3_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h4_p4_s1_seed2026 | DLinearAllFeatures | 1h→1h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h4_p4_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h4_p4_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h4_p4_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h8_p8_s1_seed2026 | DLinearAllFeatures | 2h→2h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h8_p8_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h8_p8_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h8_p8_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h12_p12_s1_seed2026 | DLinearAllFeatures | 3h→3h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h12_p12_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h12_p12_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h12_p12_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h16_p16_s1_seed2026 | DLinearAllFeatures | 4h→4h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h16_p16_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h16_p16_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h16_p16_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h32_p32_s1_seed2026 | DLinearAllFeatures | 8h→8h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h32_p32_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h32_p32_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h32_p32_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h48_p48_s1_seed2026 | DLinearAllFeatures | 12h→12h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h48_p48_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h48_p48_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h48_p48_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower15min | dlinear_all_features_h96_p96_s1_seed2026 | DLinearAllFeatures | 24h→24h | 1 | completed | `results/reports/GuangningWindPower15min/dlinear_all_features_h96_p96_s1_seed2026.log` | `results/records/GuangningWindPower15min/dlinear_all_features_h96_p96_s1_seed2026.json` | `checkpoints/GuangningWindPower15min/dlinear_all_features_h96_p96_s1_seed2026` | `agent_workspace/summaries/9.14 dlinear all features.md` |
| GuangningWindPower10min | dlinear_all_features_h6_p1_s1_seed2026 | DLinearAllFeatures | 60min历史预测10min | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h6_p1_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h6_p1_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h6_p1_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h6_p2_s1_seed2026 | DLinearAllFeatures | 60min历史预测20min | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h6_p2_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h6_p2_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h6_p2_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h6_p3_s1_seed2026 | DLinearAllFeatures | 60min历史预测30min | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h6_p3_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h6_p3_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h6_p3_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h6_p6_s1_seed2026 | DLinearAllFeatures | 60min历史预测60min | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h6_p6_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h6_p6_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h6_p6_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h12_p12_s1_seed2026 | DLinearAllFeatures | 2h历史预测2h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h12_p12_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h12_p12_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h12_p12_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h18_p18_s1_seed2026 | DLinearAllFeatures | 3h历史预测3h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h18_p18_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h18_p18_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h18_p18_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h24_p24_s1_seed2026 | DLinearAllFeatures | 4h历史预测4h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h24_p24_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h24_p24_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h24_p24_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h48_p48_s1_seed2026 | DLinearAllFeatures | 8h历史预测8h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h48_p48_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h48_p48_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h48_p48_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h72_p72_s1_seed2026 | DLinearAllFeatures | 12h历史预测12h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h72_p72_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h72_p72_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h72_p72_s1_seed2026/round_1.pt` | - |
| GuangningWindPower10min | dlinear_all_features_h144_p144_s1_seed2026 | DLinearAllFeatures | 24h历史预测24h | 1 | planned | `results/reports/GuangningWindPower10min/dlinear_all_features_h144_p144_s1_seed2026.log` | `results/records/GuangningWindPower10min/dlinear_all_features_h144_p144_s1_seed2026.json` | `checkpoints/GuangningWindPower10min/dlinear_all_features_h144_p144_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h12_p1_s1_seed2026 | DLinearAllFeatures | 60min历史预测5min | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h12_p1_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h12_p1_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h12_p1_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h12_p2_s1_seed2026 | DLinearAllFeatures | 60min历史预测10min | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h12_p2_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h12_p2_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h12_p2_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h12_p3_s1_seed2026 | DLinearAllFeatures | 60min历史预测15min | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h12_p3_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h12_p3_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h12_p3_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h12_p12_s1_seed2026 | DLinearAllFeatures | 60min历史预测60min | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h12_p12_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h12_p12_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h12_p12_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h24_p24_s1_seed2026 | DLinearAllFeatures | 2h历史预测2h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h24_p24_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h24_p24_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h24_p24_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h36_p36_s1_seed2026 | DLinearAllFeatures | 3h历史预测3h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h36_p36_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h36_p36_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h36_p36_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h48_p48_s1_seed2026 | DLinearAllFeatures | 4h历史预测4h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h48_p48_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h48_p48_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h48_p48_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h96_p96_s1_seed2026 | DLinearAllFeatures | 8h历史预测8h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h96_p96_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h96_p96_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h96_p96_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h144_p144_s1_seed2026 | DLinearAllFeatures | 12h历史预测12h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h144_p144_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h144_p144_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h144_p144_s1_seed2026/round_1.pt` | - |
| GuangningWindPower5min | dlinear_all_features_h288_p288_s1_seed2026 | DLinearAllFeatures | 24h历史预测24h | 1 | planned | `results/reports/GuangningWindPower5min/dlinear_all_features_h288_p288_s1_seed2026.log` | `results/records/GuangningWindPower5min/dlinear_all_features_h288_p288_s1_seed2026.json` | `checkpoints/GuangningWindPower5min/dlinear_all_features_h288_p288_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h60_p1_s1_seed2026 | DLinearAllFeatures | 60min历史预测1min | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h60_p1_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h60_p1_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h60_p1_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h60_p2_s1_seed2026 | DLinearAllFeatures | 60min历史预测2min | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h60_p2_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h60_p2_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h60_p2_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h60_p3_s1_seed2026 | DLinearAllFeatures | 60min历史预测3min | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h60_p3_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h60_p3_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h60_p3_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h60_p60_s1_seed2026 | DLinearAllFeatures | 60min历史预测60min | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h60_p60_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h60_p60_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h60_p60_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h120_p120_s1_seed2026 | DLinearAllFeatures | 2h历史预测2h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h120_p120_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h120_p120_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h120_p120_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h180_p180_s1_seed2026 | DLinearAllFeatures | 3h历史预测3h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h180_p180_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h180_p180_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h180_p180_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h240_p240_s1_seed2026 | DLinearAllFeatures | 4h历史预测4h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h240_p240_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h240_p240_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h240_p240_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h480_p480_s1_seed2026 | DLinearAllFeatures | 8h历史预测8h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h480_p480_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h480_p480_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h480_p480_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h720_p720_s1_seed2026 | DLinearAllFeatures | 12h历史预测12h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h720_p720_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h720_p720_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h720_p720_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | dlinear_all_features_h1440_p1440_s1_seed2026 | DLinearAllFeatures | 24h历史预测24h | 1 | planned | `results/reports/GuangningWindPower1min/dlinear_all_features_h1440_p1440_s1_seed2026.log` | `results/records/GuangningWindPower1min/dlinear_all_features_h1440_p1440_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/dlinear_all_features_h1440_p1440_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h60_p1_s1_seed2026 | PatchMLPAllFeatures | 60min历史预测1min，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p1_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p1_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p1_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h60_p2_s1_seed2026 | PatchMLPAllFeatures | 60min历史预测2min，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p2_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p2_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p2_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h60_p3_s1_seed2026 | PatchMLPAllFeatures | 60min历史预测3min，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p3_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p3_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p3_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h60_p60_s1_seed2026 | PatchMLPAllFeatures | 60min历史预测60min，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p60_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p60_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h60_p60_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h120_p120_s1_seed2026 | PatchMLPAllFeatures | 2h历史预测2h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h120_p120_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h120_p120_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h120_p120_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h180_p180_s1_seed2026 | PatchMLPAllFeatures | 3h历史预测3h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h180_p180_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h180_p180_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h180_p180_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h240_p240_s1_seed2026 | PatchMLPAllFeatures | 4h历史预测4h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h240_p240_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h240_p240_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h240_p240_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h480_p480_s1_seed2026 | PatchMLPAllFeatures | 8h历史预测8h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h480_p480_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h480_p480_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h480_p480_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h720_p720_s1_seed2026 | PatchMLPAllFeatures | 12h历史预测12h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h720_p720_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h720_p720_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h720_p720_s1_seed2026/round_1.pt` | - |
| GuangningWindPower1min | patchmlp_all_features_dbloss_w0p5_h1440_p1440_s1_seed2026 | PatchMLPAllFeatures | 24h历史预测24h，功率DBLoss权重0.5 | 1 | planned | `results/reports/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h1440_p1440_s1_seed2026.log` | `results/records/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h1440_p1440_s1_seed2026.json` | `checkpoints/GuangningWindPower1min/patchmlp_all_features_dbloss_w0p5_h1440_p1440_s1_seed2026/round_1.pt` | - |

## Confirmed Decisions

- Primary metric: validation scaled MSE for checkpoint selection; report test Acc30, MAE, MSE, RMSE and MAPE.
- Sampling: 900 seconds, prediction origin stride one point.
- Split: chronological 70%/10%/20%; unchanged from the current repository.
- Normalization: per-turbine train StandardScaler followed by window RevIN.
- Features: all 56 no-missing processed fields except constant `传动链-液压站压力`; four circular angles are replaced by sin/cos, yielding 59 model channels.
- Seeds: one formal round with seed 2026 per history/horizon setting.
- Checkpoint selection: minimum validation scaled MSE, patience 10.
- Significance: unavailable for one seed; no significance claim.
- Result naming: `dlinear_all_features_h{history}_p{horizon}_s1_seed2026`.
- Planned sampling extensions: 600/300/60 seconds, each with prediction origin stride one point; each launcher rebuilds the shared `dataset/processed`, so resolution campaigns must run sequentially.
- PatchMLPAllFeatures uses all 59 historical channels and the existing cross-variable MLP; no Transformer is introduced.
- Power DBLoss: `--loss DBLoss` optimizes `MSE_power + --dbloss-weight * DBLoss_power`; the shell fixes weight 0.5 while the CLI keeps it configurable, and checkpoint selection remains validation scaled MSE.
- PatchMLP DBLoss result naming: `patchmlp_all_features_dbloss_w0p5_h{history}_p{horizon}_s1_seed2026`.

## Latest Conclusions

- One-epoch H4/P1 smoke run completed with finite 59-channel data and end-to-end test evaluation.
- All ten formal settings completed in approximately 3 minutes 52 seconds.
- Acc30 improved by 1.74 to 3.20 percentage points for 15-minute through 2-hour horizons.
- Acc30 decreased for every 3-hour through 24-hour setting; the simple all-feature projection is not a universal replacement for the baseline.
- Exact comparison: `agent_workspace/summaries/9.14 dlinear all features.md`.

## Next Actions

1. Run multiple seeds before making a statistical model-selection claim.
2. Run the planned DLinearAllFeatures 10/5/1-minute scripts one at a time and summarize their JSON records.
3. Keep `src/scripts/9.14 patchmlp_all_features_dbloss_1min.sh` unstarted until the user requests execution.
