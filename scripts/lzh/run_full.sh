#!/bin/bash
set -e
# 兼容入口：等价于 run_full_10s.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT/run_full_10s.sh"
