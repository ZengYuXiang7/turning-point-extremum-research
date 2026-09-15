from __future__ import annotations

import os
from pathlib import Path

from config import PROJECT_ROOT
from tasks.multi_turbine.configuration import parse_args


# 模型元数据和动态模块缓存统一留在项目目录。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),)


def main():
    args = parse_args()

    # 联合面板训练与测试只进入多风机任务链。
    from tasks.multi_turbine.task import train_and_test

    train_and_test(args)

    # 训练结束后用同一个联合窗口生成十台风机预测图。
    if args.mode == "train":
        from tasks.multi_turbine.visualization import visualize_checkpoint_predictions

        visualize_checkpoint_predictions(Path(args.run_dir), args.visualization_split, args.visualization_seed, args.visualization_device,)


if __name__ == "__main__":
    main()
