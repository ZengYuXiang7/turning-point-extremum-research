from __future__ import annotations

import os
from pathlib import Path

from config import PROJECT_ROOT
from tasks.single.configuration import parse_args


# 模型元数据和动态模块缓存统一留在项目目录。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),)


def main():
    args = parse_args()

    # 常规单风机及天气任务只进入常规任务链。
    from tasks.single.task import train_and_test

    train_and_test(args)

    # 训练结束后生成固定随机样本的预测图。
    if (
        args.mode == "train"
        and args.scenario != "ProvidedFutureWeather"
        and args.visualize_after_train
    ):
        from tasks.single.visualization import visualize_checkpoint_predictions

        visualize_checkpoint_predictions(Path(args.run_dir), args.visualization_split, args.visualization_seed, args.visualization_device,)


if __name__ == "__main__":
    main()
