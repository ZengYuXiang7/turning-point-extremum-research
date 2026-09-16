import json
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT


def print_saved_artifacts(checkpoint_path: Path, prediction_pdf_path: Path, split: str, selection_seed: int, selected_turbine_id: np.ndarray,) -> None:
    # 汇报 single 任务本次生成的固定窗口图件。
    display_checkpoint_path = checkpoint_path.resolve()
    if display_checkpoint_path.is_relative_to(PROJECT_ROOT):
        display_checkpoint_path = display_checkpoint_path.relative_to(PROJECT_ROOT)
    display_pdf_path = prediction_pdf_path.relative_to(PROJECT_ROOT).as_posix()
    print(json.dumps({ "event": "single_visualization_saved", "checkpoint": display_checkpoint_path.as_posix(), "pdf": display_pdf_path, "split": split, "selection_seed": selection_seed, "turbines": selected_turbine_id.tolist(), }, ensure_ascii=False,), flush=True,)
