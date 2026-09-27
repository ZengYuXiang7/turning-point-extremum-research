from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "models" / "pretrained" / "TimeMoE-50M"
REVISION = "446753ee48ff3726d0606a81d0092d54acee995e"


if __name__ == "__main__":
    snapshot_download(
        repo_id="Maple728/TimeMoE-50M",
        revision=REVISION,
        local_dir=TARGET,
    )
    print(f"TimeMoE-50M downloaded to {TARGET}")
