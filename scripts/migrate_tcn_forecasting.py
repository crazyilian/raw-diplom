import json
import shutil
from pathlib import Path

import torch
import tqdm


SOURCE_FAMILY = Path("runs/TCN-forecasting-bug")
TARGET_FAMILY = Path("runs/TCN-forecasting")


def conv_weight(weight: torch.Tensor) -> torch.Tensor:
    channels, length1, length2 = weight.shape
    return weight.permute(0, 2, 1).contiguous().view(channels * length2, 1, length1)


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        if key.endswith("forecast_head.weight"):
            key = key.replace("forecast_head.weight", "forecast_head.projection.weight")
            value = conv_weight(value)
        elif key.endswith("forecast_head.bias"):
            key = key.replace("forecast_head.bias", "forecast_head.projection.bias")
            value = value.reshape(-1)
        converted[key] = value
    return converted


def update_json(path: Path, run_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "run" in data:
        data["run"]["dir"] = str(run_dir)
    if "wandb" in data:
        data["wandb"]["group"] = TARGET_FAMILY.name
    if "family" in data:
        data["family"] = TARGET_FAMILY.name
    if "run_dir" in data:
        data["run_dir"] = str(run_dir)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def migrate_run(source_run_dir: Path) -> None:
    target_run_dir = TARGET_FAMILY / source_run_dir.name
    shutil.copytree(source_run_dir, target_run_dir)
    for checkpoint_path in (target_run_dir / "checkpoints").glob("*.pt"):
        torch.save(convert_state_dict(torch.load(checkpoint_path, map_location="cpu")), checkpoint_path)
    update_json(target_run_dir / "config.json", target_run_dir)
    update_json(target_run_dir / "summary.json", target_run_dir)


run_dirs = sorted(path for path in SOURCE_FAMILY.iterdir() if path.is_dir())
for run_dir in tqdm.tqdm(run_dirs):
    migrate_run(run_dir)
print(f"migrated {len(run_dirs)} TCN forecasting runs")
