import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)


base_name = "PatchTST-forecasting"
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "PatchTST-style Transformer forecasting for PU.",
    "tags": ["transformer", "patchtst", "forecasting", "pu"],
}
base_config["task"] = {
    "type": "forecasting",
}
base_config["model"] = {
    "name": "patch_tst",
    "params": {
        "patch_len": None,
        "patch_stride": None,
        "d_model": None,
        "nhead": 4,
        "num_layers": None,
        "dim_feedforward": None,
        "dropout": 0.0,
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
}
base_config["scheduler"] = {
    "name": "warmup_cosine",
    "params": {
        "warmup_epochs": 10,
        "min_lr": 1e-6,
        "start_factor": 1e-4,
    },
}
base_config["trainer"] = {
    "epochs": 110,
    "eval_every_n_epochs": 10,
    "device": "cuda",
    "amp": False,
    "clip_grad_norm": 1.0,
    "checkpoint": {
        "metric": "val/roc_auc",
        "mode": "max",
        "min_delta": 0.0,
    },
    "early_stopping": {
        "enabled": False,
        "metric": "val/roc_auc",
        "mode": "max",
        "patience": 12,
        "min_delta": 0.0,
    },
}
base_config["wandb"] = {
    "enabled": False,
    "project": "project-name",
    "entity": "",
    "mode": "offline",
    "group": base_name,
    "tags": base_config["run"]["tags"],
}


seeds = [42, 43, 44]
horizons = [8, 16, 24, 32]

search_space = [
    {"patch_len": 12, "patch_stride": 12, "d_model": 32, "num_layers": 1, "dim_feedforward": 64},
    {"patch_len": 12, "patch_stride": 12, "d_model": 64, "num_layers": 2, "dim_feedforward": 128},
    {"patch_len": 12, "patch_stride": 6, "d_model": 32, "num_layers": 2, "dim_feedforward": 64},
    {"patch_len": 12, "patch_stride": 6, "d_model": 64, "num_layers": 2, "dim_feedforward": 128},
    {"patch_len": 24, "patch_stride": 12, "d_model": 64, "num_layers": 2, "dim_feedforward": 128},
]

sweep_configs = []
for horizon_size in [16, 32, 64]:
for config_id, model_params in enumerate(search_space, start=1):
    for horizon_size in horizons:
        for seed in seeds:
            config = copy.deepcopy(base_config)
            name = (
                f"{config_id:03d}-s{seed}-hor{horizon_size}"
                f"-p{model_params['patch_len']}-s{model_params['patch_stride']}"
                f"-d{model_params['d_model']}-lay{model_params['num_layers']}"
            )
            config["run"] = {
                **config["run"],
                "name": name,
                "dir": str(Path(config["run"]["dir"]) / name),
                "seed": seed,
            }
            config["dataset"]["params"]["horizon_size"] = horizon_size
            config["model"]["params"] = dict(model_params, nhead=4, dropout=0.1, activation="gelu", norm_first=True)
            sweep_configs.append(config)

print("total planned runs:", len(sweep_configs))

import sys

args = [None if value == "None" else int(value) for value in sys.argv[1:]]
if not args:
    args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned runs after slice:", len(sweep_configs))

run_experiments(sweep_configs, skip_existing=True)
