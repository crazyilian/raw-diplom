import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = 'GRU-reconstruction'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "GRU reconstruction for PU.",
    "tags": ["gru", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "gru_ae",
    "params": {
        "dropout": 0.0,

        "hidden_size": None, 
        "latent_size": None,
        "num_layers": None,
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 3e-3,
        "weight_decay": 1e-4,
    },
}
base_config["scheduler"] =  {
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
    "tags": ["tcn", "reconstruction", "pu"],
}

##### hyperparameters prepare #####

seeds = [42, 43, 44]

sweep_configs = []
cur_id = 0
for batch_size in [1024, 512]:
    for epochs in [110, 60, 210]:
        for lr in [3e-4, 1e-3, 3e-2]:
            for wd in [1e-4]:
                cur_id += 1
                for seed in seeds:
                    config = copy.deepcopy(base_config)
                    name = f"{cur_id:0>3}-s{seed}-batch{batch_size}-lr{lr:.2e}-wd{wd:.2e}-ep{epochs}"
                    config["run"] = {
                        **config["run"],
                        "name": name,
                        "dir": str(Path(config["run"]["dir"]) / name),
                        "seed": seed,
                    }
                    config["optimizer"]["params"]["lr"] = lr
                    config["optimizer"]["params"]["weight_decay"] = wd
                    config["trainer"]["epochs"] = epochs
                    config["dataset"]["loader"]["batch_size"] = batch_size

                    config["model"]["params"]["hidden_size"] = 96
                    config["model"]["params"]["latent_size"] = 48
                    config["model"]["params"]["num_layers"] = 1

                    sweep_configs.append(config)

print("total planned runs:", len(sweep_configs))

##### slice hyperparameters #####

import sys
args = list(map(lambda x: None if x == "None" else int(x), sys.argv[1:]))
if not args: args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned runs after slice:", len(sweep_configs))

##### run experiments #####

run_experiments(sweep_configs, skip_existing=True)
