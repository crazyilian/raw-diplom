import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = 'debug-TCN-reconstruction'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "TCN reconstruction for PU.",
    "tags": ["tcn", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "tcn_ae",
    "params": {
        "activation": "relu",
        "dropout": 0.0,

        "hidden_channels": 32, 
        "latent_channels": 8,
        "num_blocks": 5,
        "kernel_size": 3,
        "separable": None, # true, false
        "norm": None, # batch, layer, none

        "dilations": None # None for 2^i - classic
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 1e-3,
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
    "epochs": 210,
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
for separable in [False]:
    for norm in ["none"]:
        for lr in [1e-4, 1e-3, 1e-2]:
            for weight_decay in [1e-4, 1e-3]:
                for epochs in [110, 210]:
                    cur_id += 1
                    for seed in seeds:
                        config = copy.deepcopy(base_config)
                        
                        name = f"{cur_id:0>3}-s{seed}-sep{int(separable)}-norm_{norm}-lr{lr:e}-wd{weight_decay:e}-ep{epochs}"
                        config["run"] = {
                            **config["run"],
                            "name": name,
                            "dir": str(Path(config["run"]["dir"]) / name),
                            "seed": seed,
                        }
                        config["model"]["params"]["separable"] = separable
                        config["model"]["params"]["norm"] = norm
                        config["optimizer"]["params"]["lr"] = lr
                        config["optimizer"]["params"]["weight_decay"] = weight_decay
                        config["trainer"]["epochs"] = epochs

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
