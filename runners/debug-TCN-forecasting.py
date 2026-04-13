import copy
from pathlib import Path

from industrial_ad import DEFAULT_EXPERIMENT_CONFIG, clone_config, run_experiments


# BASE_NAME = "TCN-forecasting"
# SEEDS = [42, 43, 44]
# HORIZON_SIZES = [8, 16, 32]
# SEARCH_SPACE = [
#     {"hidden_channels": 32, "num_blocks": 4, "kernel_size": 5, "separable": True, "dropout": 0.0},
#     {"hidden_channels": 32, "num_blocks": 4, "kernel_size": 5, "separable": True, "dropout": 0.1},
#     {"hidden_channels": 48, "num_blocks": 5, "kernel_size": 3, "separable": True, "dropout": 0.0},
#     {"hidden_channels": 48, "num_blocks": 5, "kernel_size": 3, "separable": False, "dropout": 0.0},
#     {"hidden_channels": 64, "num_blocks": 4, "kernel_size": 7, "separable": True, "dropout": 0.0},
#     {"hidden_channels": 64, "num_blocks": 5, "kernel_size": 5, "separable": True, "dropout": 0.1},
# ]


base_name = 'debug-TCN-forecasting'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "TCN forecasting for PU.",
    "tags": ["tcn", "forecasting", "pu"],
}
base_config["task"] = {
    "type": "forecasting",
}
base_config["model"] = {
    "name": "tcn_forecaster",
    "params": {
        "activation": "relu",
        "hidden_channels": None,
        "num_blocks": None,
        "kernel_size": None,
        "dropout": 0.0,
        "separable": None, # True, False
        "norm": "layer", # group, layer
        "dilations": None, # auto
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
    "tags": base_config["run"]["tags"],
}


##### hyperparameters prepare #####

seeds = [42, 43, 44]

sweep_configs = []
cur_id = 0


for horizon_size in [16, 32]:
    for lr in [3e-3, 1e-3]:
        for norm in ["layer", "group"]:
            for separable in [True]:
                for hidden_channels in [32]:
                    for num_blocks in [4]:
                        for kernel_size in [3]:
                            
                            dilations = [2**index for index in range(num_blocks)]
                            receptive_field = 1 + 4 * (kernel_size - 1) * sum(dilations)
                            if receptive_field > 250 or receptive_field < 70:
                                continue

                            cur_id += 1
                            for seed in seeds:
                                config = copy.deepcopy(base_config)
                                
                                name = f"{cur_id:0>3}-s{seed}-h{horizon_size}-lr{lr}-norm{norm}"
                                config["run"] = {
                                    **config["run"],
                                    "name": name,
                                    "dir": str(Path(config["run"]["dir"]) / name),
                                    "seed": seed,
                                }

                                config["dataset"]["params"]["horizon_size"] = horizon_size
                                config["model"]["params"]["separable"] = separable
                                config["model"]["params"]["hidden_channels"] = hidden_channels
                                config["model"]["params"]["num_blocks"] = num_blocks
                                config["model"]["params"]["kernel_size"] = kernel_size

                                config["optimizer"]["params"]["lr"] = lr
                                config["model"]["params"]["norm"] = norm

                                sweep_configs.append(config)

print("total planned runs:", len(sweep_configs))

##### slice hyperparameters #####

import sys
args = list(map(lambda x: None if x == "None" else int(x), sys.argv[1:]))
if not args: args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned runs after slice:", len(sweep_configs))

##### run experiments #####

run_experiments(sweep_configs, skip_existing=True, dry_run=False)
