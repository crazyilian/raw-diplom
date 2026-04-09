import copy
from pathlib import Path


from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = 'debug-Transformer-reconstruction'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "Transformer reconstruction for PU.",
    "tags": ["transformer", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "transformer_ae",
    "params": {
        "d_model": None,
        "nhead": 4,
        "num_layers": None,
        "latent_dim": None,
        "dim_feedforward": None,
        "dropout": 0.0,
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 6e-4,
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

for num_layers in [2]:
    for d_model in [48]:
        for latent_dim in [12]:
            for dim_feedforward in [96]:
                for lr in [6e-4, 3e-4, 1e-4]:
                    for epochs in [60, 110, 210]:

                        cur_id += 1
                        for seed in seeds:
                            config = copy.deepcopy(base_config)
                            name = f"{cur_id:0>3}-s{seed}-lay{num_layers}-mod{d_model}-lat{latent_dim}-ff{dim_feedforward}-lr{lr:.2}-ep{epochs}"
                            config["run"] = {
                                **config["run"],
                                "name": name,
                                "dir": str(Path(config["run"]["dir"]) / name),
                                "seed": seed,
                            }
                            config["model"]["params"]["num_layers"] = num_layers
                            config["model"]["params"]["d_model"] = d_model
                            config["model"]["params"]["latent_dim"] = latent_dim
                            config["model"]["params"]["dim_feedforward"] = dim_feedforward

                            config["optimizer"]["params"]["lr"] = lr
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
