from __future__ import annotations

from typing import Any

from industrial_ad.utils import clone_config


DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "",
        "dir": "",
        "seed": 42,
        "notes": "",
        "tags": [],
    },
    "task": {
        "type": "",
    },
    "dataset": {
        "name": "pu",
        "params": {
            "root": "./data/PU",
            "train_patterns": ["K0*/N15_*/"],
            "val_patterns": ["K0*/N15_*/", "KA*/N15_*_1/"],
            "test_patterns": ["K0*/N15_*/", "KA*/N15_*_1/"],
            "window_size": 120,
            "window_overlap": 97,
            "horizon_size": 0,
            "files_per_chunk": 1,
            "bytes_cache_limit": 3e9,
            "feature_pipeline": {
                "name": "classic_27d_v1",
                "params": {
                    "aggregation_window": 64,
                    "aggregation_step": 16,
                    "sampling_rate": 40000,
                    "drop_edges": 640,
                    "raw_channel_indices": [0, 1, 2],
                    "use_preset_scaler": True,
                },
            },
        },
        "loader": {
            "batch_size": 1024,
            "num_workers": 8,
            "persistent_workers": True,
            "pin_memory": True,
        },
    },
    "debug": {
        "dataset": {
            "train_file_limit": None,
            "val_file_limit": None,
            "test_file_limit": None,
        },
        "trainer": {
            "max_train_batches": None,
            "max_eval_batches": None,
        },
    },
    "model": {
        "name": "",
        "params": {},
    },
    "loss": {
        "name": "mse",
        "params": {},
    },
    "optimizer": {
        "name": "",
        "params": {},
    },
    "scheduler": {
        "name": "warmup_cosine",
        "params": {
            "warmup_epochs": 10,
            "min_lr": 1e-6,
            "start_factor": 1e-4,
        },
    },
    "trainer": {
        "epochs": 0,
        "eval_every_n_epochs": 1,
        "device": "",
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
    },
    "scoring": {
        "error_reducer": {
            "name": "mean_abs",
            "params": {
                "reduce_dims": [1],
            },
        },
        "score_estimator": {
            "name": "mahalanobis",
            "params": {
                "eps": 1e-5,
            },
        },
        "threshold": {
            "name": "best_f1",
            "params": {},
        },
    },
    "wandb": {
        "enabled": False,
        "project": "",
        "entity": "",
        "mode": "offline",
        "group": "",
        "tags": [],
    },
    "benchmark": {
        "enabled": False,
        "device": "cpu",
        "num_threads": 1,
        "warmup_runs": 50,
        "num_runs": 2000,
        "profile_memory": True,
    },
}


DEFAULT_QUANTIZATION_CONFIG: dict[str, Any] = {
    "run": {
        "name": "",
        "dir": "",
        "seed": 42,
        "notes": "",
        "tags": [],
    },
    "source": {
        "run_dir": "",
        "checkpoint": "best",
    },
    "evaluation": {
        "metric": "val/roc_auc",
        "max_batches": None,
    },
    "quantization": {
        "backend": "fbgemm",
        "dtype": "qint8",
        "calibration_batches": None,
    },
    "benchmark": clone_config(DEFAULT_EXPERIMENT_CONFIG["benchmark"]),
}


def is_pca_config(config: dict[str, Any]) -> bool:
    return str(config["model"]["name"]).lower() == "pca"


def make_default_config() -> dict[str, Any]:
    """Return a writable copy of the project config template."""
    return clone_config(DEFAULT_EXPERIMENT_CONFIG)


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate the fields that must be explicitly set before a run starts."""
    is_pca_model = is_pca_config(config)
    required_string_fields = {
        "run.name": config["run"]["name"],
        "run.dir": config["run"]["dir"],
        "task.type": config["task"]["type"],
        "dataset.name": config["dataset"]["name"],
        "model.name": config["model"]["name"],
        "trainer.device": config["trainer"]["device"],
    }
    if not is_pca_model:
        required_string_fields["optimizer.name"] = config["optimizer"]["name"]
    for field_name, value in required_string_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    if int(config["trainer"]["epochs"]) <= 0:
        raise ValueError("trainer.epochs must be positive.")
    if int(config["trainer"]["eval_every_n_epochs"]) <= 0:
        raise ValueError("trainer.eval_every_n_epochs must be positive.")
    if int(config["dataset"]["loader"]["batch_size"]) <= 0:
        raise ValueError("dataset.loader.batch_size must be positive.")
    if int(config["dataset"]["loader"]["num_workers"]) == 0 and bool(config["dataset"]["loader"]["persistent_workers"]):
        raise ValueError("dataset.loader.persistent_workers requires dataset.loader.num_workers > 0.")

    trainer_device = config["trainer"]["device"]
    if bool(config["trainer"]["amp"]) and trainer_device.split(":", 1)[0].lower() != "cuda":
        raise ValueError("trainer.amp can only be enabled for CUDA devices.")

    task_type = config["task"]["type"].lower()
    if task_type not in {"reconstruction", "forecasting"}:
        raise ValueError(f"Unsupported task type: {config['task']['type']}")
    if task_type == "forecasting" and int(config["dataset"]["params"]["horizon_size"]) <= 0:
        raise ValueError("Forecasting experiments require dataset.params.horizon_size > 0.")


def make_default_quantization_config() -> dict[str, Any]:
    """Return a writable copy of the quantization config template."""
    return clone_config(DEFAULT_QUANTIZATION_CONFIG)


def validate_quantization_config(config: dict[str, Any]) -> None:
    """Validate the fields that must be explicitly set before quantization starts."""
    required_string_fields = {
        "run.name": config["run"]["name"],
        "run.dir": config["run"]["dir"],
        "source.run_dir": config["source"]["run_dir"],
        "source.checkpoint": config["source"]["checkpoint"],
        "evaluation.metric": config["evaluation"]["metric"],
        "quantization.backend": config["quantization"]["backend"],
        "quantization.dtype": config["quantization"]["dtype"],
    }
    for field_name, value in required_string_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    if config["quantization"]["calibration_batches"] is not None and int(config["quantization"]["calibration_batches"]) <= 0:
        raise ValueError("quantization.calibration_batches must be positive.")

    max_batches = config["evaluation"].get("max_batches")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("evaluation.max_batches must be positive or None.")

    if bool(config["benchmark"]["enabled"]):
        for field_name in ["num_threads", "warmup_runs", "num_runs"]:
            if int(config["benchmark"][field_name]) <= 0:
                raise ValueError(f"benchmark.{field_name} must be positive.")
