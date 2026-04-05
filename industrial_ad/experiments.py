from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import tqdm
from industrial_ad.datasets.PU.dataset import build_pu_dataloaders
from industrial_ad.config import validate_experiment_config
from industrial_ad.models import build_model
from industrial_ad.scoring import (
    AnomalyDetectorWrapper,
    build_error_reducer,
    build_score_estimator,
)
from industrial_ad.training import build_criterion, build_optimizer, build_scheduler, train_anomaly_detector
from industrial_ad.utils import clone_config, count_parameters, dump_json, load_json, parameter_size_bytes, seed_everything


def _build_detector(config: dict[str, Any], sample_input: torch.Tensor, sample_target: torch.Tensor):
    """Build the predictor model and wrap it with anomaly-scoring logic."""
    input_shape = tuple(int(value) for value in sample_input.shape[1:])
    target_shape = tuple(int(value) for value in sample_target.shape[1:])
    model = build_model(config["model"], input_shape=input_shape, target_shape=target_shape)
    error_reducer = build_error_reducer(config["scoring"]["error_reducer"])

    # One forward pass on a sample batch is enough to know how many features the
    # scorer will receive after error reduction.
    with torch.no_grad():
        sample_errors = error_reducer(model(sample_input[:1].float()), sample_target[:1].float())
    feature_dim = int(sample_errors.shape[1])

    detector = AnomalyDetectorWrapper(
        model=model,
        error_reducer=error_reducer,
        score_estimator=build_score_estimator(config["scoring"]["score_estimator"], feature_dim=feature_dim),
        threshold_config=config["scoring"]["threshold"],
    )
    runtime = {
        "input_shape": list(input_shape),
        "target_shape": list(target_shape),
        "score_feature_dim": feature_dim,
    }
    return detector, runtime


def _init_wandb(config: dict[str, Any]):
    """Initialize W&B when it is enabled in the config."""
    wandb_config = config["wandb"]
    if not wandb_config["enabled"]:
        return None
    if not wandb_config["project"]:
        raise ValueError("wandb.project must be set when wandb.enabled is True.")

    import wandb

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"] or None,
        mode=wandb_config["mode"],
        name=config["run"]["name"],
        group=wandb_config["group"] or Path(config["run"]["dir"]).parent.name,
        tags=list(config["run"]["tags"]) + list(wandb_config["tags"]),
        config=clone_config(config),
        reinit=True,
    )
    if run is None:
        raise RuntimeError("wandb.init returned None.")
    return wandb


def _build_summary(run_dir: Path, config: dict[str, Any], runtime: dict[str, Any], detector, train_summary: dict[str, Any]):
    """Enrich the training summary with run metadata and model size stats."""
    summary = dict(train_summary)
    summary.update(
        {
            "family": run_dir.parent.name,
            "run_name": config["run"]["name"],
            "run_dir": str(run_dir),
            "model_name": config["model"]["name"],
            "task_type": config["task"]["type"],
            "input_shape": runtime["input_shape"],
            "target_shape": runtime["target_shape"],
            "score_feature_dim": runtime["score_feature_dim"],
            "parameter_count": count_parameters(detector),
            "parameter_size_bytes": parameter_size_bytes(detector),
        }
    )
    dump_json(run_dir / "summary.json", summary)
    return summary


def run_experiment(
        config: dict[str, Any], *,
        overwrite: bool = False, 
        skip_existing: bool = False, 
        data_bundle: dict[str, Any] | None = None,
        dry_run: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    if dry_run:
        validate_experiment_config(config)
        if data_bundle is None:
            data_bundle = build_pu_dataloaders(config)
        sample_batch = next(iter(data_bundle["loaders"]["train"]))
        detector, runtime = _build_detector(config, sample_batch[0].float(), sample_batch[1].float())
        print(config["run"]["name"], sum([p.numel() for p in detector.model.parameters()]))
        return {}, data_bundle

    """Run one experiment described by a fully explicit config."""
    validate_experiment_config(config)
    run_dir = Path(config["run"]["dir"])
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and skip_existing:
        return load_json(summary_path), data_bundle
    if run_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the original config first so failed runs still leave a readable artifact.
    config_snapshot = clone_config(config)
    dump_json(run_dir / "config.json", config_snapshot)
    seed_everything(int(config["run"]["seed"]))

    # Dataset dispatch stays explicit: right now the project only supports PU.
    if config["dataset"]["name"].lower() != "pu":
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")

    # Build the data first, then infer all runtime-only shapes from a real batch.
    if data_bundle is None:
        data_bundle = build_pu_dataloaders(config)
    sample_batch = next(iter(data_bundle["loaders"]["train"]))
    detector, runtime = _build_detector(config, sample_batch[0].float(), sample_batch[1].float())

    config_snapshot["runtime"] = {
        **runtime,
        **data_bundle["metadata"],
    }
    dump_json(run_dir / "config.json", config_snapshot)

    optimizer = build_optimizer(config["optimizer"], detector.model.parameters())
    criterion = build_criterion(config["loss"])
    scheduler = build_scheduler(config["scheduler"], optimizer, total_epochs=int(config["trainer"]["epochs"]))
    wandb_run = _init_wandb(config_snapshot)

    try:
        _, train_summary = train_anomaly_detector(
            detector,
            data_bundle["loaders"]["train"],
            data_bundle["loaders"]["val"],
            data_bundle["loaders"]["test"],
            optimizer,
            scheduler,
            criterion,
            config_snapshot["trainer"],
            config_snapshot["debug"]["trainer"],
            run_dir,
            wandb_run=wandb_run,
            config_snapshot=config_snapshot,
        )
        summary = _build_summary(run_dir, config_snapshot, config_snapshot["runtime"], detector, train_summary)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return summary, data_bundle


def run_experiments(
    configs: list[dict[str, Any]],
    *,
    overwrite: bool = False,
    skip_existing: bool = True,
    stop_on_error: bool = True,
    share_data_bundle: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run a list of explicit experiment configs one by one."""
    results = []
    data_bundle = None
    for config in tqdm.tqdm(configs, desc='experiments'):
        try:
            summary, new_data_bundle = run_experiment(config, overwrite=overwrite, skip_existing=skip_existing, data_bundle=data_bundle, dry_run=dry_run)
            if share_data_bundle and new_data_bundle is not None:
                data_bundle = new_data_bundle
            results.append(summary)
        except Exception as exc:
            if stop_on_error:
                raise
            results.append(
                {
                    "run_name": config["run"]["name"],
                    "run_dir": config["run"]["dir"],
                    "error": True,
                    "exception": repr(exc),
                }
            )
    return results


def load_detector_from_run(run_dir: str | Path, checkpoint: str = "best"):
    """Reload a saved detector and the config that produced it."""
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.json")
    runtime = config["runtime"]

    model = build_model(
        config["model"],
        input_shape=tuple(runtime["input_shape"]),
        target_shape=tuple(runtime["target_shape"]),
    )
    detector = AnomalyDetectorWrapper(
        model=model,
        error_reducer=build_error_reducer(config["scoring"]["error_reducer"]),
        score_estimator=build_score_estimator(config["scoring"]["score_estimator"], runtime["score_feature_dim"]),
        threshold_config=config["scoring"]["threshold"],
    )

    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    if not checkpoint_path.exists() and checkpoint != "last":
        checkpoint_path = run_dir / "checkpoints" / "last.pt"
    detector.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    detector.eval()
    return detector, config

