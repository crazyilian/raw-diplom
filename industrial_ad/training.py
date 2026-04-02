from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
import warnings

from industrial_ad.utils import dump_json


def build_criterion(loss_config: dict[str, Any]):
    """Build the loss function selected in the config."""
    losses = {
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
    }
    name = loss_config["name"].lower()
    if name not in losses:
        raise ValueError(f"Unknown loss: {loss_config['name']}")
    return losses[name](**loss_config["params"])


def build_optimizer(optimizer_config: dict[str, Any], parameters):
    """Build the optimizer selected in the config."""
    optimizers = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }
    name = optimizer_config["name"].lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    return optimizers[name](parameters, **optimizer_config["params"])


def build_scheduler(scheduler_config: dict[str, Any], optimizer, total_epochs: int):
    """Build the learning-rate schedule selected in the config."""
    name = scheduler_config["name"].lower()
    if name in {"none", "constant"}:
        return None
    if name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")

    params = scheduler_config["params"]
    warmup_epochs = int(params["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=float(params["min_lr"]),
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(params["start_factor"]),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _metric_is_better(candidate: float, best: float | None, mode: str, min_delta: float) -> bool:
    """Return whether a metric improved enough to count as a new best checkpoint."""
    if np.isnan(candidate):
        return False
    if best is None:
        return True
    if mode == "max":
        return candidate > best + min_delta
    if mode == "min":
        return candidate < best - min_delta
    raise ValueError(f"Unknown optimization mode: {mode}")


def _save_figures(figures: dict[str, Any], base_dir: Path, prefix: str, epoch: int) -> None:
    """Save all plots for one split and one evaluation epoch."""
    for name, figure in figures.items():
        target_dir = base_dir / f"{prefix}_{name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(target_dir / f"epoch_{epoch:04d}.png", bbox_inches="tight")


def _close_figures(figures: dict[str, Any]) -> None:
    """Close matplotlib figures once they have been saved and optionally logged."""
    for figure in figures.values():
        plt.close(figure)


def _log_wandb_figures(wandb_run, log_payload: dict[str, Any], figures_by_split: dict[str, dict[str, Any]]) -> None:
    """Attach evaluation plots to the current W&B log payload."""
    if wandb_run is None:
        return
    for split_name, figures in figures_by_split.items():
        for figure_name, figure in figures.items():
            log_payload[f"{split_name}/{figure_name.upper()}"] = wandb_run.Image(figure)


def _build_summary(history: dict[str, Any], checkpoint_metric: str, best_evaluation: dict[str, Any] | None) -> dict[str, Any]:
    """Build the compact run summary saved next to the full history."""
    return {
        "best_epoch": history["best_epoch"],
        "best_metric_name": checkpoint_metric,
        "best_metric_value": history["best_metric_value"],
        "stopped_early": history["stopped_early"],
        "total_time_seconds": history["total_time_seconds"],
        "last_epoch": history["train_epochs"][-1]["epoch"] if history["train_epochs"] else 0,
        "best_metrics": best_evaluation["metrics"] if best_evaluation is not None else {},
        "last_metrics": history["evaluations"][-1]["metrics"] if history["evaluations"] else {},
    }


def train_anomaly_detector(
    detector,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    trainer_config: dict[str, Any],
    debug_trainer_config: dict[str, Any],
    save_dir: str | Path,
    *,
    wandb_run=None,
    config_snapshot: dict[str, Any] | None = None,
):
    """Train a detector, evaluate it on schedule and save local/W&B artifacts."""
    save_dir = Path(save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    plots_dir = save_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if config_snapshot is not None:
        dump_json(save_dir / "config.json", config_snapshot)

    device = torch.device(trainer_config["device"])
    detector.to(device)

    # AMP stays on the same code path as FP32 training; with `enabled=False`
    # autocast and GradScaler simply behave like no-ops.
    scaler = torch.amp.GradScaler("cuda", enabled=bool(trainer_config["amp"]))

    checkpoint_config = trainer_config["checkpoint"]
    early_stopping = trainer_config["early_stopping"]
    max_train_batches = debug_trainer_config["max_train_batches"]
    max_eval_batches = debug_trainer_config["max_eval_batches"]

    history = {
        "train_epochs": [],
        "evaluations": [],
        "best_epoch": None,
        "best_metric_name": checkpoint_config["metric"],
        "best_metric_value": None,
        "stopped_early": False,
        "total_time_seconds": 0.0,
    }
    best_evaluation = None
    best_early_metric = None
    patience_left = int(early_stopping["patience"])
    total_time = 0.0

    if wandb_run is not None:
        wandb_run.define_metric("epoch")

    for epoch in tqdm.tqdm(range(1, int(trainer_config["epochs"]) + 1), desc=save_dir.name):
        epoch_start = time.time()
        detector.model.train()
        batch_losses = []
        batch_grad_norms = []

        # Train for one epoch, optionally truncating the epoch for smoke runs.
        for batch_index, (x, y, _) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(trainer_config["amp"])):
                loss = criterion(detector.model(x), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(detector.model.parameters(), float(trainer_config["clip_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(float(loss.item()))
            batch_grad_norms.append(float(grad_norm.item()))

        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()

        # Train metrics are aggregated once per epoch; per-batch loss logging is intentionally omitted.
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        train_record = {
            "epoch": epoch,
            "loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "grad_norm": float(np.mean(batch_grad_norms)) if batch_grad_norms else float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_seconds": epoch_time,
        }
        history["train_epochs"].append(train_record)
        history["total_time_seconds"] = total_time

        log_payload = {
            "epoch": epoch,
            "train/loss": train_record["loss"],
            "train/grad_norm": train_record["grad_norm"],
            "train/lr": train_record["lr"],
            "train/epoch_time": train_record["epoch_time_seconds"],
        }

        should_evaluate = epoch % int(trainer_config["eval_every_n_epochs"]) == 0 or epoch == int(trainer_config["epochs"])
        if should_evaluate:
            # Validation serves two purposes: it fits the anomaly scorer and then
            # provides the metrics used for checkpointing and early stopping.
            detector.fit_score_estimator(val_loader, max_batches=max_eval_batches)
            detector.fit_threshold(val_loader, max_batches=max_eval_batches)
            val_metrics, val_figures = detector.evaluate(val_loader, prefix="val", max_batches=max_eval_batches)
            test_metrics, test_figures = detector.evaluate(test_loader, prefix="test", max_batches=max_eval_batches)

            evaluation_record = {"epoch": epoch, "metrics": {**val_metrics, **test_metrics}, "is_best": False}
            current_metric = float(evaluation_record["metrics"][checkpoint_config["metric"]])
            if _metric_is_better(current_metric, history["best_metric_value"], checkpoint_config["mode"], float(checkpoint_config["min_delta"])):
                history["best_metric_value"] = current_metric
                history["best_epoch"] = epoch
                evaluation_record["is_best"] = True
                best_evaluation = evaluation_record
                torch.save(detector.state_dict(), checkpoints_dir / "best.pt")

            # Early stopping uses the same evaluation snapshot, but may track a different metric.
            if early_stopping["enabled"]:
                early_value = float(evaluation_record["metrics"][early_stopping["metric"]])
                if _metric_is_better(early_value, best_early_metric, early_stopping["mode"], float(early_stopping["min_delta"])):
                    best_early_metric = early_value
                    patience_left = int(early_stopping["patience"])
                else:
                    patience_left -= 1

            # Local figures are part of the run artifacts and are saved on every evaluation.
            history["evaluations"].append(evaluation_record)
            log_payload.update(evaluation_record["metrics"])
            _save_figures(val_figures, plots_dir, "val", epoch)
            _save_figures(test_figures, plots_dir, "test", epoch)
            dump_json(save_dir / "history.json", history)
            _log_wandb_figures(wandb_run, log_payload, {"val": val_figures, "test": test_figures})
            _close_figures(val_figures)
            _close_figures(test_figures)

            # Early stopping is checked only after a full evaluation snapshot is saved.
            if early_stopping["enabled"] and patience_left < 0:
                history["stopped_early"] = True
                torch.save(detector.state_dict(), checkpoints_dir / "last.pt")
                if wandb_run is not None:
                    wandb_run.log(log_payload)
                break

        # `last.pt` is refreshed every epoch so interrupted runs still leave the most recent weights behind.
        torch.save(detector.state_dict(), checkpoints_dir / "last.pt")
        if wandb_run is not None:
            wandb_run.log(log_payload)

    # Save both the full history and a short summary that notebooks can read quickly.
    summary = _build_summary(history, checkpoint_config["metric"], best_evaluation)
    dump_json(save_dir / "history.json", history)
    dump_json(save_dir / "summary.json", summary)
    return history, summary
