from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an experiment config."""
    return copy.deepcopy(config)


def flatten_dict(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary for dataframe-friendly summaries."""
    items: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


def ensure_json_serializable(value: Any) -> Any:
    """Convert numpy and torch values into JSON-friendly Python objects."""
    if isinstance(value, dict):
        return {str(key): ensure_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact next to experiment outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ensure_json_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact produced by the experiment code."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible smoke runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable and non-trainable parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_size_bytes(module: torch.nn.Module) -> int:
    """Estimate the raw parameter footprint in bytes."""
    total = 0
    for parameter in module.parameters():
        total += parameter.numel() * parameter.element_size()
    return total
