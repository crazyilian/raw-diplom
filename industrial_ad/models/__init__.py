from __future__ import annotations

from typing import Any

from industrial_ad.models.mlp import WindowMLP


def build_model(model_config: dict[str, Any], input_shape: tuple[int, ...], target_shape: tuple[int, ...]):
    """Build a model family selected by `model.name`."""
    model_name = model_config["name"].lower()
    model_params = dict(model_config["params"])

    if model_name in {"window_mlp", "mlp_autoencoder"}:
        return WindowMLP(input_shape=input_shape, output_shape=target_shape, **model_params)

    raise ValueError(f"Unknown model: {model_config['name']}")


__all__ = ["WindowMLP", "build_model"]
