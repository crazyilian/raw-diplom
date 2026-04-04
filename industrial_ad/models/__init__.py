from __future__ import annotations

from typing import Any

from industrial_ad.models.mlp_ae import MLPAutoencoder
from industrial_ad.models.tcn_ae import TCNAutoencoder


def build_model(model_config: dict[str, Any], input_shape: tuple[int, ...], target_shape: tuple[int, ...]):
    """Build a model family selected by `model.name`."""
    model_name = model_config["name"].lower()
    model_params = dict(model_config["params"])

    if model_name in ["window_mlp", "mlp_ae"]:
        return MLPAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "tcn_ae":
        return TCNAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)

    raise ValueError(f"Unknown model: {model_config['name']}")


__all__ = ["build_model", "MLPAutoencoder", "TCNAutoencoder"]
