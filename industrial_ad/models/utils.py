from torch import nn


def build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name.lower()]()

