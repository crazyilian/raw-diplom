from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Parameter-free sinusoidal positional encoding from the original Transformer."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)


class PatchTSTForecaster(nn.Module):
    """PatchTST-style Transformer forecaster.

    The model follows the main PatchTST design choices: patch segmentation of the
    input window, channel-independent processing with shared Transformer weights,
    and a direct linear head that predicts the whole forecast horizon.

    Input shape:  (batch, input_steps, channels)
    Output shape: (batch, horizon_steps, channels)
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        patch_len: int = 12,
        patch_stride: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("Expected shapes in the form (time_steps, channels).")
        if self.input_shape[1] != self.output_shape[1]:
            raise ValueError("PatchTSTForecaster expects the same channel count in input and output.")
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive.")
        if patch_len > self.input_shape[0]:
            raise ValueError("patch_len must not exceed the input sequence length.")
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("d_model, nhead and num_layers must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if activation.lower() not in {"relu", "gelu"}:
            raise ValueError("TransformerEncoderLayer supports activation='relu' or 'gelu' in this implementation.")

        self.input_steps, self.channels = self.input_shape
        self.horizon_steps = int(self.output_shape[0])
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)
        self.num_patches = 1 + (self.input_steps - self.patch_len) // self.patch_stride
        if self.num_patches <= 0:
            raise ValueError("No patches can be extracted with the chosen patch_len and patch_stride.")

        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=self.num_patches)
        self.input_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward) if dim_feedforward is not None else 2 * self.d_model,
            dropout=float(dropout),
            activation=activation.lower(),
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(self.num_patches * self.d_model, self.horizon_steps)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (*, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}."
            )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, channels) -> (batch, channels, num_patches, patch_len)
        return x.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=self.patch_stride).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        patches = self._patchify(x)
        batch_size, channels, num_patches, patch_len = patches.shape

        tokens = patches.reshape(batch_size * channels, num_patches, patch_len)
        tokens = self.patch_embedding(tokens)
        tokens = self.positional_encoding(tokens)
        tokens = self.input_dropout(tokens)
        tokens = self.encoder(tokens)

        forecast = self.head(tokens.reshape(batch_size, channels, num_patches * self.d_model))
        return forecast.transpose(1, 2)
