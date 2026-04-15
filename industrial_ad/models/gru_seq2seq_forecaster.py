from typing import Iterable

import torch
from torch import nn


class GRUSeq2seqForecaster(nn.Module):
    """GRU encoder-decoder forecaster for multi-step prediction.

    The model follows the classical seq2seq recipe: an encoder GRU reads the
    observed window, and a decoder GRU autoregressively generates the forecast
    horizon. During training the decoder can use teacher forcing; during
    evaluation it always free-runs on its own predictions.

    Input shape:  (batch, input_steps, channels)
    Output shape: (batch, horizon_steps, channels)
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        teacher_forcing_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("Expected shapes in the form (time_steps, channels).")
        if self.input_shape[1] != self.output_shape[1]:
            raise ValueError("GRUSeq2seqForecaster expects the same channel count in input and output.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")

        self.input_steps, self.channels = self.input_shape
        self.horizon_steps = int(self.output_shape[0])
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.channels)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (*, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}."
            )

    def _validate_target(self, target: torch.Tensor) -> None:
        if target.ndim != 3:
            raise ValueError(f"Expected a 3D target tensor, got shape {tuple(target.shape)}.")
        if target.shape[1:] != self.output_shape:
            raise ValueError(
                f"Expected target shape (*, {self.output_shape[0]}, {self.output_shape[1]}), got {tuple(target.shape)}."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        return hidden

    def decode(self, hidden: torch.Tensor, start_token: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        prev = start_token
        predictions = []

        for step in range(self.horizon_steps):
            out, hidden = self.decoder(prev, hidden)
            pred = self.output_projection(out)
            predictions.append(pred)

            if step + 1 == self.horizon_steps:
                continue

            if self.training and target is not None and self.teacher_forcing_ratio > 0.0:
                teacher_mask = torch.rand(pred.shape[0], 1, 1, device=pred.device) < self.teacher_forcing_ratio
                prev = torch.where(teacher_mask, target[:, step : step + 1, :], pred)
            else:
                prev = pred

        return torch.cat(predictions, dim=1)

    def forward_train(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        self._validate_target(target)
        hidden = self.encode(x)
        start_token = x[:, -1:, :]
        return self.decode(hidden=hidden, start_token=start_token, target=target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        hidden = self.encode(x)
        start_token = x[:, -1:, :]
        return self.decode(hidden=hidden, start_token=start_token, target=None)
