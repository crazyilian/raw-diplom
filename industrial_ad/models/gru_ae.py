from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class GRUAutoencoder(nn.Module):
    """Canonical GRU encoder-decoder for fixed-window reconstruction.

    Input and output shapes are ``(batch, time_steps, channels)``.

    Main differences from the previous hybrid GRU implementation:
    1. The decoder is conditioned through its initial hidden state, as in classical
       seq2seq encoder-decoder models for reconstruction.
    2. The decoder consumes a shifted sequence of previous values during training
       (teacher forcing / scheduled sampling) and free-runs during evaluation.
    3. Multi-layer hidden states are kept in the correct ``(layers, batch, hidden)``
       layout with no batch mixing.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional_encoder: bool = False,
        teacher_forcing_ratio: float = 0.0,
        reverse_target: bool = False,
        learned_start: bool = False,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("Expected shapes in the form (time_steps, channels).")
        if self.input_shape != self.output_shape:
            raise ValueError("This implementation is intended for reconstruction only.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")

        self.time_steps, self.channels = self.input_shape
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional_encoder = bool(bidirectional_encoder)
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)
        self.reverse_target = bool(reverse_target)
        self.learned_start = bool(learned_start)

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0

        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=self.bidirectional_encoder,
        )

        if self.bidirectional_encoder:
            self.encoder_to_decoder = nn.Linear(2 * self.hidden_size, self.hidden_size)
        else:
            self.encoder_to_decoder = nn.Identity()

        self.decoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.channels)

        if self.learned_start:
            self.start_token = nn.Parameter(torch.zeros(1, 1, self.channels))
        else:
            self.register_buffer("start_token", torch.zeros(1, 1, self.channels), persistent=False)

    def _prepare_target(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=(1,)) if self.reverse_target else x

    def _init_decoder_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        if not self.bidirectional_encoder:
            return encoder_hidden

        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        hidden = self.encoder_to_decoder(hidden)
        return hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        return self._init_decoder_hidden(hidden)

    def decode(self, target: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        device = target.device
        dtype = target.dtype

        prev = self.start_token.expand(batch_size, 1, self.channels).to(device=device, dtype=dtype)
        predictions = []

        for t in range(self.time_steps):
            out, hidden = self.decoder(prev, hidden)
            pred = self.output_projection(out)
            predictions.append(pred)

            if t + 1 == self.time_steps:
                continue

            if self.training and self.teacher_forcing_ratio > 0.0:
                teacher_mask = (torch.rand(batch_size, 1, 1, device=device) < self.teacher_forcing_ratio)
                prev = torch.where(teacher_mask, target[:, t : t + 1, :], pred)
            else:
                prev = pred

        return torch.cat(predictions, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (*, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}."
            )

        target = self._prepare_target(x)
        hidden = self.encode(x)
        reconstruction = self.decode(target=target, hidden=hidden)
        if self.reverse_target:
            reconstruction = reconstruction.flip(dims=(1,))
        return reconstruction


# if __name__ == "__main__":
#     model = WindowGRUSeq2SeqAutoencoder(
#         input_shape=(120, 27),
#         output_shape=(120, 27),
#         hidden_size=64,
#         num_layers=2,
#         dropout=0.1,
#         bidirectional_encoder=False,
#         teacher_forcing_ratio=0.0,
#         reverse_target=False,
#         learned_start=False,
#     )
#     x = torch.randn(4, 120, 27)
#     y = model(x)
#     print("Input shape:", tuple(x.shape))
#     print("Output shape:", tuple(y.shape))
#     print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))