
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class GRURepeatedAutoencoder(nn.Module):
    """GRU encoder-decoder for window reconstruction.

    Expects input with shape ``(batch, time_steps, channels)`` and returns the same shape.
    The encoder compresses the whole window into a single latent vector. The decoder
    reconstructs the complete sequence from that latent representation.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        decoder_input: str = "latent",
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("GRUAutoencoder expects shapes in the form (time_steps, channels).")
        if self.input_shape != self.output_shape:
            raise ValueError("This implementation is intended for reconstruction, so input_shape must equal output_shape.")
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError("hidden_size and latent_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if decoder_input not in {"latent", "zeros"}:
            raise ValueError("decoder_input must be either 'latent' or 'zeros'.")

        time_steps, channels = self.input_shape
        self.time_steps = time_steps
        self.channels = channels
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.num_layers = int(num_layers)
        self.decoder_input_mode = decoder_input

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.to_latent = nn.Linear(self.hidden_size, self.latent_size)
        self.latent_activation = nn.Tanh()

        self.decoder_input_projection = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder_hidden_projection = nn.Linear(self.latent_size, self.num_layers * self.hidden_size)
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.output_shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        top_hidden = hidden[-1]
        latent = self.latent_activation(self.to_latent(top_hidden))
        return latent

    def _build_decoder_inputs(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        if self.decoder_input_mode == "zeros":
            return torch.zeros(
                batch_size,
                self.time_steps,
                self.hidden_size,
                device=latent.device,
                dtype=latent.dtype,
            )

        repeated = self.decoder_input_projection(latent).unsqueeze(1)
        return repeated.repeat(1, self.time_steps, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoder_inputs = self._build_decoder_inputs(latent)
        hidden = (
            self.decoder_hidden_projection(latent)
            .view(latent.shape[0], self.num_layers, self.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        decoded, _ = self.decoder(decoder_inputs, hidden)
        return self.output_projection(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"GRUAutoencoder expects a 3D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (*, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}."
            )
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction


# if __name__ == "__main__":
#     model = GRUAutoencoder(
#         input_shape=(120, 27),
#         output_shape=(120, 27),
#         hidden_size=64,
#         latent_size=32,
#         num_layers=1,
#         dropout=0.0,
#         decoder_input="latent",
#     )
#     x = torch.randn(4, 120, 27)
#     y = model(x)
#     print("Input shape:", tuple(x.shape))
#     print("Output shape:", tuple(y.shape))
#     print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
