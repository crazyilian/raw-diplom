from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from .utils import build_activation
import math


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for tensors shaped as `(batch, channels, time)`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    """1D convolution that only looks to the past."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)

class ChannelWiseLinear(nn.Module):
    def __init__(self, num_channels: int, length1: int, length2: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_channels, length1, length2))  # (C, L1, L2)
        if bias:
            self.bias = nn.Parameter(torch.empty(num_channels, length2))          # (C, L2)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))  # 1/sqrt(L1)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L1)
        y = torch.einsum("bcl,clm->bcm", x, self.weight)  # -> (B, C, L2)
        if self.bias is not None:
            y = y + self.bias  # broadcast по batch
        return y


class DepthwiseSeparableCausalConv1d(nn.Module):
    """Depthwise-separable causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TemporalResidualBlock(nn.Module):
    """Residual causal TCN block with two temporal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        activation: str,
        dropout: float,
        separable: bool,
        norm: str,
    ) -> None:
        super().__init__()
        conv = DepthwiseSeparableCausalConv1d if separable else CausalConv1d
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.norm1 = self._build_norm(norm, out_channels)
        self.norm2 = self._build_norm(norm, out_channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    @staticmethod
    def _build_norm(name: str, channels: int) -> nn.Module:
        key = name.lower()
        if key == "batch":
            return nn.BatchNorm1d(channels)
        if key == "layer":
            return ChannelLayerNorm(channels)
        if key == "group":
            return nn.GroupNorm(num_groups=1, num_channels=channels)
        if key == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported norm: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation(y)
        y = self.dropout(y)
        return y + residual


class TCNForecaster(nn.Module):
    """Direct TCN forecaster for `(time, channels)` windows."""

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        separable: bool = True,
        norm: str = "layer",
        dilations: Sequence[int] | None = None,
        final_steps: int | None = None
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.output_shape = tuple(int(value) for value in output_shape)
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("TCNForecaster expects shapes in the form (time_steps, channels).")
        if self.input_shape[1] != self.output_shape[1]:
            raise ValueError("TCNForecaster requires the same number of input and output channels.")
        if hidden_channels <= 0 or num_blocks <= 0:
            raise ValueError("hidden_channels and num_blocks must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.history_steps, self.channels = self.input_shape
        self.forecast_steps = int(self.output_shape[0])
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.num_blocks = int(num_blocks)
        self.final_steps = int(final_steps) if final_steps is not None else self.forecast_steps

        if dilations is None:
            self.dilations = [2 ** index for index in range(self.num_blocks)]
        else:
            self.dilations = [int(value) for value in dilations]
            if len(self.dilations) != self.num_blocks:
                raise ValueError("len(dilations) must match num_blocks.")

        self.input_projection = nn.Conv1d(self.channels, self.hidden_channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=float(dropout),
                    separable=bool(separable),
                    norm=norm,
                )
                for dilation in self.dilations
            ]
        )
        self.output_projection = nn.Conv1d(self.hidden_channels, self.channels, kernel_size=1, bias=False)
        self.forecast_head = ChannelWiseLinear(self.channels, self.final_steps, self.forecast_steps)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCNForecaster expects a 3D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (*, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}."
            )

        y = x.transpose(1, 2)
        y = self.input_projection(y)
        y = self.blocks(y)
        y = self.output_projection(y)
        context = y[:, :, -self.final_steps :]
        prediction = self.forecast_head(context)
        return prediction.transpose(1, 2)
