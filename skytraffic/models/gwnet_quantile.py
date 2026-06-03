from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange

from .gwnet import GWNET


def _quantile_integration_weights(quantile_levels: torch.Tensor) -> torch.Tensor:
    if quantile_levels.numel() == 1:
        return torch.ones_like(quantile_levels)

    weights = torch.zeros_like(quantile_levels)
    weights[0] = (quantile_levels[1] - quantile_levels[0]) / 2
    weights[-1] = (quantile_levels[-1] - quantile_levels[-2]) / 2
    if quantile_levels.numel() > 2:
        weights[1:-1] = (quantile_levels[2:] - quantile_levels[:-2]) / 2
    return weights / weights.sum()


class GWNET_Quantile(GWNET):
    def __init__(
        self,
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        randomadj: bool = True,
        aptonly: bool = True,
        kernel_size: int = 2,
        nhid: int = 32,
        residual_channels: int = None,
        dilation_channels: int = None,
        skip_channels: int = None,
        end_channels: int = None,
        apt_layer: bool = True,
        feature_dim: int = 2,
        output_dim: int = 1,
        quantiles: List[float] = None,
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
        metadata: dict = None,
    ):
        super().__init__(
            dropout=dropout,
            blocks=blocks,
            layers=layers,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            randomadj=randomadj,
            aptonly=aptonly,
            kernel_size=kernel_size,
            nhid=nhid,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            apt_layer=apt_layer,
            feature_dim=feature_dim,
            output_dim=output_dim,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            metadata=metadata,
        )

        if quantiles is None:
            quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        quantile_levels = torch.as_tensor(quantiles, dtype=torch.float32)
        if torch.any(quantile_levels[1:] <= quantile_levels[:-1]):
            raise ValueError("quantiles must be strictly increasing")

        del self.end_conv_2
        self.register_buffer("quantile_levels", quantile_levels)
        self.quantile_conv = nn.Conv2d(
            in_channels=self.end_channels,
            out_channels=self.pred_steps * quantile_levels.numel(),
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(source)
        x = self.quantile_conv(x)[..., -1]
        return rearrange(
            x,
            "N (T Q) P -> N T P Q",
            T=self.pred_steps,
            Q=self.quantile_levels.numel(),
        )

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        quantiles = self(source)
        valid = ~target.isnan()
        errors = target.unsqueeze(-1) - quantiles
        levels = self.quantile_levels.reshape(*([1] * (errors.ndim - 1)), -1)
        pinball = torch.maximum(levels * errors, (levels - 1.0) * errors)
        return {"quantile_loss": pinball[valid].mean()}

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        quantiles = torch.sort(self(source), dim=-1).values
        weights = _quantile_integration_weights(self.quantile_levels)
        pred = (quantiles * weights.reshape(*([1] * (quantiles.ndim - 1)), -1)).sum(dim=-1)
        return {
            "pred": pred,
            "quantiles": quantiles,
            "quantile_levels": self.quantile_levels,
        }
