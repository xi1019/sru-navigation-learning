#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Path-encoding module for compressing a sequence of waypoints into a latent vector."""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to a (B, L, D) sequence."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the first L positions of x, shape (B, L, D)."""
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class PathEncoder(nn.Module):
    """Encodes a path of shape (B, max_len, input_dim) into a compact latent vector.

    Steps:
    - project raw waypoint features to d_model
    - add sinusoidal positional embedding
    - apply a 1-layer Transformer encoder block (self-attention + FFN)
    - pool the sequence with a learned cross-attention query

    Output: (B, d_model)
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 32,
        num_heads: int = 4,
        max_len: int = 15,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.input_linear = nn.Linear(input_dim, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ELU(inplace=True),
            nn.Linear(d_model * 2, d_model),
            nn.ELU(inplace=True),
        )

        self.latent_query = nn.Parameter(torch.randn(d_model))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, path: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            path: Tensor of shape (B, max_len, input_dim)
            query: Optional external query of shape (B, d_model).
                None uses the original learned query.

        Returns:
            Latent vector of shape (B, d_model)
        """
        x = self.input_linear(path)
        x = self.pos_encoding(x)

        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        if query is None:
            q = self.latent_query.unsqueeze(0).expand(path.size(0), -1).unsqueeze(1)
        else:
            q = query.unsqueeze(1)
        latent, _ = self.cross_attn(q, x, x, need_weights=False)
        return latent.squeeze(1)
