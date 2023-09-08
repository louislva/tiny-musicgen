import torch
from torch import nn
import typing as tp
import torch.nn.functional as F
from multiheadattention import MultiheadAttention, SimpleAttention

USE_CUSTOM_ATTENTION = True

def create_sin_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions
    adim = torch.arange(half_dim, device=positions.device, dtype=positions.dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], 10000, device=positions.device)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024)
        self.self_attn = MultiheadAttention(
            embed_dim=1024,
            causal=True, # big difference!
            cross_attention=False,
            num_heads=16,
        )
        self.norm_cross = nn.LayerNorm(1024)
        self.cross_attention = SimpleAttention(
            embed_dim=1024,
            causal=False,
            cross_attention=True,
            num_heads=16,
        )

        self.norm2 = nn.LayerNorm(1024)
        self.linear1 = nn.Linear(1024, 4096, bias=False)
        self.linear2 = nn.Linear(4096, 1024, bias=False)

    def forward(self, x, cross_attention_src):
        x_ = self.norm1(x)
        x = x + self.self_attn(x_, x_, x_)[0]

        x_ = self.norm_cross(x)
        x = x + self.cross_attention(x_, cross_attention_src, cross_attention_src)[0]

        x_ = self.norm2(x)
        x = x + self.linear2(F.gelu(self.linear1(x_)))
        return x

class Transformer(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 24, ff_dim: int = 4096, heads: int = 16):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(depth)
        ])

    def pos_embed(self, x):
        B, T, dim = x.shape
        positions = torch.arange(T, device=x.device).view(1, -1, 1).to(x.dtype).to(x.device)
        x = x + create_sin_embedding(positions, dim)
        return x
    def forward(self, x, cross_attention_src):
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, cross_attention_src=cross_attention_src)
        return x
    