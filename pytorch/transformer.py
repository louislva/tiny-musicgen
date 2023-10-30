import torch
from torch import nn
import torch.nn.functional as F
import xformers.ops as xops

import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool, cross_attention: bool):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention
        self.decoder = True

        self.in_proj_weight = nn.Parameter(torch.empty((embed_dim * 3, embed_dim)))
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_cache = None
        self.value_cache = None

    def get_causal_mask(self, size: int):
        queries_pos = torch.arange(size).view(-1, 1)
        keys_pos = torch.arange(size).view(1, -1)
        return torch.where(
            (queries_pos - keys_pos) >= 0,
            torch.zeros([]),
            torch.full([], float('-inf'))
        )
    
    def compute_key(self, key):
        # If batch size changed, or the seq length has gone down, invalidate the cache
        if self.key_cache is None or self.key_cache.shape[0] != key.shape[0] or key.shape[1] < self.key_cache.shape[1]:
            self.key_cache = torch.empty((key.shape[0], 0, key.shape[-1]), dtype=torch.float16).to(key.device)

        key_computed = F.linear(key[:, self.key_cache.shape[1]:], self.in_proj_weight[self.embed_dim: 2 * self.embed_dim])
        self.key_cache = torch.cat([
            self.key_cache,
            key_computed
        ], dim=1)

        return self.key_cache
    def compute_value(self, value):
        # If batch size changed, or the seq length has gone down, invalidate the cache
        if self.value_cache is None or self.value_cache.shape[0] != value.shape[0] or value.shape[1] < self.value_cache.shape[1]:
            self.value_cache = torch.empty((value.shape[0], 0, value.shape[-1]), dtype=torch.float16).to(value.device)

        value_computed = F.linear(value[:, self.value_cache.shape[1]:], self.in_proj_weight[2 * self.embed_dim:])
        self.value_cache = torch.cat([
            self.value_cache,
            value_computed
        ], dim=1)

        return self.value_cache
    
    def forward(self, query, key, value):
        if not self.cross_attention:
            key = query
            value = query
        
        q = F.linear(query, self.in_proj_weight[:self.embed_dim])
        if self.decoder:
            q = F.linear(query[:, -1:], self.in_proj_weight[:self.embed_dim])
        else:
            q = F.linear(query, self.in_proj_weight[:self.embed_dim])
        k = self.compute_key(key)
        v = self.compute_value(value)
        q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads) for x in [q, k, v]]

        B, T, h, d = q.shape
        x = xops.memory_efficient_attention(q, k, v)
        x = x.view(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x, None

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
        self.cross_attention = MultiheadAttention(
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