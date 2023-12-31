from tinygrad.tensor import Tensor
from tinygrad import nn
import math

def linear(x, w):
    return x.linear(w.transpose(0,1))

class MultiheadAttention():
    def __init__(self, embed_dim: int, num_heads: int, causal: bool, cross_attention: bool):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention

        self.in_proj_weight = Tensor.kaiming_uniform(embed_dim * 3, embed_dim, a=math.sqrt(5))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    

    def get_causal_mask(self, size: int):
        queries_pos = Tensor.arange(size).reshape(-1, 1)
        keys_pos = Tensor.arange(size).reshape(1, -1)
        return Tensor.where(
            (queries_pos - keys_pos) >= 0,
            Tensor.zeros([]),
            Tensor.full([], float('-inf'))
        )

    def __call__(self, query, key, value):
        if self.cross_attention:
            q = query.linear(self.in_proj_weight[:self.embed_dim])
            k = key.linear(self.in_proj_weight[self.embed_dim: 2 * self.embed_dim])
            v = value.linear(self.in_proj_weight[2 * self.embed_dim:])
            q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads).transpose(1, 2) for x in [q, k, v]]
        else:
            projected = linear(query, self.in_proj_weight)
            packed = projected.reshape(projected.shape[0], projected.shape[1], 3, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 3)
            q, k, v = [packed[:,:,i,:,:] for i in range(3)]

        B, h, T, d = q.shape
        # Logic: every head is effectively another batch item,
        # e.g. you should only interact with your own head index,
        # that's why it goes first together with batch
        q = q / math.sqrt(self.embed_dim // self.num_heads)
        attention = q.matmul(k.transpose(2, 3)) 
        if self.causal:
            attn_mask = self.get_causal_mask(query.shape[1]).unsqueeze(0).unsqueeze(0).to(q.device).cast(q.dtype)
            attention += attn_mask
        activation = attention.softmax(axis=-1)
        x = (v.unsqueeze(2).repeat([1,1,T,1,1]) * activation.unsqueeze(-1).repeat([1,1,1,1,64])).sum(axis=3)
        x = x.transpose(1,2).reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x, None

def create_sin_embedding(positions: Tensor, dim: int) -> Tensor:
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions
    adim = Tensor.arange(half_dim, device=positions.device, dtype=positions.dtype).reshape(1, 1, -1)
    max_period_tensor = Tensor.full([], 10000, device=positions.device)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return phase.cos().cat(phase.sin(), dim=-1)

class TransformerLayer():
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

    def __call__(self, x, cross_attention_src):
        x_ = self.norm1(x)
        x = x + self.self_attn(x_, x_, x_)[0]

        x_ = self.norm_cross(x)
        x = x + self.cross_attention(x_, cross_attention_src, cross_attention_src)[0]

        x_ = self.norm2(x)
        x = x + self.linear2(Tensor.gelu(self.linear1(x_)))
        return x

class Transformer():
    def __init__(self, dim: int = 1024, depth: int = 24, ff_dim: int = 4096, heads: int = 16):
        super().__init__()

        self.layers = [
            TransformerLayer() for _ in range(depth)
        ]

    def pos_embed(self, x):
        B, T, dim = x.shape
        positions = Tensor.arange(T, device=x.device).reshape(1, -1, 1).cast(x.dtype)
        x = x + create_sin_embedding(positions, dim)
        return x
    def __call__(self, x, cross_attention_src):
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, cross_attention_src=cross_attention_src)
        return x