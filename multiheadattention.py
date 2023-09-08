from einops import rearrange
from xformers import ops
import torch
from torch import nn
from torch.nn import init
import typing as tp
import torch.nn.functional as F
from audiocraft.modules.streaming import StreamingModule
import math

def _get_attention_time_dimension():
    return 2
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool, cross_attention: bool):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention

        self.in_proj_weight = nn.Parameter(torch.empty((embed_dim * 3, embed_dim)))
        init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    

    def _get_mask(self, current_steps: int, device: torch.device, dtype: torch.dtype):
        # Return a causal mask, accounting for potentially stored past keys/values
        # We actually return a bias for the attention score, as this has the same
        # convention both in the builtin MHA in Pytorch, and Xformers functions.
        queries_pos = torch.arange(
            0, current_steps, device=device).view(-1, 1)
        keys_pos = torch.arange(current_steps, device=device).view(1, -1)
        delta = queries_pos - keys_pos
        valid = delta >= 0
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def forward(self, query, key, value):
        if self.cross_attention:
            # Project
            q = F.linear(query, self.in_proj_weight[:self.embed_dim])
            k = F.linear(key, self.in_proj_weight[self.embed_dim: 2 * self.embed_dim])
            v = F.linear(value, self.in_proj_weight[2 * self.embed_dim:])

            # Split into heads
            q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads).transpose(1, 2) for x in [q, k, v]]
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
            bound_layout = "b h p t d"
            packed = rearrange(projected, f"b t (p h d) -> {bound_layout}", p=3, h=self.num_heads)
            q, k, v = ops.unbind(packed, dim=2)

        # q.shape = [B, h, T, d]
        # Logic: every head is effectively another batch item; e.g. you should only interact with your own head index; 
        # that's why it goes first together with batch
        q = q / math.sqrt(self.embed_dim // self.num_heads)
        attention = q.matmul(k.transpose(2, 3)) 
        if self.causal:
            attention += self._get_mask(query.shape[1], q.device, q.dtype).unsqueeze(0).unsqueeze(0)
        activation = torch.softmax(attention, dim=-1)
        layout = "b h t d"
        key_layout = layout.replace('t', 'k')
        x = torch.einsum(f"b h t k, {key_layout} -> {layout}", activation, v)
        x = rearrange(x, f"{layout} -> b t (h d)", h=self.num_heads)        
        x = self.out_proj(x)
        return x, None


class MultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        cross_attention: Should be true when used as a cross attention.
            All keys and values must be available at once, streaming is only for the queries.
            Cannot be used with `causal` or `rope` (as it wouldn't make sens to
            interpret the time steps in the keys relative to those in the queries).
        safe_streaming (bool): Bug fix, will go away with xformers update.
        qk_layer_norm (bool): Layer normalization applied to queries and keys before dot product.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 causal: bool = False, attention_as_float32: bool = False, cross_attention: bool = False,
                 safe_streaming: bool = True, kv_repeat: int = 1,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.causal = causal
        self.past_context = None
        self.attention_as_float32 = attention_as_float32
        self.cross_attention = cross_attention
        self.safe_streaming = safe_streaming
        self.num_heads = num_heads

        if cross_attention:
            assert not causal, "Causal cannot work with cross attention."

        # CUstom implementation i guess
        out_dim = embed_dim
        assert num_heads % kv_repeat == 0
        assert not cross_attention or kv_repeat == 1
        num_kv = num_heads // kv_repeat
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        in_proj = nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

    def _get_mask(self, current_steps: int, device: torch.device, dtype: torch.dtype):
        # Return a causal mask, accounting for potentially stored past keys/values
        # We actually return a bias for the attention score, as this has the same
        # convention both in the builtin MHA in Pytorch, and Xformers functions.
        time_dim = _get_attention_time_dimension()
        if self._streaming_state:
            past_keys = self._streaming_state['past_keys']
            past_steps = past_keys.shape[time_dim]
        else:
            past_steps = 0

        queries_pos = torch.arange(
            past_steps, current_steps + past_steps, device=device).view(-1, 1)
        keys_pos = torch.arange(past_steps + current_steps, device=device).view(1, -1)
        delta = queries_pos - keys_pos
        valid = delta >= 0
        if self.past_context is not None:
            valid &= (delta <= self.past_context)
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def _complete_kv(self, k, v):
        time_dim = _get_attention_time_dimension()
        if self.cross_attention:
            # With cross attention we assume all keys and values
            # are already available, and streaming is with respect
            # to the queries only.
            return k, v
        # Complete the key/value pair using the streaming state.
        if self._streaming_state:
            pk = self._streaming_state['past_keys']
            nk = torch.cat([pk, k], dim=time_dim)
            if v is k:
                nv = nk
            else:
                pv = self._streaming_state['past_values']
                nv = torch.cat([pv, v], dim=time_dim)
        else:
            nk = k
            nv = v

        assert nk.shape[time_dim] == nv.shape[time_dim]
        offset = 0
        if self.past_context is not None:
            offset = max(0, nk.shape[time_dim] - self.past_context)
        if self._is_streaming:
            self._streaming_state['past_keys'] = nk[:, offset:]
            if v is not k:
                self._streaming_state['past_values'] = nv[:, offset:]
            if 'offset' in self._streaming_state:
                self._streaming_state['offset'] += offset
            else:
                self._streaming_state['offset'] = torch.tensor(0)
        return nk, nv

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        assert attn_mask is None
        assert not is_causal, ("New param added in torch 2.0.1 not supported, "
                               "use the causal args in the constructor.")

        time_dim = _get_attention_time_dimension()
        layout = "b h t d"
        dtype = query.dtype
        if self._is_streaming:
            assert self.causal or self.cross_attention, \
                "Streaming only available for causal or cross attention"

        if self.causal:
            # At the moment we specialize only for the self-attention case.
            assert query.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            assert value.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            attn_mask = self._get_mask(query.shape[1], query.device, query.dtype)

        # custom implementation
        assert need_weights is False
        assert key_padding_mask is None
        if False:
            # Different queries, keys, values, we have to spit manually the weights
            # before applying the linear.
            dim = self.in_proj_weight.shape[0] // 3
            q = nn.functional.linear(query, self.in_proj_weight[:dim])
            # todo: when streaming, we could actually save k, v and check the shape actually match.
            k = nn.functional.linear(query, self.in_proj_weight[dim: 2 * dim])
            v = nn.functional.linear(query, self.in_proj_weight[2 * dim:])
            q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads).transpose(1, 2) for x in [q, k, v]]
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
            bound_layout = "b h p t d"
            packed = rearrange(projected, f"b t (p h d) -> {bound_layout}", p=3, h=self.num_heads)
            q, k, v = ops.unbind(packed, dim=2)

            # k, v = self._complete_kv(k, v)
     
        # We include the dot product as float32, for consistency
        # with the other implementations that include that step
        # as part of the attention. Note that when using `autocast`,
        # the einsums would be done as bfloat16, but the softmax
        # would be done as bfloat16, so `attention_as_float32` will
        # extend a bit the range of operations done in float32,
        # although this should make no difference.
        q = q / math.sqrt(self.embed_dim // self.num_heads)
        key_layout = layout.replace('t', 'k')
        query_layout = layout
        pre_w = q.matmul(k.transpose(2, 3)) 
        pre_w = pre_w + self._get_mask(query.shape[1], q.device, q.dtype).unsqueeze(0).unsqueeze(0)
        w = torch.softmax(pre_w, dim=-1)
        # Key and value have the same format.

        x = torch.einsum(f"b h t k, {key_layout} -> {layout}", w, v)
        # x = x.to(dtype)
        x = rearrange(x, f"{layout} -> b t (h d)", h=self.num_heads)
        # print("x.shape", x.shape)
        x = self.out_proj(x)

        return x, None

