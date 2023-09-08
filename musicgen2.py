from einops import rearrange
from xformers import ops
import torch
from torch import nn
import typing as tp
import torch.nn.functional as F
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import StreamingTransformerLayer
from tqdm import trange
import random
import numpy as np
import copy
from audiocraft.modules.streaming import StreamingModule

# set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5, top_k=2048)

def _get_attention_time_dimension():
    return 2
class MultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        dropout (float): Dropout level.
        bias (bool): Use bias in projections.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None, custom: bool = False, attention_as_float32: bool = False, cross_attention: bool = False,
                 safe_streaming: bool = True, qk_layer_norm: bool = False, kv_repeat: int = 1,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if past_context is not None:
            assert causal

        self.embed_dim = embed_dim
        self.causal = causal
        self.past_context = past_context
        self.attention_as_float32 = attention_as_float32
        self.cross_attention = cross_attention
        self.safe_streaming = safe_streaming
        self.num_heads = num_heads
        self.dropout = dropout
        self.kv_repeat = kv_repeat
        if cross_attention:
            assert not causal, "Causal cannot work with cross attention."
            assert rope is None, "Rope cannot work with cross attention."

        self.custom = custom
        if self.custom:
            out_dim = embed_dim
            assert num_heads % kv_repeat == 0
            assert not cross_attention or kv_repeat == 1
            num_kv = num_heads // kv_repeat
            kv_dim = (embed_dim // num_heads) * num_kv
            out_dim += 2 * kv_dim
            in_proj = nn.Linear(embed_dim, out_dim, bias=bias, **factory_kwargs)
            # We try to follow the default PyTorch MHA convention, to easily compare results.
            self.in_proj_weight = in_proj.weight
            self.in_proj_bias = in_proj.bias
            print("bias is:", bias)
            if bias:
                self.in_proj_bias.data.zero_()  # Following Pytorch convention
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            if bias:
                self.out_proj.bias.data.zero_()
        else:
            assert not qk_layer_norm
            assert kv_repeat == 1
            self.mha = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True,
                **factory_kwargs)
        self.qk_layer_norm = qk_layer_norm
        if qk_layer_norm:
            assert self.custom
            assert kv_repeat == 1
            ln_dim = embed_dim
            self.q_layer_norm = nn.LayerNorm(ln_dim)
            self.k_layer_norm = nn.LayerNorm(ln_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not self.custom:
            # Support compat with regular MHA
            keys = [n for n, _ in self.mha.named_parameters()]
            for key in keys:
                if prefix + key in state_dict:
                    state_dict[prefix + "mha." + key] = state_dict.pop(prefix + key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        if time_dim == 2:
            layout = "b h t d"
        else:
            layout = "b t h d"
        dtype = query.dtype
        if self._is_streaming:
            assert self.causal or self.cross_attention, \
                "Streaming only available for causal or cross attention"

        if self.causal:
            # At the moment we specialize only for the self-attention case.
            assert query.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            assert value.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            attn_mask = self._get_mask(query.shape[1], query.device, query.dtype)

        if self.custom:
            # custom implementation
            assert need_weights is False
            assert key_padding_mask is None
            if self.cross_attention:
                # Different queries, keys, values, we have to spit manually the weights
                # before applying the linear.
                dim = self.in_proj_weight.shape[0] // 3
                if self.in_proj_bias is None:
                    bias_q, bias_k, bias_v = None, None, None
                else:
                    bias_q = self.in_proj_bias[:dim]
                    bias_k = self.in_proj_bias[dim: 2 * dim]
                    bias_v = self.in_proj_bias[2 * dim:]
                q = nn.functional.linear(query, self.in_proj_weight[:dim], bias_q)
                # todo: when streaming, we could actually save k, v and check the shape actually match.
                k = nn.functional.linear(key, self.in_proj_weight[dim: 2 * dim], bias_k)
                v = nn.functional.linear(value, self.in_proj_weight[2 * dim:], bias_v)
                if self.qk_layer_norm is True:
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                q, k, v = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k, v]]
            else:
                # profiling breaks that propertysomehow.
                assert query is key, "specialized implementation"
                assert value is key, "specialized implementation"
                projected = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
                if self.kv_repeat == 1:
                    if time_dim == 2:
                        bound_layout = "b h p t d"
                    else:
                        bound_layout = "b t p h d"
                    packed = rearrange(projected, f"b t (p h d) -> {bound_layout}", p=3, h=self.num_heads)
                    q, k, v = ops.unbind(packed, dim=2)
                else:
                    embed_dim = self.embed_dim
                    per_head_dim = (embed_dim // self.num_heads)
                    kv_heads = self.num_heads // self.kv_repeat
                    q = projected[:, :, :embed_dim]
                    start = embed_dim
                    end = start + per_head_dim * kv_heads
                    k = projected[:, :, start: end]
                    v = projected[:, :, end:]
                    q = rearrange(q, f"b t (h d) -> {layout}", h=self.num_heads)
                    k = rearrange(k, f"b t (h d) -> {layout}", h=kv_heads)
                    v = rearrange(v, f"b t (h d) -> {layout}", h=kv_heads)

                if self.qk_layer_norm is True:
                    assert self.kv_repeat == 1
                    q, k = [rearrange(x, f"{layout} -> b t (h d)") for x in [q, k]]
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                    q, k = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k]]
                k, v = self._complete_kv(k, v)
                if self.kv_repeat > 1:
                    k = expand_repeated_kv(k, self.kv_repeat)
                    v = expand_repeated_kv(v, self.kv_repeat)
            if self.attention_as_float32:
                q, k, v = [x.float() for x in [q, k, v]]
            else:
                # We include the dot product as float32, for consistency
                # with the other implementations that include that step
                # as part of the attention. Note that when using `autocast`,
                # the einsums would be done as bfloat16, but the softmax
                # would be done as bfloat16, so `attention_as_float32` will
                # extend a bit the range of operations done in float32,
                # although this should make no difference.
                q = q / q.shape[-1] ** 0.5
                key_layout = layout.replace('t', 'k')
                query_layout = layout
                if self._is_streaming and self.safe_streaming and q.device.type == 'cuda':
                    with torch.autocast(device_type=q.device.type, dtype=torch.float32):
                        pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                else:
                    pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                if attn_mask is not None:
                    pre_w = pre_w + attn_mask
                w = torch.softmax(pre_w, dim=-1)
                w = F.dropout(w, self.dropout, training=self.training).to(v)
                # Key and value have the same format.
                x = torch.einsum(f"b h t k, {key_layout} -> {layout}", w, v)
            x = x.to(dtype)
            x = rearrange(x, f"{layout} -> b t (h d)", h=self.num_heads)
            x = self.out_proj(x)
        else:
            key, value = self._complete_kv(key, value)
            if self.attention_as_float32:
                query, key, value = [x.float() for x in [query, key, value]]
            x, _ = self.mha(
                query, key, value, key_padding_mask,
                need_weights, attn_mask, average_attn_weights)
            x = x.to(dtype)

        return x, None

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024)
        # self.self_attn = nn.MultiheadAttention(
        #     embed_dim=1024,
        #     num_heads=16,
        #     bias=False,
        #     batch_first=True
        # )
        self.self_attn = MultiheadAttention(
            embed_dim=1024,
            causal=True, # big difference!
            # past_context=None,
            # memory_efficient=True, # negligble difference, but a difference
            # attention_as_float32=False,
            # rope=None,
            cross_attention=False,
            # safe_streaming=True,
            num_heads=16,
            dropout=0.0,
            # kv_repeat=1,
            bias=False,
            custom=True
        )

    def forward(self, x, cross_attention_src):
        x = self.norm1(x)
        return self.self_attn(x, x, x)[0]

class Transformer(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 24, ff_dim: int = 4096, heads: int = 16):
        super().__init__()

        # self.layers = nn.ModuleList([
        #     nn.TransformerEncoderLayer(
        #         d_model=dim, dim_feedforward=ff_dim, nhead=heads,
        #         dropout=0.0,
        #         # memory_format=torch.channels_last, 
        #         # dtype=torch.float16,
        #         # layout='TN',
        #         # device='cuda',
        #         activation=F.gelu,
        #         batch_first=True,
        #         norm_first=True,
        #         # pin_memory=True,
        #         # requires_grad=True,
        #     ) for _ in range(depth)
        # ])
        # self.layers = nn.ModuleList([
        #     StreamingTransformerLayer(
        #         d_model=dim, dim_feedforward=ff_dim, num_heads=heads,
        #         dropout=0.0,
        #         # memory_format=torch.channels_last, 
        #         # dtype=torch.float16,
        #         # layout='TN',
        #         # device='cuda',
        #         activation=F.gelu,
        #         bias_ff=False,
        #         bias_attn=False,
        #         # batch_first=True,
        #         norm_first=True,
        #         cross_attention=True,
        #         # pin_memory=True,
        #         # requires_grad=True,
        #     ) for _ in range(depth)
        # ])
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(depth)
        ])
        # self.layers = model.lm.transformer.layers

    def forward(self, x, cross_attention_src):
        # TODO: undo stupid naming
        z=cross_attention_src

        # x = x.transpose(0, 1)
        # z = z.transpose(0, 1)
        # x.shape = [T, B, dim]
        # x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, cross_attention_src=z)
        # x = x.transpose(0, 1)
        # z = z.transpose(0, 1)
        # x.shape = [B, T, dim]
        return x
    
    def copy_weights(self):
        reference = model.lm.transformer.layers
        for i, layer in enumerate(self.layers):
            # odict_keys(['self_attn.in_proj_weight', 'self_attn.out_proj.weight', 'linear1.weight', 'linear2.weight', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'cross_attention.in_proj_weight', 'cross_attention.out_proj.weight', 'norm_cross.weight', 'norm_cross.bias'])
            # Missing key(s) in state_dict: "self_attn.mha.in_proj_bias", "self_attn.mha.out_proj.bias", "linear1.bias", "linear2.bias", "cross_attention.mha.in_proj_bias", "cross_attention.mha.out_proj.bias"
            # print(reference[i].state_dict().keys())
            layer_state_dict = layer.state_dict()
            reference_state_dict = {}
            reference_state_dict_full = reference[i].state_dict()
            for k in reference_state_dict_full:
                if k in layer_state_dict:
                    # print("copying", k)
                    reference_state_dict[k] = reference_state_dict_full[k]
            # print("reference_state_dict", reference_state_dict)
            self.layers[i].load_state_dict(reference_state_dict)
    
their_transformer = model.lm.transformer.float()
my_transformer = Transformer().cuda().float()
my_transformer.copy_weights()
sample = torch.randn(1, 4, 1024).cuda()

# print("their_transformer.layers[0].self_attn.weight.data", their_transformer.layers[0].self_attn.weight.data)
# print("my_transformer.layers[0].self_attn.weight.data", my_transformer.layers[0].self_attn.weight.data)

with torch.no_grad():
    their_out = their_transformer(sample, cross_attention_src=sample)
    my_out = my_transformer(sample, cross_attention_src=sample)
    diff = (their_out.half() - my_out.half()).abs().sum()
    print("diff", diff)