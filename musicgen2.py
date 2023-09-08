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
from multiheadattention import MultiheadAttention

# set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5, top_k=2048)

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024)
        self.self_attn = MultiheadAttention(
            embed_dim=1024,
            causal=True, # big difference!
            cross_attention=False,
            num_heads=16,
            dropout=0.0,
            bias=False,
            custom=True
        )

    def forward(self, x, cross_attention_src):
        x_ = self.norm1(x)
        x = x + self.self_attn(x_, x_, x_)[0]
        return x

class Transformer(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 24, ff_dim: int = 4096, heads: int = 16):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(depth)
        ])

    def forward(self, x, cross_attention_src):
        # x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, cross_attention_src=cross_attention_src)
        return x
    
    def copy_weights(self):
        reference = model.lm.transformer.layers
        for i, layer in enumerate(self.layers):
            # odict_keys(['self_attn.in_proj_weight', 'self_attn.out_proj.weight', 'linear1.weight', 'linear2.weight', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'cross_attention.in_proj_weight', 'cross_attention.out_proj.weight', 'norm_cross.weight', 'norm_cross.bias'])
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

with torch.no_grad():
    their_out = their_transformer(sample, cross_attention_src=sample)
    my_out = my_transformer(sample, cross_attention_src=sample)
    diff = (their_out.half() - my_out.half()).abs().sum()
    print("diff", diff)