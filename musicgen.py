import torch
from torch import nn
import torch.nn.functional as F
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.models.lm import LMModel
from tqdm import trange
import random
import numpy as np
from transformer import Transformer

DEBUG = False

# set seed
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# The model
class LouisGen(nn.Module):
    def __init__(self, lm: LMModel, depth=24, codebook_count=4, codebook_size=2048, dim=1024):
        super().__init__()

        self.codebook_count = codebook_count
        self.codebook_size = codebook_size
        self.dim = dim
        self.depth = depth
        
        # Copying layers over
        # self.cfg_dropout = lm.cfg_dropout
        # self.att_dropout = lm.att_dropout
        # self.condition_provider = lm.condition_provider
        self.fuser = lm.fuser
        self.emb = nn.ModuleList([nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)])
        self.transformer = Transformer(dim=dim, depth=24)
        self.out_norm = nn.LayerNorm(dim)
        self.linears = nn.ModuleList([nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)])

    def forward(self, x):
        # TODO: masking for transformer training
        # x.shape = [B, C, T]
        DEBUG and print("0", x.shape)

        x = self.emb[0](x[:,0,:]) + \
            self.emb[1](x[:,1,:]) + \
            self.emb[2](x[:,2,:]) + \
            self.emb[3](x[:,3,:])
        # x.shape = [B, T, dim]
        DEBUG and print("A", x.shape)

        # the zeros had a ?batch_sze+?? of 2 not 1 in meta's code, so need to look into what that was
        x, z = self.fuser(x, {
            "description": (torch.zeros(2, 1, 1024).to(x.device), torch.zeros(2, 1).to(x.device)),
        })
        # z.shape = [B, T, dim]
        DEBUG and print("A2.x", x.shape)
        DEBUG and print("A2.z", z.shape)

        x = self.transformer(x, cross_attention_src=z) # torch.zeros(x.shape[0], 0, x.shape[2], device=x.device))
        # x.shape = [B, T, dim]
        DEBUG and print("B", x.shape)

        x = self.out_norm(x)
        # x.shape = [B, T, dim]
        DEBUG and print("C", x.shape)

        x = torch.stack([
            self.linears[0](x),
            self.linears[1](x),
            self.linears[2](x),
            self.linears[3](x),
        ], dim=1)
        # x.shape = [C, B, T, codebook_size]
        DEBUG and print("D", x.shape)
        
        # x.shape = [B, C, T, codebook_size]
        DEBUG and print("E", x.shape)

        return x
    
    def load_pretrained(self):
        with torch.no_grad():
            path = '/home/louislva/.cache/huggingface/hub/models--facebook--musicgen-small/snapshots/2610ed09b7335026d4c2f977003a0dbc2c815272/state_dict.bin'
            _values = torch.load(path, map_location='cuda')
            values = _values["best_state"]
            for i in range(self.codebook_count):
                self.emb[i].weight.data.copy_(values[f"emb.{i}.weight"])
                self.linears[i].weight.data.copy_(values[f"linears.{i}.weight"])
            self.out_norm.weight.data.copy_(values["out_norm.weight"])
            self.out_norm.bias.data.copy_(values["out_norm.bias"])

            transformer_dict = {}
            for key in values:
                if(key.startswith("transformer")):
                    transformer_dict[key.replace("transformer.", "")] = values[key]
            self.transformer.load_state_dict(transformer_dict)

model = MusicGen.get_pretrained('facebook/musicgen-small')            
louisgen = LouisGen(model.lm)
louisgen = louisgen.cuda()
louisgen.load_pretrained()

def sample_louisgen():
    TOP_K = 250
    TEMPERATURE = 1.0
    SPECIAL_TOKEN_ID = 2048
    
    tokens = (torch.ones(2, 4, 1).long() * SPECIAL_TOKEN_ID).cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in trange(250 + 3):
                if(i <= 3):
                    # Unsure if this is neccessary
                    tokens[:,0:,0:1] = SPECIAL_TOKEN_ID
                    tokens[:,1:,0:2] = SPECIAL_TOKEN_ID
                    tokens[:,2:,0:3] = SPECIAL_TOKEN_ID
                    tokens[:,3:,0:4] = SPECIAL_TOKEN_ID
                
                logits = louisgen.forward(tokens)
                topk, indices = logits[:, :, -1, :].topk(TOP_K, dim=-1)
                topk = F.softmax((topk / TEMPERATURE), dim=-1)
                samples = torch.multinomial(topk.view((-1, TOP_K)), 1).view(topk.shape[:-1] + (1,))
                new_tokens = torch.gather(indices, dim=-1, index=samples)
                tokens = torch.cat([tokens, new_tokens], dim=2)

    tokens = torch.stack([
        tokens[:,0,0:-3],
        tokens[:,1,1:-2],
        tokens[:,2,2:-1],
        tokens[:,3,3:],
    ], dim=1)[:,:,1:]

    return tokens

if __name__ == '__main__':
    # set seed to SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    tokens = sample_louisgen()
    manual_audio = model.compression_model.decode(tokens)
    audio_write('new', manual_audio[0].cpu(), sample_rate=32000)
#     for i in range(8):
#         tokens = sample_louisgen()
#         manual_audio = model.compression_model.decode(tokens)
#         audio_write('louis' + str(i), manual_audio[0].cpu(), sample_rate=32000)

#     for i in range(8):
#         _, tokens = model.generate_unconditional(1, progress=True, return_tokens=True)
#         manual_audio = model.compression_model.decode(tokens)
#         audio_write('zuckerberg' + str(i), manual_audio[0].cpu(), sample_rate=32000)