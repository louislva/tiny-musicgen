import torch
from torch import nn
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.models.lm import LMModel
from tqdm import trange

DEBUG = False

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5, top_k=2048)
print(model.lm.transformer)

class LouisGen(nn.Module):
    def __init__(self, lm: LMModel):
        super().__init__()
        
        # Copying layers over
        self.cfg_dropout = lm.cfg_dropout
        self.att_dropout = lm.att_dropout
        # self.condition_provider = lm.condition_provider
        self.fuser = lm.fuser
        self.emb = nn.ModuleList([
            lm.emb[0],
            lm.emb[1],
            lm.emb[2],
            lm.emb[3],
        ])
        self.transformer = lm.transformer
        self.out_norm = lm.out_norm
        self.linears = nn.ModuleList([
            lm.linears[0],
            lm.linears[1],
            lm.linears[2],
            lm.linears[3],
        ])

    def forward(self, x):
        # x.shape = [B, Q, T]
        DEBUG and print("0", x.shape)

        x = self.emb[0](x[:,0,:]) + \
            self.emb[1](x[:,1,:]) + \
            self.emb[2](x[:,2,:]) + \
            self.emb[3](x[:,3,:])
        # x.shape = [B, T, d_model]
        DEBUG and print("A", x.shape)

        # the zeros had a ?batch_sze+?? of 2 not 1 in meta's code, so need to look into what that was
        x, z = self.fuser(x, {
            "description": (torch.zeros(1, 1, 1024).to(x.device), torch.zeros(1, 1).to(x.device)),
        })
        # z.shape = [B, 2, d_model]
        DEBUG and print("A2.x", x.shape)
        DEBUG and print("A2.z", z.shape)

        x = self.transformer.forward(x, cross_attention_src=z) # torch.zeros(x.shape[0], 0, x.shape[2], device=x.device))
        # x.shape = [B, T, d_model]
        DEBUG and print("B", x.shape)

        x = self.out_norm(x)
        # x.shape = [B, T, d_model]
        DEBUG and print("C", x.shape)

        x = torch.stack([
            self.linears[0](x),
            self.linears[1](x),
            self.linears[2](x),
            self.linears[3](x),
        ], dim=1)
        # x.shape = [Q, B, T, codebook_size]
        DEBUG and print("D", x.shape)
        
        # x.shape = [B, Q, T, codebook_size]
        DEBUG and print("E", x.shape)

        return x

def sample_louisgen(model):
    louisgen = LouisGen(model.lm)

    tokens = (torch.ones(1, 4, 1).long() * model.lm.special_token_id).cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in trange(250 + 3):
                # x.shape = [B, Q, T]
                DEBUG and print("input", tokens.shape)
                x = louisgen.forward(tokens)
                probs = (x).softmax(dim=-1)
                new_tokens = probs[:, :, -1, :].view((-1, probs.shape[-1])).multinomial(num_samples=1).view((x.shape[0], x.shape[1], 1))
                # x.shape = [B, Q, T]
                tokens = torch.cat([tokens, new_tokens], dim=-1)

    tokens = torch.stack([
        tokens[:,0,0:-3],
        tokens[:,1,1:-2],
        tokens[:,2,2:-1],
        tokens[:,3,3:],
    ], dim=1)[:,:,1:]

    return tokens

tokens = sample_louisgen(model)
# _, tokens = model.generate_unconditional(1, progress=True, return_tokens=True)

manual_audio = model.compression_model.decode(tokens)
audio_write('manual_audio', manual_audio[0].cpu(), sample_rate=32000)