import torch
from torch import nn
from transformer import Transformer
from huggingface_hub import hf_hub_download

class MusicGen(nn.Module):
    def __init__(self, depth=24, codebook_count=4, codebook_size=2048, dim=1024):
        super().__init__()

        self.codebook_count = codebook_count
        self.codebook_size = codebook_size
        self.dim = dim
        self.depth = depth

        self.emb = nn.ModuleList([nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)])
        self.transformer = Transformer(dim=dim, depth=24)
        self.out_norm = nn.LayerNorm(dim)
        self.linears = nn.ModuleList([nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)])

    def forward(self, x):
        x = sum([emb(x[:,i,:]) for i, emb in enumerate(self.emb)])
        x = self.transformer(x, cross_attention_src=torch.zeros((2, 1, 1024)).to(x.device))
        x = self.out_norm(x)
        x = torch.stack([linear(x) for linear in self.linears], dim=1)
        return x
    
    def load_pretrained(self, device, model='facebook/musicgen-small'):
        path = hf_hub_download(repo_id=model, filename='state_dict.bin', cache_dir=None)
        _values = torch.load(path, map_location=device)
        state_dict = {
            k: v for k, v in _values["best_state"].items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict)