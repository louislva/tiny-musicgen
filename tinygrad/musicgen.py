from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, get_state_dict
from tinygrad.helpers import dtypes
from transformer import Transformer
from huggingface_hub import hf_hub_download
import torch

class MusicGen():
    def __init__(self, depth=24, codebook_count=4, codebook_size=2048, dim=1024):
        super().__init__()

        self.codebook_count = codebook_count
        self.codebook_size = codebook_size
        self.dim = dim
        self.depth = depth
        
        self.emb = [nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)]
        self.transformer = Transformer(dim=dim, depth=24)
        self.out_norm = nn.LayerNorm(dim)
        self.linears = [nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)]

    def __call__(self, x):
        x = sum([emb(x[:,i,:]) for i, emb in enumerate(self.emb)])
        x = self.transformer(x, cross_attention_src=Tensor.zeros(1, 1, 1024).to(x.device))
        x = self.out_norm(x)
        x = Tensor.stack([linear(x) for linear in self.linears], dim=1)
        return x
    
    def load_pretrained(self, model='facebook/musicgen-small'):
        path = hf_hub_download(repo_id=model, filename='state_dict.bin', cache_dir=None)
        _state_dict = torch.load(path, map_location='cpu')
        self_state_dict = get_state_dict(self)
        state_dict = {
            k: Tensor(v.cpu().detach().numpy()) for k, v in _state_dict["best_state"].items() if k in self_state_dict
        }
        load_state_dict(self, state_dict)

if __name__ == "__main__":
    print("hello")
    musicgen = MusicGen()
    x = Tensor.zeros([1, 4, 5], dtype=dtypes.int32).to("cuda")
    y = musicgen(x).realize()
    print(y)
    x = Tensor.zeros([1, 4, 5], dtype=dtypes.int32).to("cuda")
    y = musicgen(x).realize()
    print(y)
    x = Tensor.zeros([1, 4, 5], dtype=dtypes.int32).to("cuda")
    y = musicgen(x).realize()
    print(y)