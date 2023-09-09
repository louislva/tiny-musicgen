import torch
from tinygrad.tensor import Tensor
from tqdm import trange
import random
import numpy as np
from musicgen import MusicGen
from torch_compression import Encodec as TorchEncodec
import torchaudio

# set seed
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

musicgen = MusicGen()
musicgen.load_pretrained()
encodec = TorchEncodec()
encodec.load_pretrained('cpu')

def sample_musicgen():
    Tensor.training = False
    TOP_K = 250
    TEMPERATURE = 1.0
    SPECIAL_TOKEN_ID = 2048
    
    tokens = np.ones((2, 4, 1), dtype=np.int32) * SPECIAL_TOKEN_ID
    for i in trange(75 + 3):
        logits = musicgen(Tensor(tokens)).realize().numpy()
        topk, indices = torch.tensor(logits[:, :, -1, :]).topk(TOP_K, dim=-1)
        topk = torch.softmax((topk / TEMPERATURE), dim=-1)
        samples = torch.multinomial(topk.view((-1, TOP_K)), 1).view(topk.shape[:-1] + (1,))
        new_tokens = torch.gather(indices, dim=-1, index=samples).numpy()
        tokens = np.concatenate([tokens, new_tokens], axis=2)

    tokens = np.stack([
        tokens[:,0,0:-3],
        tokens[:,1,1:-2],
        tokens[:,2,2:-1],
        tokens[:,3,3:],
    ], axis=1)[:,:,1:]

    return tokens

if __name__ == '__main__':
    # set seed to SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate
    tokens = sample_musicgen()

    # The compression model (for converting from tokens to audio) is implemented only in PyTorch
    # Simple reason was that Tinygrad doesn't support LSTMs, and no way in hell I'm rewriting that
    manual_audio = encodec.decode(torch.tensor(tokens))
    torchaudio.save('new.wav', manual_audio[0].cpu(), sample_rate=32000)