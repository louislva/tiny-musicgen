import torch
import torch.nn.functional as F
from tqdm import trange
import random
import numpy as np
from musicgen import MusicGen
from compression import Encodec
import torchaudio

DEVICE = 'cpu'

# set seed
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

musicgen = MusicGen()
musicgen = musicgen.to(DEVICE)
musicgen.load_pretrained(DEVICE)
encodec = Encodec()
encodec = encodec.to(DEVICE)
encodec.load_pretrained(DEVICE)

def sample_musicgen():
    TOP_K = 250
    TEMPERATURE = 1.0
    SPECIAL_TOKEN_ID = 2048
    
    tokens = (torch.ones(2, 4, 1).long() * SPECIAL_TOKEN_ID).to(DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in trange(10 + 3):
                if(i <= 3):
                    # Unsure if this is neccessary
                    tokens[:,0:,0:1] = SPECIAL_TOKEN_ID
                    tokens[:,1:,0:2] = SPECIAL_TOKEN_ID
                    tokens[:,2:,0:3] = SPECIAL_TOKEN_ID
                    tokens[:,3:,0:4] = SPECIAL_TOKEN_ID
                
                logits = musicgen.forward(tokens)
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
    tokens = sample_musicgen()

    # Still need compression model from fb
    manual_audio = encodec.decode(tokens)
    torchaudio.save('new-ta.wav', manual_audio[0].cpu(), 32000)