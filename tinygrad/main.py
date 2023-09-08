from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from audiocraft.models import EncodecModel
from audiocraft.data.audio import audio_write
from tqdm import trange
import random
import numpy as np
from musicgen import MusicGen
import torch

# set seed
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

musicgen = MusicGen()
musicgen.load_pretrained()

def sample_musicgen():
    Tensor.training = False
    TOP_K = 250
    TEMPERATURE = 1.0
    SPECIAL_TOKEN_ID = 2048
    
    tokens = np.ones((2, 4, 1), dtype=np.int32) * SPECIAL_TOKEN_ID
    for i in trange(75 + 3):
        # if(i <= 3):
        #     # Unsure if this is neccessary
        #     tokens[:,0:,0:1] = SPECIAL_TOKEN_ID
        #     tokens[:,1:,0:2] = SPECIAL_TOKEN_ID
        #     tokens[:,2:,0:3] = SPECIAL_TOKEN_ID
        #     tokens[:,3:,0:4] = SPECIAL_TOKEN_ID
        
        logits = musicgen(Tensor(tokens)).realize().numpy()
        topk, indices = torch.tensor(logits[:, :, -1, :]).topk(TOP_K, dim=-1)
        topk = torch.softmax((topk / TEMPERATURE), dim=-1)
        samples = torch.multinomial(topk.view((-1, TOP_K)), 1).view(topk.shape[:-1] + (1,))
        new_tokens = torch.gather(indices, dim=-1, index=samples).numpy()
        tokens = np.concatenate([tokens, new_tokens], axis=2)

    tokens = Tensor.stack([
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
    tokens = torch.tensor(tokens.realize().numpy()).cuda()

    # Still need compression model from fb
    encodec = EncodecModel.get_pretrained('facebook/encodec_32khz').cuda()
    manual_audio = encodec.decode(tokens)
    audio_write('new', manual_audio[0].cpu(), sample_rate=32000)
#     for i in range(8):
#         tokens = sample_musicgen()
#         manual_audio = model.compression_model.decode(tokens)
#         audio_write('louis' + str(i), manual_audio[0].cpu(), sample_rate=32000)

#     for i in range(8):
#         _, tokens = model.generate_unconditional(1, progress=True, return_tokens=True)
#         manual_audio = model.compression_model.decode(tokens)
#         audio_write('zuckerberg' + str(i), manual_audio[0].cpu(), sample_rate=32000)