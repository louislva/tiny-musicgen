from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5)
_, tokens = model.generate_unconditional(1, progress=True, return_tokens=True)
manual_audio = model.compression_model.decode(tokens)
audio_write('manual_audio', manual_audio[0].cpu(), sample_rate=32000)