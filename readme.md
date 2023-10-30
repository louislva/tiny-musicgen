# Tiny Musicgen

This codebase implements [MusicGen](https://arxiv.org/abs/2306.05284) in both [Torch](https://pytorch.org/) & [Tinygrad](https://github.com/tinygrad/tinygrad). Both implementations are about ~200 lines, depending on how you count. Should note, the Tinygrad implementation is unreasonably slow on NVIDIA, but can do 2 tokens / second on my M2 MacBook.

My aim was more to implement it very simply & readably, and so I've skipped any semblence of:

- Flexibility/reusability in the transformer
- Einsum, and other sorcery that confuses me
- Conditioning the model on genre / instruments / etc - this one will just autoregressively generate random songs

I originally didn't have caching or flash attention, but I've added that back in to make it usable for longer sequences.
