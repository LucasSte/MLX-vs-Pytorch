import torch
import torch.nn as nn
from typing import Optional
import math
from dataset.simple_transformers import load_ptb


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
            self,
            dims: int,
            min_freq: float = 0.0001,
            max_freq: float = 1,
            scale: Optional[float] = None,
            cos_first: bool = False,
            full_turns: bool = False,
    ):
        super().__init__()
        one_zero = 1 - torch.arange(0, dims // 2) / (dims // 2 - 1)
        min_freq = math.log(min_freq)
        max_freq = math.log(max_freq)

        self._sigmas = torch.exp(one_zero * (max_freq - min_freq) + min_freq)
        if full_turns:
            self._sigmas = self._sigmas * (2 * math.pi)

        self.scale = scale or (2 / dims) ** 0.5
        self.cos_first = cos_first

    def forward(self, x):
        y = x[..., None] * self._sigmas
        cosy = torch.cos(y)
        siny = torch.sin(y)

        if self.cos_first:
            y = torch.cat((cosy, siny), -1)
        else:
            y = torch.cat((siny, cosy), -1)

        if self.scale != 1:
            y = y * self.scale

        return y


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
            dims: int,
            num_heads: int,
            checkpoint: bool
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = SinusoidalPositionalEncoding(dims)

        encoder_layer = nn.TransformerEncoderLayer(
            dims, nhead=num_heads, dim_feedforward=dims*4, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: torch.dtype = torch.float32):
        indices = torch.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.to(dtype) * -1e9
        return mask

    def forward(self, x):
        l_shape = x.shape[1]
        mask = self.create_additive_causal_mask(l_shape)
        x = self.embedding(x)
        x = x + self.pe(torch.arange(l_shape))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def train(
    num_blocks,
    batch_size,
    context_size,
    dim,
    num_heads,
    checkpoint,
    learning_rate,
    weight_decay,
    num_iters,
    lr_warmup,
):
    vocab, train, valid, test = load_ptb()
    model = TransformerLM(len(vocab), num_blocks, dim, num_heads, checkpoint)

