# File copied (and slightly modified) from: https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py

from dataset.simple_transformers import load_ptb
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            dims: int,
            num_heads: int,
            checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        l_shape = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(l_shape))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s: s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def train(num_blocks, dim, num_heads, checkpoint, learning_rate, weight_decay):
    vocab, train, valid, test = load_ptb()

    model = TransformerLM(
        len(vocab), num_blocks, dim, num_heads, checkpoint
    )
    mx.eval(model.parameters())

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )