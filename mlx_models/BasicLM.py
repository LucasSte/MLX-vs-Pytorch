# File copied (and slightly modified) from: https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py

from dataset.simple_transformers import load_ptb
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from functools import partial
import math


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
        mask = nn.MultiHeadAttention.create_additive_causal_mask(l_shape)
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
    mx.eval(model.parameters())

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def eval_fn(dataset):
        inputs, targets = map(mx.array, to_samples(context_size, dataset))
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s: s + batch_size], targets[s: s + batch_size]
            bx, by = map(mx.array, (bx, by))
            losses = loss_fn(model, bx, by, reduce=False)
            loss += mx.sum(losses).item()
        return loss / len(targets)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    for it, (inputs, targets) in zip(range(num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        optimizer.learning_rate = min(1, it / lr_warmup) * learning_rate
        loss = step(inputs, targets)
        mx.eval(state)
        losses.append(loss.item())
        # TODO: Remove prints when everything is working.
        # concatenate everything to an array of strings and print at the end.
        if (it + 1) % 5 == 0:
            train_loss = np.mean(losses)
            print(f"Iter {it + 1}: Train loss {train_loss:.3f}, ")
            val_loss = eval_fn(valid)
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
            )

    test_loss = eval_fn(test)
    test_ppl = math.exp(test_loss)
    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
