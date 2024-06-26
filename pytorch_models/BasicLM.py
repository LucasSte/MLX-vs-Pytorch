import torch
import torch.nn as nn
from typing import Optional
import math
from dataset.simple_transformers import load_ptb
from mlx_models.BasicLM import to_samples, iterate_batches
import numpy as np

mps_dev = torch.device('mps')


# Adapted from
# https://github.com/ml-explore/mlx/blob/c4a471c99d0c6e6b085ff944ffef149905296a14/python/mlx/nn/layers/positional_encoding.py#L57
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
        one_zero = 1 - torch.arange(0, dims // 2, device=mps_dev) / (dims // 2 - 1)
        min_freq = torch.log(torch.tensor(min_freq))
        max_freq = torch.log(torch.tensor(max_freq))

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
        checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = SinusoidalPositionalEncoding(dims)

        encoder_layer = nn.TransformerEncoderLayer(
            dims, nhead=num_heads, dim_feedforward=dims * 4, norm_first=True, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(dims, vocab_size)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: torch.dtype = torch.float32):
        # Adapted from
        # https://github.com/ml-explore/mlx/blob/c4a471c99d0c6e6b085ff944ffef149905296a14/python/mlx/nn/layers/transformer.py#L102
        indices = torch.arange(N, device=mps_dev)
        mask = indices[:, None] < indices[None]
        mask = mask.to(dtype) * -1e9
        return mask

    def forward(self, x):
        l_shape = x.shape[1]
        mask = self.create_additive_causal_mask(l_shape)
        x = self.embedding(x)
        x = x + self.pe(torch.arange(l_shape, device=mps_dev))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def mps_tensor(x):
    return torch.tensor(x, device=mps_dev)


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
    ptb_data
):
    vocab, train, valid, test = ptb_data
    model = TransformerLM(len(vocab), num_blocks, dim, num_heads, checkpoint).to(mps_dev)

    def loss_fn(model, x, y):
        logits = model(x)
        losses = nn.functional.cross_entropy(logits.permute(0, 2, 1), y)
        return losses

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def eval_fn(dataset):
        inputs, targets = to_samples(context_size, dataset)
        inputs = torch.tensor(inputs, dtype=torch.int32, device=mps_dev)
        targets = torch.tensor(targets, dtype=torch.int32, device=mps_dev)
        loss = 0
        model.train(False)
        with torch.no_grad():
            for s in range(0, min(targets.shape[0], 1024), batch_size):
                bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
                losses = loss_fn(model, bx, by)
                loss += torch.sum(losses).item()
        return loss / len(targets)

    lr_lambda = lambda epoch: min(1, epoch / lr_warmup) * learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # @torch.compile
    def step(inputs, targets, it):
        optimizer.zero_grad()
        loss = loss_fn(model, inputs, targets)
        loss.backward()
        scheduler.step(it)
        return loss

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    for it, (inputs, targets) in zip(range(num_iters), train_iterator):
        model.train(True)
        inputs = torch.tensor(inputs, dtype=torch.int32, device=mps_dev)
        targets = torch.tensor(targets, device=mps_dev, dtype=torch.int32)
        loss = step(inputs, targets, it)
        losses.append(loss.item())
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
