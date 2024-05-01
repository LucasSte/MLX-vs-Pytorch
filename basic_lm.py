from dataset.simple_transformers import load_ptb
from mlx_models.BasicLM import train as mlx_train
from pytorch_models.BasicLM import train as pytorch_train
import time

context_size = 1024
num_blocks = 12
dim = 1024
num_heads = 16
num_iters = 2
learning_rate = 3e-4
weight_decay = 1e-5
lr_warmup = 200
batch_size = 32

if __name__ == '__main__':
    data = load_ptb()

    start = time.time()
    mlx_train(num_blocks, batch_size, context_size, dim, num_heads, False,
              learning_rate, weight_decay, num_iters, lr_warmup, data)
    end = time.time()
    print(f"MLX time: {end - start}s")

    # import torch
    # emb = torch.nn.Embedding(32, 2, device='mps')
    # print(emb)
    # a = torch.tensor([0, 1], dtype=torch.int32)
    # print(emb(a))

