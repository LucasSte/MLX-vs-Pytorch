from dataset.simple_transformers import load_ptb
from mlx_models.BasicLM import train as mlx_train
from pytorch_models.BasicLM import train as pytorch_train
from utils.initializer import initialize
import numpy as np
import time

context_size = 1024
num_blocks = 12
dim = 1024
num_heads = 16
epochs = 5
learning_rate = 3e-4
weight_decay = 1e-5
lr_warmup = 200
batch_size = 32

if __name__ == "__main__":
    args, times = initialize()

    data = load_ptb()

    for i in range(0, args.iter):
        if args.framework == "mlx":
            start = time.time()
            mlx_train(
                num_blocks,
                batch_size,
                context_size,
                dim,
                num_heads,
                False,
                learning_rate,
                weight_decay,
                epochs,
                lr_warmup,
                data,
            )
            end = time.time()
            elapsed = end - start
            print(f"MLX time: {elapsed}s")
            times[i] = elapsed
        else:
            start = time.time()
            pytorch_train(
                num_blocks,
                batch_size,
                context_size,
                dim,
                num_heads,
                False,
                learning_rate,
                weight_decay,
                epochs,
                lr_warmup,
                data,
            )
            end = time.time()
            elapsed = end - start
            print(f"Pytorch time: {elapsed}s")
            times[i] = elapsed

    print(f"\nLLM train test: ran {args.iter} times")
    print(
        f"Framework: {args.framework}\n\tAverage: {np.mean(times)}s - Median: {np.median(times)}s"
    )
