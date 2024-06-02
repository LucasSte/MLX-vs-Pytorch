from dataset.BERT import load_nli
from utils.initializer import initialize
from pytorch_models.MiniBERT import train as pytorch_train
from mlx_models.MiniBERT import train as mlx_train
import numpy as np
import time

num_epochs = 5
batch_size = 8
num_labels = 3
lr = 5e-5
bert_config = {
    "hidden_size": 128,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 512,
    "vocab_size": 30522,
}

if __name__ == "__main__":
    args, times = initialize()

    dataset = load_nli()

    for i in range(0, args.iter):
        if args.framework == "mlx":
            start = time.time()
            mlx_train(num_epochs, batch_size, num_labels, bert_config, lr, dataset)
            end = time.time()

            elapsed = end - start
            times[i] = elapsed
            print(f"MLX time: {elapsed}s")
        else:
            start = time.time()
            pytorch_train(num_epochs, batch_size, num_labels, bert_config, lr, dataset)
            end = time.time()

            elapsed = end - start
            times[i] = elapsed
            print(f"Pytorch time: {elapsed}s")

    print(f"\nBERT fine tune test: ran {args.iter} times")
    print(
        f"Framework: {args.framework}\n\tAverage: {np.mean(times)}s - Median: {np.median(times)}s"
    )
