from dataset.BERT import load_nli
import argparse
from pytorch_models.MiniBERT import train as pytorch_train
from mlx_models.MiniBERT import train as mlx_train
import time

num_epochs = 3
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
    parser = argparse.ArgumentParser(description="Language model training benchmark")
    parser.add_argument(
        "--framework",
        help="Which framework to use: pytorch or mlx",
        type=str,
    )

    args = parser.parse_args()

    if args.framework not in ["pytorch", "mlx"]:
        raise Exception("Unexpected option")

    dataset = load_nli()

    if args.framework == "mlx":
        start = time.time()
        mlx_train(num_epochs, batch_size, num_labels, bert_config, lr, dataset)
        end = time.time()
        print(f"MLX time: {end - start}s")
    else:
        start = time.time()
        pytorch_train(num_epochs, batch_size, num_labels, bert_config, lr, dataset)
        end = time.time()
        print(f"Pytorch time: {end - start}s")
