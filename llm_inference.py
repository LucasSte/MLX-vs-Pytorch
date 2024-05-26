from mlx_models.tiny_llama import MLXLlama
from pytorch_models.tiny_llama import TorchLLama
import argparse
import time

out_filename = "_llm_out.txt"

prompts = [
    "How to get in a good university?",
    # "What is artificial intelligence?",
    # "What is a computer?",
    # "How to land in an awesome job?",
    # "How to change the world?",
]

temp = 0.7
sample = True
top_k = 50
top_p = 0.95
max_tokens = 1024


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

    if args.framework == "mlx":
        # TODO: Update the mlx case
        mlx_model = MLXLlama(max_tokens, temp, "mlx" + out_filename)
        start = time.time()
        for item in prompts:
            mlx_model.generate_and_save(item)
        end = time.time()
        mlx_model.finish()
        print(f"MLX time: {end - start}s")
    else:
        torch_model = TorchLLama(
            max_tokens, temp, top_k, top_p, sample, "torch" + out_filename
        )
        start = time.time()
        for item in prompts:
            torch_model.generate_and_save(item)
        end = time.time()
        torch_model.finish()
        print(f"Pytorch time: {end - start}s")
