from mlx_models.tiny_llama import MLXLlama
from pytorch_models.tiny_llama import TorchLLama
from utils.initializer import initialize
import numpy as np
import time

out_filename = "_llm_out.txt"

prompts = [
    "How to get in a good university?",
    "What is artificial intelligence?",
    "What is a computer?",
    "How to land in an awesome job?",
    "How to change the world?",
]

max_tokens = 1024


def run_model(model) -> float:
    start = time.time()
    for item in prompts:
        model.generate_and_save(item)
    end = time.time()

    return end - start


if __name__ == "__main__":
    args, times = initialize()

    for i in range(0, args.iter):
        if args.framework == "mlx":
            mlx_model = MLXLlama(max_tokens, "mlx" + out_filename)
            elapsed = run_model(mlx_model)
            mlx_model.finish()
            times[i] = elapsed
            print(f"MLX time: {elapsed}s")
        else:
            torch_model = TorchLLama(max_tokens, "pytorch" + out_filename)
            elapsed = run_model(torch_model)
            torch_model.finish()

            times[i] = elapsed
            print(f"Pytorch time: {elapsed}s")

    print(f"\nLLM inference test: ran {args.iter} times")
    print(
        f"Framework: {args.framework}\n\tAverage: {np.mean(times)}s - Median: {np.median(times)}s"
    )
