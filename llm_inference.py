from mlx_models.tiny_llama import MLXLlama
import argparse
import time

out_filename = "llm_out.txt"

prompts = [
    "How to get in a good university?",
    "What is artificial intelligence?",
    "What is a computer?",
    "How to land in an awesome job?",
    "How to change the world?",
]

temp = 0.0
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
        mlx_model = MLXLlama(max_tokens, temp, "mlx" + out_filename)
        start = time.time()
        for item in prompts:
            mlx_model.generate_and_save(item)
        end = time.time()
        print(f"MLX time: {end - start}s")
