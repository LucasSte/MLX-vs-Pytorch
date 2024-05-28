import argparse
import time
from pytorch_models.whisper import TorchWhisper
from mlx_models.whisper import MLXWhisper


iterations = 50
out_filename = "_whisper.txt"

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
        mlx_model = MLXWhisper(iterations, "mlx" + out_filename)
        start = time.time()
        mlx_model.generate()
        end = time.time()
        mlx_model.finish()
        print(f"MLX time: {end - start}s")
    else:
        torch_model = TorchWhisper(iterations, "pytorch" + out_filename)
        start = time.time()
        torch_model.generate()
        end = time.time()
        torch_model.finish()
        print(f"Pytorch time: {end - start}s")
