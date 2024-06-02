from utils.initializer import initialize
import time
from pytorch_models.whisper import TorchWhisper
from mlx_models.whisper import MLXWhisper
import numpy as np


num_examples = 70
out_filename = "_whisper.txt"

if __name__ == "__main__":
    args, times = initialize()

    for i in range(args.iter):
        if args.framework == "mlx":
            mlx_model = MLXWhisper(num_examples, "mlx" + out_filename)
            start = time.time()
            mlx_model.generate()
            end = time.time()
            mlx_model.finish()
            elapsed = end - start
            print(f"MLX time: {elapsed}s")
            times[i] = elapsed
        else:
            torch_model = TorchWhisper(num_examples, "pytorch" + out_filename)
            start = time.time()
            torch_model.generate()
            end = time.time()
            torch_model.finish()
            elapsed = end - start
            print(f"Pytorch time: {elapsed}s")
            times[i] = elapsed

    print(f"\nWhisper inference test: ran {args.iter} times")
    print(
        f"Framework: {args.framework}\n\tAverage: {np.mean(times)}s - Median: {np.median(times)}s"
    )
