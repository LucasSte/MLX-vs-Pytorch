from utils.initializer import initialize
from mlx_models.switch import run_test as mlx_run
from pytorch_models.switch import run_test as pytorch_run
import numpy as np


loop_times = 50
size = 10000
filename = "_switch.txt"

if __name__ == "__main__":
    args, times = initialize()

    for i in range(args.iter):
        if args.framework == "mlx":
            times[i] = mlx_run(loop_times, size, "mlx" + filename)
        else:
            times[i] = pytorch_run(loop_times, size, "pytorch" + filename)

    print(f"\nSwitch test: ran {args.iter} times")
    print(
        f"Framework: {args.framework}\n\tAverage: {np.mean(times)}s - Median: {np.median(times)}s"
    )
