import argparse
from mlx_models.switch import run_test as mlx_run
from pytorch_models.switch import run_test as pytorch_run


iterations = 50
size = 10000
filename = "_switch.txt"

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
        mlx_run(iterations, size, "mlx" + filename)
    else:
        pytorch_run(iterations, size, "pytorch" + filename)
