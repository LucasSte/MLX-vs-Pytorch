import argparse
import numpy as np


def initialize():
    parser = argparse.ArgumentParser(description="Language model training benchmark")
    parser.add_argument(
        "--framework",
        help="Which framework to use: pytorch or mlx",
        type=str,
    )
    parser.add_argument(
        "--iter", help="How many times to run the test", default=1, type=int
    )

    args = parser.parse_args()
    times = np.zeros(args.iter)

    return args, times
