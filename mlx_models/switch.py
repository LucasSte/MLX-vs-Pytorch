import mlx.core as mx
import time


def multiply_and_divide(op1: mx.array, op2: mx.array, stream) -> mx.array:
    mult = mx.matmul(op1, op2, stream=stream)
    div = mx.divide(mult, mult.max(0, stream=stream), stream=stream)
    return div


def multiply_items(op1: mx.array, op2: mx.array, op3: mx.array, stream):
    res_1 = multiply_and_divide(op1, op2, stream)
    res_2 = multiply_and_divide(op2, op3, stream)
    res_3 = multiply_and_divide(op3, op1, stream)
    return res_1, res_2, res_3


def run_test(iterations: int, size: int, filename: str) -> float:
    a = mx.random.uniform(shape=(size, size), stream=mx.cpu, dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), stream=mx.cpu, dtype=mx.float32)
    c = mx.random.uniform(shape=(size, size), stream=mx.cpu, dtype=mx.float32)

    start = time.time()

    for _ in range(0, iterations):
        mps_1, mps_2, mps_3 = multiply_items(a, b, c, mx.gpu)
        mx.eval(mps_1, mps_2, mps_3)
        a, b, c = multiply_items(mps_1, mps_2, mps_3, mx.cpu)
        mx.eval(a, b, c)

    end = time.time()

    with open(filename, "w") as file:
        print(a, file=file, flush=True)
        print(b, file=file, flush=True)
        print(c, file=file, flush=True)

    duration = end - start
    print(f"MLX time: {duration}")
    return duration
