import torch
import time


def multiply_and_divide(op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
    mult = torch.matmul(op1, op2)
    div = torch.divide(mult, torch.max(mult, 0).values)
    return div


def multiply_items(op1: torch.Tensor, op2: torch.Tensor, op3: torch.Tensor):
    r_1 = multiply_and_divide(op1, op2)
    r_2 = multiply_and_divide(op2, op3)
    r_3 = multiply_and_divide(op1, op3)
    return r_1, r_2, r_3


mps_device = torch.device("mps")
cpu_device = torch.device("cpu")


def run_test(iterations: int, size: int, filename: str):
    res_1 = torch.rand(size, size, device=cpu_device, dtype=torch.float32)
    res_2 = torch.rand(size, size, device=cpu_device, dtype=torch.float32)
    res_3 = torch.rand(size, size, device=cpu_device, dtype=torch.float32)

    start = time.time()

    for _ in range(0, iterations):
        a = res_1.to(mps_device)
        b = res_2.to(mps_device)
        c = res_3.to(mps_device)

        mps_1, mps_2, mps_3 = multiply_items(a, b, c)

        cpu_1 = mps_1.to(cpu_device)
        cpu_2 = mps_2.to(cpu_device)
        cpu_3 = mps_3.to(cpu_device)

        res_1, res_2, res_3 = multiply_items(cpu_1, cpu_2, cpu_3)

    end = time.time()

    with open(filename, "w") as file:
        print(res_1, file=file, flush=True)
        print(res_2, file=file, flush=True)
        print(res_3, file=file, flush=True)

    duration = end - start
    print(f"Pytorch time: {duration}")
    return duration
