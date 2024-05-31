import torch
import time


def multiply_items(op1: torch.Tensor, op2: torch.Tensor, op3: torch.Tensor):
    res_1 = torch.matmul(op1, op2)
    res_1 = torch.divide(res_1, torch.max(res_1, 0).values)
    res_2 = torch.matmul(op2, op3)
    res_2 = torch.divide(res_2, torch.max(res_2, 0).values)
    res_3 = torch.matmul(op3, op1)
    res_3 = torch.divide(res_3, torch.max(res_3, 0).values)
    return res_1, res_2, res_3


mps_device = torch.device("mps")
cpu_device = torch.device("cpu")

res_1 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)
res_2 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)
res_3 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)

start = time.time()

for _ in range(0, 50):
    a = res_1.to(mps_device)
    b = res_2.to(mps_device)
    c = res_3.to(mps_device)

    mps_1 = torch.matmul(a, b)
    mps_1 = torch.divide(mps_1, torch.max(mps_1, 0).values)
    mps_2 = torch.matmul(b, c)
    mps_2 = torch.divide(mps_2, torch.max(mps_2, 0).values)
    mps_3 = torch.matmul(a, c)
    mps_3 = torch.divide(mps_3, torch.max(mps_3, 0).values)

    cpu_1 = mps_1.to(cpu_device)
    cpu_2 = mps_2.to(cpu_device)
    cpu_3 = mps_3.to(cpu_device)

    res_1 = torch.matmul(cpu_1, cpu_2)
    res_1 = torch.divide(res_1, torch.max(res_1, 0).values)
    res_2 = torch.matmul(cpu_2, cpu_3)
    res_2 = torch.divide(res_2, torch.max(res_2, 0).values)
    res_3 = torch.matmul(cpu_3, cpu_1)
    res_3 = torch.divide(res_3, torch.max(res_3, 0).values)

end = time.time()

print(res_1)
print(res_2)
print(res_3)

print(f"Time: {end-start}")
