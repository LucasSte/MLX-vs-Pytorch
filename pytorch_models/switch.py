import torch

mps_device = torch.device("mps")
cpu_device = torch.device("cpu")

res_1 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)
res_2 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)
res_3 = torch.rand(10000, 10000, device=cpu_device, dtype=torch.float32)

for _ in range(0, 50):
    a = res_1.to(mps_device)
    b = res_2.to(mps_device)
    c = res_3.to(mps_device)

    mps_1 = torch.matmul(a, b) / 1000
    mps_2 = torch.matmul(b, c) / 1000
    mps_3 = torch.matmul(a, c) / 1000

    cpu_1 = mps_1.to(cpu_device)
    cpu_2 = mps_2.to(cpu_device)
    cpu_3 = mps_3.to(cpu_device)

    res_1 = torch.matmul(cpu_1, cpu_2) / 1000
    res_2 = torch.matmul(cpu_2, cpu_3) / 1000
    res_3 = torch.matmul(cpu_3, cpu_1) / 1000

print(res_1)
print(res_2)
print(res_3)
