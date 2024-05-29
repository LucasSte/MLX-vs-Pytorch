import mlx.core as mx
import time

res_1 = mx.random.uniform(shape=(10000, 10000), stream=mx.cpu)
res_2 = mx.random.uniform(shape=(10000, 10000), stream=mx.cpu)
res_3 = mx.random.uniform(shape=(10000, 10000), stream=mx.cpu)


start = time.time()
for _ in range(0, 50):
    # TODO: This only copies the pointer, right?
    a = res_1
    b = res_2
    c = res_3

    mps_1 = mx.matmul(a, b, stream=mx.gpu)
    mps_1 = mx.divide(mps_1, mps_1.max(0, stream=mx.gpu), stream=mx.gpu)
    mps_2 = mx.matmul(b, c, stream=mx.gpu)
    mps_2 = mx.divide(mps_2, mps_2.max(0, stream=mx.gpu), stream=mx.gpu)
    mps_3 = mx.matmul(a, c, stream=mx.gpu)
    mps_3 = mx.divide(mps_3, mps_3.max(0, stream=mx.gpu), stream=mx.gpu)

    mx.eval(mps_1, mps_2, mps_3)

    # TODO: This only copies the pointer, right?
    cpu_1 = mps_1
    cpu_2 = mps_2
    cpu_3 = mps_3

    res_1 = mx.matmul(cpu_1, cpu_2, stream=mx.cpu)
    res_1 = mx.divide(res_1, res_1.max(0, stream=mx.cpu), stream=mx.cpu)
    res_2 = mx.matmul(cpu_2, cpu_3, stream=mx.cpu)
    res_2 = mx.divide(res_2, res_2.max(0, stream=mx.cpu), stream=mx.cpu)
    res_3 = mx.matmul(cpu_3, cpu_1, stream=mx.cpu)
    res_3 = mx.divide(res_3, res_3.max(0, stream=mx.cpu), stream=mx.cpu)

    mx.eval(res_1, res_2, res_3)
end = time.time()

print(res_1)
print(res_2)
print(res_3)

print(f"Time: {end-start}")
