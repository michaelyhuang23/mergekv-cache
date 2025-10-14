import torch
import time

def benchmark_matmul(dtype=torch.float32, size=16384):  # Larger size for A100 (adjust if OOM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("No GPU detected!")
        return

    print(f"Running {dtype} matmul on {torch.cuda.get_device_name()}")
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        c = torch.mm(a, b)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):  # Average over 10 runs
        c = torch.mm(a, b)
        torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / 10
    flops = 2 * size ** 3  # 2 FLOPs per element in matmul
    gflops = (flops / avg_time) / 1e9
    print(f"Average time: {avg_time:.4f}s")
    print(f"Achieved: {gflops:.2f} GFLOPS")

# Run FP32
benchmark_matmul(torch.float32)

# Run FP16 (uses Tensor Cores)
benchmark_matmul(torch.float16)
