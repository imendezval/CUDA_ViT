import torch

from vector_add_ext import load_vector_add


def benchmark_ms(op, warmup=100, iterations=1_000):
    for _ in range(warmup):
        op()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        op()

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def main():
    ext = load_vector_add()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print()

    for n in [1_024, 65_536, 1_000_000, 16_000_000]:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)

        with torch.inference_mode():
            pytorch_ms = benchmark_ms(lambda: torch.add(a, b))
            custom_ms = benchmark_ms(lambda: ext.vector_add(a, b))

        speedup = pytorch_ms / custom_ms
        bytes_per_call = 3 * n * 4  # read a + read b + write output
        bandwidth_gbs = bytes_per_call / (custom_ms * 1e-3) / 1e9

        print(
            f"n={n:>10,} | "
            f"PyTorch={pytorch_ms:.4f} ms | "
            f"custom={custom_ms:.4f} ms | "
            f"speedup={speedup:.2f}x | "
            f"custom bandwidth={bandwidth_gbs:.1f} GB/s"
        )


if __name__ == "__main__":
    main()