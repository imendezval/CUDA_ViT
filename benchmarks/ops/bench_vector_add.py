from __future__ import annotations

import argparse

import torch

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    check_close,
    effective_bandwidth_gbs,
    format_comparison,
    format_correctness,
    format_run_header,
    format_table,
    time_cuda,
)
from cuda_vit.ops.vector_add_ext import load_vector_add


SIZES = (1_024, 65_536, 1_000_000, 16_000_000)


def make_inputs(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    return a, b


def logical_bytes(n: int) -> int:
    return 3 * n * 4


def benchmark_size(
    ext: object,
    n: int,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    a, b = make_inputs(n)
    expected = torch.add(a, b)
    correctness = check_close(
        "vector_add",
        ext.vector_add(a, b),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    timings = (
        time_cuda(
            "pytorch_add",
            lambda: torch.add(a, b),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "vector_add",
            lambda: ext.vector_add(a, b),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )

    bytes_per_call = logical_bytes(n)

    print(f"\nn={n}")
    print(format_correctness(correctness))
    print(format_table(timings))
    for timing in timings:
        bandwidth = effective_bandwidth_gbs(bytes_per_call, timing)
        print(f"{timing.name}: logical_bandwidth={bandwidth:.1f} GB/s")
    print(format_comparison(timings[0], timings[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)

    config = BenchmarkConfig(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(format_run_header("Vector Add Benchmark", BenchmarkEnv.current(), config))

    ext = load_vector_add()

    for n in SIZES:
        benchmark_size(
            ext,
            n,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
