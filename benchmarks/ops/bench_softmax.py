from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

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
from cuda_vit.ops.softmax_ext import load_softmax


SHAPES = (
    (1, 31),
    (2, 256),
    (2, 257),
    (8, 197),
    (2 * 4 * 197, 197),
    (2 * 8 * 384, 384),
)


def make_input(shape: tuple[int, int]) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=torch.float32)


def logical_bytes(shape: tuple[int, int]) -> int:
    rows, cols = shape
    return 2 * rows * cols * 4


def benchmark_shape(
    ext: object,
    shape: tuple[int, int],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x = make_input(shape)
    expected = F.softmax(x, dim=-1)
    correctness = check_close(
        "softmax",
        ext.softmax(x),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

    timings = (
        time_cuda(
            "pytorch_softmax",
            lambda: F.softmax(x, dim=-1),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "softmax",
            lambda: ext.softmax(x),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )

    bytes_per_call = logical_bytes(shape)

    print(f"\nshape={shape}")
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
    print(format_run_header("Softmax Benchmark", BenchmarkEnv.current(), config))

    ext = load_softmax()

    for shape in SHAPES:
        benchmark_shape(
            ext,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

