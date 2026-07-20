from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from benchmarks.core import (
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
from cuda_vit.ops.layernorm_ext import load_layernorm


SHAPES = (
    (1, 1, 31),
    (2, 3, 256),
    (2, 3, 257),
    (8, 197, 384),
    (2, 197, 768),
)


def make_inputs(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    gamma = torch.randn(hidden, device="cuda", dtype=torch.float32)
    beta = torch.randn(hidden, device="cuda", dtype=torch.float32)
    return x, gamma, beta


def pytorch_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=gamma,
        bias=beta,
        eps=eps,
    )


def logical_bytes(shape: tuple[int, ...]) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return 4 * elements * 4


def benchmark_shape(
    ext: object,
    shape: tuple[int, ...],
    *,
    eps: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x, gamma, beta = make_inputs(shape)
    expected = pytorch_layernorm(x, gamma, beta, eps)
    correctness = check_close(
        "layernorm",
        ext.layernorm(x, gamma, beta, eps),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    timings = (
        time_cuda(
            "pytorch_layer_norm",
            lambda: pytorch_layernorm(x, gamma, beta, eps),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "layernorm",
            lambda: ext.layernorm(x, gamma, beta, eps),
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
    parser.add_argument("--eps", type=float, default=1e-5)
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
    print(format_run_header("LayerNorm Benchmark", BenchmarkEnv.current(), config))

    ext = load_layernorm()

    for shape in SHAPES:
        benchmark_shape(
            ext,
            shape,
            eps=args.eps,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

