from __future__ import annotations

import argparse

import torch

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    check_close,
    effective_tflops,
    format_comparison,
    format_correctness,
    format_run_header,
    format_table,
    time_cuda,
)
from benchmarks.common.shapes import ATTENTION_OP_SHAPES, AttentionShape
from cuda_vit.ops.scaled_qk_ext import load_scaled_qk


def make_inputs(shape: AttentionShape) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(
        shape.batch,
        shape.heads,
        shape.tokens,
        shape.head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k = torch.randn_like(q)
    return q, k


def pytorch_scaled_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)


def benchmark_shape(
    ext: object,
    shape: AttentionShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    q, k = make_inputs(shape)
    expected = pytorch_scaled_qk(q, k)
    correctness = check_close(
        "scaled_qk",
        ext.scaled_qk(q, k),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    timings = (
        time_cuda(
            "pytorch_scaled_qk",
            lambda: pytorch_scaled_qk(q, k),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "scaled_qk",
            lambda: ext.scaled_qk(q, k),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )

    print(f"\nshape={shape.label}")
    print(format_correctness(correctness))
    print(format_table(timings))
    for timing in timings:
        throughput = effective_tflops(shape.attention_matmul_flops, timing)
        print(f"{timing.name}: estimated_throughput={throughput:.2f} TFLOP/s")
    print(format_comparison(timings[0], timings[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    config = BenchmarkConfig(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(format_run_header("Scaled QK Benchmark", BenchmarkEnv.current(), config))

    ext = load_scaled_qk()

    for shape in ATTENTION_OP_SHAPES:
        benchmark_shape(
            ext,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
