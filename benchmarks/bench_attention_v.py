from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from benchmarks.core import (
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
from benchmarks.shapes import ATTENTION_SHAPES, AttentionShape
from cuda_vit.ops.attention_v_ext import load_attention_v


def make_inputs(shape: AttentionShape) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(
        shape.batch,
        shape.heads,
        shape.tokens,
        shape.tokens,
        device="cuda",
        dtype=torch.float32,
    )
    probs = F.softmax(scores, dim=-1).contiguous()
    v = torch.randn(
        shape.batch,
        shape.heads,
        shape.tokens,
        shape.head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    return probs, v


def benchmark_shape(
    ext: object,
    shape: AttentionShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    probs, v = make_inputs(shape)
    expected = torch.matmul(probs, v)
    correctness = check_close(
        "attention_v",
        ext.attention_v(probs, v),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    timings = (
        time_cuda(
            "pytorch_attention_v",
            lambda: torch.matmul(probs, v),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "attention_v",
            lambda: ext.attention_v(probs, v),
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
    print(format_run_header("Attention V Benchmark", BenchmarkEnv.current(), config))

    ext = load_attention_v()

    for shape in ATTENTION_SHAPES:
        benchmark_shape(
            ext,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
