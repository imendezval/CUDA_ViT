from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from benchmarks.common.core import BenchmarkConfig, BenchmarkEnv, format_run_header
from benchmarks.attention.profile_attention import make_inputs, make_variant, run_warmup
from benchmarks.common.shapes import AttentionShape


Variant = Callable[[], torch.Tensor]


VARIANTS = (
    "pytorch_manual",
    "pytorch_sdpa",
    "custom_3_kernel",
    "fused_attention",
    "flashattention",
)


def measure_memory(
    op: Variant,
    *,
    iterations: int,
) -> tuple[int, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(iterations):
            op()

    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )


def benchmark_variant(
    variant: str,
    shape: AttentionShape,
    *,
    warmup: int,
    iterations: int,
) -> None:
    if variant == "flashattention" and not shape.supports_flashattention:
        print(f"{variant},{shape.label},skipped,skipped,shape_not_supported")
        return

    q, k, v = make_inputs(shape)
    op = make_variant(variant, shape, q, k, v)
    run_warmup(op, warmup)
    peak_allocated, peak_reserved = measure_memory(op, iterations=iterations)
    print(f"{variant},{shape.label},{peak_allocated},{peak_reserved},ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
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
        repeats=1,
    )
    print(format_run_header("Attention Memory Benchmark", BenchmarkEnv.current(), config))

    shape = AttentionShape(
        batch=args.batch,
        heads=args.heads,
        tokens=args.tokens,
        head_dim=args.head_dim,
    )
    print("variant,shape,peak_allocated_bytes,peak_reserved_bytes,status")

    for variant in VARIANTS:
        benchmark_variant(
            variant,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
