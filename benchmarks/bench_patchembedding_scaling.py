from __future__ import annotations

import argparse
from collections.abc import Iterable

import torch

from benchmarks.bench_patchembedding import (
    PatchEmbeddingShape,
    logical_bytes,
    make_inputs,
    pytorch_patchembedding,
)
from benchmarks.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    Timing,
    check_close,
    effective_bandwidth_gbs,
    format_run_header,
    speedup,
    time_cuda,
)
from cuda_vit.ops.patchembedding_ext import load_patchembedding
from cuda_vit.ops.patchembeddingv2_ext import load_patchembeddingv2


BATCH_SWEEP = (
    PatchEmbeddingShape(1, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(4, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(8, 3, 224, 224, 16, 384),
)

IMAGE_SWEEP = (
    PatchEmbeddingShape(2, 3, 32, 32, 16, 384),
    PatchEmbeddingShape(2, 3, 64, 64, 16, 384),
    PatchEmbeddingShape(2, 3, 128, 128, 16, 384),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 384),
)

PATCH_SWEEP = (
    PatchEmbeddingShape(2, 3, 224, 224, 8, 384),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(2, 3, 224, 224, 32, 384),
)

EMBED_SWEEP = (
    PatchEmbeddingShape(2, 3, 224, 224, 16, 64),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 768),
)

RTOL = 2e-4
ATOL = 2e-4


def print_rows(
    sweep: str,
    shape: PatchEmbeddingShape,
    timings: Iterable[Timing],
) -> None:
    rows = tuple(timings)
    baseline = rows[0]
    bytes_per_call = logical_bytes(shape)
    for timing in rows:
        bandwidth = effective_bandwidth_gbs(bytes_per_call, timing)
        print(
            f"{sweep},{shape.label},{timing.name},"
            f"{timing.median_ms:.6f},{speedup(baseline, timing):.4f},"
            f"{bandwidth:.1f}"
        )


def benchmark_shape(
    ext_v1: object,
    ext_v2: object,
    sweep: str,
    shape: PatchEmbeddingShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x, weight = make_inputs(shape)
    expected = pytorch_patchembedding(x, weight, shape)
    check_close(
        "patchembedding",
        ext_v1.patchembedding(x, weight),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )
    check_close(
        "patchembeddingv2",
        ext_v2.patchembeddingv2(x, weight),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    timings = (
        time_cuda(
            "pytorch_conv2d",
            lambda: pytorch_patchembedding(x, weight, shape),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "patchembedding",
            lambda: ext_v1.patchembedding(x, weight),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "patchembeddingv2",
            lambda: ext_v2.patchembeddingv2(x, weight),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )
    print_rows(sweep, shape, timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
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
    print(format_run_header("Patch Embedding Scaling Benchmark", BenchmarkEnv.current(), config))
    print("sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs")

    ext_v1 = load_patchembedding()
    ext_v2 = load_patchembeddingv2()

    sweeps = (
        ("batch", BATCH_SWEEP),
        ("image", IMAGE_SWEEP),
        ("patch", PATCH_SWEEP),
        ("embed", EMBED_SWEEP),
    )
    for sweep, shapes in sweeps:
        for shape in shapes:
            benchmark_shape(
                ext_v1,
                ext_v2,
                sweep,
                shape,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )


if __name__ == "__main__":
    main()
