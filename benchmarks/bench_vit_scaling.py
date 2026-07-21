from __future__ import annotations

import argparse

import torch

from benchmarks.bench_vit import (
    Variant,
    VitShape,
    load_extensions,
    make_inputs,
    make_weights,
    vit_forward,
)
from benchmarks.core import BenchmarkConfig, BenchmarkEnv, check_close, format_run_header, time_cuda


VARIANTS = (
    Variant("pytorch_manual", "pytorch", "manual", False, False, False),
    Variant("pytorch_sdpa", "pytorch", "sdpa", False, False, False),
    Variant("custom_v1_3_kernel", "custom_v1", "custom_3_kernel", True, True, True),
    Variant("custom_v2_flashattention", "custom_v2", "flash", True, True, True),
    Variant("custom_v2_flashattention_torch_linear", "custom_v2", "flash", False, True, False),
)

BATCH_SWEEP = (
    VitShape(1, 3, 128, 16, 192, 3, 4, 2, False),
    VitShape(2, 3, 128, 16, 192, 3, 4, 2, False),
    VitShape(4, 3, 128, 16, 192, 3, 4, 2, False),
    VitShape(8, 3, 128, 16, 192, 3, 4, 2, False),
)

IMAGE_SWEEP = (
    VitShape(2, 3, 128, 16, 192, 3, 4, 2, False),
    VitShape(2, 3, 256, 16, 192, 3, 4, 2, False),
    VitShape(2, 3, 384, 16, 192, 3, 4, 2, False),
    VitShape(2, 3, 512, 16, 192, 3, 4, 2, False),
)

PATCHES_SWEEP = (
    VitShape(2, 3, 256, 32, 192, 3, 4, 2, False),
    VitShape(2, 3, 256, 16, 192, 3, 4, 2, False),
    VitShape(2, 3, 256, 8, 192, 3, 4, 2, False),
)

SWEEPS = (
    ("batch", BATCH_SWEEP),
    ("image", IMAGE_SWEEP),
    ("patches", PATCHES_SWEEP),
)


def benchmark_shape(
    sweep: str,
    shape: VitShape,
    exts: dict[str, object],
    *,
    eps: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x = make_inputs(shape)
    weights = make_weights(shape)
    expected = vit_forward(x, weights, shape, VARIANTS[0], exts, eps)

    for variant in VARIANTS:
        if variant.attention == "flash" and not shape.supports_flashattention:
            print(f"{sweep},{shape.label},{variant.name},skipped,,,,,,,")
            continue
        actual = vit_forward(x, weights, shape, variant, exts, eps)
        correctness = check_close(
            variant.name,
            actual,
            expected,
            rtol=2e-3,
            atol=2e-3,
        )
        timing = time_cuda(
            variant.name,
            lambda variant=variant: vit_forward(x, weights, shape, variant, exts, eps),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        print(
            f"{sweep},{shape.label},{variant.name},ok,"
            f"{timing.median_ms:.6f},{timing.mean_ms:.6f},"
            f"{timing.min_ms:.6f},{timing.max_ms:.6f},"
            f"{correctness.max_abs_error:.6g},{correctness.mean_abs_error:.6g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
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
    print(format_run_header("Whole ViT Scaling Benchmark", BenchmarkEnv.current(), config))
    print(
        "sweep,shape,variant,status,median_ms,mean_ms,min_ms,max_ms,"
        "max_abs_error,mean_abs_error"
    )

    exts = load_extensions()
    for sweep, shapes in SWEEPS:
        for shape in shapes:
            benchmark_shape(
                sweep,
                shape,
                exts,
                eps=args.eps,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )


if __name__ == "__main__":
    main()
