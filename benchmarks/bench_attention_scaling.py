from __future__ import annotations

import argparse
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from benchmarks.bench_attention import custom_attention, pytorch_attention, validate_outputs
from benchmarks.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    Timing,
    format_run_header,
    speedup,
    throughput_scale,
    time_cuda,
)
from benchmarks.shapes import AttentionShape
from cuda_vit.ops.attention_v_ext import load_attention_v
from cuda_vit.ops.flashattention_ext import load_flashattention
from cuda_vit.ops.fused_attention_ext import load_fused_attention
from cuda_vit.ops.scaled_qk_ext import load_scaled_qk
from cuda_vit.ops.softmax_ext import load_softmax


BATCH_SWEEP = (
    AttentionShape(1, 3, 192, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(4, 3, 192, 64),
    AttentionShape(8, 3, 192, 64),
)

SEQUENCE_SWEEP = (
    AttentionShape(2, 3, 32, 64),
    AttentionShape(2, 3, 64, 64),
    AttentionShape(2, 3, 128, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(2, 3, 197, 64),
    AttentionShape(2, 3, 256, 64),
)

HEAD_SWEEP = (
    AttentionShape(2, 1, 192, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(2, 6, 192, 64),
    AttentionShape(2, 12, 192, 64),
)

HEAD_DIM_SWEEP = (
    AttentionShape(2, 3, 128, 32),
    AttentionShape(2, 3, 128, 64),
    AttentionShape(2, 3, 128, 128),
)


def make_inputs(shape: AttentionShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        shape.batch,
        shape.heads,
        shape.tokens,
        shape.head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def print_rows(
    sweep: str,
    baseline_shape: AttentionShape,
    baseline_timings: dict[str, Timing],
    shape: AttentionShape,
    timings: Iterable[Timing],
) -> None:
    rows = tuple(timings)
    baseline = rows[0]
    for timing in rows:
        if timing.name in baseline_timings:
            scaling = throughput_scale(
                baseline_shape.attention_matmul_flops,
                baseline_timings[timing.name],
                shape.attention_matmul_flops,
                timing,
            )
            scaling_text = f"{scaling:.4f}"
        else:
            scaling_text = "nan"
        print(
            f"{sweep},{shape.label},{timing.name},"
            f"{timing.median_ms:.6f},{speedup(baseline, timing):.4f},"
            f"{scaling_text}"
        )


def benchmark_shape(
    fused_ext: object,
    flash_ext: object,
    scaled_qk_ext: object,
    softmax_ext: object,
    attention_v_ext: object,
    sweep: str,
    shape: AttentionShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[Timing, ...]:
    q, k, v = make_inputs(shape)
    validate_outputs(
        fused_ext,
        flash_ext,
        scaled_qk_ext,
        softmax_ext,
        attention_v_ext,
        shape,
        q,
        k,
        v,
    )

    timings = [
        time_cuda(
            "pytorch_sdpa",
            lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
            ),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "pytorch_manual",
            lambda: pytorch_attention(q, k, v),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "custom_3_kernel",
            lambda: custom_attention(
                scaled_qk_ext,
                softmax_ext,
                attention_v_ext,
                q,
                k,
                v,
            ),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "fused_attention",
            lambda: fused_ext.fused_attention(q, k, v),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    ]

    if shape.supports_flashattention:
        timings.append(
            time_cuda(
                "flashattention",
                lambda: flash_ext.FlashAttention(q, k, v),
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
        )

    return tuple(timings)


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
    print(format_run_header("Attention Scaling Benchmark", BenchmarkEnv.current(), config))
    print("sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale")

    fused_ext = load_fused_attention()
    flash_ext = load_flashattention()
    scaled_qk_ext = load_scaled_qk()
    softmax_ext = load_softmax()
    attention_v_ext = load_attention_v()

    sweeps = (
        ("batch", BATCH_SWEEP),
        ("sequence", SEQUENCE_SWEEP),
        ("heads", HEAD_SWEEP),
        ("head_dim", HEAD_DIM_SWEEP),
    )
    for sweep, shapes in sweeps:
        baseline_shape = shapes[0]
        baseline_timings = None
        for shape in shapes:
            timings = benchmark_shape(
                fused_ext,
                flash_ext,
                scaled_qk_ext,
                softmax_ext,
                attention_v_ext,
                sweep,
                shape,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            if baseline_timings is None:
                baseline_timings = {timing.name: timing for timing in timings}
            print_rows(
                sweep,
                baseline_shape,
                baseline_timings,
                shape,
                timings,
            )


if __name__ == "__main__":
    main()
