from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from benchmarks.bench_attention import custom_attention, pytorch_attention
from benchmarks.core import BenchmarkConfig, BenchmarkEnv, format_run_header
from benchmarks.shapes import AttentionShape
from cuda_vit.ops.attention_v_ext import load_attention_v
from cuda_vit.ops.flashattention_ext import load_flashattention
from cuda_vit.ops.fused_attention_ext import load_fused_attention
from cuda_vit.ops.scaled_qk_ext import load_scaled_qk
from cuda_vit.ops.softmax_ext import load_softmax


Variant = Callable[[], torch.Tensor]


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


def cuda_event_count(prof: object) -> int:
    count = 0
    for event in prof.events():
        device_type = getattr(event, "device_type", None)
        if getattr(device_type, "name", "") == "CUDA":
            count += 1
    return count


def make_variant(
    name: str,
    shape: AttentionShape,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Variant:
    if name == "pytorch_manual":
        return lambda: pytorch_attention(q, k, v)
    if name == "pytorch_sdpa":
        return lambda: F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
    if name == "custom_3_kernel":
        scaled_qk_ext = load_scaled_qk()
        softmax_ext = load_softmax()
        attention_v_ext = load_attention_v()
        return lambda: custom_attention(scaled_qk_ext, softmax_ext, attention_v_ext, q, k, v)
    if name == "fused_attention":
        fused_ext = load_fused_attention()
        return lambda: fused_ext.fused_attention(q, k, v)
    if name == "flashattention":
        if not shape.supports_flashattention:
            raise ValueError("flashattention requires Dh=64 and T divisible by 32")
        flash_ext = load_flashattention()
        return lambda: flash_ext.FlashAttention(q, k, v)
    raise ValueError(f"unknown variant: {name}")


def run_warmup(op: Variant, warmup: int) -> None:
    with torch.inference_mode():
        for _ in range(warmup):
            op()
    torch.cuda.synchronize()


def run_profile(
    op: Variant,
    *,
    iterations: int,
    trace: Path | None,
) -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(iterations):
                op()
                prof.step()

    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )
    print(f"cuda_event_count={cuda_event_count(prof)}")
    print(f"peak_cuda_memory_allocated_bytes={peak_allocated}")
    print(f"peak_cuda_memory_reserved_bytes={peak_reserved}")

    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        print(f"trace={trace}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[
            "pytorch_manual",
            "pytorch_sdpa",
            "custom_3_kernel",
            "fused_attention",
            "flashattention",
        ],
        default="custom_3_kernel",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace", type=Path, default=None)
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
    print(format_run_header("Attention Profiler", BenchmarkEnv.current(), config))

    shape = AttentionShape(
        batch=args.batch,
        heads=args.heads,
        tokens=args.tokens,
        head_dim=args.head_dim,
    )
    print(f"variant={args.variant}")
    print(f"shape={shape.label}")

    q, k, v = make_inputs(shape)
    op = make_variant(args.variant, shape, q, k, v)
    run_warmup(op, args.warmup)
    run_profile(op, iterations=args.iterations, trace=args.trace)


if __name__ == "__main__":
    main()
