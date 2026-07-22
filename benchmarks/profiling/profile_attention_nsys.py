from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from benchmarks.common.shapes import AttentionShape


PRESENTATION_SHAPE = AttentionShape(2, 3, 512, 64)
PROFILE_VARIANTS = (
    "PyTorch SDPA",
    "Custom 3 Part Kernel",
    "Fused Attention",
    "FlashAttention",
)


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def run_variant(name: str, op: Callable[[], torch.Tensor], iterations: int) -> None:
    with nvtx_range(name):
        for _ in range(iterations):
            op()


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


def custom_attention(
    scaled_qk_ext: object,
    softmax_ext: object,
    attention_v_ext: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    scores = scaled_qk_ext.scaled_qk(q, k)
    batch, heads, tokens, _ = scores.shape
    probs = softmax_ext.softmax(scores.reshape(batch * heads * tokens, tokens))
    probs = probs.reshape(batch, heads, tokens, tokens).contiguous()
    return attention_v_ext.attention_v(probs, v)


def profile_attention(
    *,
    warmup: int,
    iterations: int,
    use_cuda_profiler: bool,
) -> None:
    from cuda_vit.ops.attention_v_ext import load_attention_v
    from cuda_vit.ops.flashattention_ext import load_flashattention
    from cuda_vit.ops.fused_attention_ext import load_fused_attention
    from cuda_vit.ops.scaled_qk_ext import load_scaled_qk
    from cuda_vit.ops.softmax_ext import load_softmax

    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    shape = PRESENTATION_SHAPE
    if not shape.supports_flashattention:
        raise RuntimeError(f"shape does not support FlashAttention: {shape.label}")

    q, k, v = make_inputs(shape)
    fused_ext = load_fused_attention()
    flash_ext = load_flashattention()
    scaled_qk_ext = load_scaled_qk()
    softmax_ext = load_softmax()
    attention_v_ext = load_attention_v()
    ops = (
        (
            "PyTorch SDPA",
            lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False),
        ),
        (
            "Custom 3 Part Kernel",
            lambda: custom_attention(scaled_qk_ext, softmax_ext, attention_v_ext, q, k, v),
        ),
        ("Fused Attention", lambda: fused_ext.fused_attention(q, k, v)),
        ("FlashAttention", lambda: flash_ext.FlashAttention(q, k, v)),
    )

    with torch.inference_mode():
        for _, op in ops:
            for _ in range(warmup):
                op()
        torch.cuda.synchronize()

        if use_cuda_profiler:
            torch.cuda.profiler.start()
        with nvtx_range(f"attention:{shape.label}"):
            for name, op in ops:
                run_variant(name, op, iterations)
        torch.cuda.synchronize()
        if use_cuda_profiler:
            torch.cuda.profiler.stop()

    print(f"shape={shape.label}")
    print("variants=" + ",".join(PROFILE_VARIANTS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--no-cuda-profiler",
        action="store_true",
        help="Do not emit cudaProfilerStart/Stop markers for Nsight capture ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    profile_attention(
        warmup=args.warmup,
        iterations=args.iterations,
        use_cuda_profiler=not args.no_cuda_profiler,
    )


if __name__ == "__main__":
    main()
