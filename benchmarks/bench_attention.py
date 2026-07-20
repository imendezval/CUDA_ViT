from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmarks.core import format_comparison, format_table, time_cuda
from cuda_vit.ops.attention_v_ext import load_attention_v
from cuda_vit.ops.flashattention_ext import load_flashattention
from cuda_vit.ops.fused_attention_ext import load_fused_attention
from cuda_vit.ops.scaled_qk_ext import load_scaled_qk
from cuda_vit.ops.softmax_ext import load_softmax


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads: int
    tokens: int
    head_dim: int

    @property
    def label(self) -> str:
        return (
            f"B{self.batch}_H{self.heads}_T{self.tokens}_Dh{self.head_dim}"
        )

    @property
    def supports_flashattention(self) -> bool:
        return self.head_dim == 64 and self.tokens % 32 == 0


SHAPES = (
    AttentionShape(1, 1, 32, 64),
    AttentionShape(2, 3, 64, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(2, 3, 197, 64),
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


def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


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


def validate_outputs(
    fused_ext: object,
    flash_ext: object,
    scaled_qk_ext: object,
    softmax_ext: object,
    attention_v_ext: object,
    shape: AttentionShape,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    expected = pytorch_attention(q, k, v)

    torch.testing.assert_close(
        custom_attention(scaled_qk_ext, softmax_ext, attention_v_ext, q, k, v),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        fused_ext.fused_attention(q, k, v),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    if shape.supports_flashattention:
        torch.testing.assert_close(
            flash_ext.FlashAttention(q, k, v),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )


def benchmark_shape(
    fused_ext: object,
    flash_ext: object,
    scaled_qk_ext: object,
    softmax_ext: object,
    attention_v_ext: object,
    shape: AttentionShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
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
            "pytorch_manual",
            lambda: pytorch_attention(q, k, v),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
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

    print(f"\nshape={shape.label}")
    if not shape.supports_flashattention:
        print("flashattention skipped: requires Dh=64 and T divisible by 32")
    print(format_table(timings))
    print(format_comparison(timings[0], timings[1]))
    print(format_comparison(timings[0], timings[2]))
    print(format_comparison(timings[0], timings[3]))
    if shape.supports_flashattention:
        print(format_comparison(timings[0], timings[4]))


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

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    fused_ext = load_fused_attention()
    flash_ext = load_flashattention()
    scaled_qk_ext = load_scaled_qk()
    softmax_ext = load_softmax()
    attention_v_ext = load_attention_v()

    for shape in SHAPES:
        benchmark_shape(
            fused_ext,
            flash_ext,
            scaled_qk_ext,
            softmax_ext,
            attention_v_ext,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
