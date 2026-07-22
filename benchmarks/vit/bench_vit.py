from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    Timing,
    check_close,
    format_run_header,
    time_cuda,
)


@dataclass(frozen=True)
class VitShape:
    batch: int
    channels: int
    image_size: int
    patch: int
    embed_dim: int
    heads: int
    mlp_ratio: int
    depth: int
    class_token: bool

    def __post_init__(self) -> None:
        if self.image_size % self.patch != 0:
            raise ValueError("image_size must be divisible by patch")
        if self.embed_dim % self.heads != 0:
            raise ValueError("embed_dim must be divisible by heads")

    @property
    def patches(self) -> int:
        return (self.image_size // self.patch) ** 2

    @property
    def tokens(self) -> int:
        return self.patches + int(self.class_token)

    @property
    def patch_elements(self) -> int:
        return self.channels * self.patch * self.patch

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads

    @property
    def mlp_dim(self) -> int:
        return self.embed_dim * self.mlp_ratio

    @property
    def supports_flashattention(self) -> bool:
        return self.head_dim == 64 and self.tokens % 32 == 0

    @property
    def label(self) -> str:
        cls = "cls" if self.class_token else "nocls"
        return (
            f"B{self.batch}_I{self.image_size}_P{self.patch}_T{self.tokens}_"
            f"D{self.embed_dim}_H{self.heads}_L{self.depth}_{cls}"
        )


@dataclass(frozen=True)
class VitWeights:
    patch: torch.Tensor
    pos: torch.Tensor
    cls: torch.Tensor | None
    ln1_gamma: tuple[torch.Tensor, ...]
    ln1_beta: tuple[torch.Tensor, ...]
    qkv: tuple[torch.Tensor, ...]
    proj: tuple[torch.Tensor, ...]
    ln2_gamma: tuple[torch.Tensor, ...]
    ln2_beta: tuple[torch.Tensor, ...]
    mlp1_weight: tuple[torch.Tensor, ...]
    mlp1_bias: tuple[torch.Tensor, ...]
    mlp2: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Variant:
    name: str
    patch: str
    attention: str
    custom_linear: bool
    custom_layernorm: bool
    custom_mlp: bool


SHAPES = {
    "smoke": VitShape(1, 3, 64, 16, 192, 3, 4, 1, False),
    "flash": VitShape(2, 3, 128, 16, 192, 3, 4, 2, False),
    "vit_like": VitShape(2, 3, 224, 16, 192, 3, 4, 2, True),
}

VARIANTS = (
    Variant("pytorch_manual", "pytorch", "manual", False, False, False),
    Variant("pytorch_sdpa", "pytorch", "sdpa", False, False, False),
    Variant("custom_v1_3_kernel", "custom_v1", "custom_3_kernel", True, True, True),
    Variant("custom_v2_flashattention", "custom_v2", "flash", True, True, True),
    Variant("custom_v2_flashattention_torch_linear", "custom_v2", "flash", False, True, False),
)


def make_inputs(shape: VitShape) -> torch.Tensor:
    return torch.randn(
        shape.batch,
        shape.channels,
        shape.image_size,
        shape.image_size,
        device="cuda",
        dtype=torch.float32,
    )


def make_weights(shape: VitShape) -> VitWeights:
    def normal(*size: int, std: float = 0.02) -> torch.Tensor:
        return torch.randn(*size, device="cuda", dtype=torch.float32) * std

    def ones(*size: int) -> torch.Tensor:
        return torch.ones(*size, device="cuda", dtype=torch.float32)

    def zeros(*size: int) -> torch.Tensor:
        return torch.zeros(*size, device="cuda", dtype=torch.float32)

    cls = normal(1, 1, shape.embed_dim) if shape.class_token else None
    return VitWeights(
        patch=normal(shape.embed_dim, shape.patch_elements),
        pos=normal(1, shape.tokens, shape.embed_dim),
        cls=cls,
        ln1_gamma=tuple(ones(shape.embed_dim) for _ in range(shape.depth)),
        ln1_beta=tuple(zeros(shape.embed_dim) for _ in range(shape.depth)),
        qkv=tuple(normal(3 * shape.embed_dim, shape.embed_dim) for _ in range(shape.depth)),
        proj=tuple(normal(shape.embed_dim, shape.embed_dim) for _ in range(shape.depth)),
        ln2_gamma=tuple(ones(shape.embed_dim) for _ in range(shape.depth)),
        ln2_beta=tuple(zeros(shape.embed_dim) for _ in range(shape.depth)),
        mlp1_weight=tuple(normal(shape.mlp_dim, shape.embed_dim) for _ in range(shape.depth)),
        mlp1_bias=tuple(zeros(shape.mlp_dim) for _ in range(shape.depth)),
        mlp2=tuple(normal(shape.embed_dim, shape.mlp_dim) for _ in range(shape.depth)),
    )


def pytorch_patch_embed(x: torch.Tensor, weight: torch.Tensor, shape: VitShape) -> torch.Tensor:
    conv_weight = weight.view(shape.embed_dim, shape.channels, shape.patch, shape.patch)
    patches = F.conv2d(x, conv_weight, bias=None, stride=shape.patch)
    return patches.permute(0, 2, 3, 1).reshape(
        shape.batch,
        shape.patches,
        shape.embed_dim,
    ).contiguous()


def torch_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight, bias=None)


def custom_linear(linear_ext: object, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    out = linear_ext.linear_forward(rows, weight)
    return out.reshape(*x.shape[:-1], weight.shape[0]).contiguous()


def torch_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)


def split_qkv(qkv: torch.Tensor, shape: VitShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = qkv.reshape(
        shape.batch,
        shape.tokens,
        3,
        shape.heads,
        shape.head_dim,
    )
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
    return qkv[0], qkv[1], qkv[2]


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def custom_3_kernel_attention(
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


def merge_heads(x: torch.Tensor, shape: VitShape) -> torch.Tensor:
    return x.transpose(1, 2).reshape(shape.batch, shape.tokens, shape.embed_dim).contiguous()


def vit_forward(
    x: torch.Tensor,
    weights: VitWeights,
    shape: VitShape,
    variant: Variant,
    exts: dict[str, object],
    eps: float,
) -> torch.Tensor:
    if variant.patch == "pytorch":
        x = pytorch_patch_embed(x, weights.patch, shape)
    elif variant.patch == "custom_v1":
        x = exts["patch_v1"].patchembedding(x, weights.patch)
    elif variant.patch == "custom_v2":
        x = exts["patch_v2"].patchembeddingv2(x, weights.patch)
    elif variant.patch == "custom_v3":
        x = exts["patch_v3"].patchembeddingv3(x, weights.patch)
    else:
        raise ValueError(f"unknown patch implementation: {variant.patch}")

    if shape.class_token:
        cls = weights.cls.expand(shape.batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
    x = x + weights.pos

    for layer in range(shape.depth):
        residual = x
        if variant.custom_layernorm:
            x = exts["layernorm"].layernorm(
                x,
                weights.ln1_gamma[layer],
                weights.ln1_beta[layer],
                eps,
            )
        else:
            x = torch_layernorm(x, weights.ln1_gamma[layer], weights.ln1_beta[layer], eps)

        linear = (
            lambda data, weight: custom_linear(exts["linear"], data, weight)
            if variant.custom_linear
            else torch_linear(data, weight)
        )
        q, k, v = split_qkv(linear(x, weights.qkv[layer]), shape)
        if variant.attention == "manual":
            attn = manual_attention(q, k, v)
        elif variant.attention == "sdpa":
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        elif variant.attention == "custom_3_kernel":
            attn = custom_3_kernel_attention(
                exts["scaled_qk"],
                exts["softmax"],
                exts["attention_v"],
                q,
                k,
                v,
            )
        elif variant.attention == "fused":
            attn = exts["fused_attention"].fused_attention(q, k, v)
        elif variant.attention == "flash":
            attn = exts["flashattention"].FlashAttention(q, k, v)
        else:
            raise ValueError(f"unknown attention implementation: {variant.attention}")
        x = residual + linear(merge_heads(attn, shape), weights.proj[layer])

        residual = x
        if variant.custom_layernorm:
            x = exts["layernorm"].layernorm(
                x,
                weights.ln2_gamma[layer],
                weights.ln2_beta[layer],
                eps,
            )
        else:
            x = torch_layernorm(x, weights.ln2_gamma[layer], weights.ln2_beta[layer], eps)
        if variant.custom_mlp:
            hidden = exts["mlp"].fused_MLPlinear_GELU(
                x.reshape(-1, shape.embed_dim).contiguous(),
                weights.mlp1_weight[layer],
                weights.mlp1_bias[layer],
            ).reshape(shape.batch, shape.tokens, shape.mlp_dim)
        else:
            hidden = F.gelu(
                F.linear(x, weights.mlp1_weight[layer], weights.mlp1_bias[layer]),
                approximate="tanh",
            )
        x = residual + linear(hidden, weights.mlp2[layer])

    return x


def format_rows(
    rows: list[tuple[str, str, Timing | None, float | None, float | None, float | None, float | None]],
) -> str:
    header = (
        "variant,shape,status,median_ms,mean_ms,min_ms,max_ms,"
        "max_abs_error,mean_abs_error,speedup_vs_pytorch_manual,speedup_vs_pytorch_sdpa"
    )
    body = []
    timings = {name: timing for name, _, timing, *_ in rows if timing is not None}
    manual = timings.get("pytorch_manual")
    sdpa = timings.get("pytorch_sdpa")
    for name, shape, timing, max_err, mean_err, speed_manual, speed_sdpa in rows:
        if timing is None:
            body.append(f"{name},{shape},skipped,,,,,,,,")
            continue
        if manual is not None:
            speed_manual = manual.median_ms / timing.median_ms
        if sdpa is not None:
            speed_sdpa = sdpa.median_ms / timing.median_ms
        body.append(
            f"{name},{shape},ok,{timing.median_ms:.6f},{timing.mean_ms:.6f},"
            f"{timing.min_ms:.6f},{timing.max_ms:.6f},{max_err:.6g},{mean_err:.6g},"
            f"{speed_manual:.6f},{speed_sdpa:.6f}"
        )
    return "\n".join([header, *body])


def benchmark_shape(
    shape: VitShape,
    variants: tuple[Variant, ...],
    exts: dict[str, object],
    *,
    eps: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> str:
    x = make_inputs(shape)
    weights = make_weights(shape)
    expected = vit_forward(x, weights, shape, variants[0], exts, eps)

    rows = []
    for variant in variants:
        if variant.attention == "flash" and not shape.supports_flashattention:
            rows.append((variant.name, shape.label, None, None, None, None, None))
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
        rows.append(
            (
                variant.name,
                shape.label,
                timing,
                correctness.max_abs_error,
                correctness.mean_abs_error,
                None,
                None,
            )
        )
    return format_rows(rows)


def load_extensions() -> dict[str, object]:
    from cuda_vit.ops.attention_v_ext import load_attention_v
    from cuda_vit.ops.flashattention_ext import load_flashattention
    from cuda_vit.ops.fused_attention_ext import load_fused_attention
    from cuda_vit.ops.layernorm_ext import load_layernorm
    from cuda_vit.ops.linear_forward_ext import load_linear_forward
    from cuda_vit.ops.mlp_linear_gelu_ext import load_fused_mlp_linear_gelu
    from cuda_vit.ops.patchembedding_ext import load_patchembedding
    from cuda_vit.ops.patchembeddingv2_ext import load_patchembeddingv2
    from cuda_vit.ops.patchembeddingv3_ext import load_patchembeddingv2 as load_patchembeddingv3
    from cuda_vit.ops.scaled_qk_ext import load_scaled_qk
    from cuda_vit.ops.softmax_ext import load_softmax

    return {
        "attention_v": load_attention_v(),
        "flashattention": load_flashattention(),
        "fused_attention": load_fused_attention(),
        "layernorm": load_layernorm(),
        "linear": load_linear_forward(),
        "mlp": load_fused_mlp_linear_gelu(),
        "patch_v1": load_patchembedding(),
        "patch_v2": load_patchembeddingv2(),
        "patch_v3": load_patchembeddingv3(),
        "scaled_qk": load_scaled_qk(),
        "softmax": load_softmax(),
    }


def selected_shapes(name: str) -> tuple[VitShape, ...]:
    if name == "all":
        return tuple(SHAPES.values())
    return (SHAPES[name],)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=(*SHAPES.keys(), "all"), default="flash")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
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
    print(format_run_header("Whole ViT Benchmark", BenchmarkEnv.current(), config))

    exts = load_extensions()
    for shape in selected_shapes(args.shape):
        print()
        print(benchmark_shape(
            shape,
            VARIANTS,
            exts,
            eps=args.eps,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        ))


if __name__ == "__main__":
    main()
