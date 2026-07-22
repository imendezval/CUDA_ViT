from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median

import torch
import torch.nn.functional as F

from benchmarks.common.core import BenchmarkConfig, BenchmarkEnv, check_close, format_run_header
from benchmarks.vit.bench_vit import (
    Variant,
    VitShape,
    VitWeights,
    custom_3_kernel_attention,
    custom_linear,
    load_extensions,
    make_inputs,
    make_weights,
    merge_heads,
    pytorch_patch_embed,
    split_qkv,
    torch_layernorm,
    torch_linear,
    vit_forward,
)


SHAPES = {
    "flash": VitShape(2, 3, 128, 16, 192, 3, 4, 2, False),
    "large_tokens": VitShape(2, 3, 256, 16, 192, 3, 4, 2, False),
}

VARIANTS = (
    Variant("pytorch_sdpa", "pytorch", "sdpa", False, False, False),
    Variant("custom_flash_own_linear", "custom_v2", "flash", True, True, True),
    Variant("custom_flash_cublas_linear", "custom_v2", "flash", False, True, False),
    Variant("custom_v3_flash_own_linear", "custom_v3", "flash", True, True, True),
    Variant("custom_v3_flash_cublas_linear", "custom_v3", "flash", False, True, False),
)

COMPONENTS = (
    "patch_embedding",
    "token_setup",
    "layernorm",
    "qkv_projection",
    "attention",
    "output_projection",
    "mlp",
    "residual_add",
)


@dataclass(frozen=True)
class ComponentTiming:
    component: str
    samples_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return median(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return mean(self.samples_ms)


def timed_component(
    timings: dict[str, list[torch.cuda.Event]],
    component: str,
    op: Callable[[], torch.Tensor],
) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = op()
    end.record()
    timings[component].extend((start, end))
    return out


def linear_for_variant(variant: Variant, exts: dict[str, object]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if variant.custom_linear:
        return lambda data, weight: custom_linear(exts["linear"], data, weight)
    return torch_linear


def patch_for_variant(
    x: torch.Tensor,
    weights: VitWeights,
    shape: VitShape,
    variant: Variant,
    exts: dict[str, object],
) -> torch.Tensor:
    if variant.patch == "pytorch":
        return pytorch_patch_embed(x, weights.patch, shape)
    if variant.patch == "custom_v1":
        return exts["patch_v1"].patchembedding(x, weights.patch)
    if variant.patch == "custom_v2":
        return exts["patch_v2"].patchembeddingv2(x, weights.patch)
    if variant.patch == "custom_v3":
        return exts["patch_v3"].patchembeddingv3(x, weights.patch)
    raise ValueError(f"unknown patch implementation: {variant.patch}")


def attention_for_variant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: Variant,
    exts: dict[str, object],
) -> torch.Tensor:
    if variant.attention == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    if variant.attention == "custom_3_kernel":
        return custom_3_kernel_attention(
            exts["scaled_qk"],
            exts["softmax"],
            exts["attention_v"],
            q,
            k,
            v,
        )
    if variant.attention == "flash":
        return exts["flashattention"].FlashAttention(q, k, v)
    raise ValueError(f"unsupported breakdown attention implementation: {variant.attention}")


def instrumented_forward(
    x: torch.Tensor,
    weights: VitWeights,
    shape: VitShape,
    variant: Variant,
    exts: dict[str, object],
    eps: float,
) -> tuple[torch.Tensor, dict[str, list[torch.cuda.Event]]]:
    timings: dict[str, list[torch.cuda.Event]] = defaultdict(list)
    linear = linear_for_variant(variant, exts)

    x = timed_component(
        timings,
        "patch_embedding",
        lambda: patch_for_variant(x, weights, shape, variant, exts),
    )
    x = timed_component(
        timings,
        "token_setup",
        lambda: torch.cat((weights.cls.expand(shape.batch, -1, -1), x), dim=1) + weights.pos
        if shape.class_token
        else x + weights.pos,
    )

    for layer in range(shape.depth):
        residual = x
        x = timed_component(
            timings,
            "layernorm",
            lambda layer=layer, x=x: exts["layernorm"].layernorm(
                x,
                weights.ln1_gamma[layer],
                weights.ln1_beta[layer],
                eps,
            )
            if variant.custom_layernorm
            else torch_layernorm(x, weights.ln1_gamma[layer], weights.ln1_beta[layer], eps),
        )
        q, k, v = split_qkv(
            timed_component(
                timings,
                "qkv_projection",
                lambda layer=layer, x=x: linear(x, weights.qkv[layer]),
            ),
            shape,
        )
        attn = timed_component(
            timings,
            "attention",
            lambda q=q, k=k, v=v: attention_for_variant(q, k, v, variant, exts),
        )
        projected = timed_component(
            timings,
            "output_projection",
            lambda layer=layer, attn=attn: linear(merge_heads(attn, shape), weights.proj[layer]),
        )
        x = timed_component(
            timings,
            "residual_add",
            lambda residual=residual, projected=projected: residual + projected,
        )

        residual = x
        x = timed_component(
            timings,
            "layernorm",
            lambda layer=layer, x=x: exts["layernorm"].layernorm(
                x,
                weights.ln2_gamma[layer],
                weights.ln2_beta[layer],
                eps,
            )
            if variant.custom_layernorm
            else torch_layernorm(x, weights.ln2_gamma[layer], weights.ln2_beta[layer], eps),
        )

        def mlp(layer: int = layer, x: torch.Tensor = x) -> torch.Tensor:
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
            return linear(hidden, weights.mlp2[layer])

        mlp_out = timed_component(timings, "mlp", mlp)
        x = timed_component(
            timings,
            "residual_add",
            lambda residual=residual, mlp_out=mlp_out: residual + mlp_out,
        )

    return x, dict(timings)


def component_elapsed_ms(events: list[torch.cuda.Event]) -> float:
    return sum(
        events[idx].elapsed_time(events[idx + 1])
        for idx in range(0, len(events), 2)
    )


def benchmark_variant(
    x: torch.Tensor,
    weights: VitWeights,
    shape: VitShape,
    variant: Variant,
    exts: dict[str, object],
    *,
    eps: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, ComponentTiming]:
    with torch.inference_mode():
        for _ in range(warmup):
            vit_forward(x, weights, shape, variant, exts, eps)
        torch.cuda.synchronize()

        samples = {component: [] for component in COMPONENTS}
        for _ in range(repeats):
            repeat_events = {component: [] for component in COMPONENTS}
            for _ in range(iterations):
                _, events = instrumented_forward(x, weights, shape, variant, exts, eps)
                for component, component_events in events.items():
                    repeat_events[component].extend(component_events)
            torch.cuda.synchronize()
            for component in COMPONENTS:
                samples[component].append(component_elapsed_ms(repeat_events[component]) / iterations)

    return {
        component: ComponentTiming(component, tuple(values))
        for component, values in samples.items()
    }


def print_rows(variant: Variant, shape: VitShape, timings: dict[str, ComponentTiming]) -> None:
    total_median = sum(timing.median_ms for timing in timings.values())
    total_mean = sum(timing.mean_ms for timing in timings.values())
    for component in COMPONENTS:
        timing = timings[component]
        share = 100.0 * timing.median_ms / total_median if total_median else 0.0
        print(
            f"{variant.name},{shape.label},{component},"
            f"{timing.median_ms:.6f},{timing.mean_ms:.6f},{share:.2f}"
        )
    print(f"{variant.name},{shape.label},total,{total_median:.6f},{total_mean:.6f},100.00")


def selected_shape(name: str) -> VitShape:
    return SHAPES[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES.keys(), default="large_tokens")
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

    shape = selected_shape(args.shape)
    if not shape.supports_flashattention:
        raise RuntimeError(f"selected shape does not support FlashAttention: {shape.label}")

    config = BenchmarkConfig(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(format_run_header("Whole ViT Component Breakdown Benchmark", BenchmarkEnv.current(), config))
    print("variant,shape,component,median_ms,mean_ms,share_pct")

    exts = load_extensions()
    x = make_inputs(shape)
    weights = make_weights(shape)
    expected = vit_forward(x, weights, shape, VARIANTS[0], exts, args.eps)
    for variant in VARIANTS:
        actual = vit_forward(x, weights, shape, variant, exts, args.eps)
        check_close(variant.name, actual, expected, rtol=2e-3, atol=2e-3)
        timings = benchmark_variant(
            x,
            weights,
            shape,
            variant,
            exts,
            eps=args.eps,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print_rows(variant, shape, timings)


if __name__ == "__main__":
    main()
