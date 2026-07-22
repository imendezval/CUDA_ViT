from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from benchmarks.common.report import (
    ATTENTION_MEMORY_HEADER,
    ATTENTION_MEMORY_SCALING_HEADER,
    ATTENTION_SCALING_HEADER,
    PATCH_SCALING_HEADER,
    VIT_SCALING_HEADER,
    VIT_BREAKDOWN_HEADER,
    read_rows,
)


COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)

SERIES_COLORS = {
    "pytorch_sdpa": "#2563eb",
    "custom_3_kernel": "#16a34a",
    "fused_attention": "#9333ea",
    "flashattention": "#ea580c",
}


DISPLAY_NAMES = {
    "flashattention": "FlashAttention",
    "fused_attention": "Fused Attention",
    "custom_3_kernel": "Custom 3 Part Kernel",
    "pytorch_sdpa": "PyTorch SDPA",
    "pytorch_manual": "PyTorch Manual",
    "pytorch_conv2d": "PyTorch Conv2d",
    "patchembedding": "PatchEmbedding v1",
    "patchembeddingv2": "PatchEmbedding v2",
    "patchembeddingv3": "PatchEmbedding v3",
    "custom_v1_3_kernel": "PatchEmbedding v1 + 3 Part Kernel",
    "custom_v2_3_kernel": "PatchEmbedding v2 + 3 Part Kernel",
    "custom_v2_fused_attention": "PatchEmbedding v2 + Fused Attention",
    "custom_v2_flashattention": "PatchEmbedding v2 + FlashAttention",
    "custom_v2_3_kernel_torch_linear": "Custom v2 3 Part + cuBLAS Linear",
    "custom_v2_fused_attention_torch_linear": "Custom v2 Fused + cuBLAS Linear",
    "custom_v2_flashattention_torch_linear": "PatchEmbedding v2 + FlashAttention + cuBLAS Linear",
    "custom_flash_own_linear": "Custom FlashAttention + Custom Linear",
    "custom_flash_cublas_linear": "Custom FlashAttention + cuBLAS Linear",
    "custom_v3_flash_own_linear": "PatchEmbedding v3 + FlashAttention + Custom Linear",
    "custom_v3_flash_cublas_linear": "PatchEmbedding v3 + FlashAttention + cuBLAS Linear",
}

AMDAHL_DISPLAY_NAMES = {
    "custom_flash_own_linear": "Custom ViT + Custom Linear",
    "custom_flash_cublas_linear": "Custom ViT + cuBLAS Linear",
}

VIT_BREAKDOWN_PLOT_DISPLAY_NAMES = {
    "custom_v3_flash_cublas_linear": "Custom ViT + cuBLAS Linear",
    "pytorch_sdpa": "PyTorch Baseline",
}

COMPONENT_NAMES = {
    "patch_embedding": "Patch Embedding",
    "layernorm": "LayerNorm",
    "qkv_projection": "QKV Projection",
    "attention": "Attention",
    "output_projection": "Output Projection",
    "mlp": "MLP",
    "residual_add": "Residual Adds",
    "total": "Total",
}

AMDAHL_LOCAL_SPEEDUPS = (2.0, 5.0, 10.0, float("inf"))
VIT_PRESENTATION_COMPONENTS = {
    "patch_embedding",
    "layernorm",
    "qkv_projection",
    "attention",
    "output_projection",
    "mlp",
    "residual_add",
}
VIT_AMDAHL_COMPONENTS = VIT_PRESENTATION_COMPONENTS - {"residual_add"}
VIT_BREAKDOWN_PLOT_VARIANTS = (
    "custom_v3_flash_cublas_linear",
    "pytorch_sdpa",
)

ATTENTION_PLOT_VARIANTS = (
    "pytorch_sdpa",
    "custom_3_kernel",
    "fused_attention",
    "flashattention",
)

ATTENTION_CUSTOM_PLOT_VARIANTS = (
    "custom_3_kernel",
    "fused_attention",
    "flashattention",
)

VIT_PLOT_VARIANTS = (
    "pytorch_manual",
    "pytorch_sdpa",
    "custom_v2_flashattention",
    "custom_v2_flashattention_torch_linear",
)

VIT_PLOT_VARIANTS_WITH_PEV1_3PART = (
    "pytorch_manual",
    "pytorch_sdpa",
    "custom_v1_3_kernel",
    "custom_v2_flashattention",
    "custom_v2_flashattention_torch_linear",
)

METRIC_LABELS = {
    "median_ms": "Latency (ms)",
    "speedup_vs_pytorch_sdpa": "Speedup vs PyTorch SDPA (x)",
    "speedup_vs_pytorch_conv2d": "Speedup vs PyTorch Conv2d (x)",
    "throughput_scale": "Throughput Scaling (x)",
    "images_per_s": "Images / s",
    "tokens_per_s": "Tokens / s",
    "peak_allocated_mib": "Peak Allocated Memory (MiB)",
    "extra_peak_allocated_vs_sdpa_mib": "Extra Peak Memory vs PyTorch SDPA (MiB)",
}

METRIC_TITLES = {
    "median_ms": "Latency",
    "speedup_vs_pytorch_sdpa": "Speedup",
    "speedup_vs_pytorch_conv2d": "Speedup",
    "throughput_scale": "Throughput Scaling",
    "images_per_s": "Image Throughput",
    "tokens_per_s": "Token Throughput",
    "peak_allocated_mib": "Peak Memory",
    "extra_peak_allocated_vs_sdpa_mib": "Extra Peak Memory",
}

SWEEP_LABELS = {
    "batch": "Batch Size",
    "image": "Image Size (px)",
    "patch": "Patch Size (px)",
    "embed": "Embedding Dimension",
    "sequence": "Sequence Length (tokens)",
    "heads": "Attention Heads",
    "head_dim": "Head Dimension",
    "patches": "Number of Patches",
}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def amdahl_display_name(name: str) -> str:
    return AMDAHL_DISPLAY_NAMES.get(name, display_name(name))


def vit_breakdown_display_name(name: str) -> str:
    return VIT_BREAKDOWN_PLOT_DISPLAY_NAMES.get(name, display_name(name))


def series_color(name: str, index: int) -> str:
    return SERIES_COLORS.get(name, COLORS[index % len(COLORS)])


def ordered_series(
    series: dict[str, list[tuple[str, float]]],
    order: tuple[str, ...],
) -> dict[str, list[tuple[str, float]]]:
    ordered = {name: series.get(name, []) for name in order}
    for name, values in series.items():
        if name not in ordered:
            ordered[name] = values
    return ordered


def component_name(name: str) -> str:
    return COMPONENT_NAMES.get(name, name)


def local_speedup_name(value: float) -> str:
    return "Infinite Component Speedup" if value == float("inf") else f"{value:g}x Component Speedup"


def amdahl_speedup_from_share(share_pct: float, local_speedup: float) -> float:
    fraction = share_pct / 100.0
    if local_speedup == float("inf"):
        return 1.0 / (1.0 - fraction) if fraction < 1.0 else float("inf")
    return 1.0 / ((1.0 - fraction) + fraction / local_speedup)


def axis_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def chart_title(subject: str, metric: str, sweep: str | None) -> str:
    metric_title = METRIC_TITLES.get(metric, metric)
    if sweep is None:
        return f"{subject} {metric_title}"
    return f"{subject} {metric_title} vs {SWEEP_LABELS.get(sweep, sweep)}"


def x_axis_label(sweep: str | None) -> str:
    return SWEEP_LABELS.get(sweep, "Shape")


def nice_axis_max(value: float) -> float:
    if value <= 0:
        return 1.0
    scaled = value * 1.05
    exponent = math.floor(math.log10(scaled))
    base = 10 ** exponent
    for multiple in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        candidate = multiple * base
        if candidate >= scaled:
            return candidate
    return 10.0 * base


def tick_text(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def point_line_plot(
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[str, float]]],
    *,
    width: int = 1280,
    height: int = 720,
    y_min: float = 0.0,
) -> str:
    margin_left = 110
    margin_right = 440
    margin_top = 70
    margin_bottom = 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    labels = []
    for values in series.values():
        for label, _ in values:
            if label not in labels:
                labels.append(label)

    y_values = [value for values in series.values() for _, value in values]
    y_max = max(y_values) if y_values else 1.0
    y_max = nice_axis_max(y_max)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_pos(label: str) -> float:
        if len(labels) == 1:
            return margin_left + plot_w / 2
        return margin_left + labels.index(label) * plot_w / (len(labels) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="38" font-family="sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{margin_left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(x_label)}</text>',
        f'<text x="24" y="{margin_top + plot_h / 2}" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_h / 2})">{escape(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for idx in range(5):
        value = y_min + (y_max - y_min) * idx / 4
        y = y_pos(value)
        out.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick_text(value)}</text>')

    for label in labels:
        x = x_pos(label)
        out.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(label)}</text>')

    for idx, (name, values) in enumerate(series.items()):
        color = series_color(name, idx)
        if values:
            points = " ".join(f"{x_pos(label):.2f},{y_pos(value):.2f}" for label, value in values)
            out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for label, value in values:
                out.append(f'<circle cx="{x_pos(label):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}"/>')
        legend_y = margin_top + idx * 24
        legend_x = width - margin_right + 40
        out.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        out.append(f'<text x="{legend_x + 22}" y="{legend_y + 3}" font-family="sans-serif" font-size="14">{escape(display_name(name))}</text>')

    out.append("</svg>")
    return "\n".join(out)


def bar_plot(
    title: str,
    x_label: str,
    y_label: str,
    values: list[tuple[str, float]],
    *,
    width: int = 960,
    height: int = 720,
) -> str:
    margin_left = 110
    margin_right = 40
    margin_top = 70
    margin_bottom = 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_max = nice_axis_max(max((value for _, value in values), default=1.0))
    bar_w = plot_w / max(len(values), 1) * 0.65

    def y_pos(value: float) -> float:
        return margin_top + plot_h - (value / y_max) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="38" font-family="sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{margin_left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(x_label)}</text>',
        f'<text x="24" y="{margin_top + plot_h / 2}" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_h / 2})">{escape(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for idx in range(5):
        value = y_max * idx / 4
        y = y_pos(value)
        out.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick_text(value)}</text>')

    for idx, (label, value) in enumerate(values):
        slot = plot_w / max(len(values), 1)
        x = margin_left + idx * slot + (slot - bar_w) / 2
        y = y_pos(value)
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{margin_top + plot_h - y:.2f}" fill="{COLORS[idx % len(COLORS)]}"/>')
        out.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick_text(value)}</text>')
        out.append(f'<text x="{x + bar_w / 2:.2f}" y="{margin_top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(display_name(label))}</text>')

    out.append("</svg>")
    return "\n".join(out)


def stacked_bar_plot(
    title: str,
    x_label: str,
    y_label: str,
    rows: list[dict[str, str]],
    metric: str,
    *,
    width: int = 1280,
    height: int = 720,
) -> str:
    margin_left = 110
    margin_right = 420
    margin_top = 70
    margin_bottom = 105
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    variants = []
    components = []
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["component"] not in VIT_PRESENTATION_COMPONENTS:
            continue
        if row["variant"] not in VIT_BREAKDOWN_PLOT_VARIANTS:
            continue
        if row["variant"] not in variants:
            variants.append(row["variant"])
        if row["component"] not in components:
            components.append(row["component"])
        values[(row["variant"], row["component"])] = float(row[metric])

    totals = {
        variant: sum(values.get((variant, component), 0.0) for component in components)
        for variant in VIT_BREAKDOWN_PLOT_VARIANTS
        if variant in variants
    }
    variants = [variant for variant in VIT_BREAKDOWN_PLOT_VARIANTS if variant in variants]
    y_max = 100.0 if metric == "share_pct" else nice_axis_max(max(totals.values(), default=1.0))
    bar_w = min(120.0, plot_w / max(len(variants), 1) * 0.45)

    def y_pos(value: float) -> float:
        return margin_top + plot_h - (value / y_max) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="38" font-family="sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{margin_left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(x_label)}</text>',
        f'<text x="24" y="{margin_top + plot_h / 2}" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_h / 2})">{escape(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for idx in range(5):
        value = y_max * idx / 4
        y = y_pos(value)
        out.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick_text(value)}</text>')

    for variant_idx, variant in enumerate(variants):
        slot = plot_w / max(len(variants), 1)
        x = margin_left + variant_idx * slot + (slot - bar_w) / 2
        cumulative = 0.0
        for component_idx, component in enumerate(components):
            value = values.get((variant, component), 0.0)
            y1 = y_pos(cumulative)
            y2 = y_pos(cumulative + value)
            color = COLORS[component_idx % len(COLORS)]
            out.append(
                f'<rect x="{x:.2f}" y="{y2:.2f}" width="{bar_w:.2f}" '
                f'height="{y1 - y2:.2f}" fill="{color}"/>'
            )
            cumulative += value
        label_x = x + bar_w / 2
        out.append(f'<text x="{label_x:.2f}" y="{margin_top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(vit_breakdown_display_name(variant))}</text>')
        if metric == "median_ms":
            out.append(f'<text x="{label_x:.2f}" y="{y_pos(cumulative) - 8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick_text(cumulative)}</text>')

    for idx, component in enumerate(components):
        color = COLORS[idx % len(COLORS)]
        legend_y = margin_top + idx * 24
        legend_x = width - margin_right + 40
        out.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        out.append(f'<text x="{legend_x + 22}" y="{legend_y + 3}" font-family="sans-serif" font-size="14">{escape(component_name(component))}</text>')

    out.append("</svg>")
    return "\n".join(out)


def extract_sweep_label(sweep: str | None, shape: str) -> str:
    if sweep is None:
        return shape

    patterns = {
        "batch": r"B(\d+)",
        "image": r"(?:I|H)(\d+)",
        "patch": r"P(\d+)",
        "embed": r"D(\d+)",
        "sequence": r"T(\d+)",
        "heads": r"_H(\d+)_",
        "head_dim": r"Dh(\d+)",
        "patches": r"T(\d+)",
    }
    pattern = patterns.get(sweep)
    if pattern is None:
        return shape
    match = re.search(pattern, shape)
    return match.group(1) if match else shape


def _extract_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"could not parse shape value with pattern {pattern}: {text}")
    return int(match.group(1))


def metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "images_per_s":
        batch = _extract_int(r"B(\d+)", row["shape"])
        return batch * 1000.0 / float(row["median_ms"])
    if metric == "tokens_per_s":
        batch = _extract_int(r"B(\d+)", row["shape"])
        tokens = _extract_int(r"T(\d+)", row["shape"])
        return batch * tokens * 1000.0 / float(row["median_ms"])
    if metric == "peak_allocated_mib":
        return int(row["peak_allocated_bytes"]) / (1024 * 1024)
    return float(row[metric])


def scaling_series(
    rows: list[dict[str, str]],
    metric: str,
    *,
    sweep: str | None,
) -> dict[str, list[tuple[str, float]]]:
    series = defaultdict(list)
    for row in rows:
        if sweep is not None and row["sweep"] != sweep:
            continue
        label = (
            extract_sweep_label(sweep, row["shape"])
            if sweep is not None
            else f'{row["sweep"]}:{row["shape"]}'
        )
        value = metric_value(row, metric)
        series_key = row["name"] if "name" in row else row["variant"]
        series[series_key].append((label, value))
    return dict(series)


def plot_attention_scaling(
    path: Path,
    output: Path,
    metric: str,
    *,
    sweep: str | None,
) -> None:
    rows = [
        row
        for row in read_rows(path, ATTENTION_SCALING_HEADER)
        if row["name"] in ATTENTION_PLOT_VARIANTS
    ]
    output.write_text(
        point_line_plot(
            chart_title("Attention", metric, sweep),
            x_axis_label(sweep),
            axis_label(metric),
            ordered_series(scaling_series(rows, metric, sweep=sweep), ATTENTION_PLOT_VARIANTS),
        )
    )


def plot_patch_scaling(
    path: Path,
    output: Path,
    metric: str,
    *,
    sweep: str | None,
) -> None:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    output.write_text(
        point_line_plot(
            chart_title("Patch Embedding", metric, sweep),
            x_axis_label(sweep),
            axis_label(metric),
            scaling_series(rows, metric, sweep=sweep),
        )
    )


def plot_attention_memory(path: Path, output: Path) -> None:
    rows = read_rows(path, ATTENTION_MEMORY_HEADER)
    values = [
        (row["variant"], int(row["peak_allocated_bytes"]) / (1024 * 1024))
        for row in rows
        if row["status"] == "ok" and row["variant"] in ATTENTION_PLOT_VARIANTS
    ]
    output.write_text(
        bar_plot(
            "Attention Peak Memory by Implementation",
            "Implementation",
            "Peak Allocated Memory (MiB)",
            values,
        )
    )


def plot_attention_memory_scaling(path: Path, output: Path) -> None:
    rows = [
        row
        for row in read_rows(path, ATTENTION_MEMORY_SCALING_HEADER)
        if row["status"] == "ok" and row["variant"] in ATTENTION_PLOT_VARIANTS
    ]
    output.write_text(
        point_line_plot(
            "Attention Peak Memory vs Sequence Length",
            x_axis_label("sequence"),
            axis_label("peak_allocated_mib"),
            ordered_series(
                scaling_series(rows, "peak_allocated_mib", sweep="sequence"),
                ATTENTION_PLOT_VARIANTS,
            ),
        )
    )


def extra_memory_series_vs_sdpa(rows: list[dict[str, str]]) -> dict[str, list[tuple[str, float]]]:
    baselines = {
        row["shape"]: int(row["peak_allocated_bytes"])
        for row in rows
        if row["variant"] == "pytorch_sdpa" and row["status"] == "ok"
    }
    series = defaultdict(list)
    for row in rows:
        if (
            row["status"] != "ok"
            or row["shape"] not in baselines
            or row["variant"] not in ATTENTION_PLOT_VARIANTS
            or row["variant"] == "pytorch_sdpa"
        ):
            continue
        label = extract_sweep_label("sequence", row["shape"])
        extra_mib = (int(row["peak_allocated_bytes"]) - baselines[row["shape"]]) / (1024 * 1024)
        series[row["variant"]].append((label, extra_mib))
    return ordered_series(dict(series), ATTENTION_CUSTOM_PLOT_VARIANTS)


def plot_attention_extra_memory_scaling(path: Path, output: Path) -> None:
    rows = read_rows(path, ATTENTION_MEMORY_SCALING_HEADER)
    output.write_text(
        point_line_plot(
            "Attention Extra Peak Memory vs Sequence Length",
            x_axis_label("sequence"),
            axis_label("extra_peak_allocated_vs_sdpa_mib"),
            extra_memory_series_vs_sdpa(rows),
        )
    )


def plot_vit_scaling(
    path: Path,
    output: Path,
    *,
    sweep: str,
    include_pev1_3part: bool = False,
) -> None:
    variants = VIT_PLOT_VARIANTS_WITH_PEV1_3PART if include_pev1_3part else VIT_PLOT_VARIANTS
    title = chart_title("Whole ViT Inference", "median_ms", sweep)
    if include_pev1_3part:
        title += " with PEv1 3 Part Kernel"
    rows = [
        row
        for row in read_rows(path, VIT_SCALING_HEADER)
        if row["status"] == "ok" and row["sweep"] == sweep and row["variant"] in variants
    ]
    output.write_text(
        point_line_plot(
            title,
            x_axis_label(sweep),
            axis_label("median_ms"),
            scaling_series(rows, "median_ms", sweep=sweep),
        )
    )


def plot_vit_breakdown(path: Path, output: Path, *, metric: str) -> None:
    rows = read_rows(path, VIT_BREAKDOWN_HEADER)
    title = "Whole ViT Component Breakdown"
    y_label = "Latency (ms)" if metric == "median_ms" else "Runtime Share (%)"
    if metric == "share_pct":
        title += " by Runtime Share"
    output.write_text(
        stacked_bar_plot(
            title,
            "Implementation",
            y_label,
            rows,
            metric,
        )
    )


def amdahl_series(rows: list[dict[str, str]], variant: str) -> dict[str, list[tuple[str, float]]]:
    variant_rows = [
        row
        for row in rows
        if row["variant"] == variant and row["component"] in VIT_AMDAHL_COMPONENTS
    ]
    series = {}
    for local_speedup in AMDAHL_LOCAL_SPEEDUPS:
        series[local_speedup_name(local_speedup)] = [
            (
                component_name(row["component"]),
                amdahl_speedup_from_share(float(row["share_pct"]), local_speedup),
            )
            for row in variant_rows
        ]
    return series


def plot_vit_amdahl(path: Path, output: Path, *, variant: str) -> None:
    rows = read_rows(path, VIT_BREAKDOWN_HEADER)
    output.write_text(
        point_line_plot(
            f"Amdahl Speedup Limits for {amdahl_display_name(variant)}",
            "Optimized Component",
            "Whole-ViT Speedup Limit (x)",
            amdahl_series(rows, variant),
            width=1500,
            y_min=1.0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=(
            "attention-scaling",
            "patch-scaling",
            "attention-memory",
            "attention-memory-scaling",
            "attention-extra-memory-scaling",
            "vit-scaling",
            "vit-breakdown",
            "vit-amdahl",
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--metric",
        choices=(
            "median_ms",
            "speedup_vs_pytorch_sdpa",
            "speedup_vs_pytorch_conv2d",
            "throughput_scale",
            "images_per_s",
            "tokens_per_s",
            "share_pct",
        ),
        default="median_ms",
    )
    parser.add_argument(
        "--sweep",
        help="Plot only one scaling sweep, for example embed, batch, sequence, heads, or head_dim.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "attention-scaling":
        plot_attention_scaling(args.input, args.output, args.metric, sweep=args.sweep)
    elif args.kind == "patch-scaling":
        plot_patch_scaling(args.input, args.output, args.metric, sweep=args.sweep)
    elif args.kind == "attention-memory":
        plot_attention_memory(args.input, args.output)
    elif args.kind == "attention-memory-scaling":
        plot_attention_memory_scaling(args.input, args.output)
    elif args.kind == "attention-extra-memory-scaling":
        plot_attention_extra_memory_scaling(args.input, args.output)
    elif args.kind == "vit-scaling":
        if args.sweep is None:
            raise ValueError("--sweep is required for vit-scaling")
        plot_vit_scaling(args.input, args.output, sweep=args.sweep)
    elif args.kind == "vit-breakdown":
        if args.metric not in {"median_ms", "share_pct"}:
            raise ValueError("vit-breakdown supports --metric median_ms or share_pct")
        plot_vit_breakdown(args.input, args.output, metric=args.metric)
    elif args.kind == "vit-amdahl":
        if args.sweep is None:
            raise ValueError("--sweep must name the variant for vit-amdahl")
        plot_vit_amdahl(args.input, args.output, variant=args.sweep)
    else:
        raise AssertionError(args.kind)

    print(args.output)


if __name__ == "__main__":
    main()
