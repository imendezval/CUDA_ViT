from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from benchmarks.report import (
    ATTENTION_MEMORY_HEADER,
    ATTENTION_MEMORY_SCALING_HEADER,
    ATTENTION_SCALING_HEADER,
    PATCH_SCALING_HEADER,
    VIT_SCALING_HEADER,
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


DISPLAY_NAMES = {
    "flashattention": "FlashAttention",
    "fused_attention": "Fused Attention",
    "custom_3_kernel": "Custom 3 Part Kernel",
    "pytorch_sdpa": "PyTorch SDPA",
    "pytorch_manual": "PyTorch Manual",
    "pytorch_conv2d": "PyTorch Conv2d",
    "patchembedding": "PatchEmbedding v1",
    "patchembeddingv2": "PatchEmbedding v2",
    "custom_v1_3_kernel": "PatchEmbedding v1 + 3 Part Kernel",
    "custom_v2_3_kernel": "PatchEmbedding v2 + 3 Part Kernel",
    "custom_v2_fused_attention": "PatchEmbedding v2 + Fused Attention",
    "custom_v2_flashattention": "PatchEmbedding v2 + FlashAttention",
    "custom_v2_3_kernel_torch_linear": "Custom v2 3 Part + cuBLAS Linear",
    "custom_v2_fused_attention_torch_linear": "Custom v2 Fused + cuBLAS Linear",
    "custom_v2_flashattention_torch_linear": "PatchEmbedding v2 + FlashAttention + cuBLAS Linear",
}

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

    def x_pos(label: str) -> float:
        if len(labels) == 1:
            return margin_left + plot_w / 2
        return margin_left + labels.index(label) * plot_w / (len(labels) - 1)

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

    for label in labels:
        x = x_pos(label)
        out.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(label)}</text>')

    for idx, (name, values) in enumerate(series.items()):
        color = COLORS[idx % len(COLORS)]
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
    rows = read_rows(path, ATTENTION_SCALING_HEADER)
    output.write_text(
        point_line_plot(
            chart_title("Attention", metric, sweep),
            x_axis_label(sweep),
            axis_label(metric),
            scaling_series(rows, metric, sweep=sweep),
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
        if row["status"] == "ok"
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
        if row["status"] == "ok"
    ]
    output.write_text(
        point_line_plot(
            "Attention Peak Memory vs Sequence Length",
            x_axis_label("sequence"),
            axis_label("peak_allocated_mib"),
            scaling_series(rows, "peak_allocated_mib", sweep="sequence"),
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
        if row["status"] != "ok" or row["shape"] not in baselines:
            continue
        label = extract_sweep_label("sequence", row["shape"])
        extra_mib = (int(row["peak_allocated_bytes"]) - baselines[row["shape"]]) / (1024 * 1024)
        series[row["variant"]].append((label, extra_mib))
    return dict(series)


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
    else:
        raise AssertionError(args.kind)

    print(args.output)


if __name__ == "__main__":
    main()
