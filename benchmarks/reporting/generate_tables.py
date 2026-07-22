from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmarks.common.core import amdahl_speedup
from benchmarks.reporting.plot import display_name
from benchmarks.common.report import (
    ATTENTION_MEMORY_SCALING_HEADER,
    ATTENTION_SCALING_HEADER,
    PATCH_SCALING_HEADER,
    VIT_BREAKDOWN_HEADER,
    markdown_table,
    read_rows,
    report_attention_scaling,
    report_patch_scaling,
)


ATTENTION_PRESENTATION_SWEEPS = {"batch", "sequence", "heads"}
PATCH_VARIANTS = ("pytorch_conv2d", "patchembedding", "patchembeddingv2", "patchembeddingv3")
ATTENTION_VARIANTS = (
    "pytorch_manual",
    "pytorch_sdpa",
    "custom_3_kernel",
    "fused_attention",
    "flashattention",
)
VIT_BREAKDOWN_COMPONENTS = (
    "patch_embedding",
    "layernorm",
    "qkv_projection",
    "attention",
    "output_projection",
    "mlp",
    "residual_add",
    "total",
)
AMDAHL_LOCAL_SPEEDUPS = (2.0, 5.0, 10.0, float("inf"))
VIT_AMDAHL_COMPONENTS = set(VIT_BREAKDOWN_COMPONENTS) - {"residual_add", "total"}
PATCH_REPRESENTATIVE_SHAPE = "B2_C3_H224_W224_P16_D384"
ATTENTION_REPRESENTATIVE_SHAPE = "B2_H3_T192_Dh64"
PATCH_LARGEST = {
    "batch": "B32_C3_H224_W224_P16_D384",
    "image": "B2_C3_H512_W512_P16_D384",
    "patch": "B2_C3_H224_W224_P56_D384",
    "embed": "B2_C3_H224_W224_P16_D3072",
}
ATTENTION_LARGEST = {
    "batch": "B16_H3_T192_Dh64",
    "sequence": "B2_H3_T512_Dh64",
    "heads": "B2_H24_T192_Dh64",
}


def filter_sweeps(
    rows: list[dict[str, str]],
    allowed_sweeps: set[str],
) -> list[dict[str, str]]:
    return [row for row in rows if row["sweep"] in allowed_sweeps]


def grouped_timings(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sweep"], row["shape"])][row["name"]] = float(row["median_ms"])
    return dict(grouped)


def row_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["sweep"], row["shape"], row["name"]): row for row in rows}


def fmt_ms(value: float) -> str:
    return f"{value:.6f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.3f}x"


def fmt_local_speedup(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:g}x"


def summarize_vs_baseline(
    grouped: dict[tuple[str, str], dict[str, float]],
    baseline: str,
    variants: tuple[str, ...],
) -> list[tuple[str, str, str, str, str]]:
    summary = []
    for variant in variants:
        ratios = []
        for timings in grouped.values():
            if baseline not in timings or variant not in timings:
                continue
            ratios.append(timings[baseline] / timings[variant])
        if not ratios:
            summary.append((display_name(variant), "0/0", "", "", ""))
            continue
        wins = sum(ratio > 1.0 for ratio in ratios)
        total = len(ratios)
        summary.append(
            (
                display_name(variant),
                f"{wins}/{total}",
                fmt_ratio(median(ratios)),
                fmt_ratio(max(ratios)),
                fmt_ratio(min(ratios)),
            )
        )
    return summary


def patch_performance_summary(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    grouped = grouped_timings(rows)
    return markdown_table(
        (
            "Variant",
            "Wins vs PyTorch Conv2d",
            "Median Speed",
            "Best Speed",
            "Worst Speed",
        ),
        summarize_vs_baseline(
            grouped,
            "pytorch_conv2d",
            ("patchembedding", "patchembeddingv2", "patchembeddingv3"),
        ),
    )


def patch_custom_evolution_summary(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    grouped = grouped_timings(rows)
    comparisons = (
        ("PatchEmbedding v2 vs v1", "patchembedding", "patchembeddingv2"),
        ("PatchEmbedding v3 vs v2", "patchembeddingv2", "patchembeddingv3"),
        ("PatchEmbedding v3 vs v1", "patchembedding", "patchembeddingv3"),
    )
    table_rows = []
    for label, baseline, candidate in comparisons:
        ratios = [
            timings[baseline] / timings[candidate]
            for timings in grouped.values()
            if baseline in timings and candidate in timings
        ]
        if not ratios:
            table_rows.append((label, "0/0", "", "", ""))
            continue
        table_rows.append(
            (
                label,
                f"{sum(ratio > 1.0 for ratio in ratios)}/{len(ratios)}",
                fmt_ratio(median(ratios)),
                fmt_ratio(max(ratios)),
                fmt_ratio(min(ratios)),
            )
        )
    return markdown_table(
        (
            "Comparison",
            "Wins",
            "Median Speedup",
            "Best Speedup",
            "Worst Speedup",
        ),
        table_rows,
    )


def attention_custom_evolution_summary(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    grouped = grouped_timings(rows)
    comparisons = (
        ("Fused Attention vs Custom 3 Part Kernel", "custom_3_kernel", "fused_attention"),
        ("FlashAttention vs Fused Attention", "fused_attention", "flashattention"),
        ("FlashAttention vs Custom 3 Part Kernel", "custom_3_kernel", "flashattention"),
    )
    table_rows = []
    for label, baseline, candidate in comparisons:
        ratios = [
            timings[baseline] / timings[candidate]
            for timings in grouped.values()
            if baseline in timings and candidate in timings
        ]
        if not ratios:
            table_rows.append((label, "0/0", "", "", ""))
            continue
        table_rows.append(
            (
                label,
                f"{sum(ratio > 1.0 for ratio in ratios)}/{len(ratios)}",
                fmt_ratio(median(ratios)),
                fmt_ratio(max(ratios)),
                fmt_ratio(min(ratios)),
            )
        )
    return markdown_table(
        (
            "Comparison",
            "Wins",
            "Median Speedup",
            "Best Speedup",
            "Worst Speedup",
        ),
        table_rows,
    )


def attention_performance_summary(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    grouped = grouped_timings(rows)
    variants = (
        "pytorch_sdpa",
        "custom_3_kernel",
        "fused_attention",
        "flashattention",
    )
    manual_rows = summarize_vs_baseline(grouped, "pytorch_manual", variants)
    sdpa_rows = summarize_vs_baseline(
        grouped,
        "pytorch_sdpa",
        ("pytorch_manual", "custom_3_kernel", "fused_attention", "flashattention"),
    )
    return "\n\n".join(
        [
            "## Against PyTorch Manual",
            markdown_table(
                (
                    "Variant",
                    "Wins vs Manual",
                    "Median Speed",
                    "Best Speed",
                    "Worst Speed",
                ),
                manual_rows,
            ),
            "## Against PyTorch SDPA",
            markdown_table(
                (
                    "Variant",
                    "Wins vs SDPA",
                    "Median Speed",
                    "Best Speed",
                    "Worst Speed",
                ),
                sdpa_rows,
            ),
        ]
    )


def representative_patch_latency(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    lookup = row_lookup(rows)
    table_rows = []
    for variant in PATCH_VARIANTS:
        row = lookup.get(("embed", PATCH_REPRESENTATIVE_SHAPE, variant))
        table_rows.append(
            (
                display_name(variant),
                fmt_ms(float(row["median_ms"])) if row else "",
            )
        )
    return markdown_table(("Variant", "Latency (ms)"), table_rows)


def representative_attention_latency(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    lookup = row_lookup(rows)
    table_rows = []
    for variant in ATTENTION_VARIANTS:
        row = lookup.get(("sequence", ATTENTION_REPRESENTATIVE_SHAPE, variant))
        if row is None:
            continue
        table_rows.append((display_name(variant), fmt_ms(float(row["median_ms"]))))
    return markdown_table(("Variant", "Latency (ms)"), table_rows)


def largest_patch_latency(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    lookup = row_lookup(rows)
    table_rows = []
    for sweep, shape in PATCH_LARGEST.items():
        values = [sweep, shape]
        for variant in PATCH_VARIANTS:
            row = lookup.get((sweep, shape, variant))
            values.append(fmt_ms(float(row["median_ms"])) if row else "")
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Conv2d (ms)",
            "PatchEmbedding v1 (ms)",
            "PatchEmbedding v2 (ms)",
            "PatchEmbedding v3 (ms)",
        ),
        table_rows,
    )


def largest_attention_latency(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    lookup = row_lookup(rows)
    table_rows = []
    for sweep, shape in ATTENTION_LARGEST.items():
        values = [sweep, shape]
        for variant in ATTENTION_VARIANTS:
            row = lookup.get((sweep, shape, variant))
            values.append(fmt_ms(float(row["median_ms"])) if row else "")
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Manual (ms)",
            "PyTorch SDPA (ms)",
            "Custom 3 Part Kernel (ms)",
            "Fused Attention (ms)",
            "FlashAttention (ms)",
        ),
        table_rows,
    )


def images_per_s(row: dict[str, str]) -> float:
    batch = int(row["shape"].split("_", maxsplit=1)[0][1:])
    return batch * 1000.0 / float(row["median_ms"])


def tokens_per_s(row: dict[str, str]) -> float:
    parts = row["shape"].split("_")
    batch = int(parts[0][1:])
    tokens = int(next(part[1:] for part in parts if part.startswith("T")))
    return batch * tokens * 1000.0 / float(row["median_ms"])


def fmt_rate(value: float) -> str:
    return f"{value:.1f}"


def fmt_mib(bytes_text: str) -> str:
    return f"{int(bytes_text) / (1024 * 1024):.2f}"


def component_label(component: str) -> str:
    labels = {
        "patch_embedding": "Patch Embedding",
        "token_setup": "Token Setup",
        "layernorm": "LayerNorm",
        "qkv_projection": "QKV Projection",
        "attention": "Attention",
        "output_projection": "Output Projection",
        "mlp": "MLP",
        "residual_add": "Residual Adds",
        "total": "Total",
    }
    return labels.get(component, component)


def largest_patch_throughput(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    lookup = row_lookup(rows)
    table_rows = []
    for sweep, shape in PATCH_LARGEST.items():
        values = [sweep, shape]
        for variant in PATCH_VARIANTS:
            row = lookup.get((sweep, shape, variant))
            values.append(fmt_rate(images_per_s(row)) if row else "")
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Conv2d (images/s)",
            "PatchEmbedding v1 (images/s)",
            "PatchEmbedding v2 (images/s)",
            "PatchEmbedding v3 (images/s)",
        ),
        table_rows,
    )


def largest_attention_throughput(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    lookup = row_lookup(rows)
    table_rows = []
    for sweep, shape in ATTENTION_LARGEST.items():
        values = [sweep, shape]
        for variant in ATTENTION_VARIANTS:
            row = lookup.get((sweep, shape, variant))
            values.append(fmt_rate(tokens_per_s(row)) if row else "")
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Manual (tokens/s)",
            "PyTorch SDPA (tokens/s)",
            "Custom 3 Part Kernel (tokens/s)",
            "Fused Attention (tokens/s)",
            "FlashAttention (tokens/s)",
        ),
        table_rows,
    )


def attention_memory_scaling_table(path: Path) -> str:
    rows = filter_sweeps(
        read_rows(path, ATTENTION_MEMORY_SCALING_HEADER),
        {"sequence"},
    )
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["shape"]][row["variant"]] = row

    table_rows = []
    for shape, variants in grouped.items():
        values = [shape]
        for variant in ATTENTION_VARIANTS:
            row = variants.get(variant)
            if row is None or row["status"] != "ok":
                values.append("")
            else:
                values.append(fmt_mib(row["peak_allocated_bytes"]))
        table_rows.append(tuple(values))

    return markdown_table(
        (
            "Shape",
            "PyTorch Manual (MiB)",
            "PyTorch SDPA (MiB)",
            "Custom 3 Part Kernel (MiB)",
            "Fused Attention (MiB)",
            "FlashAttention (MiB)",
        ),
        table_rows,
    )


def vit_component_breakdown_table(path: Path) -> str:
    rows = read_rows(path, VIT_BREAKDOWN_HEADER)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["component"]][row["variant"]] = row

    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    table_rows = []
    for component in VIT_BREAKDOWN_COMPONENTS:
        values = [component_label(component)]
        for variant in variants:
            row = grouped.get(component, {}).get(variant)
            if row is None:
                values.append("")
                values.append("")
            else:
                values.append(fmt_ms(float(row["median_ms"])))
                values.append(f'{float(row["share_pct"]):.2f}')
        table_rows.append(tuple(values))

    headers = ["Component"]
    for variant in variants:
        headers.append(f"{display_name(variant)} (ms)")
        headers.append(f"{display_name(variant)} (%)")
    return markdown_table(tuple(headers), table_rows)


def amdahl_component_limit_table(path: Path) -> str:
    rows = read_rows(path, VIT_BREAKDOWN_HEADER)
    table_rows = []
    for row in rows:
        if row["component"] not in VIT_AMDAHL_COMPONENTS:
            continue
        runtime_fraction = float(row["share_pct"]) / 100.0
        values = [
            display_name(row["variant"]),
            component_label(row["component"]),
            f'{float(row["share_pct"]):.2f}',
        ]
        for local_speedup in AMDAHL_LOCAL_SPEEDUPS:
            if local_speedup == float("inf"):
                speedup = 1.0 / (1.0 - runtime_fraction) if runtime_fraction < 1.0 else float("inf")
            else:
                speedup = amdahl_speedup(runtime_fraction, local_speedup)
            values.append(fmt_ratio(speedup))
        table_rows.append(tuple(values))

    return markdown_table(
        (
            "Variant",
            "Optimized Component",
            "Runtime Share (%)",
            "Whole Speedup if 2x",
            "Whole Speedup if 5x",
            "Whole Speedup if 10x",
            "Whole Speedup if inf",
        ),
        table_rows,
    )


def takeaway_summary(patch_scaling: Path, attention_scaling: Path) -> str:
    patch_rows = read_rows(patch_scaling, PATCH_SCALING_HEADER)
    patch_grouped = grouped_timings(patch_rows)
    attention_rows = filter_sweeps(
        read_rows(attention_scaling, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    attention_grouped = grouped_timings(attention_rows)

    v2_vs_v1 = [
        timings["patchembedding"] / timings["patchembeddingv2"]
        for timings in patch_grouped.values()
        if "patchembedding" in timings and "patchembeddingv2" in timings
    ]
    v2_vs_conv = [
        timings["pytorch_conv2d"] / timings["patchembeddingv2"]
        for timings in patch_grouped.values()
        if "pytorch_conv2d" in timings and "patchembeddingv2" in timings
    ]
    v3_vs_v2 = [
        timings["patchembeddingv2"] / timings["patchembeddingv3"]
        for timings in patch_grouped.values()
        if "patchembeddingv2" in timings and "patchembeddingv3" in timings
    ]
    v3_vs_conv = [
        timings["pytorch_conv2d"] / timings["patchembeddingv3"]
        for timings in patch_grouped.values()
        if "pytorch_conv2d" in timings and "patchembeddingv3" in timings
    ]
    fused_vs_custom = [
        timings["custom_3_kernel"] / timings["fused_attention"]
        for timings in attention_grouped.values()
        if "custom_3_kernel" in timings and "fused_attention" in timings
    ]
    fused_vs_sdpa = [
        timings["pytorch_sdpa"] / timings["fused_attention"]
        for timings in attention_grouped.values()
        if "pytorch_sdpa" in timings and "fused_attention" in timings
    ]
    flash_vs_manual = [
        timings["pytorch_manual"] / timings["flashattention"]
        for timings in attention_grouped.values()
        if "pytorch_manual" in timings and "flashattention" in timings
    ]
    return markdown_table(
        ("Question", "Answer"),
        [
            (
                "Does PatchEmbedding v2 improve over v1?",
                f"Yes, median {fmt_ratio(median(v2_vs_v1))} faster.",
            ),
            (
                "Does PatchEmbedding v2 beat PyTorch Conv2d?",
                f"Rarely, {sum(ratio > 1.0 for ratio in v2_vs_conv)}/{len(v2_vs_conv)} shapes.",
            ),
            *(
                [
                    (
                        "Does PatchEmbedding v3 improve over v2?",
                        f"Median {fmt_ratio(median(v3_vs_v2))} vs v2.",
                    ),
                    (
                        "Does PatchEmbedding v3 beat PyTorch Conv2d?",
                        f"{sum(ratio > 1.0 for ratio in v3_vs_conv)}/{len(v3_vs_conv)} shapes.",
                    ),
                ]
                if v3_vs_v2 and v3_vs_conv
                else []
            ),
            (
                "Does fused attention beat the 3-part kernel?",
                f"Yes, median {fmt_ratio(median(fused_vs_custom))} faster.",
            ),
            (
                "Does fused attention beat PyTorch SDPA?",
                f"No, {sum(ratio > 1.0 for ratio in fused_vs_sdpa)}/{len(fused_vs_sdpa)} presentation shapes.",
            ),
            (
                "Does FlashAttention beat PyTorch Manual?",
                f"Sometimes, {sum(ratio > 1.0 for ratio in flash_vs_manual)}/{len(flash_vs_manual)} compatible shapes.",
            ),
        ],
    )


def generate_tables(
    patch_scaling: Path,
    attention_scaling: Path,
    attention_memory_scaling: Path | None,
    output_root: Path,
    vit_breakdown: Path | None = None,
) -> tuple[Path, ...]:
    patch_dir = output_root / "patch_embedding" / "tables"
    attention_dir = output_root / "attention" / "tables"
    vit_dir = output_root / "vit" / "tables"
    patch_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
    vit_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        patch_dir / "scaling_summary.md",
        patch_dir / "performance_summary.md",
        patch_dir / "custom_evolution_summary.md",
        patch_dir / "representative_latency.md",
        patch_dir / "largest_latency.md",
        patch_dir / "largest_throughput.md",
        attention_dir / "scaling_summary.md",
        attention_dir / "performance_summary.md",
        attention_dir / "custom_evolution_summary.md",
        attention_dir / "representative_latency.md",
        attention_dir / "largest_latency.md",
        attention_dir / "largest_throughput.md",
        attention_dir / "memory_scaling.md",
        vit_dir / "component_breakdown.md",
        vit_dir / "amdahl_limits.md",
        output_root / "takeaways.md",
    ]
    def write_table(path: Path, text: str) -> None:
        path.write_text(text + "\n")

    write_table(outputs[0], report_patch_scaling(patch_scaling))
    write_table(outputs[1], patch_performance_summary(patch_scaling))
    write_table(outputs[2], patch_custom_evolution_summary(patch_scaling))
    write_table(outputs[3], representative_patch_latency(patch_scaling))
    write_table(outputs[4], largest_patch_latency(patch_scaling))
    write_table(outputs[5], largest_patch_throughput(patch_scaling))
    attention_rows = filter_sweeps(
        read_rows(attention_scaling, ATTENTION_SCALING_HEADER),
        ATTENTION_PRESENTATION_SWEEPS,
    )
    filtered_attention = attention_dir / "_presentation_attention_scaling.csv"
    filtered_attention.write_text(
        "\n".join(
            [
                ",".join(ATTENTION_SCALING_HEADER),
                *(
                    ",".join(row[column] for column in ATTENTION_SCALING_HEADER)
                    for row in attention_rows
                ),
            ]
        )
    )
    write_table(outputs[6], report_attention_scaling(filtered_attention))
    write_table(outputs[7], attention_performance_summary(attention_scaling))
    write_table(outputs[8], attention_custom_evolution_summary(attention_scaling))
    write_table(outputs[9], representative_attention_latency(attention_scaling))
    write_table(outputs[10], largest_attention_latency(attention_scaling))
    write_table(outputs[11], largest_attention_throughput(attention_scaling))
    if attention_memory_scaling is not None and attention_memory_scaling.exists():
        write_table(outputs[12], attention_memory_scaling_table(attention_memory_scaling))
    else:
        write_table(outputs[12], "No attention memory scaling data available.")
    if vit_breakdown is not None and vit_breakdown.exists():
        write_table(outputs[13], vit_component_breakdown_table(vit_breakdown))
        write_table(outputs[14], amdahl_component_limit_table(vit_breakdown))
    else:
        write_table(outputs[13], "No whole-ViT component breakdown data available.")
        write_table(outputs[14], "No whole-ViT component breakdown data available.")
    write_table(outputs[15], takeaway_summary(patch_scaling, attention_scaling))
    filtered_attention.unlink()
    return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-scaling",
        type=Path,
        default=Path("profiles/patch_embedding/scaling.csv"),
    )
    parser.add_argument(
        "--attention-scaling",
        type=Path,
        default=Path("profiles/attention/scaling.csv"),
    )
    parser.add_argument(
        "--attention-memory-scaling",
        type=Path,
        default=Path("profiles/attention/memory_scaling.csv"),
    )
    parser.add_argument(
        "--vit-breakdown",
        type=Path,
        default=Path("profiles/vit/breakdown.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_tables(
        args.patch_scaling,
        args.attention_scaling,
        args.attention_memory_scaling,
        args.output_root,
        args.vit_breakdown,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
