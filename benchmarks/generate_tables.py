from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmarks.plot import display_name
from benchmarks.report import (
    ATTENTION_MEMORY_SCALING_HEADER,
    ATTENTION_SCALING_HEADER,
    PATCH_SCALING_HEADER,
    markdown_table,
    read_rows,
    report_attention_scaling,
    report_patch_scaling,
)


ATTENTION_PRESENTATION_SWEEPS = {"batch", "sequence", "heads"}
PATCH_VARIANTS = ("pytorch_conv2d", "patchembedding", "patchembeddingv2")
ATTENTION_VARIANTS = (
    "pytorch_manual",
    "pytorch_sdpa",
    "custom_3_kernel",
    "fused_attention",
    "flashattention",
)
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
            ("patchembedding", "patchembeddingv2"),
        ),
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
        row = lookup[("embed", PATCH_REPRESENTATIVE_SHAPE, variant)]
        table_rows.append((display_name(variant), fmt_ms(float(row["median_ms"]))))
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
            values.append(fmt_ms(float(lookup[(sweep, shape, variant)]["median_ms"])))
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Conv2d (ms)",
            "PatchEmbedding v1 (ms)",
            "PatchEmbedding v2 (ms)",
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


def largest_patch_throughput(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    lookup = row_lookup(rows)
    table_rows = []
    for sweep, shape in PATCH_LARGEST.items():
        values = [sweep, shape]
        for variant in PATCH_VARIANTS:
            values.append(fmt_rate(images_per_s(lookup[(sweep, shape, variant)])))
        table_rows.append(tuple(values))
    return markdown_table(
        (
            "Sweep",
            "Largest Shape",
            "PyTorch Conv2d (images/s)",
            "PatchEmbedding v1 (images/s)",
            "PatchEmbedding v2 (images/s)",
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
) -> tuple[Path, ...]:
    patch_dir = output_root / "patch_embedding" / "tables"
    attention_dir = output_root / "attention" / "tables"
    patch_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        patch_dir / "scaling_summary.md",
        patch_dir / "performance_summary.md",
        patch_dir / "representative_latency.md",
        patch_dir / "largest_latency.md",
        patch_dir / "largest_throughput.md",
        attention_dir / "scaling_summary.md",
        attention_dir / "performance_summary.md",
        attention_dir / "representative_latency.md",
        attention_dir / "largest_latency.md",
        attention_dir / "largest_throughput.md",
        attention_dir / "memory_scaling.md",
        output_root / "takeaways.md",
    )
    def write_table(path: Path, text: str) -> None:
        path.write_text(text + "\n")

    write_table(outputs[0], report_patch_scaling(patch_scaling))
    write_table(outputs[1], patch_performance_summary(patch_scaling))
    write_table(outputs[2], representative_patch_latency(patch_scaling))
    write_table(outputs[3], largest_patch_latency(patch_scaling))
    write_table(outputs[4], largest_patch_throughput(patch_scaling))
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
    write_table(outputs[5], report_attention_scaling(filtered_attention))
    write_table(outputs[6], attention_performance_summary(attention_scaling))
    write_table(outputs[7], representative_attention_latency(attention_scaling))
    write_table(outputs[8], largest_attention_latency(attention_scaling))
    write_table(outputs[9], largest_attention_throughput(attention_scaling))
    if attention_memory_scaling is not None and attention_memory_scaling.exists():
        write_table(outputs[10], attention_memory_scaling_table(attention_memory_scaling))
    else:
        write_table(outputs[10], "No attention memory scaling data available.")
    write_table(outputs[11], takeaway_summary(patch_scaling, attention_scaling))
    filtered_attention.unlink()
    return outputs


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
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_tables(
        args.patch_scaling,
        args.attention_scaling,
        args.attention_memory_scaling,
        args.output_root,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
