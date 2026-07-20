from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ATTENTION_SCALING_HEADER = (
    "sweep",
    "shape",
    "name",
    "median_ms",
    "speedup_vs_pytorch_sdpa",
    "throughput_scale",
)
PATCH_SCALING_HEADER = (
    "sweep",
    "shape",
    "name",
    "median_ms",
    "speedup_vs_pytorch_conv2d",
    "logical_bandwidth_gbs",
    "throughput_scale",
)
ATTENTION_MEMORY_HEADER = (
    "variant",
    "shape",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "status",
)


def read_rows(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    try:
        start = lines.index(",".join(header))
    except ValueError as exc:
        raise ValueError(f"missing CSV header in {path}: {','.join(header)}") from exc

    reader = csv.DictReader(lines[start:])
    return [row for row in reader if row]


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def fmt_float(value: str, digits: int = 4) -> str:
    if value == "nan":
        return value
    return f"{float(value):.{digits}f}"


def report_attention_scaling(path: Path) -> str:
    rows = read_rows(path, ATTENTION_SCALING_HEADER)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sweep"], row["shape"])].append(row)

    summary = []
    for (sweep, shape), group in grouped.items():
        fastest = min(group, key=lambda row: float(row["median_ms"]))
        custom = next((row for row in group if row["name"] == "custom_3_kernel"), None)
        fused = next((row for row in group if row["name"] == "fused_attention"), None)
        flash = next((row for row in group if row["name"] == "flashattention"), None)
        summary.append(
            (
                sweep,
                shape,
                fastest["name"],
                fmt_float(fastest["median_ms"], 6),
                fmt_float(custom["speedup_vs_pytorch_sdpa"]) if custom else "",
                fmt_float(fused["speedup_vs_pytorch_sdpa"]) if fused else "",
                fmt_float(flash["speedup_vs_pytorch_sdpa"]) if flash else "",
            )
        )

    return markdown_table(
        (
            "Sweep",
            "Shape",
            "Fastest",
            "Fastest ms",
            "Custom vs SDPA",
            "Fused vs SDPA",
            "Flash vs SDPA",
        ),
        summary,
    )


def report_patch_scaling(path: Path) -> str:
    rows = read_rows(path, PATCH_SCALING_HEADER)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sweep"], row["shape"])].append(row)

    summary = []
    for (sweep, shape), group in grouped.items():
        fastest = min(group, key=lambda row: float(row["median_ms"]))
        v1 = next((row for row in group if row["name"] == "patchembedding"), None)
        v2 = next((row for row in group if row["name"] == "patchembeddingv2"), None)
        summary.append(
            (
                sweep,
                shape,
                fastest["name"],
                fmt_float(fastest["median_ms"], 6),
                fmt_float(v1["speedup_vs_pytorch_conv2d"]) if v1 else "",
                fmt_float(v2["speedup_vs_pytorch_conv2d"]) if v2 else "",
                fmt_float(v2["logical_bandwidth_gbs"], 1) if v2 else "",
            )
        )

    return markdown_table(
        (
            "Sweep",
            "Shape",
            "Fastest",
            "Fastest ms",
            "v1 vs conv2d",
            "v2 vs conv2d",
            "v2 GB/s",
        ),
        summary,
    )


def report_attention_memory(path: Path) -> str:
    rows = read_rows(path, ATTENTION_MEMORY_HEADER)
    summary = []
    for row in rows:
        allocated = row["peak_allocated_bytes"]
        reserved = row["peak_reserved_bytes"]
        if row["status"] == "ok":
            allocated = f"{int(allocated) / (1024 * 1024):.2f}"
            reserved = f"{int(reserved) / (1024 * 1024):.2f}"
        summary.append(
            (
                row["variant"],
                row["shape"],
                allocated,
                reserved,
                row["status"],
            )
        )

    return markdown_table(
        (
            "Variant",
            "Shape",
            "Peak Allocated MiB",
            "Peak Reserved MiB",
            "Status",
        ),
        summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=("attention-scaling", "patch-scaling", "attention-memory"),
    )
    parser.add_argument("input", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.kind == "attention-scaling":
        report = report_attention_scaling(args.input)
    elif args.kind == "patch-scaling":
        report = report_patch_scaling(args.input)
    elif args.kind == "attention-memory":
        report = report_attention_memory(args.input)
    else:
        raise AssertionError(args.kind)

    print(report)


if __name__ == "__main__":
    main()
