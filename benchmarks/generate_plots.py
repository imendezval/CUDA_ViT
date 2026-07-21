from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from benchmarks.plot import (
    plot_attention_extra_memory_scaling,
    plot_attention_memory_scaling,
    plot_attention_scaling,
    plot_patch_scaling,
    plot_vit_scaling,
)


@dataclass(frozen=True)
class PlotSpec:
    kind: str
    sweep: str
    metric: str
    filename: str


PATCH_SWEEPS = ("batch", "image", "patch", "embed")
ATTENTION_SWEEPS = ("batch", "sequence", "heads")


def plot_specs(*, include_speedup: bool) -> tuple[PlotSpec, ...]:
    specs = []
    for sweep in PATCH_SWEEPS:
        specs.append(PlotSpec("patch", sweep, "median_ms", f"{sweep}_latency.svg"))
        specs.append(
            PlotSpec(
                "patch",
                sweep,
                "images_per_s",
                f"{sweep}_images_per_s.svg",
            )
        )
        if include_speedup:
            specs.append(
                PlotSpec(
                    "patch",
                    sweep,
                    "speedup_vs_pytorch_conv2d",
                    f"{sweep}_speedup_vs_conv2d.svg",
                )
            )

    for sweep in ATTENTION_SWEEPS:
        specs.append(PlotSpec("attention", sweep, "median_ms", f"{sweep}_latency.svg"))
        specs.append(
            PlotSpec(
                "attention",
                sweep,
                "tokens_per_s",
                f"{sweep}_tokens_per_s.svg",
            )
        )
        if include_speedup:
            specs.append(
                PlotSpec(
                    "attention",
                    sweep,
                    "speedup_vs_pytorch_sdpa",
                    f"{sweep}_speedup_vs_sdpa.svg",
                )
            )
    return tuple(specs)


def managed_filenames() -> set[str]:
    filenames = {spec.filename for spec in plot_specs(include_speedup=True)}
    for sweep in PATCH_SWEEPS:
        filenames.add(f"{sweep}_throughput_scale.svg")
    for sweep in ATTENTION_SWEEPS:
        filenames.add(f"{sweep}_throughput_scale.svg")
    return filenames


def generate_plots(
    patch_scaling: Path,
    attention_scaling: Path,
    attention_memory_scaling: Path | None,
    vit_scaling: Path | None,
    output_root: Path,
    *,
    include_speedup: bool,
) -> tuple[Path, ...]:
    patch_dir = output_root / "patch_embedding" / "plots"
    attention_dir = output_root / "attention" / "plots"
    vit_dir = output_root / "vit" / "plots"
    patch_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
    vit_dir.mkdir(parents=True, exist_ok=True)
    active_specs = plot_specs(include_speedup=include_speedup)
    active_filenames = {
        spec.kind: {item.filename for item in active_specs if item.kind == spec.kind}
        for spec in active_specs
    }

    for kind, output_dir in (("patch", patch_dir), ("attention", attention_dir)):
        active = active_filenames.get(kind, set())
        for filename in managed_filenames() - active:
            stale = output_dir / filename
            if stale.exists():
                stale.unlink()

    outputs = []
    for spec in active_specs:
        if spec.kind == "patch":
            output = patch_dir / spec.filename
            plot_patch_scaling(patch_scaling, output, spec.metric, sweep=spec.sweep)
        elif spec.kind == "attention":
            output = attention_dir / spec.filename
            plot_attention_scaling(attention_scaling, output, spec.metric, sweep=spec.sweep)
        else:
            raise AssertionError(spec.kind)
        outputs.append(output)
    if attention_memory_scaling is not None and attention_memory_scaling.exists():
        output = attention_dir / "sequence_peak_memory.svg"
        plot_attention_memory_scaling(attention_memory_scaling, output)
        outputs.append(output)
        output = attention_dir / "sequence_extra_peak_memory_vs_sdpa.svg"
        plot_attention_extra_memory_scaling(attention_memory_scaling, output)
        outputs.append(output)
    if vit_scaling is not None and vit_scaling.exists():
        for sweep in ("batch", "image", "patches"):
            output = vit_dir / f"{sweep}_latency.svg"
            plot_vit_scaling(vit_scaling, output, sweep=sweep)
            outputs.append(output)
            output = vit_dir / f"{sweep}_latency_with_pev1_3part.svg"
            plot_vit_scaling(vit_scaling, output, sweep=sweep, include_pev1_3part=True)
            outputs.append(output)
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
        "--vit-scaling",
        type=Path,
        default=Path("profiles/vit/scaling.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--include-speedup",
        action="store_true",
        help="Also generate speedup-vs-baseline plots. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_plots(
        args.patch_scaling,
        args.attention_scaling,
        args.attention_memory_scaling,
        args.vit_scaling,
        args.output_root,
        include_speedup=args.include_speedup,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
