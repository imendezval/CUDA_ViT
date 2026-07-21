# Benchmarks

Benchmark implementations are grouped by the thing being measured:

- `common/`: shared timing, shape, and report helpers
- `ops/`: single-kernel and primitive-operation benchmarks
- `patch_embedding/`: patch embedding benchmarks and scaling sweeps
- `attention/`: attention benchmarks, memory checks, scaling, and profiler entrypoints
- `vit/`: whole-ViT benchmarks and scaling sweeps
- `reporting/`: plot and table generation

Run benchmark commands through the grouped modules, for example:

```bash
python -m benchmarks.vit.bench_vit_scaling
python -m benchmarks.attention.profile_attention
python -m benchmarks.reporting.generate_plots
```
