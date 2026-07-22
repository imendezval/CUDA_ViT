# Benchmarks

Benchmark implementations are grouped by the thing being measured:

- `common/`: shared timing, shape, and report helpers
- `ops/`: single-kernel and primitive-operation benchmarks
- `patch_embedding/`: patch embedding benchmarks and scaling sweeps
- `attention/`: attention benchmarks, memory checks, scaling, and profiler entrypoints
- `vit/`: whole-ViT benchmarks and scaling sweeps
- `reporting/`: plot and table generation
- `profiling/`: Nsight Systems entrypoints for fixed presentation shapes

Run benchmark commands through the grouped modules, for example:

```bash
python -m benchmarks.vit.bench_vit_scaling
python -m benchmarks.attention.profile_attention
python -m benchmarks.reporting.generate_plots
```

Nsight Systems profiling entrypoints use NVTX ranges and `cudaProfilerStart/Stop`
so the capture stays focused on the measured loop:

```bash
mkdir -p profiles/nsight/patch_embedding profiles/nsight/attention

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --force-overwrite=true \
  --output=profiles/nsight/patch_embedding/conv2d_vs_patchembeddingv3 \
  python -m benchmarks.profiling.profile_patch_embedding_nsys

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --force-overwrite=true \
  --output=profiles/nsight/attention/sdpa_vs_custom_attention \
  python -m benchmarks.profiling.profile_attention_nsys
```
