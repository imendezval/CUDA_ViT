# Nsight Profiling

Use the normal benchmark scripts for latency and speedup. Use Nsight when you
need timeline evidence or hardware-level kernel diagnostics.

## Nsight Systems

Nsight Systems answers:

- How many CUDA launches happened?
- Is the CPU leaving gaps between GPU kernels?
- Is launch overhead visible?
- Are fused kernels reducing timeline fragmentation?

Profile PyTorch SDPA:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=profiles/nsys_attention_pytorch_sdpa \
  python -m benchmarks.profile_attention \
    --variant pytorch_sdpa \
    --warmup 5 \
    --iterations 10
```

Profile the custom 3-kernel attention path:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=profiles/nsys_attention_custom_3_kernel \
  python -m benchmarks.profile_attention \
    --variant custom_3_kernel \
    --warmup 5 \
    --iterations 10
```

Profile fused attention:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=profiles/nsys_attention_fused \
  python -m benchmarks.profile_attention \
    --variant fused_attention \
    --warmup 5 \
    --iterations 10
```

Profile FlashAttention-compatible shape:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=profiles/nsys_attention_flash \
  python -m benchmarks.profile_attention \
    --variant flashattention \
    --batch 2 \
    --heads 3 \
    --tokens 192 \
    --head-dim 64 \
    --warmup 5 \
    --iterations 10
```

Open the `.nsys-rep` file in the Nsight Systems GUI. Compare:

- CUDA kernel count
- CPU launch overhead
- gaps between kernels
- total CUDA kernel time
- memory copies, if any
- whether the GPU is idle between launches

## Nsight Compute

Nsight Compute answers:

- Why is this specific CUDA kernel slow?
- Is occupancy low?
- Are global memory accesses inefficient?
- Are warps stalled on memory, barriers, or dependencies?
- Are register/shared-memory limits hurting occupancy?

Start with a small kernel set and a representative shape:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:scaled \
  --export profiles/ncu_scaled_qk \
  python -m benchmarks.bench_scaled_qk \
    --warmup 3 \
    --iterations 5 \
    --repeats 1
```

Softmax:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:softmax \
  --export profiles/ncu_softmax \
  python -m benchmarks.bench_softmax \
    --warmup 3 \
    --iterations 5 \
    --repeats 1
```

Attention @ V:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:attention \
  --export profiles/ncu_attention_v \
  python -m benchmarks.bench_attention_v \
    --warmup 3 \
    --iterations 5 \
    --repeats 1
```

Fused attention:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:fused \
  --export profiles/ncu_fused_attention \
  python -m benchmarks.bench_attention \
    --warmup 3 \
    --iterations 5 \
    --repeats 1
```

Patch embedding v2:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:Patch \
  --export profiles/ncu_patchembedding \
  python -m benchmarks.bench_patchembedding \
    --warmup 3 \
    --iterations 5 \
    --repeats 1
```

Open `.ncu-rep` files in the Nsight Compute GUI. Focus on:

- achieved occupancy
- SM utilization
- warp execution efficiency
- global load/store efficiency
- memory throughput
- L1/L2 cache hit rates
- register usage
- shared memory usage
- top stall reasons

## Presentation Use

Use benchmark scripts for the headline numbers:

- latency
- speedup
- TFLOP/s or GB/s
- scaling behavior

Use Nsight Systems screenshots to show:

- custom 3-kernel attention has multiple launches
- fused attention reduces launch count
- FlashAttention has a compact timeline when the shape is supported

Use Nsight Compute screenshots to explain:

- memory access efficiency
- warp stalls
- occupancy limits
- whether the bottleneck is memory, compute, or launch overhead
