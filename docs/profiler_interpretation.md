# Profiler Interpretation

Use profiler output to explain benchmark results, not to replace them. The
benchmark scripts give the headline numbers. Profilers explain where the time
and memory went.

## Torch Profiler

Run:

```bash
python -m benchmarks.attention.profile_attention --variant custom_3_kernel
```

Useful columns:

- `Self CPU`: CPU time spent directly in that row.
- `CPU total`: CPU time for that row including child operations.
- `Self CUDA`: CUDA time spent directly in that row.
- `CUDA total`: CUDA time for that row including child kernels.
- `CUDA time avg`: average CUDA time per call.
- `# of Calls`: how many times that operation or kernel occurred.
- `CUDA Mem`: CUDA memory attributed to that row including children.
- `Self CUDA Mem`: CUDA memory allocated directly by that row.

Use:

- Compare `# of Calls` between manual, custom 3-kernel, fused, and flash paths.
- Use `CUDA total` to explain which operation dominates GPU time.
- Use `cudaLaunchKernel` rows as evidence of CPU launch overhead.
- Use memory columns to explain intermediate allocation differences.


## Attention Memory Benchmark

Run:

```bash
python -m benchmarks.attention.bench_attention_memory
```

Columns:

- `variant`: implementation being measured.
- `shape`: benchmark shape.
- `peak_allocated_bytes`: high-water active CUDA allocation.
- `peak_reserved_bytes`: high-water memory reserved by PyTorch's caching allocator.
- `status`: `ok` or skipped reason.

Use `peak_allocated_bytes` for presentation comparisons. It is closer to the
actual tensors needed by the implementation.

Use `peak_reserved_bytes` as allocator context only. Reserved memory can stay
high because PyTorch keeps freed blocks for reuse.

Expected attention pattern:

- manual attention can allocate large `[B, H, T, T]` intermediates.
- custom 3-kernel attention also materializes scores and probabilities.
- fused and flash variants should reduce intermediate pressure when implemented
  that way.

## Nsight Systems

Nsight Systems is a timeline profiler. Use it to answer:

- Are there CPU gaps between CUDA kernels?
- How many kernel launches are visible?
- Is launch overhead significant?
- Do fused kernels make the timeline more compact?

Presentation use:

- Show one timeline screenshot for `custom_3_kernel`.
- Show one timeline screenshot for `fused_attention` or `flashattention`.
- Highlight launch gaps and the number of visible kernels.

Do not use Nsight Systems to diagnose low occupancy or memory coalescing. Use
Nsight Compute for that.

## Nsight Compute

Nsight Compute is a per-kernel hardware profiler. Use it to explain why a
specific custom CUDA kernel is slow or fast.

Useful metrics:

- `Achieved Occupancy`: fraction of possible active warps actually active.
- `SM Utilization`: how busy the streaming multiprocessors were.
- `Warp Execution Efficiency`: how well threads in each warp did useful work.
- `Global Load/Store Efficiency`: whether global memory accesses are coalesced.
- `Memory Throughput`: achieved memory bandwidth.
- `L1/L2 Hit Rate`: cache reuse and locality.
- `Registers Per Thread`: register pressure.
- `Shared Memory Per Block`: shared-memory pressure.
- `Top Stall Reasons`: why warps waited.

Interpretation examples:

- Low occupancy plus high register count suggests register pressure.
- Low global load efficiency suggests uncoalesced memory access.
- High memory stalls suggest the kernel is memory-bound.
- High barrier stalls suggest synchronization overhead.
- Good TFLOP/s with poor memory metrics can still be fine for compute-bound ops.
