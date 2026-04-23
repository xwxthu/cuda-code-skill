# CUDA C++ Programming Guide — Search Guide

**Location:** `references/cuda-prog-guide-docs/` (709 files, 5.9MB)  
**Source:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/  
**Structure:** 26 top-level chapters, each split into per-section `.md` files.

## Chapter Map

| Chapter | Directory | Key Topics |
|---------|-----------|------------|
| 5 | `5-programming-model/` | Kernels, thread hierarchy, block clusters, memory hierarchy, async SIMT |
| 6 | `6-programming-interface/` | nvcc, CUDA runtime, streams, graphs, multi-device, unified VA |
| 7 | `7-hardware-implementation/` | SIMT architecture, hardware multithreading, warp execution |
| 8 | `8-performance-guidelines/` | Maximize utilization/memory/instruction throughput |
| 10 | `10-c-language-extensions/` | `__global__`, `__shared__`, atomics, warp shuffle/vote/reduce, barriers, TMA, async copies |
| 11 | `11-cooperative-groups/` | thread_block, cluster_group, grid_group, tiled_partition, collectives |
| 12 | `12-cluster-launch-control/` | Thread block clusters, cancellation |
| 13 | `13-cuda-dynamic-parallelism/` | Device-side kernel launch |
| 14 | `14-virtual-memory-management/` | cuMem* APIs, physical/virtual mapping, multicast |
| 15 | `15-stream-ordered-memory-allocator/` | cudaMallocAsync, memory pools |
| 16 | `16-graph-memory-nodes/` | CUDA graph memory allocation nodes |
| 20 | `20-compute-capabilities/` | Feature table by architecture (sm_50 through sm_120) |
| 24 | `24-unified-memory-programming/` | cudaMallocManaged, `__managed__`, migration, hints |

## Common Search Queries

```bash
# Thread/block/grid hierarchy
grep -r "blockDim\|gridDim\|threadIdx\|blockIdx" references/cuda-prog-guide-docs/5-programming-model/

# SIMT architecture and warp execution
grep -r "warp\|SIMT\|divergence" references/cuda-prog-guide-docs/7-hardware-implementation/

# Memory model: coalescing, L2, shared memory
grep -r "coalesced\|global memory" references/cuda-prog-guide-docs/8-performance-guidelines/

# Warp shuffle / vote / reduce functions
grep -r "__shfl\|__ballot\|__any\|__all\|__reduce" references/cuda-prog-guide-docs/10-c-language-extensions/

# Async barrier (cuda::barrier / mbarrier)
grep -r "cuda::barrier\|mbarrier\|arrive_and_wait" references/cuda-prog-guide-docs/10-c-language-extensions/

# TMA (Tensor Memory Accelerator)
grep -r "TMA\|cudaTmaDesc\|__cp_async_bulk" references/cuda-prog-guide-docs/10-c-language-extensions/

# memcpy_async / pipeline
grep -r "memcpy_async\|cuda::pipeline" references/cuda-prog-guide-docs/10-c-language-extensions/

# Cooperative groups
grep -r "cooperative_groups\|tiled_partition\|cluster_group" references/cuda-prog-guide-docs/11-cooperative-groups/

# CUDA Graphs
grep -r "cudaGraph\|stream capture" references/cuda-prog-guide-docs/6-programming-interface/

# Atomic functions
grep -r "atomicAdd\|atomicCAS\|atomicExch" references/cuda-prog-guide-docs/10-c-language-extensions/10.14-atomic-functions/

# launch_bounds / occupancy
grep -r "__launch_bounds__\|cudaOccupancy" references/cuda-prog-guide-docs/10-c-language-extensions/

# Compute capability feature tables
grep -r "sm_90\|sm_100\|Hopper\|Blackwell" references/cuda-prog-guide-docs/20-compute-capabilities/

# Unified memory
grep -r "cudaMallocManaged\|__managed__\|cudaMemAdvise" references/cuda-prog-guide-docs/24-unified-memory-programming/

# Stream ordered allocator
grep -r "cudaMallocAsync\|cudaFreeAsync\|cudaMemPool" references/cuda-prog-guide-docs/15-stream-ordered-memory-allocator/

# Virtual memory management
grep -r "cuMemCreate\|cuMemMap\|cuMemAddressReserve" references/cuda-prog-guide-docs/14-virtual-memory-management/

# CUDA environment variables
grep -r "CUDA_VISIBLE_DEVICES\|CUDA_LAUNCH_BLOCKING" references/cuda-prog-guide-docs/22-cuda-environment-variables/

# Performance guidelines
grep -r "occupancy\|register\|instruction throughput" references/cuda-prog-guide-docs/8-performance-guidelines/
```

## Key Files

- `5-programming-model/5.2-thread-hierarchy.md` — thread/block/grid/cluster model
- `7-hardware-implementation/7.1-simt-architecture.md` — warp execution, divergence
- `8-performance-guidelines/8.3-maximize-memory-throughput.md` — memory access patterns
- `8-performance-guidelines/8.3.2-device-memory-accesses.md` — coalescing, shared memory, L2
- `10-c-language-extensions/10.14-atomic-functions/10.14.1-arithmetic-functions.md` — all atomic ops
- `10-c-language-extensions/10.22-warp-shuffle-functions.md` — `__shfl_sync`, `__shfl_xor_sync`
- `10-c-language-extensions/10.26-asynchronous-barrier.md` — `cuda::barrier`, arrive/wait
- `10-c-language-extensions/10.29-asynchronous-data-copies-using-the-tensor-memory-accelerator-tma.md` — TMA usage
- `11-cooperative-groups/11.4-group-types.md` — all group types reference
- `20-compute-capabilities/20.2-features-and-technical-specifications.md` — specs table
