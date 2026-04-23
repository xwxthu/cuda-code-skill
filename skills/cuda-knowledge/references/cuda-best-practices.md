# CUDA C++ Best Practices Guide — Search Guide

**Location:** `references/cuda-best-practices-docs/` (138 files, 1.0MB)  
**Source:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/  
**Structure:** 20 top-level chapters, per-section `.md` files.

## Chapter Map

| Chapter | Directory | Key Topics |
|---------|-----------|------------|
| 4 | `4-application-profiling/` | Amdahl's law, strong/weak scaling, profiling workflow |
| 9 | `9-performance-metrics/` | CPU/GPU timing, bandwidth calculation |
| 10 | `10-memory-optimizations/` | H2D transfers, coalescing, L2 cache, shared memory, registers |
| 11 | `11-execution-configuration-optimizations/` | Occupancy, block size heuristics, concurrent kernels |
| 12 | `12-instruction-optimization/` | Arithmetic throughput, control flow, division, math libs |
| 13 | `13-control-flow/` | Branch divergence, predication |
| 19 | `19-recommendations-and-best-practices/` | Prioritized optimization checklist |

## Common Search Queries

```bash
# Memory coalescing patterns and examples
grep -r "coalesced\|misaligned\|strided" references/cuda-best-practices-docs/10-memory-optimizations/

# Shared memory bank conflicts
grep -r "bank conflict\|shared memory" references/cuda-best-practices-docs/10-memory-optimizations/10.2.3-shared-memory/

# Pinned (page-locked) memory for H2D transfers
grep -r "pinned\|page-locked\|cudaMallocHost" references/cuda-best-practices-docs/10-memory-optimizations/

# Occupancy calculation and tuning
grep -r "occupancy\|__launch_bounds__\|block size" references/cuda-best-practices-docs/11-execution-configuration-optimizations/

# Warp divergence and branch predication
grep -r "divergence\|predication\|branch" references/cuda-best-practices-docs/13-control-flow/

# Arithmetic throughput (integer vs float, div/mod cost)
grep -r "throughput\|division\|modulo\|reciprocal" references/cuda-best-practices-docs/12-instruction-optimization/

# Bandwidth: theoretical vs effective
grep -r "bandwidth\|theoretical\|effective" references/cuda-best-practices-docs/9-performance-metrics/

# Register pressure
grep -r "register\|spilling\|maxrregcount" references/cuda-best-practices-docs/10-memory-optimizations/10.2.7-registers/

# Asynchronous transfers (overlap compute + copy)
grep -r "asynchronous\|overlap\|cudaMemcpyAsync" references/cuda-best-practices-docs/10-memory-optimizations/

# L2 cache window tuning
grep -r "L2.*window\|accessPolicyWindow\|hitRatio" references/cuda-best-practices-docs/10-memory-optimizations/10.2.2-l2-cache/

# Amdahl's law / scaling
grep -r "Amdahl\|strong scaling\|weak scaling" references/cuda-best-practices-docs/4-application-profiling/

# Priority checklist for optimization
grep -r "priority\|recommendation" references/cuda-best-practices-docs/19-recommendations-and-best-practices/
```

## Key Files

- `9-performance-metrics/9.2.1-theoretical-bandwidth-calculation.md` — roofline baseline
- `10-memory-optimizations/10.2.1-coalesced-access-to-global-memory.md` — coalescing rules
- `10-memory-optimizations/10.2.3.1-shared-memory-and-memory-banks.md` — bank conflict diagnosis
- `10-memory-optimizations/10.2.3.2-shared-memory-in-matrix-multiplication-cab.md` — tiling example
- `10-memory-optimizations/10.2.3.4-asynchronous-copy-from-global-memory-to-shared-memory.md` — async copy
- `11-execution-configuration-optimizations/11.1-occupancy.md` — occupancy theory
- `11-execution-configuration-optimizations/11.3-thread-and-block-heuristics.md` — block size rules
- `12-instruction-optimization/12.1.1-throughput-of-native-arithmetic-instructions.md` — op costs
- `13-control-flow/13.1-branching-and-divergence.md` — divergence patterns to avoid
- `19-recommendations-and-best-practices/19.1-overall-performance-optimization-strategies.md` — checklist
