---
name: cuda-knowledge
description: "CUDA kernel development, debugging, performance optimization, linear algebra, and multi-GPU communication for Claude Code. Use when writing, debugging, or optimizing CUDA code, GPU kernels, parallel algorithms, or CUDA library calls. Covers CUDA programming model (threads, blocks, grids, warps, SIMT), memory hierarchy (global/shared/register/L2), CUDA C++ language extensions (__global__, __shared__, __launch_bounds__), atomic functions, warp shuffle/vote/reduce, cooperative groups, async barriers, TMA, CUDA Graphs, stream-ordered allocator, virtual memory management, unified memory, compute capabilities, performance best practices (coalescing, occupancy, bank conflicts, instruction throughput), cuBLAS/cuBLASLt GEMM operations, CUDA Math API (half, bfloat16, FP8, FP6, FP4), NCCL multi-GPU collectives, non-interactive profiling with nsys/ncu, debugging with cuda-gdb/compute-sanitizer, binary inspection with cuobjdump, and performance analysis workflows. Triggers on CUDA, GPU programming, kernel optimization, nsys, ncu, cuda-gdb, compute-sanitizer, PTX, GPU profiling, parallel performance, programming model, thread hierarchy, memory hierarchy, warp execution, warp divergence, occupancy, bank conflict, memory coalescing, shared memory, register pressure, async barrier, cuda::barrier, TMA, tensor memory accelerator, CUDA Graphs, cudaGraph, cooperative groups, dynamic parallelism, unified memory, cudaMallocManaged, stream ordered allocator, cudaMallocAsync, virtual memory management, compute capability, cuBLAS, cublasLtMatmul, GEMM, GemmEx, FP8, bfloat16, half precision, __half, __nv_bfloat16, cublasGemmEx, cublasGemmStridedBatchedEx, NCCL, ncclAllReduce, ncclReduceScatter, ncclAllGather, ncclCommInitRank, tensor parallel, pipeline parallel, all-reduce, vLLM CUDA kernels."
---

# CUDA Programming Skill

## Core Philosophy

**Measure before guessing.** GPU performance is deeply counterintuitive. Profile first, hypothesize second, change third, verify fourth.

**Small, isolated changes.** CUDA bugs compound. Make one change, test it, commit it. Resist the urge to "fix everything at once."

**printf is your strongest tool.** When debuggers fail, when tools produce inscrutable output, printf in device code reveals truth. Don't be embarrassed to use it extensively.

**Sometimes, stare at the diff.** Inscrutable segfaults are common. Tools often don't help. The human approach: minimize the diff, read it carefully, see the bug. This is legitimate and often faster than tooling.

## Debugging Workflow

### First Response to a Bug

1. **Reproduce minimally** — Isolate the failing kernel with smallest possible input
2. **Add printf** — Before any tool, add `printf` in device code to trace execution
3. **Run compute-sanitizer** — Catch memory errors non-interactively:

   ```bash
   compute-sanitizer --tool memcheck ./your_program
   compute-sanitizer --tool racecheck ./your_program  # for race conditions
   compute-sanitizer --tool initcheck ./your_program  # uninitialized memory
   ```

4. **If still stuck**, try cuda-gdb non-interactively for backtrace:

   ```bash
   cuda-gdb -batch -ex "run" -ex "bt" ./your_program
   ```

5. **When tools fail** — Minimize the diff between working and broken code. Read it. The bug is in the diff.

### printf in Device Code

```cuda
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  // Limit output
        printf("Kernel launched, n=%d, data[0]=%f\n", n, data[0]);
    }
    // ... kernel logic ...
    if (idx < 10) {  // Sample a few threads
        printf("Thread %d: result=%f\n", idx, someValue);
    }
}
```

**Key patterns:**

- Guard with `if (idx == 0)` or `if (idx < N)` to avoid output flood
- Print at kernel entry to confirm launch
- Print intermediate values at suspected failure points
- Flush is automatic at kernel completion

### compute-sanitizer Quick Reference

**Common gotcha:** "Invalid **shared** write... out of bounds" usually means insufficient dynamic shared memory allocation in the kernel launch, not wrong array indexing. Check `<<<grid, block, smem_size>>>`.

```bash
# Memory errors (most common)
compute-sanitizer --tool memcheck ./program

# Other tools: racecheck, initcheck, synccheck
# For detailed options, see references/debugging-tools.md
```

### cuda-gdb Non-Interactive

```bash
# Get backtrace on crash
cuda-gdb -batch -ex "run" -ex "bt" ./program

# For breakpoints, thread inspection, see references/debugging-tools.md
```

**Compile with debug info:**

```bash
nvcc -g -G -lineinfo program.cu -o program
```

### cuobjdump for Binary Inspection

```bash
# Dump PTX and SASS
cuobjdump -ptx ./program
cuobjdump -sass ./program

# For resource usage, symbol listing, see references/debugging-tools.md
```

**For complete debugging tool reference:** See `references/debugging-tools.md` for detailed compute-sanitizer options, cuda-gdb workflows, and cuobjdump analysis patterns.

## Performance Optimization Workflow

### Golden Rule

**Never optimize without profiling first.** Intuition about GPU bottlenecks is almost always wrong. The profile → fix → verify loop is the actual optimization work, not a preliminary step.

### Performance Investigation Steps

1. **Establish baseline** — Time the operation, record it
2. **Profile with nsys** — Get timeline, identify which kernels matter
3. **Deep-dive with ncu** — Analyze specific bottleneck kernels
4. **Hypothesize** — Based on metrics, form specific hypothesis
5. **Change one thing** — Make a single targeted change
6. **Verify** — Re-profile, confirm improvement
7. **Repeat**

### nsys (Nsight Systems) — Timeline Profiling

Use nsys for: "Where is time being spent?" — CPU/GPU interaction, kernel launch patterns, memory transfers, overall timeline.

```bash
# Basic profile
nsys profile -o report ./program
nsys stats report.nsys-rep --report cuda_gpu_kern_sum

# With NVTX markers
nsys profile --trace=cuda,nvtx -o report ./program

# Key reports: cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_mem_time_sum, nvtx_sum
# For detailed usage, see references/nsys-guide.md
```

**For detailed nsys analysis patterns:** See `references/nsys-guide.md` for timeline interpretation, identifying common bottlenecks, and analysis workflows.

### ncu (Nsight Compute) — Kernel Analysis

Use ncu for: "Why is this kernel slow?" — Detailed metrics, roofline, memory analysis, occupancy.

```bash
# Profile specific kernel
ncu --kernel-name "myKernel" -o report ./program

# Quick summary to stdout
ncu --set basic ./program

# Sets: basic, full, memory, launch, roofline
# Sections: ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, Occupancy
# For detailed metrics and interpretation, see references/ncu-guide.md
```

**Warning:** ncu expert system recommendations can be misleading. Always verify with actual metrics and experiments.

**Scale matters:** Optimizations that help at large scale can hurt at small scale. Always profile at your actual problem size, not theoretical maximums.

**For detailed ncu metric interpretation:** See `references/ncu-guide.md` for understanding roofline analysis, memory bottlenecks, occupancy limits, and warp scheduling.

### NVTX for Custom Instrumentation

When you need finer granularity than kernel-level, use NVTX:

```cuda
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("Operation Name");
// ... code to profile ...
nvtxRangePop();
```

**Compile:** `-lnvToolsExt` | **Profile:** `nsys profile --trace=cuda,nvtx`

**For complete patterns:** See `references/nvtx-patterns.md` for nested ranges, colors, and analysis workflows.

### Common Performance Patterns

| Symptom                | Likely Cause                           | Investigation                                |
| ---------------------- | -------------------------------------- | -------------------------------------------- |
| Low GPU utilization    | Kernel launch overhead, CPU bottleneck | nsys timeline, look for gaps                 |
| Memory bound           | Poor access patterns, low cache hit    | ncu memory section, check coalescing         |
| Compute bound but slow | Low occupancy, register pressure       | ncu occupancy, reduce registers              |
| Lots of small kernels  | Launch overhead dominates              | nsys timeline, consider fusion               |
| High memcpy time       | Excessive H2D/D2H transfers            | nsys cuda_gpu_mem, batch transfers           |
| Most cycles stalled    | Bank conflicts, memory stalls          | ncu SchedulerStatistics, check shared memory |
| High sectors/request   | Poor coalescing (>4 sectors/req)       | ncu memory metrics, use vectorized loads     |

**Critical traps:** Bank conflicts and memory coalescing issues often dominate performance but aren't obvious without profiling. See `references/performance-traps.md` for detailed diagnosis and fixes.

**Reality check:** Budget 80% of optimization time for problems you didn't predict. Profile-driven iteration discovers the real bottlenecks.

## Compilation Reference

```bash
# Debug build
nvcc -g -G -lineinfo -O0 program.cu -o program_debug

# Release build
nvcc -O3 -lineinfo program.cu -o program

# Specific architecture
nvcc -arch=sm_80 program.cu -o program  # Ampere
nvcc -arch=sm_89 program.cu -o program  # Ada Lovelace
nvcc -arch=sm_90 program.cu -o program  # Hopper

# Generate PTX (inspect it)
nvcc -ptx program.cu

# Verbose compilation (see register usage)
nvcc --ptxas-options=-v program.cu

# With NVTX
nvcc program.cu -lnvToolsExt -o program
```

**Always compile with `-lineinfo` for production profiling** — minimal overhead, enables source correlation.

## Local API Documentation

Complete reference documentation available for grep-based search:

**PTX ISA 9.1** — `references/ptx-docs/` (405 files, 2.3MB)

- Search guide: `references/ptx-isa.md`
- Use for: Instruction-level optimization, inline PTX, TensorCore operations (WMMA, WGMMA, TMA), memory swizzling

**CUDA Runtime API 13.1** — `references/cuda-runtime-docs/` (107 files, 0.9MB)

- Search guide: `references/cuda-runtime.md`
- Use for: Error codes, API parameters, device properties (`cudaDeviceProp`), memory management, stream behavior

**CUDA Driver API 13.1** — `references/cuda-driver-docs/` (128 files, 0.8MB)

- Search guide: `references/cuda-driver.md`
- Use for: Context management (`cuCtxCreate`), module loading (`cuModuleLoad`), virtual memory, Driver errors (`CUDA_ERROR_*`), advanced features

**cuBLAS 13.2** — `references/cublas-docs/` (319 files, 2.9MB)

- Search guide: `references/cublas.md`
- Chapters: `1-introduction/`, `2-using-the-cublas-api/`, `3-using-the-cublaslt-api/`, `4-using-the-cublasxt-api/`
- Use for: GEMM operations (`cublas<t>gemm`, `cublasGemmEx`), cuBLASLt fused GEMM with custom epilogues (`cublasLtMatmul`), FP8/BF16 narrow-precision GEMM, batched GEMM, matrix layouts and data types
- Key files:
  - `2-using-the-cublas-api/2.7-cublas-level-3-function-reference.md` — GEMM, TRSM, SYMM, SYRK
  - `3-using-the-cublaslt-api/3.4-cublaslt-api-reference.md` — cublasLtMatmul and all Lt descriptors
  - `3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md` — cublasLtEpilogue_t, layout attributes
  - `2-using-the-cublas-api/2.8-blas-like-extension.md` — GemmEx, GemmBatchedEx, GemmStridedBatchedEx

**CUDA Math API** — `references/cuda-math-docs/` (40 files, 0.4MB)

- Search guide: `references/cuda-math.md`
- Modules: `modules/` (14 files) — single/double precision, intrinsics for half, bfloat16, FP8, FP6, FP4, SIMD, cast, integer
- Data structures: `data-structures/` (26 files) — `__half`, `__half2`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp6_*`, `__nv_fp4_*`
- Use for: Device math functions (`sinf`, `__expf`, `__fmaf_rn`), narrow-precision type layouts (FP8/FP6/FP4 E2M1/E2M3/E3M2/E4M3/E5M2/E8M0), half/bfloat16 arithmetic intrinsics, SIMD byte/short operations
- Key files:
  - `modules/group__cuda__math__single.md` — standard single-precision math functions
  - `modules/group__cuda__math__intrinsic__fp8.md` — FP8 conversion and arithmetic
  - `modules/group__cuda__math__intrinsic__half.md` — `__half` arithmetic operations
  - `modules/group__cuda__math__intrinsic__bfloat16.md` — `__nv_bfloat16` operations

**NCCL** — `references/nccl-docs/` (33 files, 0.5MB)

- Search guide: `references/nccl.md`
- Structure: `usage/` (13 files — communicators, collectives, streams, P2P, CUDA graphs), `api/` (12 files — colls, comms, p2p, types, device API), top-level guides (overview, env, troubleshooting, examples, mpi)
- Use for: `ncclAllReduce` / `ncclReduceScatter` / `ncclAllGather` signatures, communicator setup (`ncclCommInitRank`, `ncclGetUniqueId`), P2P send/recv for pipeline parallel, environment variable tuning (`NCCL_DEBUG`, `NCCL_ALGO`, `NCCL_IB_*`), device-initiated communication (GIN)
- Key files:
  - `api/colls.md` — all collective function signatures
  - `api/comms.md` — communicator creation and management
  - `api/types.md` — `ncclDataType_t`, `ncclResult_t`, `ncclRedOp_t`
  - `env.md` — full environment variable reference
  - `troubleshooting.md` — hang diagnosis patterns

**CUDA C++ Programming Guide** — `references/cuda-prog-guide-docs/` (709 files, 5.9MB)

- Search guide: `references/cuda-prog-guide.md`
- Structure: 26 numbered chapter directories (e.g. `5-programming-model/`, `10-c-language-extensions/`, `20-compute-capabilities/`)
- Use for: CUDA programming model (thread/block/grid/cluster hierarchy), SIMT architecture and warp execution, memory hierarchy concepts, C++ language extensions (`__global__`, `__shared__`, `__device__`, `__launch_bounds__`), atomic functions, warp intrinsics (shuffle/vote/reduce), cooperative groups, async barrier (`cuda::barrier`), TMA async copies, `memcpy_async` / `cuda::pipeline`, CUDA Graphs, stream-ordered allocator (`cudaMallocAsync`), virtual memory management (`cuMemCreate`, `cuMemMap`), unified memory (`cudaMallocManaged`), compute capability feature tables (sm_50 through sm_120), CUDA environment variables
- Key files:
  - `5-programming-model/5.2-thread-hierarchy.md` — thread/block/grid/cluster model
  - `7-hardware-implementation/7.1-simt-architecture.md` — warp execution, divergence
  - `8-performance-guidelines/8.3.2-device-memory-accesses.md` — coalescing, shared memory, L2
  - `10-c-language-extensions/10.22-warp-shuffle-functions.md` — `__shfl_sync`, `__shfl_xor_sync`
  - `10-c-language-extensions/10.26-asynchronous-barrier.md` — `cuda::barrier` arrive/wait patterns
  - `10-c-language-extensions/10.29-asynchronous-data-copies-using-the-tensor-memory-accelerator-tma.md` — TMA usage
  - `11-cooperative-groups/11.4-group-types.md` — all cooperative group types
  - `20-compute-capabilities/20.2-features-and-technical-specifications.md` — architecture specs table

**CUDA C++ Best Practices Guide** — `references/cuda-best-practices-docs/` (138 files, 1.0MB)

- Search guide: `references/cuda-best-practices.md`
- Structure: 20 numbered chapter directories covering profiling, memory, execution configuration, instruction optimization, deployment
- Use for: Official optimization methodology (Assess → Parallelize → Optimize → Deploy), memory coalescing rules and examples, shared memory bank conflict diagnosis, pinned memory for H2D transfers, occupancy calculation and block size heuristics, warp divergence patterns to avoid, arithmetic instruction throughput reference, bandwidth calculation (theoretical vs effective), register pressure management, L2 cache window tuning, deployment and compatibility guidance
- Key files:
  - `9-performance-metrics/9.2.1-theoretical-bandwidth-calculation.md` — roofline baseline
  - `10-memory-optimizations/10.2.1-coalesced-access-to-global-memory.md` — coalescing rules with examples
  - `10-memory-optimizations/10.2.3.1-shared-memory-and-memory-banks.md` — bank conflict diagnosis
  - `10-memory-optimizations/10.2.3.2-shared-memory-in-matrix-multiplication-cab.md` — tiling worked example
  - `11-execution-configuration-optimizations/11.1-occupancy.md` — occupancy theory and tuning
  - `11-execution-configuration-optimizations/11.3-thread-and-block-heuristics.md` — block size selection rules
  - `12-instruction-optimization/12.1.1-throughput-of-native-arithmetic-instructions.md` — per-op cost table
  - `13-control-flow/13.1-branching-and-divergence.md` — divergence patterns to avoid
  - `19-recommendations-and-best-practices/19.1-overall-performance-optimization-strategies.md` — prioritized checklist

Each search guide contains grep examples, documentation structure, and common usage patterns.

**Search strategy:** Use grep/ripgrep to search directly in the `*-docs/` directories. The search guides (`.md` files) provide navigation patterns and common queries.

```bash
# cuBLAS search examples
grep -r "cublasGemmEx" references/cublas-docs/
grep -r "cublasLtMatmul" references/cublas-docs/3-using-the-cublaslt-api/
grep -r "CUBLAS_COMPUTE_" references/cublas-docs/          # compute types
grep -r "CUBLASLT_EPILOGUE_" references/cublas-docs/       # epilogue options (bias, ReLU, GELU)
grep -r "FP8\|fp8\|E4M3\|E5M2" references/cublas-docs/    # FP8 narrow precision

# CUDA Math API search examples
grep -r "__expf\|__logf\|__sinf" references/cuda-math-docs/   # fast intrinsics
grep -r "__nv_fp8_e4m3\|__nv_fp8_e5m2" references/cuda-math-docs/  # FP8 types
grep -r "__half2\|__hadd\|__hmul" references/cuda-math-docs/        # half precision
grep -r "__nv_bfloat16" references/cuda-math-docs/                  # bfloat16 ops

# NCCL search examples
grep -r "ncclAllReduce" references/nccl-docs/api/
grep -r "ncclFloat16\|ncclBfloat16" references/nccl-docs/api/types.md  # FP16/BF16 support
grep -r "ncclGroupStart\|ncclGroupEnd" references/nccl-docs/           # group calls
grep -r "^## NCCL_" references/nccl-docs/env.md                        # env vars
```

## Additional References

- `references/performance-traps.md` — Bank conflicts, memory coalescing, scale-dependent optimizations
- `references/debugging-tools.md` — compute-sanitizer, cuda-gdb, cuobjdump detailed usage
- `references/nsys-guide.md` — nsys timeline analysis and bottleneck identification
- `references/ncu-guide.md` — ncu metrics, roofline, occupancy interpretation
- `references/nvtx-patterns.md` — NVTX instrumentation and profiling patterns
- `references/cuda-prog-guide.md` — CUDA C++ Programming Guide search guide (programming model, language extensions, compute capabilities)
- `references/cuda-best-practices.md` — CUDA C++ Best Practices Guide search guide (coalescing, occupancy, instruction throughput, optimization checklist)

## Checklist Before Optimizing

- [ ] Established reproducible baseline timing
- [ ] Profiled with nsys to identify hotspots
- [ ] Know which kernel(s) dominate runtime
- [ ] Profiled target kernel with ncu
- [ ] Identified specific bottleneck (memory? compute? latency?)
- [ ] Formed specific, testable hypothesis
- [ ] Plan to change ONE thing
