# CUDA Toolkit Documentation 13.2 Update 1

**Source:** https://docs.nvidia.com/cuda/index.html

---

# CUDA Toolkit Documentation 13.2 Update 1[](#cuda-toolkit-documentation-v12-8 "Permalink to this headline")

**Develop, Optimize and Deploy GPU-Accelerated Apps**

The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to deploy your application.

Using built-in capabilities for distributing computations across multi-GPU configurations, scientists and researchers can develop applications that scale from single GPU workstations to cloud installations with thousands of GPUs.

* * *

[Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
    

The Release Notes for the CUDA Toolkit.

* * *

## CUDA Installation Guides[](#installation-guides "Permalink to this headline")

[Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
    

This guide provides the minimal first-steps instructions for installation and verifying CUDA on a standard system.

[Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    

This guide discusses how to install and check for correct operation of the CUDA Development Tools on GNU/Linux systems.

[Installation Guide Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
    

This guide discusses how to install and check for correct operation of the CUDA Development Tools on Microsoft Windows systems.

* * *

## CUDA Programming Guides[](#programming-guides "Permalink to this headline")

[ CUDA Programming Guide ](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
    

This guide provides a detailed discussion of the CUDA programming model and programming interface. It also describes the hardware implementation and provides guidance on achieving maximum performance. 

[ CUDA Best Practices Guide ](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
    

This guide presents established parallelization and optimization techniques and explains coding idioms that simplify programming for CUDA-capable GPUs. It provides guidelines for obtaining the best performance from NVIDIA GPUs using the CUDA Toolkit. 

[ cuTile Python ](https://docs.nvidia.com/cuda/cutile-python/index.html)
    

This guide provides documentation of cuTile Python, the DSL for tile programming in Python. 

[ PTX ISA ](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
    

This guide provides detailed instructions on the use of PTX, a low-level parallel thread execution virtual machine and instruction set architecture (ISA). PTX exposes the GPU as a data-parallel computing device. 

[ CUDA Tile IR ](https://docs.nvidia.com/cuda/tile-ir/index.html)
    

This guide provides documentation of CUDA Tile IR, a portable, low-level tile virtual machine and instruction set that models the GPU as a tile-based processor. 

* * *

## CUDA Architecture Guides[](#architecture-guides "Permalink to this headline")

[Ada Compatibility Guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html)
    

This application note is intended to help developers ensure that their NVIDIA CUDA applications will run properly on the Ada GPUs. This document provides guidance to ensure that your software applications are compatible with Ada architecture.

[Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
    

The NVIDIA® Ada GPU architecture is NVIDIA’s 10th-generation architecture for CUDA® compute applications. The NVIDIA Ada GPU architecture retains and extends the same CUDA programming model provided by previous NVIDIA GPU architectures such as NVIDIA Ampere and Turing architectures, and applications that follow the best practices for those architectures should typically see speedups on the NVIDIA Ada architecture without any code changes. This guide summarizes the ways that an application can be fine-tuned to gain additional speedups by leveraging the NVIDIA Ada GPU architecture’s features.

[Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
    

This application note is intended to help developers ensure that their NVIDIA CUDA applications will run properly on the Blackwell GPUs. This document provides guidance to ensure that your software applications are compatible with Blackwell architecture.

[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
    

The NVIDIA® Blackwell GPU architecture is NVIDIA’s latest architecture for CUDA® compute applications. The NVIDIA Blackwell GPU architecture retains and extends the same CUDA programming model provided by previous NVIDIA GPU architectures such as NVIDIA Ampere and Turing architectures, and applications that follow the best practices for those architectures should typically see speedups on the NVIDIA Blackwell architecture without any code changes. This guide summarizes the ways that an application can be fine-tuned to gain additional speedups by leveraging the NVIDIA Blackwell GPU architecture’s features.

[Hopper Compatibility Guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html)
    

This application note is intended to help developers ensure that their NVIDIA CUDA applications will run properly on the Hopper GPUs. This document provides guidance to ensure that your software applications are compatible with Hopper architecture.

[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
    

Hopper GPU Architecture is NVIDIA’s 9th-generation architecture for CUDA compute applications. This guide summarizes the ways that applications can be fine-tuned to gain additional speedups by leveraging Hopper GPU Architecture’s features.

[Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
    

This document shows how to inline PTX (parallel thread execution) assembly language statements into CUDA code. It describes available assembler statement parameters and constraints, and the document also provides a list of some pitfalls that you may encounter.

[NVIDIA Ampere GPU Architecture Compatibility Guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)
    

This application note is intended to help developers ensure that their NVIDIA CUDA applications will run properly on GPUs based on the NVIDIA Ampere GPU Architecture. This document provides guidance to ensure that your software applications are compatible with NVIDIA Ampere GPU architecture.

[NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
    

NVIDIA Ampere GPU Architecture is NVIDIA’s 8th-generation architecture for CUDA compute applications. This guide summarizes the ways that applications can be fine-tuned to gain additional speedups by leveraging NVIDIA Ampere GPU Architecture’s features.

[PTX Interoperability](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html)
    

This document shows how to write PTX that is ABI-compliant and interoperable with other CUDA code.

[Turing Compatibility Guide](https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html)
    

This application note is intended to help developers ensure that their NVIDIA CUDA applications will run properly on GPUs based on the NVIDIA Turing Architecture. This document provides guidance to ensure that your software applications are compatible with Turing.

[Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
    

Turing is NVIDIA’s 7th-generation architecture for CUDA compute applications. This guide summarizes the ways that applications can be fine-tuned to gain additional speedups by leveraging Turing architectural features.

* * *

## CUDA API References[](#cuda-api-references "Permalink to this headline")

[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
    

Fields in structures might appear in order that is different from the order of declaration.

[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
    

Fields in structures might appear in order that is different from the order of declaration.

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
    

The CUDA math API.

[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
    

The cuBLAS library is an implementation of BLAS (Basic Linear Algebra Subprograms) on top of the NVIDIA CUDA runtime. It allows the user to access the computational resources of NVIDIA Graphical Processing Unit (GPU), but does not auto-parallelize across multiple GPUs.

[cuDLA API](https://docs.nvidia.com/cuda/cudla-api/index.html)
    

The cuDLA API.

[NVBLAS](https://docs.nvidia.com/cuda/nvblas/index.html)
    

The NVBLAS library is a multi-GPUs accelerated drop-in BLAS (Basic Linear Algebra Subprograms) built on top of the NVIDIA cuBLAS Library.

[nvJPEG](https://docs.nvidia.com/cuda/nvjpeg/index.html)
    

The nvJPEG Library provides high-performance GPU accelerated JPEG decoding functionality for image formats commonly used in deep learning and hyperscale multimedia applications.

[cuFFT](https://docs.nvidia.com/cuda/cufft/index.html)
    

The cuFFT library user guide.

[CUB](https://nvlabs.github.io/cub/)
    

The user guide for CUB.

[CUDA C++ Standard Library](https://nvidia.github.io/libcudacxx/)
    

The API reference for libcu++, the CUDA C++ standard library.

[cuFile API Reference Guide](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
    

The NVIDIA® GPUDirect® Storage cuFile API Reference Guide provides information about the preliminary version of the cuFile API reference guide that is used in applications and frameworks to leverage GDS technology and describes the intent, context, and operation of those APIs, which are part of the GDS technology.

[cuRAND](https://docs.nvidia.com/cuda/curand/index.html)
    

The cuRAND library user guide.

[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)
    

The cuSPARSE library user guide.

[NPP](https://docs.nvidia.com/cuda/npp/index.html)
    

NVIDIA NPP is a library of functions for performing CUDA accelerated processing. The initial set of functionality in the library focuses on imaging and video processing and is widely applicable for developers in these areas. NPP will evolve over time to encompass more of the compute heavy tasks in a variety of problem domains. The NPP library is written to maximize flexibility, while maintaining high performance.

[nvJitLink](https://docs.nvidia.com/cuda/nvjitlink/index.html)
    

The user guide for the nvJitLink library.

[nvFatbin](https://docs.nvidia.com/cuda/nvfatbin/index.html)
    

The user guide for the nvFatbin library.

[NVRTC (Runtime Compilation)](https://docs.nvidia.com/cuda/nvrtc/index.html)
    

NVRTC is a runtime compilation library for CUDA C++. It accepts CUDA C++ source code in character string form and creates handles that can be used to obtain the PTX. The PTX string generated by NVRTC can be loaded by cuModuleLoadData and cuModuleLoadDataEx, and linked with other modules by cuLinkAddData of the CUDA Driver API. This facility can often provide optimizations and performance not possible in a purely offline static compilation.

[Thrust](https://nvidia.github.io/cccl/thrust/)
    

The C++ parallel algorithms library.

[cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html)
    

The cuSOLVER library user guide.

* * *

## PTX Compiler API References[](#ptx-compiler-api-references "Permalink to this headline")

[PTX Compiler APIs](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html)
    

This guide shows how to compile a PTX program into GPU assembly code using APIs provided by the static PTX Compiler library.

* * *

## Miscellaneous[](#miscellaneous "Permalink to this headline")

[CUDA Demo Suite](https://docs.nvidia.com/cuda/demo-suite/index.html)
    

This document describes the demo applications shipped with the CUDA Demo Suite.

[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
    

This guide is intended to help users get started with using NVIDIA CUDA on Windows Subsystem for Linux (WSL 2). The guide covers installation and running CUDA applications and containers in this environment.

[Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)
    

This edition of the user guide describes the Multi-Instance GPU feature of the NVIDIA® A100 GPU.

[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
    

This document describes CUDA Compatibility, including CUDA Enhanced Compatibility and CUDA Forward Compatible Upgrade.

[CUPTI](https://docs.nvidia.com/cupti/index.html)
    

The CUPTI-API. The CUDA Profiling Tools Interface (CUPTI) enables the creation of profiling and tracing tools that target CUDA applications.

[Debugger API](https://docs.nvidia.com/cuda/debugger-api/index.html)
    

The CUDA debugger API.

[GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
    

A technology introduced in Kepler-class GPUs and CUDA 5.0, enabling a direct path for communication between the GPU and a third-party peer device on the PCI Express bus when the devices share the same upstream root complex using standard features of PCI Express. This document introduces the technology and describes the steps necessary to enable a GPUDirect RDMA connection to NVIDIA GPUs within the Linux device driver model.

[GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/index.html)
    

The documentation for GPUDirect Storage.

[vGPU](https://docs.nvidia.com/cuda/vGPU/index.html)
    

vGPUs that support CUDA.

* * *

## Tools[](#tools "Permalink to this headline")

[NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
    

This is a reference document for nvcc, the CUDA compiler driver. nvcc accepts a range of conventional compiler options, such as for defining macros and include/library paths, and for steering the compilation process.

[CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
    

The NVIDIA tool for debugging CUDA applications running on Linux and QNX, providing developers with a mechanism for debugging CUDA applications running on actual hardware. CUDA-GDB is an extension to the x86-64 port of GDB, the GNU Project debugger.

[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/index.html)
    

The user guide for Compute Sanitizer.

[Nsight Eclipse Plugins Installation Guide](https://docs.nvidia.com/cuda/nsightee-plugins-install-guide/index.html)
    

Nsight Eclipse Plugins Installation Guide

[Nsight Eclipse Plugins Edition](https://docs.nvidia.com/cuda/nsight-eclipse-plugins-guide/index.html)
    

Nsight Eclipse Plugins Edition getting started guide

[Nsight Systems](https://docs.nvidia.com/nsight-systems/index.html)
    

The documentation for Nsight Systems.

[Nsight Compute](https://docs.nvidia.com/nsight-compute/index.html)
    

The NVIDIA Nsight Compute is the next-generation interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool.

[Nsight Visual Studio Edition](https://docs.nvidia.com/nsight-visual-studio-edition/index.html)
    

The documentation for Nsight Visual Studio Edition.

[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
    

The application notes for cuobjdump, nvdisasm, and nvprune.

[CUDA Compile Time Advisor](https://docs.nvidia.com/cuda/cuda-compile-time-advisor/index.html)
    

The application notes for Compile Time Advisor (ctadvisor).

* * *

## White Papers[](#white-papers "Permalink to this headline")

[Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)
    

A number of issues related to floating point accuracy and compliance are a frequent source of confusion on both CPUs and GPUs. The purpose of this white paper is to discuss the most common issues related to NVIDIA GPUs and to supplement the documentation in the CUDA Programming Guide.

[Incomplete-LU and Cholesky Preconditioned Iterative Methods](https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html)
    

In this white paper we show how to use the cuSPARSE and cuBLAS libraries to achieve a 2x speedup over CPU in the incomplete-LU and Cholesky preconditioned iterative methods. We focus on the Bi-Conjugate Gradient Stabilized and Conjugate Gradient iterative methods, that can be used to solve large sparse nonsymmetric and symmetric positive definite linear systems, respectively. Also, we comment on the parallel sparse triangular solve, which is an essential building block in these algorithms.

* * *

## Application Notes[](#application-notes "Permalink to this headline")

[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)
    

This application note provides an overview of NVIDIA® Tegra® memory architecture and considerations for porting code from a discrete GPU (dGPU) attached to an x86 system to the Tegra® integrated GPU (iGPU). It also discusses EGL interoperability.

* * *

## Compiler SDK[](#compiler-sdk "Permalink to this headline")

[libNVVM API](https://docs.nvidia.com/cuda/libnvvm-api/index.html)
    

The libNVVM API.

[libdevice User’s Guide](https://docs.nvidia.com/cuda/libdevice-users-guide/index.html)
    

The libdevice library is an LLVM bitcode library that implements common functions for GPU kernels.

[NVVM IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html)
    

NVVM IR is a compiler IR (intermediate representation) based on the LLVM IR. The NVVM IR is designed to represent GPU compute kernels (for example, CUDA kernels). High-level language front-ends, like the CUDA C compiler front-end, can generate NVVM IR.

* * *

## CUDA Archives[](#cuda-archives "Permalink to this headline")

[ CUDA Features Archive ](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
    

The list of CUDA features by release.

[ CUDA C++ Programming Guide (Legacy) ](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
    

This legacy guide documents the earlier CUDA C/C++ programming model and is retained for reference for existing applications.

* * *

## Legal Notices[](#legal-notices "Permalink to this headline")

[EULA](https://docs.nvidia.com/cuda/eula/index.html)
    

The CUDA Toolkit End User License Agreement applies to the NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA Display Driver, NVIDIA Nsight tools (Visual Studio Edition), and the associated documentation on CUDA APIs, programming model and development tools. If you do not agree with the terms and conditions of the license agreement, then do not download or use the software.