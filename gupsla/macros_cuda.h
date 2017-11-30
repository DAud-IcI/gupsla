#pragma once

#include "cuda_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA EXTENSION
#ifdef __CUDACC__
#define DEV3(grid, block, shared_memory) <<<(grid), (block), (shared_memory)>>>
#define DEV4(grid, block, shared_memory, stream) <<<grid, block, shared_memory, stream>>>
#define syncthreads() __syncthreads()
#else
#define DEV3(grid, block, shared_memory)
#define DEV4(grid, block, shared_memory, stream)
#define syncthreads()
#endif

#define gpuOk(code) assert(gpuAssert((code), __FILE__, __LINE__) == cudaSuccess)