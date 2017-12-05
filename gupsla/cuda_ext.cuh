#pragma once

#include <stdio.h>
#include <stdlib.h>

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

extern "C"
{
//	inline cudaError_t gpuAssert(cudaError_t code, const char * file, int line);
	inline cudaError_t gpuAssert(cudaError_t code, const char * file, int line)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "CUDA:\n*****\n%s\n%s\nat %s : %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
			getchar(); exit(-1);
		}
		return code;
	}
}