#include <stdio.h>
#include <math.h>
#include "macros.h"
#include "cuda_ext.cuh"

void PrintTest()
{
	cudaDeviceProp gpu_info;
	gpuOk(cudaGetDeviceProperties(&gpu_info, 0));

	printf("sharedMemPerBlock: %g kB\n", gpu_info.sharedMemPerBlock / 1024.0);
	printf("sharedMemPerBlockOptin: %zd\n", gpu_info.sharedMemPerBlockOptin);
	printf("sharedMemPerMultiprocessor: %g kB\n", gpu_info.sharedMemPerMultiprocessor / 1024.0);
	printf("maxThreadsPerBlock: %d\n", gpu_info.maxThreadsPerBlock);

	int block_width = (int)sqrt(gpu_info.maxThreadsPerBlock) - 2;
	int sh_width = (int)(sqrt(gpu_info.sharedMemPerBlock + 4) - 2);
	if (sh_width < block_width) block_width = sh_width;
	printf("block_width: %d\n", block_width);
}