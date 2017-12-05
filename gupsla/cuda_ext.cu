#include "cuda_ext.cuh"
/* LNK4221
inline cudaError_t gpuAssert(cudaError_t code, const char * file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA:\n*****\n%s\n%s\nat %s : %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
		getchar(); exit(-1);
	}
	return code;
}
// */