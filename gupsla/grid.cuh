#pragma once
#include "grid.h"
#include "macros_cuda.h"

__global__ void Grid_D_GoLStep(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z);
__global__ void Grid_D_Rule90Step(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z);

struct Coordinates
{
	bool invalid;
	
	unsigned int x;
	unsigned int y;
	unsigned int z;

	dim3 shared_size;

	unsigned int global_idx;
	unsigned int shared_idx;
};

#pragma region GridKernel helpers

// initializes the Coordinates struct referenced as "c", terminates out-of-bounds threads and sets up the shared memory.
#define Grid_D_Prepare(dev_grid, size_x, size_y, size_z, c) Grid_D_InitCoordinates(dev_grid, size_x, size_y, size_z, &(c)); if (c.invalid) return; Grid_D_PrepareDevice(dev_grid, size_x, size_y, size_z, &(c))

// Generates indicies and boundary info
__device__ __forceinline__ void Grid_D_InitCoordinates(byte * dev_grid, unsigned int size_x, unsigned int size_y, unsigned int size_z, Coordinates * c);

// Allocates shared memory
__device__ __forceinline__ void Grid_D_PrepareDevice(byte * dev_grid, unsigned int size_x, unsigned int size_y, unsigned int size_z, Coordinates * c);

// Sums up the values of neighbouring cells
__device__ __forceinline__ int Grid_D_CountNeighbours(byte * dev_grid, Coordinates * c);

// Sets the cell's value and updates the device_idle variable if necessary
__device__ __forceinline__ void Grid_D_UpdateCell(byte * dev_grid, Coordinates * c, byte value, bool * device_idle);

// Macro that contains the opening boilerplate for a GridKernel type function. 
// Short version of Grid_D_Prepare using the default parameter and variable names.
// Creates struct Coordinates c and shared byte sha_block[].
#define Grid_D_DefaultBegin() Coordinates c; Grid_D_InitCoordinates(dev_grid, size_x, size_y, size_z, &c); if (c.invalid) return; extern __shared__ byte sha_block[]; Grid_D_PrepareDevice(dev_grid, size_x, size_y, size_z, &c)

// Macro that contains the closing boilerplate for a GridKernel type function. 
#define Grid_D_DefaultEnd() Grid_D_UpdateCell(dev_grid, &c, value, device_idle)

#pragma endregion

#pragma region Neighbour coordinates
#define OFF_NW(c) c.shared_idx - 1 - c.shared_size.x
#define OFF_WW(c) c.shared_idx - 1
#define OFF_SW(c) c.shared_idx - 1 + c.shared_size.x
#define OFF_NN(c) c.shared_idx     - c.shared_size.x
#define OFF_CC(c) c.shared_idx
#define OFF_SS(c) c.shared_idx     + c.shared_size.x
#define OFF_NE(c) c.shared_idx + 1 - c.shared_size.x
#define OFF_EE(c) c.shared_idx + 1
#define OFF_SE(c) c.shared_idx + 1 + c.shared_size.x
#pragma endregion


