#include "grid.cuh"

#include <stdio.h>
#include <time.h>
#include <math.h>

#include <string>
#include <map>
#include <iterator>

#define CELLS_SIZE grid->Tables * grid->Rows * grid->Columns * sizeof(byte)
#define BOOL_SIZE sizeof(bool)
#define BYTE_SIZE sizeof(byte)
#define SetIdleTrue()  grid->Idle = true ; gpuOk(cudaMemcpy(grid->dev_Idle, &True , BOOL_SIZE, cudaMemcpyHostToDevice))
#define SetIdleFalse() grid->Idle = false; gpuOk(cudaMemcpy(grid->dev_Idle, &False, BOOL_SIZE, cudaMemcpyHostToDevice))

bool False = false;
bool True  = true;

char grid_chars[255]{ '_', 'X' };

bool init = false;
cudaDeviceProp gpu_info;
int block_width;
dim3 threads;
size_t shared_memory_size;

Grid * Grid_Create(int tables, int rows, int columns, const char * kernel_name)
{
	Grid * grid = alloc(1, Grid);

	grid->Tables = tables;
	grid->Rows = rows;
	grid->Columns = columns;
	grid->Cells = alloc(tables * rows * columns, byte);

	if (!init)
	{
		srand((unsigned int)time(NULL));
		gpuOk(cudaGetDeviceProperties(&gpu_info, 0));
		init = true;

		block_width = (int)sqrt(gpu_info.maxThreadsPerBlock) - 2;
		int sh_width = (int)(sqrt(gpu_info.sharedMemPerBlock + 4) - 2);
		if (sh_width < block_width) block_width = sh_width;

		threads = dim3(block_width, block_width, 1);
		shared_memory_size = tables * (block_width + 2) * (block_width + 2) * BYTE_SIZE;
	}
	gpuOk(cudaMalloc(&(grid->dev_Cells), CELLS_SIZE));

	//grid->kernel_Step = &Grid_D_GoLStep;
	Grid_SetKernelStep(grid, kernel_name);
	gpuOk(cudaMalloc(&(grid->dev_Idle), BOOL_SIZE));
	SetIdleFalse();

	return grid;
}

void Grid_Destroy(Grid ** grid_ptr)
{
	Grid * grid = *grid_ptr;

	free(grid->Cells);
	gpuOk(cudaFree(grid->dev_Cells));
	free(grid);

	gpuOk(cudaFree(grid->dev_Idle));

	grid_ptr = NULL;
}

void Grid_Randomize(Grid * grid)
{
	for (int t = 0; t < grid->Tables; t++)
		for (int r = 0; r < grid->Rows; r++)
			for (int c = 0; c < grid->Columns; c++)
				grid->Cells[TRCTC(t, r, c, grid->Rows, grid->Columns)] = rand() % 2;
}

void Grid_Print(Grid * grid, bool draw_outline)
{
	if (draw_outline)
	{
		for (int c = 0; c < grid->Columns + 2; c++)
			putchar('#');
		putchar('\n');
	}

	for (int r = 0; r < grid->Rows; r++)
	{
		if (draw_outline) putchar('#');
		for (int c = 0; c < grid->Columns; c++)
		{
			putchar(grid_chars[grid->Cells[RCC(r, c, grid->Columns)]]);
		}
		if (draw_outline) putchar('#');
		putchar('\n');
	}

	if (draw_outline)
	{
		for (int c = 0; c < grid->Columns + 2; c++)
			putchar('#');
		putchar('\n');
	}
}

void Grid_Upload(Grid * grid)
{
	gpuOk(cudaMemcpy(grid->dev_Cells, grid->Cells, CELLS_SIZE, cudaMemcpyHostToDevice));
}

void Grid_Download(Grid * grid)
{
	gpuOk(cudaMemcpy(grid->Cells, grid->dev_Cells, CELLS_SIZE, cudaMemcpyDeviceToHost));
}

#define devideCeiling(a, b) (int)ceil(((double)(a)) / (b))
void Grid_Step(Grid * grid, bool print)
{
	dim3 blocks(devideCeiling(grid->Columns, block_width), devideCeiling(grid->Rows, block_width));
	//dim3 grid_size(grid->Columns, grid->Rows, grid->Tables);

	if (print)
	{
		printf("SIZE   : %d x %d x %d\n", grid->Columns, grid->Rows, grid->Tables);
		printf("BLOCKS : %d x %d x %d\n", blocks.x, blocks.y, blocks.z);
		printf("THREADS: %d x %d x %d\n", threads.x, threads.y, threads.z);
		printf("SHARED : %g kB\n", ceil(shared_memory_size / 1024.0));
	}
	
	SetIdleTrue();
	grid->kernel_Step DEV3(blocks, threads, shared_memory_size) (grid->dev_Cells, grid->dev_Idle, grid->Columns, grid->Rows, grid->Tables);
	gpuOk(cudaMemcpy(&(grid->Idle), grid->dev_Idle, BOOL_SIZE, cudaMemcpyDeviceToHost)); // update idle
}

__device__ __forceinline__ void Grid_D_InitCoordinates(byte * dev_grid, unsigned int size_x, unsigned int size_y, unsigned int size_z, Coordinates * c)
{
	c->x = threadIdx.x + blockIdx.x * blockDim.x;
	c->y = threadIdx.y + blockIdx.y * blockDim.y;
	if (c->invalid = (c->x >= size_x || c->y >= size_y))
		return;

	c->z = threadIdx.z;

	c->shared_size = dim3(blockDim.x + 2, blockDim.y + 2, size_z);

	c->global_idx = GLOXY(c->x, c->y);
	c->shared_idx = SHAXY(threadIdx.x + 1, threadIdx.y + 1);
}

__device__ __forceinline__ void Grid_D_PrepareDevice(byte * dev_grid, unsigned int size_x, unsigned int size_y, unsigned int size_z, Coordinates * c)
{
	extern __shared__ byte sha_block[];
	sha_block[c->shared_idx] = dev_grid[c->global_idx];

	

	bool block_edge_left  = threadIdx.x == 0;
	bool block_edge_right = threadIdx.x == (blockDim.x - 1);
	bool block_edge_up    = threadIdx.y == 0;
	bool block_edge_down  = threadIdx.y == (blockDim.y - 1);

	#define GLOBAL_EDGE_LEFT  c->x == 0
	#define GLOBAL_EDGE_RIGHT c->x == (size_x - 1)

	if (block_edge_left)
		sha_block[SHA_WW(*c)] = GLOBAL_EDGE_LEFT ?
			0 : dev_grid[GLO_WW(*c)];
	else if (block_edge_right)
		sha_block[SHA_EE(*c)] = GLOBAL_EDGE_RIGHT ?
			0 : dev_grid[GLO_EE(*c)];
	//*
	if (block_edge_up)
	{
		bool global_edge_up = c->y == 0;

		if (block_edge_left)
			sha_block[SHA_NW(*c)] = (global_edge_up || GLOBAL_EDGE_LEFT) ?
				0 : dev_grid[GLO_NW(*c)];
		else if (block_edge_right)
			sha_block[SHA_NE(*c)] = (global_edge_up || GLOBAL_EDGE_RIGHT) ?
				0 : dev_grid[0];
		
		sha_block[SHA_NN(*c)] = global_edge_up ?
			0 : dev_grid[GLO_NN(*c)];
	}// */
	//*
	else if (block_edge_down)
	{
		bool global_edge_down = c->y == (size_y - 1);

		if (block_edge_left)
			sha_block[SHA_SW(*c)] = (global_edge_down || GLOBAL_EDGE_LEFT) ?
			0 : dev_grid[GLO_SW(*c)];
		else if (block_edge_right)
			sha_block[SHA_SE(*c)] = (global_edge_down || GLOBAL_EDGE_RIGHT) ?
			0 : dev_grid[GLO_SE(*c)];

		sha_block[SHA_SS(*c)] = global_edge_down ?
			0 : dev_grid[GLO_SS(*c)];
	}// */

	/*
	if (threadIdx.x == 0)
	{
		if (threadIdx.y == 0) sha_block[0] = c->x > 0 && c->y > 0 ? dev_grid[GLOXY(c->x - 1, c->y - 1)] : 0;
		sha_block[SHAXY(0, threadIdx.y + 1)] = c->x > 0 ? dev_grid[GLOXY(c->x - 1, c->y)] : 0;
	}

	if (threadIdx.y == 0)
	{
		if (threadIdx.x == c->shared_size.x - 2) sha_block[SHAXY(c->shared_size.x - 1, 0)] = c->x < size_x - 1 && c->y > 0 ? dev_grid[GLOXY(c->x + 1, c->y - 1)] : 0;
		sha_block[SHAXY(threadIdx.x + 1, 0)] = c->y > 0 ? dev_grid[GLOXY(c->x, c->y - 1)] : 0;
	}
	
	if (threadIdx.x == c->shared_size.x - 2)
	{
		if (threadIdx.y == c->shared_size.y - 2) sha_block[SHAXY(c->shared_size.x - 2, c->shared_size.y - 2)] = c->x < size_x - 1 && c->y < size_y - 1 ? dev_grid[GLOXY(c->x + 1, c->y + 1)] : 0;
		sha_block[SHAXY(c->shared_size.x - 1, threadIdx.y + 1)] = c->x < size_x - 1 ? dev_grid[GLOXY(c->x + 1, c->y)] : 0;
	}

	if (threadIdx.y == c->shared_size.y - 2)
	{
		if (threadIdx.x == 0) sha_block[SHAXY(0, c->shared_size.y - 2)] = c->x > 0 && c->y < size_y - 1 ? dev_grid[GLOXY(c->x - 1, c->y + 1)] : 0;
		sha_block[SHAXY(threadIdx.x + 1, c->shared_size.y - 1)] = c->y < size_y - 1 ? dev_grid[GLOXY(c->x, c->y + 1)] : 0;
	}
	// */
	syncthreads();
	//printf("THREAD: %ux%u\n", c->shared_size.x, c->shared_size.y);
}

__device__ __forceinline__ int Grid_D_CountNeighbours(byte * dev_grid, Coordinates * c)
{
	extern __shared__ byte sha_block[];

	#pragma region calculate neighbours
	int neighbours = 0;
	neighbours += sha_block[c->shared_idx - 1 - c->shared_size.x];
	neighbours += sha_block[c->shared_idx - 1];
	neighbours += sha_block[c->shared_idx - 1 + c->shared_size.x];

	neighbours += sha_block[c->shared_idx - c->shared_size.x];

	neighbours += sha_block[c->shared_idx + c->shared_size.x];

	neighbours += sha_block[c->shared_idx + 1 - c->shared_size.x];
	neighbours += sha_block[c->shared_idx + 1];
	neighbours += sha_block[c->shared_idx + 1 + c->shared_size.x];
	#pragma endregion

	syncthreads();

	return neighbours;
}

__device__ __forceinline__ void Grid_D_UpdateCell(byte * dev_grid, Coordinates * c, byte value, bool * device_idle)
{
	if (dev_grid[c->global_idx] != value)
	{
		dev_grid[c->global_idx] = value;
		*device_idle = false;
	}
}

__global__ void Grid_D_GoLStep(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
	Grid_D_DefaultBegin();

	bool state = sha_block[c.shared_idx] == 1;
	int neighbours = Grid_D_CountNeighbours(dev_grid, &c);
	byte value = neighbours == 3 || (state && neighbours == 2);

	Grid_D_DefaultEnd();
}

__global__ void Grid_D_Rule90Step(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
	Grid_D_DefaultBegin();

	if (c.y == 0) return;
	byte value = (c.x == 0 || c.x == size_x - 1) ? 0 :
		(sha_block[SHA_NW(c)] + sha_block[SHA_NE(c)]) == 1;
	//byte value = sha_block[SHA_NE(c)];
	
	Grid_D_DefaultEnd();
}

__global__ void Grid_D_Rule90Step
(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
	Grid_D_DefaultBegin();

	if (c.y == 0) return;
	byte value = (c.x == 0 || c.x == size_x - 1) ? 0 :
		(sha_block[SHA_NW(c)] + sha_block[SHA_NE(c)]) == 1;

	Grid_D_DefaultEnd();
}

#pragma region GridKernel management
std::map<std::string, GridKernel> grid_kernels;
bool grid_kernels_initialized = false;

inline void GridKernelsMapInitialize()
{
	if (!grid_kernels_initialized)
	{
		grid_kernels.insert_or_assign("gol", Grid_D_GoLStep);
		grid_kernels.insert_or_assign("rule90", Grid_D_Rule90Step);

		grid_kernels_initialized = true;
	}
}

void AddKernelStep(const char * name, GridKernel kernel)
{ GridKernelsMapInitialize(); grid_kernels.insert_or_assign(name, kernel); }
void PrintKernelSteps()
{
	GridKernelsMapInitialize();
	printf("LIST OF VALID STEP FUNCTIONS:\n\n");
	for (auto i = grid_kernels.begin(); i != grid_kernels.end(); i++)
		printf("- %s\n", i->first.c_str());
}
void Grid_SetKernelStep(Grid * grid, const char * name)
{
	GridKernelsMapInitialize();
	auto i = grid_kernels.find(name);
	if (i != grid_kernels.end())
		grid->kernel_Step = i->second;
}
#pragma endregion