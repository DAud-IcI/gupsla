#include "grid.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#define CELLS_SIZE grid->Tables * grid->Rows * grid->Columns * sizeof(byte)


char grid_chars[255]{ '_', 'X' };

bool init = false;
cudaDeviceProp gpu_info;
int block_width;
dim3 threads;
size_t shared_memory_size;

Grid * Grid_Create(int tables, int rows, int columns)
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
		shared_memory_size = block_width * (block_width + 4) * sizeof(byte);
	}
	gpuOk(cudaMalloc(&(grid->dev_Cells), CELLS_SIZE));

	grid->Step = &Grid_D_GoLStep;

	return grid;
}

void Grid_Destroy(Grid ** grid_ptr)
{
	Grid * grid = *grid_ptr;

	free(grid->Cells);
	cudaFree(grid->dev_Cells);
	free(grid);

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

#define devide_ceiling(a, b) (int)ceil((double)(a) / (b))
void Grid_Step(Grid * grid)
{
	dim3 blocks(devide_ceiling(grid->Columns, block_width), devide_ceiling(grid->Rows, block_width));
	dim3 grid_size(grid->Columns, grid->Rows, grid->Tables);

	printf("SIZE   : %d x %d x %d\n", grid_size.x, grid_size.y, grid_size.z);
	printf("BLOCKS : %d x %d x %d\n", blocks.x, blocks.y, blocks.z);
	printf("THREADS: %d x %d x %d\n", threads.x, threads.y, threads.z);
	printf("SHARED : %g kB\n", ceil(shared_memory_size / 1024.0));
	
	grid->Step DEV3(blocks, threads, shared_memory_size) (grid->dev_Cells, grid_size);
}

__global__ void Grid_D_GoLStep(byte * device_grid, dim3 size)
{
	int neighbours = 0;
	bool state = false;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= size.x || y >= size.y)
	{
		syncthreads();
		return;
	}

	int i = XYW(x, y, size.x);

	state = device_grid[i] == 1;
	bool space_up = threadIdx.y > 0;
	bool space_down = threadIdx.y < size.y - 1;

	if (threadIdx.x > 0)
	{
		if (space_up) neighbours += device_grid[i - 1 - size.x];
		neighbours += device_grid[i - 1];
		if (space_down) neighbours += device_grid[i - 1 + size.x];
	}

	// if (true)
	{
		if (space_up) neighbours += device_grid[i - size.x];
		if (space_down) neighbours += device_grid[i + size.x];
	}

	if (threadIdx.x < size.x)
	{
		if (space_up) neighbours += device_grid[i + 1 - size.x];
		neighbours += device_grid[i + 1];
		if (space_down) neighbours += device_grid[i + 1 + size.x];
	}

	syncthreads();

	device_grid[i] = neighbours == 3 || (state && neighbours == 2);

}