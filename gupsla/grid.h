#pragma once
#include "macros.h"

typedef void (*GridKernel)(byte*, bool*, dim3); // __global__ void kernel(byte * dev_grid, bool * device_idle, dim3 size)

struct Grid
{
	int Tables;
	int Rows;
	int Columns;
	
	byte * Cells;
	byte * dev_Cells;

	GridKernel kernel_Step;
	bool Idle;
	bool * dev_Idle;
};

Grid * Grid_Create(int tables, int rows, int columns);
void Grid_Destroy(Grid ** grid_ptr);

void Grid_Randomize(Grid * grid);
void Grid_Print(Grid * grid, bool draw_outline = true);
void Grid_Upload(Grid * grid);
void Grid_Download(Grid * grid);
void Grid_Step(Grid * grid);

__global__ void Grid_D_GoLStep(byte * dev_grid, bool * device_idle, dim3 size);