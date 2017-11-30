#pragma once
#include "macros.h"

typedef void (*GridKernel)(byte*, bool*, unsigned int, unsigned int, unsigned int); // __global__ void kernel(byte * dev_grid, bool * device_idle, unsigned int size_x, unsigned int size_y, unsigned int size_z)

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
void Grid_Step(Grid * grid
#if _DEBUG
	, bool print = true
#else
	, bool print = false
#endif
);