// gupslagolcli.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <Windows.h>

#include <macros.h>
#include <grid.h>
#include <tools.h>
#include <math.h>

extern "C" {
#include "gifenc.h"
}

#define MAX_CYCLE 1000

void task(const char * file, int rows, int cols, const char * kernel_name, byte palette[], int palette_length, int delay, bool last_frame)
{
	PrintKernelSteps();

	// create gif file
	ge_GIF *gif = ge_new_gif(file, cols, rows, palette, (int) ceil(log2(palette_length)), 0);

	Grid* grid = Grid_Create(1, rows, cols, kernel_name);
	Grid_Randomize(grid);
	Grid_Upload(grid);
	if (rows < 30 && cols < 120) Grid_Print(grid, false);

	int i = 0;
	for (; i < MAX_CYCLE && !grid->Idle; i++)
	{
		Grid_Step(grid, false);
		Grid_Download(grid);
		if (!last_frame)
		{
			gif->frame = grid->Cells;
			ge_add_frame(gif, delay);
		}
		//printf("cycle: %d\n", i);
	}
	if (last_frame)
	{
		gif->frame = grid->Cells;
		ge_add_frame(gif, delay);
	}
	printf("cycles: %d\n", i);
	// */

	ge_close_gif(gif);

	printf("DONE\n");
}

int main()
{
	PrintTest();

	//int rows = 256, cols = 256, delay = 0;
	int rows = 256, cols = 80, delay = 0;
	char * file = "test.gif";
	//char * kernel_name = "gol";
	char * kernel_name = "rule90";
	bool last_frame = false;

	byte palette[] = { 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF };
	int palette_length = 2;

	system("dir");
	task(file, rows, cols, kernel_name, palette, palette_length, delay, last_frame);

	//printf("\nPress any key to exit...");
	//getchar();
	system(file);
	return 0;
}

