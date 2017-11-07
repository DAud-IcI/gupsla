#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include "macros.h"
#include "grid.h"
#include "qdbmp.h"

void clear() { system("cls"); }
void gotoxy(int x, int y)
{
	COORD c;
	c.X = x;
	c.Y = y;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), c);
}

//void sleep(int ms) { Sleep(ms); }
#define sleep Sleep



int threads;

void test()
{
	cudaDeviceProp gpu_info;
	gpuOk(cudaGetDeviceProperties(&gpu_info, 0));

	printf("sharedMemPerBlock: %g kB\n", gpu_info.sharedMemPerBlock / 1024.0);
	printf("sharedMemPerBlockOptin: %d\n", gpu_info.sharedMemPerBlockOptin);
	printf("sharedMemPerMultiprocessor: %g kB\n", gpu_info.sharedMemPerMultiprocessor / 1024.0);
	printf("maxThreadsPerBlock: %d\n", gpu_info.maxThreadsPerBlock);

	int block_width = (int)sqrt(gpu_info.maxThreadsPerBlock) - 2;
	int sh_width = (int)(sqrt(gpu_info.sharedMemPerBlock + 4) - 2);
	if (sh_width < block_width) block_width = sh_width;
	printf("block_width: %d\n", block_width);

	getchar();
}


int main()
{
	//BMP* bmp = BMP_ReadFile("console_gol.bmp");

	test();
	clear();

	Grid* grid = Grid_Create(1, 32, 62);
	Grid_Randomize(grid);
	Grid_Upload(grid);
	Grid_Print(grid, false);

	clear();
	for(int i = 0; i < INT_MAX - 1 && !grid->Idle; i++)
	{
		gotoxy(0, 0);
		Grid_Step(grid);
		Grid_Download(grid);
		Grid_Print(grid, false);
		//sleep(20);
	}
	// */

	

	getchar();
	return 0;
}