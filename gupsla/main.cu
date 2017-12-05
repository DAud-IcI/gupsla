#include <stdio.h>
#include <Windows.h>

#include "macros.h"
#include "cuda_ext.cuh"
#include "grid.h"
#include "tools.h"


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

int main()
{
	//BMP* bmp = BMP_ReadFile("console_gol.bmp");

	PrintTest();
	getchar();
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
		//getchar();
	}
	// */

	

	getchar();
	return 0;
}