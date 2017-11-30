// gupslagolcli.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <Windows.h>

#include <macros.h>
#include <grid.h>
#include <tools.h>

#define MAX_CYCLE 10000

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

	int rows, cols;

	printf("rows: ");
	scanf_s("%d", &rows);
	printf("cols: ");
	scanf_s("%d", &cols);

	clear();

	Grid* grid = Grid_Create(1, rows, cols);
	Grid_Randomize(grid);
	Grid_Upload(grid);
	Grid_Print(grid, false);

	clear();
	for (int i = 0; i < MAX_CYCLE && !grid->Idle; i++)
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

