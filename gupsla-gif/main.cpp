// gupslagolcli.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <Windows.h>

#include <macros.h>
#include <grid.h>
#include <tools.h>

#include "gif.h"

#define MAX_CYCLE 10000

// by sigil (https://stackoverflow.com/a/19717944)
wchar_t * convertCharArrayToLPCWSTR(const char* charArray)
{
	wchar_t* wString = new wchar_t[4096];
	MultiByteToWideChar(CP_ACP, 0, charArray, -1, wString, 4096);
	return wString;
}
bool fileExists(LPCWSTR path)
{
	DWORD attrib = GetFileAttributes(path);

	return (attrib != INVALID_FILE_ATTRIBUTES &&
		!(attrib & FILE_ATTRIBUTE_DIRECTORY));
}

void task(char * file, int rows, int cols, int delay = 1)
{
	wchar_t * wfile = convertCharArrayToLPCWSTR(file);

	GifWriter gif;
	
	assert(GifBegin(&gif, file, cols, rows, delay) == true); // validate file creation
	assert(fileExists(wfile)); // further validate file creation and prevent command injection

	Grid* grid = Grid_Create(1, rows, cols);
	Grid_Randomize(grid);
	Grid_Upload(grid);
	Grid_Print(grid, false);

	int i = 0;
	for (; i < MAX_CYCLE && !grid->Idle; i++)
	{
		Grid_Step(grid, false);
		Grid_Download(grid);
		GifWriteFrame(&gif, grid->Cells, grid->Columns, grid->Rows, delay);
	}
	printf("cycles: %d", i);
	// */

	GifEnd(&gif);

	printf("DONE");
}

int main()
{
	PrintTest();

	int rows = 40, cols = 40;
	char * file = "test.gif";

	system("dir");
	task(file, rows, cols);

	getchar();
	return 0;
}

