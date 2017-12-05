// gupslagolcli.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <Windows.h>

#include <macros.h>
#include <grid.h>
#include <tools.h>
#include <math.h>

extern "C" 
{
#include "gifenc.h"
}

#define MAX_CYCLE 1000

#define equals(var, txt) strcmp(var, txt) == 0

void task(const char * file, int rows, int cols, const char * kernel_name, byte palette[], int palette_length, int delay, bool last_frame)
{
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
		if (!last_frame)
		{
			Grid_Download(grid);
			gif->frame = grid->Cells;
			ge_add_frame(gif, delay);
		}
		//printf("cycle: %d\n", i);
	}
	if (last_frame)
	{
		Grid_Download(grid);
		gif->frame = grid->Cells;
		ge_add_frame(gif, delay);
	}
	printf("cycles: %d\n", i);
	// */

	ge_close_gif(gif);

	printf("DONE\n");
}

void printHelp()
{
	printf("gupsla-gif\n\nUSAGE:\n\n");
	printf("gupsla-gif [-r|--rows 256] [-c|--cols|--columns 256] [-d|--delay 0] [--last|--last-frame] file.gif [kernel_name] [palette_file]\n");
	printf("gupsla-gif -h|--help\n");
	printf("gupsla-gif -l|--list\n");
	printf("\n");
	
	printf("-r --rows     : the height of the picture\n");
	printf("-c --cols\n");
	printf("--columns     : the width of the picture\n");
	printf("-d --delay    : time between frames in 1/100s units (centiseconds)\n");
	printf("--last\n");
	printf("--last-frame  : save only the last frame (no animation)\n");
	printf("\n");

	printf("-h --help     : shows this screen and terminates\n");
	printf("-l --list     : shows the list of valid simulations and terminates\n");
	printf("\n");

	printf("file.gif     : the file where the GIF is saved\n");
	printf("kernel_name  : the name of the model to be used (use -l or --list to get the valid names)\n");
	printf("palette_file : the file contains hex byte triplets that indicate the colour of each state (each line is like \"ff ff ff\")\n");
}

int updatePalette(byte palette[], const char * file_name)
{
	FILE *file;
	errno_t err = fopen_s(&file, file_name, "r");

	if (!file)
	{
		printf("ERROR OPENING FILE \"%s\" : %d", file_name, err);
		exit(2);
	}

	byte r, g, b;
	int i = 0;
	while (fscanf_s(file, "%hhx %hhx %hhx", &r, &g, &b) == 3)
	{
		palette[i++] = r;
		palette[i++] = g;
		palette[i++] = b;
	}

	return i / 3;
}

int main(int argc, char ** argv)
{
	int rows = 256, cols = 256, delay = 0;
	bool last_frame = false;
	char *file = "test.gif", *kernel_name = "gol";

	byte palette[256] = { 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF };
	int palette_length = 2;

	int arg_state = 0;

	for (int i = 1; i < argc; ++i)
	{
		char * x = argv[i];
		for (char * p = x; *p; ++p) *p = tolower(*p);

		if (equals(x, "-r") || equals(x, "--rows"))
			rows = atoi(argv[++i]);
		else if (equals(x, "-c") || equals(x, "--cols") || equals(x, "--columns"))
			cols = atoi(argv[++i]);
		else if (equals(x, "-d") || equals(x, "--delay"))
			delay = atoi(argv[++i]);
		else if (equals(x, "--last") || equals(x, "--last-frame"))
			last_frame = true;

		else if (equals(x, "-h") || equals(x, "--help"))
		{ printHelp(); return 0; }
		else if (equals(x, "-l") || equals(x, "--list"))
		{ PrintKernelSteps(); return 0; }

		else if (arg_state == 0)
		{
			file = argv[i];
			arg_state++;
		}
		else if (arg_state == 1)
		{
			kernel_name = argv[i];
			arg_state++;
		}
		else if(arg_state == 2)
		{
			palette_length = updatePalette(palette, argv[i]);
			arg_state++;
		}
		else
		{
			printf("INCORRECT ARGUMENT\n\n");
			printHelp();
			return 1;
		}
	}

	PrintTest();

	//system("dir");
	task(file, rows, cols, kernel_name, palette, palette_length, delay, last_frame);

	//printf("\nPress any key to exit...");
	//getchar();
	system(file);
	return 0;
}

