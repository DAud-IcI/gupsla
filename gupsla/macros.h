#pragma once

// INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

// CORE LANGUAGE
typedef unsigned char byte;
#define alloc(count, type) (type*)malloc((count) * sizeof(type))

// PROJECT EXTENSION
 // 2D index generation in flat array
#define XYW(x, y, width) ((x) + (y) * (width))
#define RCC(row, column, columns) ((column) + (row) * (columns))
 // 3D index generation in flat array
#define XYZDW(x, y, z, width, height) (YXW(x, y, width) + (z) * (width) * (height))
#define TRCTC(table, row, column, rows, columns) (RCC(row, column, columns) + (table) * (rows) * (columns))