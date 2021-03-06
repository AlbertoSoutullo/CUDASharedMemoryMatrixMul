// CrearMatrices.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <intrin.h>
#include <iostream>

typedef struct Matrix_t
{
	int		rowNum;
	int		colNum;
	float** data;
}Matrix_t;

Matrix_t* createMatrix(int rowNum, int colNum)
{
	Matrix_t* newMatrix;
	newMatrix = (Matrix_t*)malloc(sizeof(Matrix_t));
	newMatrix->colNum = colNum;
	newMatrix->rowNum = rowNum;
	newMatrix->data = (float**)malloc(sizeof(float*)*rowNum);

	int i = 0;
	for (i = 0; i < rowNum; i++)
	{
		newMatrix->data[i] = (float*)malloc(sizeof(float)*colNum);
	}

	return newMatrix;
}

void createAndFillMatrix(Matrix_t** matrix, int rows, int cols, bool unity)
{
	int i = 0;
	int j = 0;
	float data;

	(*matrix) = createMatrix(rows, cols);
	
	for (i = 0; i < (*matrix)->rowNum; i++)
	{
		for (j = 0; j < (*matrix)->colNum; j++)
		{
			if(unity)
			{
				if (i == j) {
					data = 1;
				}
				else data = 0;
			}
			else {
				data = rand() % 11;
			}
			(*matrix)->data[i][j] = data;
		}
	}
}

void writeMatrix(const char* fileName, Matrix_t* m1, int transpose)
{
	FILE* f1 = NULL;

	fopen_s( &f1, fileName, "w");

	if (f1 == NULL)
	{
		printf("The file couldn't be open.\n");
	}
	else
	{
		fwrite(&(m1->rowNum), sizeof(int), 1, f1);
		fwrite(&(m1->colNum), sizeof(int), 1, f1);

		if (transpose)
		{
			int i = 0;
			for (i = 0; i < m1->rowNum; i++)
			{
				int j = 0;
				for (j = 0; j < m1->rowNum; j++)
				{
					fwrite(&(m1->data[j][i]), sizeof(float), 1, f1);
				}
			}
		}
		else
		{
			int i = 0;
			for (i = 0; i < m1->rowNum; i++)
			{
				fwrite(m1->data[i], sizeof(float), m1->colNum, f1);
			}
		}
		fclose(f1);
	}
}



int main()
{
	srand(time(NULL));
	int rows = 0, cols = 0;
	std::string name = "";

	Matrix_t *m1 = NULL;

	printf("Matrix 1 (non Identity)\n");
	printf("Row number:");
	scanf_s("%d", &rows);
	printf("Column number:");
	scanf_s("%d", &cols);

	printf("Write the name of the file for Matrix 1:");
	std::cin >> name;

	createAndFillMatrix(&m1, rows, cols, false);
	writeMatrix(name.c_str(), m1, 0);


	printf("Matrix 2 (Identity)\n");
	printf("Row number:");
	scanf_s("%d", &rows);
	printf("Column number:");
	scanf_s("%d", &cols);

	printf("Write the name of the file for Matrix 2:");
	std::cin >> name;

	createAndFillMatrix(&m1, rows, cols, true);
	writeMatrix(name.c_str(), m1, 0);



	return 0;
}
