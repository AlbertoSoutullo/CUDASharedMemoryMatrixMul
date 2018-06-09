// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <windows.h>

#define TILE_WIDTH 16

__host__  void getError()
{
	cudaError_t error = cudaGetLastError();
	if (cudaSuccess != error)
		printf("%s\n", cudaGetErrorString(error));
	fflush(stdout);
}

////////////////////////////METODOS-MATRICES/////////////////////////////////////////////////////////

__host__ int readMatrix(char* fileName, float** m1, int transpose)
{
	FILE* f1 = NULL;
	int rowNum = 0;
	int colNum = 0;

	f1 = fopen(fileName, "r");

	if (f1 == NULL)
	{
		printf("The file couldn't be open.\n");
		return 0;
	}
	else
	{
		fread(&(rowNum), sizeof(int), 1, f1);
		fread(&(colNum), sizeof(int), 1, f1);
		*m1 = (float*)malloc(sizeof(float)*colNum*rowNum);
		if (transpose)
		{
			int i = 0;
			for (i = 0; i < colNum; i++)
			{
				int j = 0;
				for (j = 0; j < rowNum; j++)
				{
					fread(&(*m1)[j*rowNum + i], sizeof(float), 1, f1);
				}
			}
		}
		else
		{

			int i = 0;
			for (i = 0; i < rowNum; i++)
			{
				int j = 0;
				for (j = 0; j < rowNum; j++)
				{
					fread(&(*m1)[i*colNum + j], sizeof(float), 1, f1);
				}

			}
		}
		fclose(f1);
		return rowNum;
	}
}

__host__ void writeMatrix(char* fileName, float* m1, int colNum, int rowNum, int transpose)
{
	FILE* f1 = NULL;

	f1 = fopen(fileName, "w");

	if (f1 == NULL)
	{
		printf("The file couldn't be open.\n");
	}
	else
	{
		fwrite(&(rowNum), sizeof(int), 1, f1);
		fwrite(&(colNum), sizeof(int), 1, f1);

		if (transpose)
		{
			int i = 0;
			for (i = 0; i < rowNum; i++)
			{
				int j = 0;
				for (j = 0; j <rowNum; j++)
				{
					fwrite(&(m1[j*rowNum + i]), sizeof(float), 1, f1);
				}
			}
		}
		else
		{
			int i = 0;
			for (i = 0; i < rowNum; i++)
			{
				int j = 0;
				for (j = 0; j <rowNum; j++)
				{
					fwrite(&(m1[i*colNum + j]), sizeof(float), 1, f1);
				}
			}
		}
		fclose(f1);
	}
}


__host__ void printMatrix(float* matrix, int rowNum, int colNum)
{
	int i = 0;
	for (i = 0; i < rowNum; i++)
	{
		int j = 0;
		for (j = 0; j < colNum; j++)
		{
			printf("%0.3f ", matrix[i*colNum + j]);
		}
		printf("\n");
	}
}

__host__ int martrixComparator(float* matrix1, float* correctMatrix, int rowNum, int colNum)
{
	int  i = 0;
	for (i = 0; i < rowNum; i++)
	{
		int  j = 0;
		for (j = 0; j < colNum; j++)
		{
			if (matrix1[i*colNum + j] != correctMatrix[i*colNum + j])
			{
				return 0;
			}
		}
	}
	return 1;
}

__host__ void multiplySimple(float* m1, float* m2, float* mres, int numFilas1, int numFilas2, int numColumnas)
{

	for (int i = 0; i < numFilas1; i++) {//iterate through a given set of rows of [A]
		for (int j = 0; j < numFilas2; j++) {//iterate through columns of [B]
			for (int k = 0; k < numColumnas; k++) {//iterate through rows of [B]
				mres[i*numColumnas + j] += (m1[i*numColumnas + k] * m2[j*numColumnas + k]);
			}
		}
	}

}


// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns) {
	
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_WIDTH + ty,
		Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
		if (Row < numARows && m*TILE_WIDTH + tx < numAColumns)
			ds_M[ty][tx] = A[Row*numAColumns + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (Col < numBColumns && m*TILE_WIDTH + ty < numBRows)
			ds_N[ty][tx] = B[(m*TILE_WIDTH + ty)*numBColumns + Col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}
	if (Row < numCRows && Col < numCColumns)
		C[Row*numCColumns + Col] = Pvalue;
}

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
	int BCols, int CRows, int CCols)
{
	float CValue = 0;

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	for (int k = 0; k < (TILE_WIDTH + ACols - 1) / TILE_WIDTH; k++) {

		if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
			As[threadIdx.y][threadIdx.x] = A[Row*ACols + k * TILE_WIDTH + threadIdx.x];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
			Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < TILE_WIDTH; ++n)
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
	}

	if (Row < CRows && Col < CCols)
		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
		(blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

int main(int argc, char ** argv) {

	//Initialazing variables
	float* h_mat1 = NULL; // The A matrix
	float* h_mat2 = NULL; // The B matrix
	float* h_matres = NULL; // The output C matrix
	float* d_mat1 = NULL;
	float* d_mat2 = NULL;
	float* d_matres = NULL;


	//File names
	/*char str1[20];
	char str2[20];
	printf("Introduce the fila name of the first matrix:");
	scanf("%s", str1);
	printf("Introduce the fila name of the second matrix:");
	scanf("%s", str2);*/
	clock_t cpu_startTimeTotal, cpu_endTimeTotal, cpu_startTimeMult, cpu_endTimeMult;
	double cpu_ElapseTimeTotal = 0;
	double cpu_ElapseTimeMult = 0;
	cpu_startTimeTotal = clock();


	int numARows; // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	numAColumns = numARows = readMatrix("500x500.bin", &h_mat1, 0);

	int numBRows; // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	numBColumns = numBRows = readMatrix("500x500I.bin", &h_mat2, 1);


	//inicializar resultado 
	h_matres = (float*)malloc(sizeof(float)*numAColumns*numBRows);


	//Reservamos memoria en GPU
	cudaMalloc(&d_mat1, sizeof(float) * numARows * numAColumns);
	cudaMalloc(&d_mat2, sizeof(float) * numBRows * numBColumns);
	cudaMalloc(&d_matres, sizeof(float) * numAColumns * numBRows);



	//pasamos las matrices a memoria de GPU
	cudaMemcpy(d_mat1, h_mat1, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
	getError();
	cudaMemcpy(d_mat2, h_mat2, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
	getError();



	//Definimos el tamaño de bloque 
	int tam = numARows * numAColumns;
	int numthreadporbloque = 1024;

	int numbloques = ((numARows / 32) * (numBRows / 32)) + 1;
	int numbloquesMax = 500;
	int totalDivision = numbloques / numbloquesMax + 1;

	cpu_startTimeMult = clock();

	
	//Multiplicamos
	MatMul << <numbloquesMax, numthreadporbloque >> >(d_mat1, d_mat2, d_matres,
		numARows, numAColumns,
		numBRows, numBColumns,
		numAColumns, numBRows);

	cudaThreadSynchronize();
	getError();


	cpu_endTimeMult = clock();


	//Pasamos la matriz resultado a cpu
	cudaMemcpy(h_matres, d_matres, sizeof(float) * numAColumns * numBRows, cudaMemcpyDeviceToHost);
	getError();

	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_matres);

	//escribimos la matriz resultado en un archivo
	writeMatrix("result.bin", h_matres, numAColumns, numBRows, 0);

	cpu_endTimeTotal = clock();
	cpu_ElapseTimeMult = ((cpu_endTimeMult - cpu_startTimeMult) / CLOCKS_PER_SEC);
	cpu_ElapseTimeTotal = ((cpu_endTimeTotal - cpu_startTimeTotal) / CLOCKS_PER_SEC);

	printf("Tiempo Total: %d", cpu_ElapseTimeTotal);
	printf("Tiempo Multiplicando: %d", cpu_ElapseTimeMult);

	//int result = martrixComparator(h_mat1, h_matres, numAColumns, numBRows);

	//if (result) printf("Matrix match.\n");

	free(h_mat1);
	free(h_mat2);
	free(h_matres);

	

	return 0;
}