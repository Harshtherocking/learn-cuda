#pragma once
#include <stdint.h>

// Fill Array with integers less than 10
void fill_rand_int (int * arr, int row, int col);


// Print 2D Matrix
void printMat(int * arr, int row, int col);
void printMat(float * arr, int row, int col);
void printMat(uint8_t* arr, int row, int col);
void printMat(long long* arr, int row, int col);


// Sequential Matrix Multiplication 
void serialMatMul (int * arr1, int * arr2, int * arr3, int r1, int c1, int r2, int c2);


// Parallel Matrix Multliplication 

// __global__ void kernalMatMul (int * a, int *b , int * c, int r1, int c1, int r2, int c2);


__global__ void kernalMatMul (int * a, int *b , int * c, int r1, int c1, int r2, int c2);



__global__ void Tiled_Mat_Multi(int * a, int * b,long long int *c, int N);

__global__ void Tiled_Mat_Multi(int * a, int * b, int *c, int N);
__global__ void Tiled_Mat_Multi(double * a, double * b, double *c, int N);