#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "mat.cuh"

#define B 5
#define TEMP 50

// Fill Array with integers less than 10
void fill_rand_int (int * arr, int row, int col){
  for (int i=0; i<row*col; i++) arr[i] = rand()%10;
}


// Print 2D Matrix
void printMat(int * arr, int row, int col){
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      printf("%d ", arr[i*col + j]);
    }
    printf("\n");
  }
}


// Sequential Matrix Multiplication 
void serialMatMul (int * arr1, int * arr2, int * arr3, int r1, int c1, int r2, int c2){

  assert(c1 == r2);

  for (int i=0; i<r1; i++){
    for (int k=0; k<c2; k++){

      int sum = 0; 
      for (int j=0; j<c1; j++) sum += arr1[i*c1 + j] + arr2[j*c2 + k];
      arr3[i*c2 + k] = sum;
    }
  }

}


// Parallel Matrix Multiplication
// __global__ void kernalMatMul (int * a, int * b, int *c, int r1, int c1, int r2, int c2){
//   assert(c1==r2);
//   int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//   int tidy = blockIdx.y * blockDim.y + threadIdx.y;
//
//   if (tidx < r1 && tidy < c2){
//     int sum = 0;
//     for (int i=0; i<c2; i++){
//       sum += a[tidx * c1 + i] * b[i* c2 + tidy];
//     }
//     c[tidx * c2 + tidy] = sum;
//   }
//
// }


// Block wise Parallel Matrix Multiplication
__global__ void kernalMatMul (int * a, int * b, int *c, int r1, int c1, int r2, int c2){
  assert(c1==r2);

  // let only one thread do the copying
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    __shared__ int shared_a[B*TEMP];
    __shared__ int shared_b[B*TEMP];

    // copy rows 
    memcpy(shared_a, a + blockIdx.x *B*c1 , B*c1);

    //copy coloums 
    for (int i=0; i<r2; i++){
      memcpy(shared_b + B, b + i*r2*c2 + blockIdx.y + B, B);
    }

  }
  
  // then wait for other threads 
  __syncthreads();

  // offset to first element of C in a particular block
  int offset = B * (blockIdx.x * c2 + blockIdx.y);

  // work for each thread 

  int sum = 0;
  for (int i=0; i<c1; i++){
    sum += shared_a[threadIdx.x * c1 + i] * shared_b[i* B + threadIdx.y];
  }
  c[offset + threadIdx.x * B + threadIdx.y] = sum;

}







