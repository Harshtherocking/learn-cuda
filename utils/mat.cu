#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "mat.cuh"
#include "env.cuh"
#include <stdint.h>

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

void printMat(float * arr, int row, int col){
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      printf("%f ", arr[i*col + j]);
    }
    printf("\n");
  }
}

void printMat(uint8_t* arr, int row, int col){
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
//     for (int i=0; i<c1; i++){
//       sum += a[tidx * c1 + i] * b[i* c2 + tidy];
//     }
//     c[tidx * c2 + tidy] = sum;
//   }
//
// }


// Block wise Parallel Matrix Multiplication
__global__ void kernalMatMul (int * a, int * b, int *c, int r1, int c1, int r2, int c2){
  assert(c1==r2);
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global indexing to insert element in Global resultant matrix
  // C[i][y]
  int i = bx * blockDim.x + tx;
  int j = by * blockDim.y + ty;

  __shared__ int sh_a [B*K];
  __shared__ int sh_b [B*K];

  //syncting so that shared memory stays defined for each thread in the block
  __syncthreads();

  // copying data only once per block 
  if (!(tx | ty)){
    int offset_a = B*bx*K;

    for (int i=0; i<B*K; i++){
      // copying rows from a to sh_a
      sh_a[i] = a[offset_a + i];

    }

    for (int i=0; i<K; i++){
      int offset_b = c2 * i + B * blockIdx.y;
      for (int j=0; j<B; j++){
         // copying coloums from b to sh_b
         sh_b [B*i + j] = b[offset_b + j];
      }
    }

  } 
  
  // syncing so that each thread has access to copied data in shared memory
   __syncthreads();


  // working for each thread
  int local_c_tx_ty =0;
  for (int i=0; i<K; i++){
    local_c_tx_ty += sh_a[tx * K + i] * sh_b[i * B + ty];
  }

  // mapping local calc to global solution array
  c[i * c2 + j] = local_c_tx_ty;

}
