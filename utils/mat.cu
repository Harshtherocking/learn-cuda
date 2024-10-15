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

// Fill Array with random double 
void fill_rand_double (double * arr, int row, int col){
  for (int i=0; i<row*col; i++) arr[i] = rand()/RAND_MAX;
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

void printMat(double * arr, int row, int col){
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      printf("%.2f ", arr[i*col + j]);
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

void printMat(long long* arr, int row, int col){
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      printf("%lld ", arr[i*col + j]);
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


void squareMatMul(double* arr1, double * arr2, double * arr3, int N){
  for (int i=0; i<N; i++){
    for (int k=0; k<N; k++){

      int sum = 0; 
      for (int j=0; j<N; j++) sum += arr1[i*N + j] + arr2[j*N + k];
      arr3[i*N + k] = sum;
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



__global__ void Tiled_Mat_Multi(int * a, int * b, int *c, int N){
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // C[i][y]
  int i = bx * blockDim.x + tx;
  int j = by * blockDim.y + ty;

  __shared__ int sh_a [B*B];
  __shared__ int sh_b [B*B];
  __syncthreads();


  // copying data only once per block 
  if (tx == 0 && ty == 0){
    int offset_a = B*bx*N + B*bz;
    int offset_b = B*bz*N + B*by;

    for (int x=0; x<B; x++){
      // copying each row in Tile
      for (int y=0; y<B; y++){
        sh_a[x * B + y] = a[offset_a + y];
        sh_b[x * B + y] = b[offset_b + y];
      }
      // offset to next row in a the Tile
      offset_a += N;
      offset_b += N;
    }

  }

  __syncthreads();

  // sum specific to a Block
  int local_sum  = 0;
  for (int k=0; k<B; k++){
    local_sum += sh_a[tx * B + k] * sh_b[k * B + ty];
  }

  // add local sum to Global resultant matrix using Atomic Add
  atomicAdd(&c[i * N + j], local_sum);
}



// __global__ void Tiled_Mat_Multi(double* a, double * b, double *c, int N){
  
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int bz = blockIdx.z;
  
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   // C[i][y]
//   int i = bx * blockDim.x + tx;
//   int j = by * blockDim.y + ty;

//   __shared__ double sh_a [B*B];
//   __shared__ double sh_b [B*B];
//   __syncthreads();


//   // copying data only once per block 
//   if (tx == 0 && ty == 0){
//     int offset_a = B*bx*N + B*bz;
//     int offset_b = B*bz*N + B*by;

//     for (int x=0; x<B; x++){
//       // copying each row in Tile
//       for (int y=0; y<B; y++){
//         sh_a[x * B + y] = a[offset_a + y];
//         sh_b[x * B + y] = b[offset_b + y];
//       }
//       // offset to next row in a the Tile
//       offset_a += N;
//       offset_b += N;
//     }

//   }

//   __syncthreads();

//   // sum specific to a Block
//   double local_sum  = 0;
//   for (int k=0; k<B; k++){
//     local_sum += sh_a[tx * B + k] * sh_b[k * B + ty];
//   }

//   // add local sum to Global resultant matrix using Atomic Add
//   atomicAdd(&c[i * N + j], local_sum);
// }