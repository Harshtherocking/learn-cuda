#include <stdio.h>
#include "./utils/mat.cuh"
#include "./utils/env.cuh"
#include <cuda_runtime.h>

#define N 4 

int main() {
  int * h_a = (int*) malloc (sizeof(int) * N * N);
  fill_rand_int(h_a, N, N); 

  printf("Printing A : \n");
  printMat(h_a, N, N);


  int * h_b = (int*) malloc (sizeof(int) * N * N);
  fill_rand_int(h_b, N, N);
  
  printf("Printing B: \n");
  printMat(h_b, N, N);

  int * h_c = (int*) malloc (sizeof(int) * N * N);

  int * d_a;
  int * d_b;
  int * d_c;

  //  __global__ 
  cudaMalloc(&d_a, sizeof(int) * N * N);
  cudaMalloc(&d_b, sizeof(int) * N * N);
  cudaMalloc(&d_c, sizeof(int) * N * N);


  cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);


  dim3 gridDim (N/B, N/B,N/B);
  dim3 blockDim (B,B,1);


  Tiled_Mat_Multi<<<gridDim,blockDim>>>(d_a, d_b, d_c, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

  printf("parallel computation finished\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  printf("Printing result : \n");
  printMat(h_c, N, N);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
