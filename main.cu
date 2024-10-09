#include <stdio.h>
#include "./utils/mat.cuh"
#include "./utils/env.cuh"
#include <cuda_runtime.h>

#define ROW1 15

#define COL2 5



int main() {
  // mat of size (ROW, COL)
  int * h_a = (int*) malloc (sizeof(int) * ROW1 * K);
  fill_rand_int(h_a, ROW1, K); 


  printf("Printing A : \n");
  printMat(h_a, ROW1, K);

  int * h_b = (int*) malloc (sizeof(int) * K * COL2);
  fill_rand_int(h_b, K, COL2);
  
  printf("Printing B: \n");
  printMat(h_b, K, COL2);

  int * p_h_c = (int*) malloc (sizeof(int) * ROW1 * COL2);

  int * d_a;
  int * d_b;
  int * d_c;

  //  __global__ 
  cudaMalloc(&d_a, sizeof(int) * ROW1 * K);
  cudaMalloc(&d_b, sizeof(int) * K * COL2);
  cudaMalloc(&d_c, sizeof(int) * ROW1 * COL2);


  cudaMemcpy(d_a, h_a, sizeof(int) * ROW1 * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * K * COL2, cudaMemcpyHostToDevice);


  dim3 gridDim (ROW1/B,COL2/B,1);
  dim3 blockDim (B,B,1);


  kernalMatMul<<<gridDim,blockDim>>>(d_a, d_b, d_c, ROW1, K, K, COL2);

  cudaDeviceSynchronize();

  cudaMemcpy(p_h_c, d_c, sizeof(int) * ROW1 * COL2, cudaMemcpyDeviceToHost);

  printf("parallel computation finished\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  printf("Printing result : \n");
  printMat(p_h_c, ROW1, COL2);

  free(h_a);
  free(h_b);
  free(p_h_c);

  return 0;
}
