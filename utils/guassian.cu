#include <math.h>

#define PIE 3.14159265 

typedef struct GuassianKernel {
  float * kernel;
  int sigma;
  int radius;
} GuassianKernel;



// to Calculate Guassian Function value for each element in the Kernel
__global__ void GaussianFunc (float * kernel, int * sigma, int * radius){

  int sig = * sigma;
  int rad = * radius;

  // threadIdx offset 
  int tx = threadIdx.x - rad;
  int ty = threadIdx.y - rad;

  if (-rad < tx < rad && -rad < ty < rad){
    kernel[tx * (2*rad + 1) + ty] = exp(-(tx*tx + ty*ty)/(2*sig*sig)) / (2 * PIE * sig*sig); 
  }

}



void init_kernel (GuassianKernel * g, int sigma, int radius){

  int kernel_size = (2*radius + 1) * (2*radius + 1);

  g->sigma = sigma;
  g->radius = radius;
  g->kernel = (float*) malloc (sizeof(float) * kernel_size);

  
  // device variable declaration
  float * d_kernel;
  int * d_radius;
  int * d_sigma; 


  // device variable memory allocation
  cudaMalloc(&d_kernel, sizeof(float) * kernel_size);
  cudaMalloc(&d_radius, sizeof(int));
  cudaMalloc(&d_sigma, sizeof(int));


  // memory copy from host to device 
  cudaMemcpy(d_radius, &g->radius, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sigma, &g->sigma, sizeof(int), cudaMemcpyHostToDevice);

  
  // Grid Dim and Block Dim preparation
  const dim3 gD (1,1,1);
  const dim3 bD (2*radius+1, 2*radius+1, 1);


  // launching a kernal 
  GuassianKernel<<<gD,bD>>>(d_kernel, d_sigma, d_radius);

  // memory copy from device to host
  cudaMemcpy (g->kernel, d_kernel, sizeof(float) * kernel_size, cudaMemcpyDeviceToHost);
  
}
