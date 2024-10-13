// #include <cstdint>
#include <stdint.h>
// #include <driver_types.h>
#include <stdlib.h>
// #include <sys/types.h>
#include "../utils/mat.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "./stb/stb_image.h"
#include "./stb/stb_image_write.h"

#include "../utils/guassian.cuh"


#define IMAGEPATH "./images/grainy.jpg"
#define NEWIMAGEPATH "./images/smooth.jpg"

#define SIGMA 3
#define RADIUS 2

int main(){

  /*
  Kernel Initialisation for Gaussian Blurring 
  Kernel of Standard Deviation = SIGMA 
  Kernel Width = 2 x RADIUS + 1
  */

  int kernel_size = 2*RADIUS +1;
  
  // initialising pointer for Kernel
  float * kernel  =  (float *) malloc (sizeof(float) * kernel_size * kernel_size);

  // device variable declaration & memory allocation
  float * d_kernel;
  cudaMalloc(&d_kernel, sizeof(float) * kernel_size * kernel_size);

  // Grid Dim and Block Dim preparation
  dim3 gridDim (1,1,1);
  dim3 blockDim (kernel_size, kernel_size, 1);
  
  // Launching GPU Kernel
  GuassianFunc<<<gridDim,blockDim>>>(d_kernel, SIGMA, RADIUS);
  cudaDeviceSynchronize();

  // memory copy from device to host
  cudaMemcpy (kernel, d_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyDeviceToHost);
  cudaFree(d_kernel);

  printMat(kernel, kernel_size , kernel_size);



  /*
  Reading black & white image and saving Width & Height
  */
  int width, height, bpp;
  uint8_t * image = stbi_load (IMAGEPATH, &width, &height, &bpp, 1);
  
  printf("Width %d, Height %d, BPP %d\n", width, height, bpp);

  
  /*
  Padding image for convolution step 
  */
  uint8_t * padd_img = (uint8_t * ) malloc (sizeof (uint8_t) * (width + 2*RADIUS) * (height + 2*RADIUS));

  Padding(padd_img, image, width, height, RADIUS);



  // /*
  // Guassian Blur 
  // */
  // // init new image
  // int * new_image = (int*) malloc (sizeof(int) * width * height);

  // // initialising new images 
  // uint8_t * d_image;
  // uint8_t * d_new_image;
  // float * d_k;

  // // memory allocation on device 
  // cudaMalloc(&d_image, sizeof (uint8_t) * width * height);
  // cudaMalloc(&d_new_image, sizeof (uint8_t) * width * height);
  // cudaMalloc(&d_k, sizeof(float) * kernel_size * kernel_size);

  // // copy data from Host to Device
  // cudaMemcpy(d_k, kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_image, image, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice);

  // // Kernel Launch
  // gridDim = dim3 (1,1,1);
  // blockDim = dim3 (width, height,kernel_size * kernel_size);
  
  // GuassianBlur<<<gridDim,blockDim>>>(d_image, d_new_image, d_k, width, height, kernel_size);
  
  // // copy data from Device to Host
  // cudaMemcpy(new_image, d_new_image, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);

  // // free-ing allocated memory locations
  // cudaFree(d_image);
  // cudaFree(d_new_image);
  // cudaFree(d_k);
  

  // printMat(new_image, width, height);

  // stbi_write_jpg(NEWIMAGEPATH, width, height, bpp, new_image, 1);

  return 0;
}
