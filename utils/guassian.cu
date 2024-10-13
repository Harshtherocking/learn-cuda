#include <math.h>
#include <stdint.h>

#define PIE 3.14 

/*
00000000000000
00000000000000
00**********00
00**********00
00**********00
00**********00
00000000000000
00000000000000
*/
void Padding(uint8_t * padd_img, uint8_t * img, int width, int height, int padd_rad){
  // add initial layer of zeros 
  int offset = 0;
  for (int i=0; i<padd_rad; i++){
    memset(padd_img + offset, 0, sizeof(uint8_t) * (width  + 2 * padd_rad));
    offset += width + 2 * padd_rad;
  }

  for (int i=0; i<height; i++){
    // sert initial zeros in starting of each row
    memset(padd_img+offset, 0, sizeof(uint8_t)*padd_rad);

    // copying i th row from image
    memcpy(padd_img + offset + padd_rad, img + (i* width), sizeof(uint8_t) * width);

    // inserting last zeros in ending of each row
    memset(padd_img + offset + padd_rad + width, 0, sizeof(uint8_t) * padd_rad);

    offset += width + 2 * padd_rad;
  }

  // add ending layer of zeros 
  for (int i=0; i<padd_rad; i++){
    memset(padd_img + offset, 0, sizeof(uint8_t) * (width  + 2 * padd_rad));
    offset += width + 2 * padd_rad;
  }
}



/*
To Calculate Guassian Function value for Each element in the Kernal  
*/
__global__ void GuassianFunc (float * kernel, int sigma, int radius){

  __shared__ float constant;

  constant = 2 * sigma * sigma;

  __syncthreads();
  
  // threadIdx offset 
  int tx = threadIdx.x - radius;
  int ty = threadIdx.y - radius;

  float value = exp(- (tx*tx  + ty*ty) / constant) / (PIE * constant); 
  kernel[threadIdx.x * (2*radius + 1) + threadIdx.y] = value; 

  __syncthreads();

}


__global__ void GuassianBlur (
  uint8_t* image, 
  uint8_t * new_image, 
  float * kernel, 
  int width,
  int height, 
  int kernel_width
  )
  { 
    __shared__ int radius;
    __syncthreads();

    // indices for the image
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // place the indices for Kernel
    if (tx == 0 && ty == 0 && tz == 0){
      radius = (kernel_width - 1)/2;
    }
    __syncthreads();

    /*
    calculation for each neighbour of Pixel (tx,ty)
    Pixel (tx-radius, ty-radius) is the starting point for Convolution window
    Check if Pixel [(tx - rad) * width + ty + tz] lies in the range
    */ 

    if (0<= tx-radius < height && 0<= ty-radius < width && 0<= tx + radius < height && 0<= ty + radius < width)
    new_image[tx * width + ty] += image[(tx-radius) * width + (ty-radius) + tz] * kernel[tz];

  }