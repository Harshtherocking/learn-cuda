#include <stdint.h>
#include "../utils/mat.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "./stb/stb_image.h"
#include "./stb/stb_image_write.h"

#include "../utils/guassian.cuh"


#define IMAGEPATH "./images/colosseum.ppm"
#define SIGMA 3
#define RADIUS 5

int main(){
  int width, height, bpp;
  uint8_t * image = stbi_load (IMAGEPATH, &width, &height, &bpp, 1);

  printf("Width %d, Height %d, BPP %d\n", width, height, bpp);

  GuassianKernel * g;
  init_kernel(g, SIGMA, RADIUS);

  printMat(g->kernel, (2*RADIUS+1) , (2*RADIUS+1));
  
  
  
  return 0;
}
