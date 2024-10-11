#include <stdio.h>
#include "./utils/mat.cuh"
#include "./utils/env.cuh"
#include <cuda_runtime.h>


#define ImagePath "./images/colosseum.ppm"
#define SIDE 720

int main() {
  int * buffer = (int *) malloc (sizeof(int) * SIDE * SIDE); 

  FILE * file_ptr;

  // read image 
  file_ptr = fopen(ImagePath, "r");
  fprintf(file_ptr, "P6 720 720 255");

  for (int i=0; i<SIDE; i++){
    for (int j=0; j<SIDE; j++){
      fputc(i, file_ptr);
      fputc(j, file_ptr);
    }
  }

  // fgets(buffer, WIDTH*WIDTH, file_ptr);

  fclose(file_ptr);


  return 0;
}
