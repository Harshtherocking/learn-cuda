#pragma once

typedef struct GuassianKernel{
  float * kernel;
  int sigma;
  int radius;
}GuassianKernel;
// #define GuassianKernel

void init_kernel(GuassianKernel * g, int sigma, int radius);
