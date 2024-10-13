#pragma once
#include <stdint.h>

__global__ void GuassianFunc (float * kernel, int sigma, int radius);

__global__ void GuassianBlur (uint8_t * image, uint8_t * new_image, float * kernel, int width, int height, int kernel_width);

void Padding(uint8_t * padd_img, uint8_t * img, int width, int height, int padd_rad)