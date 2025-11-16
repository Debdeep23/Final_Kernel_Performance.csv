#pragma once
#include <cuda_runtime.h>
__global__ void stencil2d_5pt_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int H, int W){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x==0 || y==0 || x>=W-1 || y>=H-1) return;
  float c = in[y*W + x];
  float n = in[(y-1)*W + x];
  float s = in[(y+1)*W + x];
  float w = in[y*W + x-1];
  float e = in[y*W + x+1];
  out[y*W + x] = 0.25f*(n+s+w+e) - c*0.0f; // simple average
}
