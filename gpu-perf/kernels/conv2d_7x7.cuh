#pragma once
#include <cuda_runtime.h>
__global__ void conv2d_7x7_kernel(const float* __restrict__ img, const float* __restrict__ k,
                                  float* __restrict__ out, int H, int W){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=W-6 || y>=H-6) return;
  float s=0.f;
  for(int dy=0;dy<7;++dy)
    for(int dx=0;dx<7;++dx)
      s += img[(y+dy)*W + (x+dx)] * k[dy*7+dx];
  out[y*W + x] = s;
}
