#pragma once
#include <cuda_runtime.h>
__global__ void conv2d_5x5_kernel(const float* __restrict__ img, const float* __restrict__ k,
                                  float* __restrict__ out, int H, int W){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=W-4 || y>=H-4) return;
  float s=0.f;
  #pragma unroll
  for(int dy=0;dy<5;++dy)
    #pragma unroll
    for(int dx=0;dx<5;++dx)
      s += img[(y+dy)*W + (x+dx)] * k[dy*5+dx];
  out[y*W + x] = s;
}
