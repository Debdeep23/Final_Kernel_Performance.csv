#pragma once
#include <cuda_runtime.h>
__global__ void conv2d_3x3_kernel(const float* __restrict__ img, const float* __restrict__ k,
                                  float* __restrict__ out, int H, int W){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=W-2 || y>=H-2) return;
  float s=0.f;
  #pragma unroll
  for(int dy=0;dy<3;++dy)
    #pragma unroll
    for(int dx=0;dx<3;++dx)
      s += img[(y+dy)*W + (x+dx)] * k[dy*3+dx];
  out[y*W + x] = s;
}
