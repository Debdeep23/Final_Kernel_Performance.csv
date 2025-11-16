#pragma once
#include <cuda_runtime.h>
__inline__ __device__ float warp_sum(float v){
  for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}
__global__ void reduce_warp_kernel(const float* __restrict__ in,
                                   float* __restrict__ partial,
                                   int N){
  float acc = 0.f;
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<N; i+=gridDim.x*blockDim.x)
    acc += in[i];
  acc = warp_sum(acc);
  if((threadIdx.x & 31)==0) partial[blockIdx.x* (blockDim.x/32) + (threadIdx.x>>5)] = acc;
}
