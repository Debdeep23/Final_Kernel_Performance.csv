#pragma once
#include <cuda_runtime.h>
__global__ void scan_block_excl_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int N){
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int g = blockIdx.x * blockDim.x + tid;
  float x = (g < N) ? in[g] : 0.0f;
  s[tid] = x; __syncthreads();
  // upsweep
  for(int offset=1; offset<blockDim.x; offset<<=1){
    int i = (tid+1)*offset*2 - 1;
    if(i<blockDim.x) s[i] += s[i - offset];
    __syncthreads();
  }
  // convert to exclusive
  if(tid==blockDim.x-1) s[tid] = 0.0f;
  __syncthreads();
  // downsweep
  for(int offset=blockDim.x>>1; offset>0; offset>>=1){
    int i = (tid+1)*offset*2 - 1;
    if(i<blockDim.x){
      float t = s[i - offset];
      s[i - offset] = s[i];
      s[i] += t;
    }
    __syncthreads();
  }
  if(g < N) out[g] = s[tid];
}
