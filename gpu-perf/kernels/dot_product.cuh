#pragma once
#include <cuda_runtime.h>
__global__ void dot_product_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ partial, int N){
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;
  float acc = 0.f;
  if (i < N) acc += A[i]*B[i];
  if (i + blockDim.x < N) acc += A[i+blockDim.x]*B[i+blockDim.x];
  s[tid] = acc;
  __syncthreads();
  for(int o=blockDim.x>>1;o>0;o>>=1){
    if(tid<o) s[tid]+=s[tid+o];
    __syncthreads();
  }
  if(tid==0) partial[blockIdx.x]=s[0];
}
