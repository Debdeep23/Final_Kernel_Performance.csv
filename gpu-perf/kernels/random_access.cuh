#pragma once
#include <cuda_runtime.h>
__global__ void random_access_kernel(const float* __restrict__ A, const int* __restrict__ idx,
                                     float* __restrict__ B, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) B[i] = A[idx[i]];
}
