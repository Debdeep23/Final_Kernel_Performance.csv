#pragma once
#include <cuda_runtime.h>
__global__ void strided_copy_4_kernel(const float* __restrict__ A,
                                      float* __restrict__ C, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int src = i*4;
  if(src < N) C[i] = A[src];
}
