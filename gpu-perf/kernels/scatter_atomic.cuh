#pragma once
#include <cuda_runtime.h>
__global__ void scatter_atomic_kernel(const float* __restrict__ A,
                                      const int* __restrict__ idx,
                                      float* __restrict__ B, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    int j = idx[i];
    atomicExch((int*)&B[j], __float_as_int(A[i]));
  }
}
