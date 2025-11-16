#pragma once
#include <cuda_runtime.h>
__global__ void strided_copy_8_kernel(const float* __restrict__ A,
                                      float* __restrict__ C,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 8;
  if (idx < N) C[idx] = A[idx];
}
