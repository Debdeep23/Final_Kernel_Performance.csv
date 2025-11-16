#pragma once
#include <cuda_runtime.h>
__global__ void saxpy_kernel(float a,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = a * A[i] + B[i];
}
