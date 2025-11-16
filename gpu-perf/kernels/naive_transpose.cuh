#pragma once
#include <cuda_runtime.h>
__global__ void naive_transpose_kernel(const float* __restrict__ A,
                                       float* __restrict__ B,
                                       int rows, int cols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows && c < cols) {
    B[c * rows + r] = A[r * cols + c];
  }
}
