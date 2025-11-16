#pragma once
#include <cuda_runtime.h>
#ifndef TILE
#define TILE 32
#endif
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < N; t += TILE) {
    if (row < N && t + threadIdx.x < N)
      As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
    else As[threadIdx.y][threadIdx.x] = 0.0f;
    if (t + threadIdx.y < N && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
    else Bs[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}
