#pragma once
#include <cuda_runtime.h>
#ifndef TSTRIDE
#define TSTRIDE 32
#endif
__global__ void shared_transpose_kernel(const float* __restrict__ A, float* __restrict__ B, int H, int W){
  __shared__ float tile[TSTRIDE][TSTRIDE+1];
  int x = blockIdx.x*TSTRIDE + threadIdx.x;
  int y = blockIdx.y*TSTRIDE + threadIdx.y;
  if (x<W && y<H) tile[threadIdx.y][threadIdx.x] = A[y*W + x];
  __syncthreads();
  int xo = blockIdx.y*TSTRIDE + threadIdx.x;
  int yo = blockIdx.x*TSTRIDE + threadIdx.y;
  if (xo<H && yo<W) B[yo*H + xo] = tile[threadIdx.x][threadIdx.y];
}
