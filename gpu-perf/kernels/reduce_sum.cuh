#pragma once
#include <cuda_runtime.h>
__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int N){
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;
  float sum = 0.f;
  if (i < N) sum += in[i];
  if (i + blockDim.x < N) sum += in[i + blockDim.x];
  s[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
    if (tid < offset) s[tid] += s[tid + offset];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}
