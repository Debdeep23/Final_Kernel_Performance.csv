#pragma once
#include <cuda_runtime.h>
__global__ void vector_add_divergent_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                            float* __restrict__ C, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=N) return;
  if ((i & 1)==0){
    float s = 0.f;
    #pragma unroll 16
    for(int t=0;t<128;++t) s += 0.0001f*t;
    C[i] = A[i] + B[i] + s;
  } else {
    C[i] = A[i] + B[i];
  }
}
