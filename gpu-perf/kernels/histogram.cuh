#pragma once
#include <cuda_runtime.h>
__global__ void histogram_kernel(const unsigned int* __restrict__ data, int N,
                                 unsigned int* __restrict__ bins){
  extern __shared__ unsigned int sbins[];
  int tid=threadIdx.x;
  for(int b=tid;b<256;b+=blockDim.x) sbins[b]=0;
  __syncthreads();
  int i=blockIdx.x*blockDim.x+tid;
  for(; i<N; i+=blockDim.x*gridDim.x){
    unsigned int v = data[i] & 255u;
    atomicAdd(&sbins[v], 1);
  }
  __syncthreads();
  for(int b=tid;b<256;b+=blockDim.x) atomicAdd(&bins[b], sbins[b]);
}
