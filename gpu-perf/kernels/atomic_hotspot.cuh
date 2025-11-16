#pragma once
#include <cuda_runtime.h>
__global__ void atomic_hotspot_kernel(unsigned int* __restrict__ counter, int iters){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  // everyone contends on counter[0]
  for(int i=0;i<iters;++i){
    atomicAdd(counter, 1u);
  }
}
