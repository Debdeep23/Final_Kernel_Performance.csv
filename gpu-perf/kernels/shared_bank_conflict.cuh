#pragma once
#include <cuda_runtime.h>
__global__ void shared_bank_conflict_kernel(float* __restrict__ out){
  __shared__ float s[1024];
  int tid = threadIdx.x;
  s[tid] = tid;
  __syncthreads();
  float acc=0.f;
  for(int it=0; it<1024; ++it){
    acc += s[(it*33) & 1023]; // stride 33 to trigger conflicts
  }
  out[tid] = acc;
}
