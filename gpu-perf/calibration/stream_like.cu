#include <cuda_runtime.h>
#include <cstdio>
__global__ void triad(float* __restrict__ A, const float* __restrict__ B,
                      const float* __restrict__ C, float alpha, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) A[i] = B[i] + alpha * C[i];
}
float run_once(int N){
  const int block=256, grid=(N+block-1)/block;
  size_t bytes=(size_t)N*sizeof(float);
  float *A,*B,*C; cudaMalloc(&A,bytes); cudaMalloc(&B,bytes); cudaMalloc(&C,bytes);
  cudaMemset(A,0,bytes); cudaMemset(B,0,bytes); cudaMemset(C,0,bytes);
  for(int i=0;i<20;++i) triad<<<grid,block>>>(A,B,C,2.0f,N);
  cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for(int i=0;i<50;++i) triad<<<grid,block>>>(A,B,C,2.0f,N);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=50.0f;
  cudaEventDestroy(s); cudaEventDestroy(e);
  cudaFree(A); cudaFree(B); cudaFree(C);
  double gbps = (double)N * 12.0 / (ms/1000.0) / 1e9;
  printf("N=%d  avg_ms=%.3f  GBps=%.2f\n", N, ms, gbps);
  return (float)gbps;
}
int main(){
  int sizes[] = {1<<27, 1<<26, 1<<25};
  float best=0.0f;
  for (int N : sizes){ best = fmaxf(best, run_once(N)); }
  printf("SUSTAINED_MEM_BW_GBPS=%.2f\n", best);
  return 0;
}
