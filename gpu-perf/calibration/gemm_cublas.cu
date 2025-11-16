#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
double time_gemm(int N){
  size_t bytes=(size_t)N*N*sizeof(float);
  float *A,*B,*C;
  if (cudaMalloc(&A,bytes)!=cudaSuccess) return -1;
  if (cudaMalloc(&B,bytes)!=cudaSuccess){ cudaFree(A); return -1; }
  if (cudaMalloc(&C,bytes)!=cudaSuccess){ cudaFree(A); cudaFree(B); return -1; }
  cublasHandle_t h; cublasCreate(&h);
  const float alpha=1.0f, beta=0.0f;
  for(int i=0;i<5;++i) cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,A,N,B,N,&beta,C,N);
  cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for(int i=0;i<10;++i) cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,A,N,B,N,&beta,C,N);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=10.0f;
  double flops=2.0*(double)N*N*N;
  double gflops=flops/(ms/1000.0)/1e9;
  printf("N=%d  avg_ms=%.3f  GFLOPS=%.1f\n", N, ms, gflops);
  cudaEventDestroy(s); cudaEventDestroy(e);
  cublasDestroy(h);
  cudaFree(A); cudaFree(B); cudaFree(C);
  return gflops;
}
int main(){
  int sizes[] = {8192, 6144, 4096};
  double best=0.0;
  for (int N : sizes){ double g=time_gemm(N); if(g>best) best=g; }
  printf("SUSTAINED_COMPUTE_GFLOPS=%.1f\n", best);
  return 0;
}
