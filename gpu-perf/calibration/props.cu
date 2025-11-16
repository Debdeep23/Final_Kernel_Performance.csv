#include <cuda_runtime.h>
#include <cstdio>
int main(){
  int dev=0; cudaSetDevice(dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  printf("name=%s\n", p.name);
  printf("major=%d minor=%d\n", p.major, p.minor);
  printf("multiProcessorCount=%d\n", p.multiProcessorCount);
  printf("maxThreadsPerMultiProcessor=%d\n", p.maxThreadsPerMultiProcessor);
  printf("maxThreadsPerBlock=%d\n", p.maxThreadsPerBlock);
  printf("regsPerMultiprocessor=%d\n", p.regsPerMultiprocessor);
  printf("sharedMemPerMultiprocessor=%zu\n", (size_t)p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlockOptin=%zu\n", (size_t)p.sharedMemPerBlockOptin);
  printf("maxBlocksPerMultiProcessor=%d\n", p.maxBlocksPerMultiProcessor);
  printf("warpSize=%d\n", p.warpSize);
  return 0;
}
