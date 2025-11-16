// runner/main.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// === kernel headers (present in your kernels/ folder) ===
#include "../kernels/vector_add.cuh"
#include "../kernels/saxpy.cuh"
#include "../kernels/strided_copy_8.cuh"
#include "../kernels/naive_transpose.cuh"
#include "../kernels/matmul_tiled.cuh"

#include "../kernels/reduce_sum.cuh"
#include "../kernels/dot_product.cuh"
#include "../kernels/histogram.cuh"
#include "../kernels/conv2d_3x3.cuh"
#include "../kernels/conv2d_7x7.cuh"
#include "../kernels/shared_transpose.cuh"
#include "../kernels/random_access.cuh"
#include "../kernels/vector_add_divergent.cuh"
#include "../kernels/shared_bank_conflict.cuh"
#include "../kernels/matmul_naive.cuh"
#include "../kernels/atomic_hotspot.cuh"

// ---------- helpers ----------
#define CUDA_OK(call) do {                                         \
  cudaError_t __err = (call);                                      \
  if (__err != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            "runner/main.cu", __LINE__, cudaGetErrorString(__err));\
    return 1;                                                      \
  }                                                                \
} while(0)

template<typename F>
float time_ms(F fn, int warm, int reps){
  // warmup launches
  for(int i=0;i<warm;++i) fn();
  CUDA_OK(cudaDeviceSynchronize());
  // timed launches
  cudaEvent_t s,e;
  CUDA_OK(cudaEventCreate(&s));
  CUDA_OK(cudaEventCreate(&e));
  CUDA_OK(cudaEventRecord(s));
  for(int i=0;i<reps;++i) fn();
  CUDA_OK(cudaEventRecord(e));
  CUDA_OK(cudaEventSynchronize(e));
  float ms=0.0f;
  CUDA_OK(cudaEventElapsedTime(&ms, s, e));
  CUDA_OK(cudaEventDestroy(s));
  CUDA_OK(cudaEventDestroy(e));
  return ms / (reps>0 ? reps : 1);
}

// Emit one CSV line: kernel,args,device_name,gx,gy,gz,bx,by,bz,time_ms
inline void emit_csv(const char* kernel,
                     const std::string& args,
                     const char* devname,
                     dim3 grid, dim3 blk, float ms)
{
  printf("%s,%s,%s,%u,%u,%u,%u,%u,%u,%.6f\n",
         kernel, args.c_str(), devname,
         grid.x, grid.y, grid.z,
         blk.x,  blk.y,  blk.z, ms);
  fflush(stdout);
}

int main(int argc,char**argv){
  // defaults
  std::string kernel="vector_add";
  int N = 1<<20;            // used by 1D kernels
  int block = 256;
  int warm=20, reps=100;
  int rows=2048, cols=2048; // transpose
  int matN=512;             // matmul sizes
  float alpha=2.0f;         // saxpy
  int H=1024, W=1024;       // conv
  int iters=100;            // atomic_hotspot

  // arg parsing
  for(int i=1;i<argc;++i){
    if(!strcmp(argv[i],"--kernel")&&i+1<argc) kernel=argv[++i];
    else if(!strcmp(argv[i],"--N")&&i+1<argc) N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--block")&&i+1<argc) block=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--warmup")&&i+1<argc) warm=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--rows")&&i+1<argc) rows=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--cols")&&i+1<argc) cols=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--matN")&&i+1<argc) matN=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--alpha")&&i+1<argc) alpha=static_cast<float>(atof(argv[++i]));
    else if(!strcmp(argv[i],"--H")&&i+1<argc) H=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--W")&&i+1<argc) W=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);
  }

  // device name once
  cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  const char* devname = prop.name;

  // reconstruct the arg string we want to preserve in CSV (everything after --kernel <name>)
  std::string argline;
  {
    bool skipNext=false;
    for(int i=1;i<argc;++i){
      if(skipNext){ skipNext=false; continue; }
      if(!strcmp(argv[i],"--kernel") && i+1<argc){
        // include the value but omit the --kernel token itself
        // i.e., we keep only the remaining args that affect launch/config
        ++i; // skip kernel name
        continue;
      }
      if(!argline.empty()) argline.push_back(' ');
      argline += argv[i];
    }
  }

  // ===== kernel switch =====
  if(kernel=="vector_add"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*B,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    dim3 blk((unsigned)block,1,1), grid((unsigned)((N+block-1)/block),1,1);
    auto launch=[&](){ vector_add_kernel<<<grid,blk>>>(A,B,C,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("vector_add", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if(kernel=="saxpy"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*B,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    dim3 blk((unsigned)block,1,1), grid((unsigned)((N+block-1)/block),1,1);
    auto launch=[&](){ saxpy_kernel<<<grid,blk>>>(alpha,A,B,C,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("saxpy", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if(kernel=="strided_copy_8"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    dim3 blk((unsigned)block,1,1), grid((unsigned)(((N+7)/8 + block-1)/block),1,1);
    auto launch=[&](){ strided_copy_8_kernel<<<grid,blk>>>(A,C,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("strided_copy_8", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(C); return 0;
  }

  if(kernel=="naive_transpose"){
    size_t bytes = (size_t)rows*cols*sizeof(float);
    float *A,*B; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes));
    dim3 blk(16,16,1), grid((unsigned)((cols+15)/16),(unsigned)((rows+15)/16),1);
    auto launch=[&](){ naive_transpose_kernel<<<grid,blk>>>(A,B,rows,cols); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("naive_transpose", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); return 0;
  }

  if(kernel=="shared_transpose"){
    size_t bytes = (size_t)rows*cols*sizeof(float);
    float *A,*B; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes));
    dim3 blk(32,32,1), grid((unsigned)((cols+31)/32),(unsigned)((rows+31)/32),1);
    auto launch=[&](){ shared_transpose_kernel<<<grid,blk>>>(A,B,rows,cols); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("shared_transpose", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); return 0;
  }

  if(kernel=="matmul_tiled"){
    int N_=matN;
    size_t bytes=(size_t)N_*N_*sizeof(float);
    float *A,*B,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    dim3 blk((unsigned)TILE,(unsigned)TILE,1),
         grid((unsigned)((N_+TILE-1)/TILE),(unsigned)((N_+TILE-1)/TILE),1);
    auto launch=[&](){ matmul_tiled_kernel<<<grid,blk>>>(A,B,C,N_); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("matmul_tiled", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if(kernel=="matmul_naive"){
    int N_=matN;
    size_t bytes=(size_t)N_*N_*sizeof(float);
    float *A,*B,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    dim3 blk(16,16,1),
         grid((unsigned)((N_+15)/16),(unsigned)((N_+15)/16),1);
    auto launch=[&](){ matmul_naive_kernel<<<grid,blk>>>(A,B,C,N_); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("matmul_naive", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if(kernel=="reduce_sum"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*partials; CUDA_OK(cudaMalloc(&A,bytes));
    int blkSz=block, gridSz=(N+blkSz*2-1)/(blkSz*2);
    CUDA_OK(cudaMalloc(&partials, (size_t)gridSz*sizeof(float)));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(partials,0,(size_t)gridSz*sizeof(float)));
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ reduce_sum_kernel<<<grid,blk, (size_t)blkSz*sizeof(float)>>>(A,partials,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("reduce_sum", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(partials); return 0;
  }

  if(kernel=="dot_product"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*B,*partials; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes));
    int blkSz=block, gridSz=(N+blkSz*2-1)/(blkSz*2);
    CUDA_OK(cudaMalloc(&partials, (size_t)gridSz*sizeof(float)));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(partials,0,(size_t)gridSz*sizeof(float)));
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ dot_product_kernel<<<grid,blk, (size_t)blkSz*sizeof(float)>>>(A,B,partials,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("dot_product", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(partials); return 0;
  }

  if(kernel=="histogram"){
    size_t bytes=(size_t)N*sizeof(unsigned int);
    unsigned int *data,*bins; CUDA_OK(cudaMalloc(&data,bytes)); CUDA_OK(cudaMalloc(&bins,256*sizeof(unsigned int)));
    CUDA_OK(cudaMemset(data,0,bytes)); CUDA_OK(cudaMemset(bins,0,256*sizeof(unsigned int)));
    int blkSz=block, gridSz=(N+blkSz-1)/blkSz;
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ histogram_kernel<<<grid,blk, 256*sizeof(unsigned int)>>>(data,N,bins); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("histogram", argline, devname, grid, blk, ms);
    cudaFree(data); cudaFree(bins); return 0;
  }

  if(kernel=="conv2d_3x3"){
    size_t bytes=(size_t)H*W*sizeof(float);
    float *img,*k,*out; CUDA_OK(cudaMalloc(&img,bytes)); CUDA_OK(cudaMalloc(&out,bytes)); CUDA_OK(cudaMalloc(&k,9*sizeof(float)));
    CUDA_OK(cudaMemset(img,0,bytes)); CUDA_OK(cudaMemset(out,0,bytes)); CUDA_OK(cudaMemset(k,0,9*sizeof(float)));
    dim3 blk(16,16,1), grid((unsigned)((W+15)/16),(unsigned)((H+15)/16),1);
    auto launch=[&](){ conv2d_3x3_kernel<<<grid,blk>>>(img,k,out,H,W); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("conv2d_3x3", argline, devname, grid, blk, ms);
    cudaFree(img); cudaFree(out); cudaFree(k); return 0;
  }

  if(kernel=="conv2d_7x7"){
    size_t bytes=(size_t)H*W*sizeof(float);
    float *img,*k,*out; CUDA_OK(cudaMalloc(&img,bytes)); CUDA_OK(cudaMalloc(&out,bytes)); CUDA_OK(cudaMalloc(&k,49*sizeof(float)));
    CUDA_OK(cudaMemset(img,0,bytes)); CUDA_OK(cudaMemset(out,0,bytes)); CUDA_OK(cudaMemset(k,0,49*sizeof(float)));
    dim3 blk(16,16,1), grid((unsigned)((W+15)/16),(unsigned)((H+15)/16),1);
    auto launch=[&](){ conv2d_7x7_kernel<<<grid,blk>>>(img,k,out,H,W); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("conv2d_7x7", argline, devname, grid, blk, ms);
    cudaFree(img); cudaFree(out); cudaFree(k); return 0;
  }

  if(kernel=="random_access"){
    size_t bytes=(size_t)N*sizeof(float), ibytes=(size_t)N*sizeof(int);
    float *A,*B; int *idx; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&idx,ibytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(idx,0,ibytes));
    int blkSz=block, gridSz=(N+blkSz-1)/blkSz;
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ random_access_kernel<<<grid,blk>>>(A,idx,B,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("random_access", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(idx); return 0;
  }

  if(kernel=="vector_add_divergent"){
    size_t bytes=(size_t)N*sizeof(float);
    float *A,*B,*C; CUDA_OK(cudaMalloc(&A,bytes)); CUDA_OK(cudaMalloc(&B,bytes)); CUDA_OK(cudaMalloc(&C,bytes));
    CUDA_OK(cudaMemset(A,0,bytes)); CUDA_OK(cudaMemset(B,0,bytes)); CUDA_OK(cudaMemset(C,0,bytes));
    int blkSz=block, gridSz=(N+blkSz-1)/blkSz;
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ vector_add_divergent_kernel<<<grid,blk>>>(A,B,C,N); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("vector_add_divergent", argline, devname, grid, blk, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if(kernel=="shared_bank_conflict"){
    float *out; CUDA_OK(cudaMalloc(&out, 1024*sizeof(float)));
    dim3 blk(1024,1,1), grid(1,1,1);
    auto launch=[&](){ shared_bank_conflict_kernel<<<grid,blk>>>(out); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("shared_bank_conflict", argline, devname, grid, blk, ms);
    cudaFree(out); return 0;
  }

  if(kernel=="atomic_hotspot"){
    unsigned int *ctr; CUDA_OK(cudaMalloc(&ctr, sizeof(unsigned int)));
    CUDA_OK(cudaMemset(ctr, 0, sizeof(unsigned int)));
    int blkSz=block, gridSz=(N+blkSz-1)/blkSz;
    dim3 blk((unsigned)blkSz,1,1), grid((unsigned)gridSz,1,1);
    auto launch=[&](){ atomic_hotspot_kernel<<<grid,blk>>>(ctr, iters); };
    float ms=time_ms(launch,warm,reps);
    emit_csv("atomic_hotspot", argline, devname, grid, blk, ms);
    cudaFree(ctr); return 0;
  }

  fprintf(stderr,"Unknown --kernel\n");
  return 1;
}

