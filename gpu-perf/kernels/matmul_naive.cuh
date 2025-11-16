#pragma once
#include <cuda_runtime.h>

// Naive square GEMM: C = A * B, row-major, size N x N
// One thread computes one C(i,j). Block is typically (16,16).
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    // Very simple triple loop, no shared memory
    for (int k = 0; k < N; ++k) {
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
