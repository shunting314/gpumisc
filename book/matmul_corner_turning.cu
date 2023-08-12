/*
 * A matrix and C matrix is in row-major while B matrix is in column major.
 * Need to use corner turning technique when loading a tile from B.
 *
 * Perf summary for matrix size 512x512
 * - cpu: 0.48s
 * - cuda naive: 657us
 * - cuda no corner turning: 135us
 * - cuda with corner turning: 118us (1.14x speedup compared with no corner turning)
 */

#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

void matmul_cpu(float* A, float* B, float *C, int SIZE) {
  auto start_ts = high_resolution_clock::now();
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < SIZE; ++k) {
        acc += A[i * SIZE + k] * B[k + j * SIZE];
      }
      C[i * SIZE + j] = acc;
    }
  }
  auto end_ts = high_resolution_clock::now();
  auto elapse_us = duration_cast<microseconds>(end_ts - start_ts);
  printf("CPU kernel takes %d usec\n", elapse_us.count());
}

#define ENABLE_CORNER_TURNING 1

// TODO: assumes SIZE is multiple of TILE_SIZE
__global__ void matmul_cuda_kernel(float* A, float *B, float *C, int SIZE) {
  extern __shared__ char smbuf[];
  int TILE_SIZE = blockDim.x;
  float* Atile = (float*) smbuf;
  float* Btile = (float*) (Atile + TILE_SIZE * TILE_SIZE);

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  float acc = 0.0f;
  for (int off = 0; off < SIZE; off += TILE_SIZE) {
    // load Atile
    Atile[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * SIZE + off + threadIdx.x];
    #if ENABLE_CORNER_TURNING
    // load Btile with corner turning
    Btile[threadIdx.x * TILE_SIZE + threadIdx.y] = B[off + threadIdx.x + (blockIdx.x * TILE_SIZE + threadIdx.y) * SIZE];
    #else

    // TODO this is a bug that I spend quite a while to discover
    #if 0
    // B[off + blockIdx.y][col]
    Btile[threadIdx.y * TILE_SIZE + threadIdx.x] = B[off + blockIdx.y + col * SIZE];
    #endif

    // B[off + threadIdx.y][col]
    Btile[threadIdx.y * TILE_SIZE + threadIdx.x] = B[off + threadIdx.y + col * SIZE];
    #endif

    __syncthreads();
    // compute
    for (int k = 0; k < TILE_SIZE; ++k) {
      acc += Atile[threadIdx.y * TILE_SIZE + k] * Btile[k * TILE_SIZE + threadIdx.x];
    }

    __syncthreads();
  }
  C[row * SIZE + col] = acc;
}

// 657 us for SIZE 512
__global__ void matmul_cuda_naive_kernel(float* A, float *B, float *C, int SIZE) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  float acc = 0.0f;
  for (int i = 0; i < SIZE; ++i) {
    acc += A[row * SIZE + i] * B[i + col * SIZE];
  }
  C[row * SIZE + col] = acc;
}

#define USE_NAIVE_KERNEL 0

void matmul_cuda(float* h_A, float* h_B, float *h_C, int SIZE) {
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, SIZE * SIZE * sizeof(*d_A));
  cudaMalloc(&d_B, SIZE * SIZE * sizeof(*d_B));
  cudaMalloc(&d_C, SIZE * SIZE * sizeof(*d_C));
  cudaMemcpy(d_A, h_A, SIZE * SIZE * sizeof(*d_A), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, SIZE * SIZE * sizeof(*d_B), cudaMemcpyHostToDevice);

  // launch kernel
  cudaDeviceSynchronize();
  auto start_ts = high_resolution_clock::now();
  int niter = 100;
  for (int i = 0; i < niter; ++i) {
    int tile_size = 16;
    int ntile = (SIZE + tile_size - 1) / tile_size;
    #if USE_NAIVE_KERNEL
    matmul_cuda_naive_kernel<<<dim3(ntile, ntile), dim3(tile_size, tile_size)>>>(d_A, d_B, d_C, SIZE);
    #else
    matmul_cuda_kernel<<<dim3(ntile, ntile), dim3(tile_size, tile_size), tile_size * tile_size * 2 * sizeof(float)>>>(d_A, d_B, d_C, SIZE);
    #endif
  }
  cudaDeviceSynchronize();
  auto end_ts = high_resolution_clock::now();
  auto elapse_us = duration_cast<microseconds>(end_ts - start_ts);
  printf("The CUDA kernel took %d usec\n", elapse_us / niter);
  
  cudaMemcpy(h_C, d_C, SIZE * SIZE * sizeof(*d_C), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  // int SIZE = 1024; // CPU took around 3.9 seconds
  int SIZE = 512; // CPU took 0.48 seconds
  int SIZE_IN_BYTES = SIZE * SIZE * 4;

  float *h_A, *h_B, *h_C_cpuref, *h_C_cuda;
  h_A = (float*) malloc(SIZE_IN_BYTES);
  h_B = (float*) malloc(SIZE_IN_BYTES);
  h_C_cpuref = (float*) malloc(SIZE_IN_BYTES);
  h_C_cuda = (float*) malloc(SIZE_IN_BYTES);

  for (int i = 0; i < SIZE * SIZE; ++i) {
    h_A[i] = (float) rand() / RAND_MAX;
    h_B[i] = (float) rand() / RAND_MAX;
  }

  // A in row major, B in column major, C in row major
  matmul_cpu(h_A, h_B, h_C_cpuref, SIZE);
  matmul_cuda(h_A, h_B, h_C_cuda, SIZE);

  // compare
  int nfail = 0;
  float rtol = 1e-5;
  float atol = 1e-8;
  for (int i = 0; i < SIZE * SIZE; ++i) {
    bool pass = (fabs(h_C_cpuref[i] - h_C_cuda[i]) <= atol + rtol * fabs(h_C_cuda[i])); 
    if (!pass) {
      ++nfail;
    }
  }
  printf("Found %d/%d mismatch\n", nfail, SIZE * SIZE);

  free(h_A);
  free(h_B);
  free(h_C_cpuref);
  free(h_C_cuda);
  return 0;
}
