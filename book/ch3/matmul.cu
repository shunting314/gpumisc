/*
 * Need add -lcublas to nvcc command.
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>

#ifndef USE_CUBLAS
#define USE_CUBLAS 0
#endif

#if USE_CUBLAS
#include <cublas_v2.h>
#endif

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

void matmul_cpu(float* A, float *B, float *C, int N) {
  auto start = high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
  auto end = high_resolution_clock::now();
  auto elapse_us = duration_cast<microseconds>(end - start);
  printf("CPU kernel takes %d usec\n", elapse_us.count());
}

#define THREAD_PER_ELEM 0
#define THREAD_PER_ROW 1
#define THREAD_PER_COL 2
#define CHOICE_TILING 3
// #define CHOICE CHOICE_TILING
#ifndef CHOICE
#define CHOICE THREAD_PER_ELEM
#endif

__global__ void matmul_cuda_kernel(float* A, float* B, float* C, int N) {
  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

  if (colIdx < N && rowIdx < N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
      acc += A[rowIdx * N + k] * B[k * N + colIdx];
    }
    C[rowIdx * N + colIdx] = acc;
  }
}

__global__ void matmul_cuda_kernel_tiling(float* A, float *B, float* C, int N) {
  #define tile_size 16
  __shared__ float Atile[tile_size][tile_size];
  __shared__ float Btile[tile_size][tile_size];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int off = 0; off < N; off += tile_size) {
    // setup the tile
    // Atile[threadIdx.y][threadIdx.x] correspond to A[threadIdx.y + blockIdx.y * tile_size][threadIdx.x + off]
    // Btile[threadIdx.y][threadIdx.x] correspond to B[threadIdx.y + off][threadIdx.x + blockIdx.x * tile_size]
    #if 0
    Atile[threadIdx.y][threadIdx.x] = A[(threadIdx.y + blockIdx.y * tile_size) * N + threadIdx.x + off];
    Btile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + off) * N + threadIdx.x + blockIdx.x * tile_size];
    #else
    Atile[threadIdx.y][threadIdx.x] = A[row * N + off + threadIdx.x];
    Btile[threadIdx.y][threadIdx.x] = B[(off + threadIdx.y) * N + col];
    #endif
    
    // do the computation on the tile
    __syncthreads();
    for (int k = 0; k < tile_size; ++k) {
      acc += Atile[threadIdx.y][k] * Btile[k][threadIdx.x];
    }
    __syncthreads();
  }
  // write acc
  C[row * N + col] = acc;
}

__global__ void matmul_cuda_kernel_per_row(float* A, float* B, float* C, int N) {
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowIdx < N) {
    for (int colIdx = 0; colIdx < N; ++colIdx) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += A[rowIdx * N + k] * B[k * N + colIdx];
      }
      C[rowIdx * N + colIdx] = acc;
    }
  }
}

__global__ void matmul_cuda_kernel_per_col(float* A, float* B, float* C, int N) {
  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (colIdx < N) {
    for (int rowIdx = 0; rowIdx < N; ++rowIdx) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += A[rowIdx * N + k] * B[k * N + colIdx];
      }
      C[rowIdx * N + colIdx] = acc;
    }
  }
}

void matmul_cuda(float *h_A, float* h_B, float *h_C, int N) {
  float* d_A, *d_B, *d_C;
  int nbytes = N * N * sizeof(float);
  cudaMalloc(&d_A, nbytes);
  cudaMalloc(&d_B, nbytes);
  cudaMalloc(&d_C, nbytes);
  cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = high_resolution_clock::now();
  int niter = 100;

  for (int i = 0; i < niter; ++i) {
    #if CHOICE == THREAD_PER_ELEM
    int blksize = 16;
    int nblk = (N + blksize - 1) / blksize;
    matmul_cuda_kernel<<<dim3(nblk, nblk), dim3(blksize, blksize)>>>(d_A, d_B, d_C, N);
    #endif
  
    #if CHOICE == CHOICE_TILING
    int blksize = 16;
    int nblk = (N + blksize - 1) / blksize;
    matmul_cuda_kernel_tiling<<<dim3(nblk, nblk), dim3(blksize, blksize)>>>(d_A, d_B, d_C, N);
    #endif
  
    #if CHOICE == THREAD_PER_ROW
    int blksize = 64;
    int nblk = (N + blksize - 1) / blksize;
    matmul_cuda_kernel_per_row<<<nblk, blksize>>>(d_A, d_B, d_C, N);
    #endif
  
    #if CHOICE == THREAD_PER_COL
    int blksize = 64;
    int nblk = (N + blksize - 1) / blksize;
    matmul_cuda_kernel_per_col<<<nblk, blksize>>>(d_A, d_B, d_C, N);
    #endif
  }

  cudaDeviceSynchronize();
  auto end = high_resolution_clock::now();
  auto elapse_us = duration_cast<microseconds>(end - start);
  printf("The CUDA kernel took %d usec\n", elapse_us / niter); 

  cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void transpose(float* mat, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      float tmp = mat[i * N + j];
      mat[i * N + j] = mat[j * N + i];
      mat[j * N + i] = tmp;
    }
  }
}

#if USE_CUBLAS
void matmul_cublas(float* h_A, float* h_B, float* h_C, int N) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float* d_A, *d_B, *d_C;
  float alpha = 1.0f, beta = 0.0f;
  int nbytes = N * N * sizeof(float);
  cudaMalloc(&d_A, nbytes);
  cudaMalloc(&d_B, nbytes);
  cudaMalloc(&d_C, nbytes);
  cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

  // By default, cublas matrix is column major. Need handle the transpose of input/output when needed.
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

  cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);
  transpose(h_C, N);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);
}
#endif

int calc_nfail(float* ref, float* act, int S) {
  int nfail = 0;
  float rtol = 1e-5;
  float atol = 1e-8;
  for (int i = 0; i < S; ++i) {
    bool pass = (fabs(ref[i] - act[i]) <= atol + rtol * fabs(act[i]));
    if (!pass) {
      ++nfail;
    }
  }
  return nfail;
}

void printMat(const char* prompt, float* mat, int N) {
  printf("%s\n", prompt);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf(" %6f", mat[i * N + j]);
    }
    printf("\n");
  }
}

int main(void) {
  const int SIZE = 512;
  // const int SIZE = 4;
  float* h_A = (float*) malloc(SIZE * SIZE * sizeof(float));
  float* h_B = (float*) malloc(SIZE * SIZE * sizeof(float));
  float* h_C_cpuref = (float*) malloc(SIZE * SIZE * sizeof(float));
  float* h_C_cublas = (float*) malloc(SIZE * SIZE * sizeof(float));
  float* h_C_cuda = (float*) malloc(SIZE * SIZE * sizeof(float));

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      h_A[i * SIZE + j] = float(rand()) / RAND_MAX;
      h_B[i * SIZE + j] = float(rand()) / RAND_MAX;
    }
  }

  matmul_cpu(h_A, h_B, h_C_cpuref, SIZE);
  matmul_cuda(h_A, h_B, h_C_cuda, SIZE);
  #if USE_CUBLAS
  matmul_cublas(h_A, h_B, h_C_cublas, SIZE);
  #endif

  if (SIZE < 10) {
    printMat("A:", h_A, SIZE);
    printMat("B:", h_B, SIZE);
    printMat("C_ref:", h_C_cpuref, SIZE);
    printMat("C_cuda:", h_C_cuda, SIZE);
    printMat("C_cublas:", h_C_cublas, SIZE);
  }

  int nfail = calc_nfail(h_C_cpuref, h_C_cuda, SIZE * SIZE);
  if (nfail > 0) {
    fprintf(stderr, "CUDA implementation does not match with cpu. %d mismatch\n", nfail);
    return -1;
  }

  #if USE_CUBLAS
  nfail = calc_nfail(h_C_cpuref, h_C_cublas, SIZE * SIZE);
  if (nfail > 0) {
    fprintf(stderr, "cublass implementation does not match with cpu. %d mismatch\n", nfail);
    return -1;
  }
  #endif

  free(h_A);
  free(h_B);
  free(h_C_cpuref);
  free(h_C_cublas);
  free(h_C_cuda);
  printf("bye\n");
  return 0;
}
