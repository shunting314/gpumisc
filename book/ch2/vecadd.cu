#include <stdio.h>
#include <assert.h>

void vecadd_cpu(float* A, float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

__global__ void vecadd_gpu(float* A, float* B, float* C, int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N) {
    C[id] = A[id] + B[id];
  }
}

int main(void) {
  const int SIZE = 1024;
  const int SIZE_IN_BYTES = SIZE * sizeof(float);

  float *h_A, *h_B, *h_C_ref, *h_C_fromgpu;
  h_A = (float*) malloc(SIZE_IN_BYTES);
  h_B = (float*) malloc(SIZE_IN_BYTES);
  h_C_ref = (float*) malloc(SIZE_IN_BYTES);
  h_C_fromgpu = (float*) malloc(SIZE_IN_BYTES);

  for (int i = 0; i < SIZE; ++i) {
    h_A[i] = rand() / 100.0;
    h_B[i] = rand() / 100.0;
  }

  vecadd_cpu(h_A, h_B, h_C_ref, SIZE);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, SIZE_IN_BYTES);
  cudaMalloc(&d_B, SIZE_IN_BYTES);
  cudaMalloc(&d_C, SIZE_IN_BYTES);
  cudaMemcpy(d_A, h_A, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

  int nthread = 256;
  int nblock = (SIZE + nthread - 1) / nthread;
  vecadd_gpu<<<nblock, nthread>>>(d_A, d_B, d_C, SIZE);
  cudaMemcpy(h_C_fromgpu, d_C, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

  int nfail = 0;
  for (int i = 0; i < SIZE; ++i) {
    nfail += (h_C_ref[i] != h_C_fromgpu[i]);
  }
  assert(nfail == 0);
  printf("#fail %d\n", nfail);

  free(h_A);
  free(h_B);
  free(h_C_ref);
  free(h_C_fromgpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
