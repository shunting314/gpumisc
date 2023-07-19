#include <stdio.h>
#include <stdlib.h>

void matvec_cpu(float* mat, float* vec, float* out, int N) {
  for (int i = 0; i < N; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
      acc += mat[i * N + j] * vec[j];
    }
    out[i] = acc;
  }
}

__global__ void matvec_kernel(float* mat, float* vec, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
      acc += mat[idx * N + j] * vec[j];
    }
    out[idx] = acc;
  }
}

void matvec_gpu(float *h_mat, float *h_vec, float *h_out, int N) {
  float *d_mat, *d_vec, *d_out;
  cudaMalloc(&d_mat, N * N * sizeof(float));
  cudaMalloc(&d_vec, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));

  cudaMemcpy(d_mat, h_mat, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice);

  int blksize = 32;
  int nblk = (N + blksize - 1) / blksize;
  matvec_kernel<<<nblk, blksize>>>(d_mat, d_vec, d_out, N);
  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_mat);
  cudaFree(d_vec);
  cudaFree(d_out);
}

int main(void) {
  const int SIZE = 256;
  float *h_mat = (float*) malloc(SIZE * SIZE * sizeof(float));
  float *h_vec = (float*) malloc(SIZE * sizeof(float));
  float *h_out_ref = (float*) malloc(SIZE * sizeof(float));
  float *h_out_fromgpu = (float*) malloc(SIZE * sizeof(float));

  // init
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      h_mat[i * SIZE + j] = float(rand()) / RAND_MAX;
    }
    h_vec[i] = float(rand()) / RAND_MAX;
  }

  matvec_cpu(h_mat, h_vec, h_out_ref, SIZE);
  matvec_gpu(h_mat, h_vec, h_out_fromgpu, SIZE);

  int nfail = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (fabs(h_out_ref[i] - h_out_fromgpu[i]) > 1e-8 + 1e-5 * fabs(h_out_fromgpu[i])) {
      ++nfail;
    }
  }
  printf("#fail %d\n", nfail);

  free(h_mat);
  free(h_vec);
  free(h_out_ref);
  free(h_out_fromgpu);
  return 0;
}
