#include <time.h>
#include <chrono>
#include <iostream>

// #define N (1 << 30)
#define N (1 << 26)

using namespace std::chrono;

__global__ void init(float* inp_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    inp_ptr[idx] = 1.0;
  }
}

__global__ void comp(float* inp_ptr, float* out_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out_ptr[idx] = inp_ptr[idx] * 2.0;
  }
}

__global__ void warmup() {
}

int main(void) {
  float* inp_ptr, *out_ptr;
  int64_t array_size = N * sizeof(*inp_ptr);
  cudaMalloc(&inp_ptr, N * sizeof(*inp_ptr));
  cudaMalloc(&out_ptr, N * sizeof(*out_ptr));


  /*
   * A block can have up to 1024 threads. But why??
   * Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
   */
  // int nthread = 2048; // Result in cudaErrorInvalidConfiguration
  int nthread = 1024;
  int nblock = (N + nthread - 1) / nthread;

  // with this dummy warmup kernel, the first init call will be as faster as the other two!
  warmup<<<1, 1>>>();

  // without calling warmup first, we would see the first call is slower then the others.
  for (int i = 0; i < 3; ++i) {
    cudaDeviceSynchronize();
    auto start_init = high_resolution_clock::now();
    init<<<nblock, nthread>>>(inp_ptr);
    cudaDeviceSynchronize();
    auto stop_init = high_resolution_clock::now();
    auto duration_init = duration_cast<microseconds>(stop_init - start_init);
    std::cout << "Init Elapse " << duration_init.count() << "us" << std::endl;
  }

  cudaDeviceSynchronize();
  auto start = high_resolution_clock::now();
  comp<<<nblock, nthread>>>(inp_ptr, out_ptr);
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  std::cout << "Comp Elapse " << duration.count() << "us" << std::endl;
  std::cout << "Memory: " << (array_size * 2 / 1e6) << "MB" << std::endl;
  // we can archieve around 1300GBPS. Which is pretty good since A100-40G peak memory bandwidth is 1555GBPS.
  // We achieved around 84% of peak memory bandwidth.
  std::cout << "Memory Bandwidth: " << ((array_size * 2 / 1e9) / (duration.count() / 1e6)) << "GBPS" << std::endl;

  cudaFree(inp_ptr);
  cudaFree(out_ptr);
  std::cout << "Any error? " << cudaGetErrorName(cudaGetLastError()) << std::endl;
  return 0;
}
