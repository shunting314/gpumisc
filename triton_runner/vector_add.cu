extern "C" __global__ void add_kernel_0d1d2d3d(float* lhs, float* rhs, float* out, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    for (int i = tid; i < numel; i += nthreads) {
        out[i] = lhs[i] + rhs[i];
    }
}
