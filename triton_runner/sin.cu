/*
 * Compiling command: 'nvcc -c -cubin -arch=sm_80 sin.cu'
 */
extern "C" __global__ void sin_kernel_0d1d2d(float* in, float* out, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    for (int i = tid; i < numel; i += nthreads) {
        out[i] = sin(in[i]);
    }
}
