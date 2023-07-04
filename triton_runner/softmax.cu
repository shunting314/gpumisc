/*
 * A dummy softmax implementation.
 * Super slow since reduction is done by a single thread..
 */
extern "C" __global__ void softmax_kernel_0d1d2(float* in, float* out, int numel_per_row) {
    // only use the first element for communication
    extern __shared__ float sdata[];
    in = in + numel_per_row * blockIdx.x;
    out = out + numel_per_row * blockIdx.x;

    // compute max
    float maxVal;
    if (threadIdx.x == 0) {
        maxVal = in[0];
        for (int i = 1; i < numel_per_row; ++i) {
            maxVal = max(maxVal, in[i]);
        }
        sdata[0] = maxVal;
    }
    __syncthreads();
    maxVal = sdata[0];

    int nthread = blockDim.x;
    for (int i = threadIdx.x; i < numel_per_row; i += nthread) {
        out[i] = exp(in[i] - maxVal);
    }

    // compute sum of exp
    float sumExp;
    if (threadIdx.x == 0) {
        sumExp = out[0];
        for (int i = 1; i < numel_per_row; ++i) {
            sumExp += out[i];
        }
        sdata[0] = sumExp;
    }
    __syncthreads();
    sumExp = sdata[0];

    for (int i = threadIdx.x; i < numel_per_row; i += nthread) {
        out[i] /= sumExp;
    }
}
