/*
 * Copyright 2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUDA application for uncoalesced global memory accesses.
 * Adds a floating point constant to an input array of double3 elements in
 * global memory and generates an output array of double3 in global memory.
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 256

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

__global__ void addConstDouble3(int numElements, double3 *d_in, double k, double3 *d_out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements)
    {
        double3 a = d_in[index];
        a.x += k;
        a.y += k;
        a.z += k;
        d_out[index] = a;
    }
}

__global__ void addConstDouble(int numElements, double *d_in, double k, double *d_out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements)
    {
        d_out[index] = d_in[index] + k;
    }
}

int main (int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    double constK = 10.0;

    int kernelOption = 0;
    if (argc > 1)
    {
       kernelOption = atoi(argv[1]);
    }

    int numElements = 1024*1024;
    if (argc > 2)
    {
        numElements = atoi(argv[2]);
        if (numElements <= 0)
        {
            fprintf(stderr, "Invalid number of elements(%s), should be a positive number\n", argv[2]);
            exit(EXIT_FAILURE);
        }
    }

    printf("double3 constant addition of %d elements\n", numElements);
    printf("kernelOption=%d\n", kernelOption);

    size_t size = numElements * sizeof(double3);

    // Allocate the host input array
    double3 *h_A = (double3 *)malloc(size);


    // Allocate the host output array
    double3 *h_B = (double3 *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL)
    {
        fprintf(stderr, "Failed to allocate host arrays!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i].x = rand()/(double)RAND_MAX;
        h_A[i].y = rand()/(double)RAND_MAX;
        h_A[i].z = rand()/(double)RAND_MAX;
    }

    // Allocate the device input array A
    double3 *d_A = NULL;
    RUNTIME_API_CALL(cudaMalloc((void **)&d_A, size));


    // Allocate the device output array B
    double3 *d_B = NULL;
    RUNTIME_API_CALL(cudaMalloc((void **)&d_B, size));

    // Copy the host input array A in host memory to the device input array in device memory
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Launch the CUDA Kernel
    int threadsPerBlock = BLOCK_SIZE;
    if (kernelOption == 0)
    {
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
        printf("CUDA kernel addConstDouble3 launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        addConstDouble3<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_A, constK, d_B);
    } 
    else if (kernelOption == 1)
    {
        int blocksPerGrid =(numElements*3 + threadsPerBlock - 1) / threadsPerBlock;
        printf("CUDA kernel addConstDouble launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        addConstDouble<<<blocksPerGrid, threadsPerBlock>>>(numElements*3, (double *)d_A, constK, (double *)d_B);
    }
    else
    {
        fprintf(stderr, "** Invalid kernel option %d\n", kernelOption);
        exit(EXIT_FAILURE);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result array in device memory to the host result vector
    // in host memory.
    RUNTIME_API_CALL(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));


    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if ((fabs(h_A[i].x + constK - h_B[i].x) > 1e-5) ||
            (fabs(h_A[i].y + constK - h_B[i].y) > 1e-5) ||
            (fabs(h_A[i].z + constK - h_B[i].z) > 1e-5))
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free device global memory
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));

    // Free host memory
    free(h_A);
    free(h_B);

    printf("Done\n");
    return 0;
}
