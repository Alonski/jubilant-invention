#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "proto.h"

__global__ void computeOnGPU(int *histogram, int *data, int numElements)
{
    int g_index = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int shared_data[HISTOGRAM_SIZE];
    shared_data[threadIdx.x] = 0;

    if (g_index < numElements)
        atomicAdd(&shared_data[data[g_index]], 1);

    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], shared_data[threadIdx.x]);
    __syncthreads();
}

int calculateHistogramCUDA(int *histogram, int *data, int numElements)
{
    // Used to save the error returned from CUDA
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(int);

    // Allocate on GPU and copy from the host
    int *d_data, *d_histo;
    err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy from the host to GPU
    err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_histo, HISTOGRAM_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemset(d_histo, 0, HISTOGRAM_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initiate Kernel
    int blocks = (numElements + HISTOGRAM_SIZE - 1) / HISTOGRAM_SIZE;
    computeOnGPU<<<blocks, HISTOGRAM_SIZE>>>(d_histo, d_data, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vector add -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from GPU to host.
    err = cudaMemcpy(histogram, d_histo, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from GPU to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free memory
    if (cudaFree(d_data) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_histo) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
