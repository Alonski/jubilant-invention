#include <cuda_runtime.h>
#include <helper_cuda.h>

#define HISTOGRAM_SIZE 256

__global__ void computeOnGPU(int *histogram, int *numbers, int numElements)
{
    int g_index = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int shared_numbers[HISTOGRAM_SIZE];
    shared_numbers[threadIdx.x] = 0;

    if (g_index < numElements)
        atomicAdd(&shared_numbers[numbers[g_index]], 1);

    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], shared_numbers[threadIdx.x]);
    __syncthreads();
}

int *calculateHistogramCUDA(int *numbers, int numElements)
{
    int blocks = numElements / HISTOGRAM_SIZE;
    int *local_histogram = 0, *local_numbers = 0;
    int *histogram = (int *)malloc((HISTOGRAM_SIZE + 1) * sizeof(int));

    // Used to save the error returned from CUDA
    cudaError_t err = cudaSuccess;

    if (numElements % HISTOGRAM_SIZE != 0)
    {
        blocks++;
    }

    // Allocate space for histogram on GPU
    err = cudaMalloc((void **)&local_histogram, (HISTOGRAM_SIZE + 1) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Set empty histogram from the host to GPU
    err = cudaMemset(local_histogram, 0, (HISTOGRAM_SIZE + 1) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate space for numbers on GPU
    err = cudaMalloc((void **)&local_numbers, numElements * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy numbers array from memory in host to GPU
    err = cudaMemcpy(local_numbers, numbers, numElements * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Calculate histogram array on GPU
    computeOnGPU<<<blocks, HISTOGRAM_SIZE>>>(local_histogram, local_numbers, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch GPU add -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy back histogram from GPU to host
    err = cudaMemcpy(histogram, local_histogram, (HISTOGRAM_SIZE + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from GPU to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    if (cudaFree(local_histogram) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free histogram memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(local_numbers) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free numbers memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return histogram;
}