// This program shows off a shared memory implementation of a histogram
// kernel in CUDA

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;

// Number of bins 
constexpr int BINS = 256;
constexpr int DIV = ((256 + BINS - 1) / BINS);

// GPU kernel for computing a histogram
//  a: Problem array in global memory
//  result: result array
//  N: Size of the array
__global__ void histogram(int *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate a local histogram for each TB
  __shared__ int s_result[BINS];

  // Initalize the shared memory to 0
  if (threadIdx.x < BINS) {
    s_result[threadIdx.x] = 0;
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Calculate histogram locally
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    atomicAdd(&s_result[(a[i] / DIV)], 1);
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Combine the partial results
  if (threadIdx.x < BINS) {
    atomicAdd(&result[threadIdx.x], s_result[threadIdx.x]);
  }
}

int computeOnGPU(int *data, int N) {
  // Allocate memory on the host
  vector<int> h_input(N);
  vector<int> h_result(BINS);

  // Allocate memory on the device
  int *data;
  int *d_result;
  cudaMalloc(&data, N);
  cudaMalloc(&d_result, BINS * sizeof(int));

  // Copy the array to the device
  cudaMemcpy(data, h_input.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int),
             cudaMemcpyHostToDevice);

  // Number of threads per threadblock
  int THREADS = 256;

  // Calculate the number of threadblocks
  int BLOCKS = N / THREADS;

  // Launch the kernel
  histogram<<<BLOCKS, THREADS>>>(data, d_result, N);

  // Copy the result back
  cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int), cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(data);
  cudaFree(d_result);

  return 0;
}