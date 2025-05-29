#include "spatial/spatial.hpp"
#include <cuda_runtime.h>

__global__ void increment_kernel(int* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx]++;
    }
}

void incrementArray(int* data, int size) {
    int* d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    increment_kernel<<<blocks, threads>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}