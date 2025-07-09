#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
__global__ void convert_int8_to_half_kernel(const int8_t* d_input, __half* d_output, const int n);
void convert_int8_to_half(const int8_t* d_input, __half* d_output, const int n, cudaStream_t &stream);
void update_weights(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers, const __half *d_eigenvalues, const __half *d_eigenvectors, cudaStream_t &stream);
