#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
__global__ void convert_int8_to_half_kernel(const int8_t* d_input, __half* d_output, const int n);
__global__ void update_weights_kernel(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers, const int num_channels, const int num_polarizations);
__global__ void convert_int_to_float_kernel(const int *d_input, float *d_output, const int n); 
void convert_int8_to_half(const int8_t* d_input, __half* d_output, const int n, cudaStream_t &stream);
void convert_int_to_float(const int *d_input, float *d_output, const int n, cudaStream_t &stream); 
void update_weights(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers, const int num_channels, const int num_polarizations, const float *d_eigenvalues, const float *d_eigenvectors, cudaStream_t &stream);
