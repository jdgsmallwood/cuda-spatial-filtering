#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

__global__ void convert_int8_to_half_kernel(const int8_t *d_input, __half *d_output, const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    while (idx < n) {
        // Promote int8_t to int before conversion
        int val = static_cast<int>(d_input[idx]);
        d_output[idx] = __int2half_rn(val);
        idx += stride;
    }
}

__global__ void update_weights_kernel(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers, const int num_channels) {
    
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n = num_beams * num_receivers;

    while (idx < n) {
        d_weights_output[idx] = d_weights[idx];
        idx += stride;
    }

}


__global__ void convert_int_to_float_kernel(const int *d_input, float *d_output, const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    while (idx < n) {
        d_output[idx] = __int2float_rn(d_input[idx]);
        idx += stride;
    }   

}

void convert_int8_to_half(const int8_t *d_input, __half *d_output, const int n, cudaStream_t &stream) {
    
    const int num_blocks = std::min(4, n / 1024 + 1);

    convert_int8_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input, d_output, n);

}


void convert_int_to_float(const int *d_input, float *d_output, const int n, cudaStream_t &stream) {
    
    const int num_blocks = std::min(4, n / 1024 + 1);

    convert_int_to_float_kernel<<<num_blocks, 1024, 0, stream>>>(d_input, d_output, n);


}


void update_weights(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers,const int num_channels, const float *d_eigenvalues, const float *d_eigenvectors, cudaStream_t &stream) {

    const int n = num_beams * num_receivers * num_channels;
    const int num_blocks = std::min(4, n / 1024 + 1);

    update_weights_kernel<<<num_blocks, 1024, 0, stream>>>(d_weights, d_weights_output, num_beams, num_receivers, num_channels);
}


