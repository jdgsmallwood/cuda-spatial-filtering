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

__global__ void update_weights_kernel(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers, const int num_channels, const int num_polarizations) {
    
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n = num_beams * num_receivers * num_channels * num_polarizations * 2;

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

__global__ void accumulate_visibilities_kernel(const float *d_visibilities, float *d_visibilities_accumulated, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while(idx < n) {
       d_visibilities_accumulated[idx] += d_visibilities[idx];
        idx += stride;
    }


}
void convert_int8_to_half(const int8_t *d_input, __half *d_output, const int n, cudaStream_t stream) {
    
    const int num_blocks = std::min(8, n / 1024 + 1);

    convert_int8_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input, d_output, n);

}


void convert_int_to_float(const int *d_input, float *d_output, const int n, cudaStream_t stream) {
    
    const int num_blocks = std::min(8, n / 1024 + 1);

    convert_int_to_float_kernel<<<num_blocks, 1024, 0, stream>>>(d_input, d_output, n);


}


void update_weights(const __half *d_weights, __half *d_weights_output, const int num_beams, const int num_receivers,const int num_channels, const int num_polarizations, const float *d_eigenvalues, float *d_eigenvectors, cudaStream_t &stream) {

    const int n = num_beams * num_receivers * num_channels * num_polarizations;
    const int num_blocks = std::min(8, n / 1024 + 1);

    update_weights_kernel<<<num_blocks, 1024, 0, stream>>>(d_weights, d_weights_output, num_beams, num_receivers, num_channels, num_polarizations);
}

void accumulate_visibilities(const float *d_visibilities, float *d_visibilities_accumulated, const int n, cudaStream_t stream) {

    const int num_blocks = std::min(8, n / 1024 + 1);

    accumulate_visibilities_kernel<<<num_blocks, 1024, 0, stream>>>(d_visibilities, d_visibilities_accumulated, n);
    

}


template<typename inputT, typename scaleT, typename outputT, size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS, size_t NR_TIME_STEPS_PER_PACKET, size_t NR_PACKETS> 
__global__ void scale_and_convert_to_half_kernel(const inputT *d_input, const scaleT *d_scale, const outputT *d_output)  {

    int channel_idx = blockIdx.x / NR_CHANNELS;
    int packet_idx = blockIdx.x % NR_CHANNELS;
    int receiver_idx = threadIdx.y;
    int polarization_idx = blockIdx.y;
    int time_idx = threadIdx.x / 2;
    int complex_idx = threadIdx.x % 2;

    int val = static_cast<int>(d_input[channel_idx][packet_idx][time_idx][receiver_idx][polarization_idx][complex_idx]);
    int scale_factor = static_cast<int>(d_scale[channel_idx][packet_idx][receiver_idx][polarization_idx]);

    int result = val * scale_factor;
    d_output[channel_idx][packet_idx][time_idx][receiver_idx][polarization_idx][complex_idx] = __int2half_rn(result);
};



