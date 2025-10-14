#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
__global__ void convert_int8_to_half_kernel(const int8_t *d_input,
                                            __half *d_output, const int n);
__global__ void
update_weights_kernel(const __half *d_weights, __half *d_weights_output,
                      const int num_beams, const int num_receivers,
                      const int num_channels, const int num_polarizations);
__global__ void convert_int_to_float_kernel(const int *d_input, float *d_output,
                                            const int n);
__global__ void
accumulate_visibilities_kernel(const float *d_visibilities,
                               float *d_visibilities_accumulated, const int n);
void convert_int8_to_half(const int8_t *d_input, __half *d_output, const int n,
                          cudaStream_t stream);
void convert_int_to_float(const int *d_input, float *d_output, const int n,
                          cudaStream_t stream);
void update_weights(const __half *d_weights, __half *d_weights_output,
                    const int num_beams, const int num_receivers,
                    const int num_channels, const int num_polarizations,
                    const float *d_eigenvalues, float *d_eigenvectors,
                    cudaStream_t &stream);
void accumulate_visibilities(const float *d_visibilities,
                             float *d_visibilities_accumulated, const int n,
                             cudaStream_t stream);

template <typename inputT, typename scaleT, typename outputT,
          size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_TIME_STEPS_PER_PACKET, size_t NR_PACKETS>
__global__ void scale_and_convert_to_half_kernel(const inputT *d_input,
                                                 const scaleT *d_scale,
                                                 outputT *d_output) {

  int channel_idx = blockIdx.x / NR_CHANNELS;
  int packet_idx = blockIdx.x % NR_CHANNELS;
  int receiver_idx = blockIdx.y / NR_POLARIZATIONS;
  int polarization_idx = blockIdx.y % NR_POLARIZATIONS;
  int time_idx = threadIdx.x / 2;
  int complex_idx = threadIdx.x % 2;

  int val =
      static_cast<int>(d_input[0][channel_idx][packet_idx][time_idx]
                              [receiver_idx][polarization_idx][complex_idx]);
  int scale_factor = static_cast<int>(
      d_scale[0][channel_idx][packet_idx][receiver_idx][polarization_idx]);

  int result = val * scale_factor;
  d_output[0][channel_idx][packet_idx][time_idx][receiver_idx][polarization_idx]
          [complex_idx] = __int2half_rn(result);
};
template <typename inputT, typename scaleT, typename outputT,
          size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_TIME_STEPS_PER_PACKET, size_t NR_PACKETS>
void scale_and_convert_to_half(const inputT *d_input, const scaleT *d_scale,
                               outputT *d_output, cudaStream_t stream) {

  const int num_blocks_x = NR_CHANNELS * NR_PACKETS;
  const int num_blocks_y = NR_POLARIZATIONS * NR_RECEIVERS;

  const int num_threads_x = NR_TIME_STEPS_PER_PACKET * 2; // *2 for complex

  (scale_and_convert_to_half_kernel<
      inputT, scaleT, outputT, NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
      NR_TIME_STEPS_PER_PACKET,
      NR_PACKETS>)<<<dim3(num_blocks_x, num_blocks_y, 1),
                     dim3(num_threads_x, 1, 1), 0, stream>>>(d_input, d_scale,
                                                             d_output);
}
