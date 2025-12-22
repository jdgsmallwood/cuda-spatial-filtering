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

__global__ void convert_float_to_half_kernel(const float *input, __half *output,
                                             const int n);

void convert_float_to_half(const float *d_input, __half *d_output, const int n,
                           cudaStream_t stream);

template <typename inputT, typename scaleT, typename outputT,
          size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS>
__global__ void scale_and_convert_to_half_kernel(const inputT *d_input,
                                                 const scaleT *d_scale,
                                                 outputT *d_output) {

  int channel_idx = blockIdx.x % NR_CHANNELS;
  int packet_idx = blockIdx.x / NR_CHANNELS;
  int receiver_idx = blockIdx.y / NR_POLARIZATIONS;
  int fpga_idx = receiver_idx / NR_RECEIVERS_PER_PACKET;
  int receiver_idx_in_pkt = receiver_idx % NR_RECEIVERS_PER_PACKET;
  int polarization_idx = blockIdx.y % NR_POLARIZATIONS;
  int time_idx = threadIdx.x / 2;
  int complex_idx = threadIdx.x % 2;
  // Can I cast directly from int8_t to __half and then I don't need to
  // convert from int to __half as well.
  int val = static_cast<int>(
      d_input[0][channel_idx][packet_idx][fpga_idx][time_idx]
             [receiver_idx_in_pkt][polarization_idx][complex_idx]);
  int scale_factor = static_cast<int>(
      d_scale[0][channel_idx][packet_idx][receiver_idx][polarization_idx]);

  int result = val * scale_factor;
  d_output[0][channel_idx][packet_idx][fpga_idx][time_idx][receiver_idx_in_pkt]
          [polarization_idx][complex_idx] = __int2half_rn(result);
};
template <typename inputT, typename scaleT, typename outputT,
          size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS>
void scale_and_convert_to_half(const inputT *d_input, const scaleT *d_scale,
                               outputT *d_output, cudaStream_t stream) {

  const int num_blocks_x = NR_CHANNELS * NR_PACKETS;
  const int num_blocks_y = NR_POLARIZATIONS * NR_RECEIVERS;

  const int num_threads_x = NR_TIME_STEPS_PER_PACKET * 2; // *2 for complex

  (scale_and_convert_to_half_kernel<
      inputT, scaleT, outputT, NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
      NR_RECEIVERS_PER_PACKET, NR_TIME_STEPS_PER_PACKET,
      NR_PACKETS>)<<<dim3(num_blocks_x, num_blocks_y, 1),
                     dim3(num_threads_x, 1, 1), 0, stream>>>(d_input, d_scale,
                                                             d_output);
}

template <typename T>
__global__ void
debug_kernel(typename T::InputPacketSamplesPlanarType *d_samples_entry,
             typename T::PacketScalesType *d_scales,
             typename T::HalfInputPacketSamplesPlanarType *d_samples_half,
             typename T::HalfPacketSamplesPlanarType *d_samples_padding,
             typename T::PaddedPacketSamplesPlanarType *d_samples_padded) {
  int i = 1;
};

template <typename T>
void debug_kernel_launch(
    typename T::InputPacketSamplesPlanarType *d_samples_entry,
    typename T::PacketScalesType *d_scales,
    typename T::HalfInputPacketSamplesPlanarType *d_samples_half,
    typename T::HalfPacketSamplesPlanarType *d_samples_padding,
    typename T::PaddedPacketSamplesPlanarType *d_samples_padded,
    cudaStream_t stream) {
  debug_kernel<T><<<1, 1, 0, stream>>>(d_samples_entry, d_scales,
                                       d_samples_half, d_samples_padding,
                                       d_samples_padded);
};

template <typename T>
__global__ void unpack_triangular_baseline_batch_kernel(
    const T *__restrict__ packedData, // Input: [Batch, N*(N+1)/2]
    T *__restrict__ denseData,        // Output: [Batch, N, N]
    const int N, const int batchSize) {
  // from Gemini with alterations
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const int num_baselines = (N * (N + 1)) / 2;
  const int total_elements = batchSize * num_baselines;

  for (int i = tid; i < total_elements; i += stride) {
    int b_idx = i / num_baselines; // Which matrix in the batch
    int k = i % num_baselines;     // Index inside packed array

    // Inverse mapping of k = j*(j+1)/2 + i (Column-Major Upper Packed)
    // We find column j and row i
    // Solving j^2 + j - 2k = 0 approximately
    int j = (int)((-1.0f + sqrtf(1.0f + 8.0f * k)) / 2.0f);
    int row = k - (j * (j + 1)) / 2;
    int col = j;

    // Destination index in Dense Column-Major (N*N)
    // dense[row + col*N]
    int dense_idx = b_idx * (N * N) + (col * N + row);
    denseData[dense_idx] = packedData[i];
  }
}

template <typename T>
void unpack_triangular_baseline_batch_launch(const T *packedData, T *denseData,
                                             const int N, const int batchSize,
                                             const int NR_CHANNELS,
                                             cudaStream_t stream) {
  const int num_blocks_x = NR_CHANNELS;
  const int num_threads_x = 1024;

  unpack_triangular_baseline_batch_kernel<T>
      <<<num_blocks_x, num_threads_x, 0, stream>>>(packedData, denseData, N,
                                                   batchSize);
}

template <typename InputT, typename OutputT>
__global__ void
detect_and_average_fft(const InputT *__restrict__ cufft_data,
                       OutputT *__restrict__ output_data, const int NR_CHANNELS,
                       const int NR_POLARIZATIONS, const int NR_FREQS,
                       const int NR_RECEIVERS, const int DOWNSAMPLE_FACTOR) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int channel_idx = blockIdx.x / NR_POLARIZATIONS;
  const int pol_idx = blockIdx.x % NR_POLARIZATIONS;

  const int num_output = NR_FREQS / DOWNSAMPLE_FACTOR;

  while (tid < num_output) {
    float output = 0.0f;
    int start_freq = tid * DOWNSAMPLE_FACTOR;

    for (int j = 0; j < DOWNSAMPLE_FACTOR; ++j) {
      for (int i = 0; i < NR_RECEIVERS; ++i) {
        __half2 in =
            (__half2)cufft_data[0][channel_idx][pol_idx][i][start_freq + j];
        output += sqrtf(in.x * in.x + in.y * in.y);
      }
    }
    output /= (NR_RECEIVERS * DOWNSAMPLE_FACTOR);
    output_data[0][channel_idx][pol_idx][tid] = output;
    tid += stride;
  }
};

template <typename InputT, typename OutputT>
void detect_and_average_fft_launch(const InputT *cufft_data,
                                   OutputT *output_data, const int NR_CHANNELS,
                                   const int NR_POLARIZATIONS,
                                   const int NR_FREQS, const int NR_RECEIVERS,
                                   const int DOWNSAMPLE_FACTOR,
                                   cudaStream_t stream) {

  detect_and_average_fft<InputT, OutputT>
      <<<NR_CHANNELS * NR_POLARIZATIONS, 1024, 0, stream>>>(
          cufft_data, output_data, NR_CHANNELS, NR_POLARIZATIONS, NR_FREQS,
          NR_RECEIVERS, DOWNSAMPLE_FACTOR);
}
