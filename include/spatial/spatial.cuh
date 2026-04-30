#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
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
          size_t NR_PACKETS, size_t TIME_STEPS_PER_THREAD = 1>
__global__ void scale_and_convert_to_half_kernel(const inputT *d_input,
                                                 const scaleT *d_scale,
                                                 outputT *d_output) {

  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");
  static_assert(
      NR_TIME_STEPS_PER_PACKET % TIME_STEPS_PER_THREAD == 0,
      "TIME_STEPS_PER_THREAD must evenly divide NR_TIME_STEPS_PER_PACKET");

  constexpr size_t ELEMS_PER_TIME = NR_POLARIZATIONS * 2;

  int channel_idx = blockIdx.x % NR_CHANNELS;
  int packet_idx = blockIdx.x / NR_CHANNELS;
  int fpga_idx = blockIdx.y;

  int complex_idx = threadIdx.x % 2;
  int pol_idx = (threadIdx.x / 2) % NR_POLARIZATIONS;
  int recv_in_pkt = blockIdx.z;
  int time_base = (threadIdx.x / ELEMS_PER_TIME) * TIME_STEPS_PER_THREAD;

  int receiver_idx = fpga_idx * NR_RECEIVERS_PER_PACKET + recv_in_pkt;
  int16_t scale_val =
      __ldg(&d_scale[0][channel_idx][packet_idx][receiver_idx][pol_idx]);

#pragma unroll
  for (int t = 0; t < TIME_STEPS_PER_THREAD; ++t) {
    int8_t sample =
        __ldg(&d_input[0][channel_idx][packet_idx][fpga_idx][time_base + t]
                      [recv_in_pkt][pol_idx][complex_idx]);

    d_output[0][channel_idx][packet_idx][fpga_idx][time_base + t][recv_in_pkt]
            [pol_idx][complex_idx] = __int2half_rn(static_cast<int>(sample) *
                                                   static_cast<int>(scale_val));
  }
};

template <int N> constexpr bool dependent_false = false;

template <typename inputT, typename scaleT, typename outputT,
          size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS, size_t TIME_STEPS_PER_THREAD = 1>
void scale_and_convert_to_half(const inputT *d_input, const scaleT *d_scale,
                               outputT *d_output, cudaStream_t stream) {

  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t ELEMS_PER_TIME = NR_POLARIZATIONS * 2;
  constexpr size_t THREADS =
      (NR_TIME_STEPS_PER_PACKET / TIME_STEPS_PER_THREAD) * ELEMS_PER_TIME;

  static_assert(
      THREADS <= 1024 || dependent_false<THREADS>,
      "Block size exceeds CUDA maximum — increase TIME_STEPS_PER_THREAD");
  static_assert(
      THREADS % 32 == 0,
      "Block size must be a multiple of warp size — check dimension values");

  scale_and_convert_to_half_kernel<
      inputT, scaleT, outputT, NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
      NR_RECEIVERS_PER_PACKET, NR_TIME_STEPS_PER_PACKET, NR_PACKETS,
      TIME_STEPS_PER_THREAD>
      <<<dim3(NR_CHANNELS * NR_PACKETS, NR_FPGA_SOURCES,
              NR_RECEIVERS_PER_PACKET),
         dim3(THREADS, 1, 1), 0, stream>>>(d_input, d_scale, d_output);
}

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
__global__ void get_data_for_fft(
    const InputT *__restrict__ input_data, OutputT *__restrict__ output_data,
    const int NR_CHANNELS, const int NR_POLARIZATIONS, const int NR_FREQS,
    const int NR_RECEIVERS, const int channel_to_fft, const int pol_to_fft) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  int channel_to_fft_local = channel_to_fft;
  int pol_to_fft_local = pol_to_fft;

  while (tid < NR_FREQS * NR_RECEIVERS) {
    int receiver_idx = tid % NR_RECEIVERS;
    int time_idx = tid / NR_RECEIVERS;
    float2 output = __half22float2(
        (__half2)(input_data[0][channel_to_fft_local][pol_to_fft_local]
                            [receiver_idx][time_idx]));

    output_data[0][receiver_idx][time_idx] = output;
    tid += stride;
  };
};

template <typename InputT, typename OutputT>
__global__ void
get_data_for_multi_channel_fft(const InputT *__restrict__ input_data,
                               OutputT *__restrict__ output_data, const int n) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const half2 *input_ptr = reinterpret_cast<const half2 *>(input_data);
  float2 *output_ptr = reinterpret_cast<float2 *>(output_data);
  while (tid < n) {
    float2 output = __half22float2(input_ptr[tid]);
    output_ptr[tid] = output;
    tid += stride;
  };
};

template <typename InputT, typename OutputT>
void get_data_for_fft_launch(const InputT *input_data, OutputT *output_data,
                             const int NR_CHANNELS, const int NR_POLARIZATIONS,
                             const int NR_FREQS, const int NR_RECEIVERS,
                             const int channel_to_fft, const int pol_to_fft,
                             cudaStream_t stream) {

  const int num_blocks = 16;
  const int num_threads = 1024;

  get_data_for_fft<InputT, OutputT><<<num_blocks, num_threads, 0, stream>>>(
      input_data, output_data, NR_CHANNELS, NR_POLARIZATIONS, NR_FREQS,
      NR_RECEIVERS, channel_to_fft, pol_to_fft);
}

template <typename InputT, typename OutputT>
void get_data_for_multi_channel_fft_launch(
    const InputT *input_data, OutputT *output_data, const int NR_CHANNELS,
    const int NR_POLARIZATIONS, const int NR_FREQS, const int NR_RECEIVERS,
    cudaStream_t stream) {

  const int num_blocks = 16;
  const int num_threads = 1024;
  const int n = NR_CHANNELS * NR_POLARIZATIONS * NR_RECEIVERS * NR_FREQS;

  get_data_for_multi_channel_fft<InputT, OutputT>
      <<<num_blocks, num_threads, 0, stream>>>(input_data, output_data, n);
}

template <typename InputT, typename OutputT>
__global__ void
detect_and_average_fft(const InputT *__restrict__ cufft_data,
                       OutputT *__restrict__ output_data, const int NR_FREQS,
                       const int NR_RECEIVERS, const int DOWNSAMPLE_FACTOR) {

  int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int num_output = NR_FREQS / DOWNSAMPLE_FACTOR;

  while (tid < num_output) {
    float output = 0.0f;
    int start_freq = tid * DOWNSAMPLE_FACTOR;
    int num_vals = 0;
    for (int j = 0; j < DOWNSAMPLE_FACTOR; ++j) {
      for (int i = 0; i < NR_RECEIVERS; ++i) {
        float2 in = cufft_data[0][i][start_freq + j];
        float val = sqrtf(in.x * in.x + in.y * in.y);

        if (!isnan(val)) {
          output += val;
          num_vals++;
        }
      }
    }
    if (num_vals != 0) {
      output /= (num_vals);
    } else {
      output = 0;
    }
    output_data[0][tid] = output;
    tid += stride;
  }
};

template <typename InputT, typename OutputT>
void detect_and_average_fft_launch(const InputT *cufft_data,
                                   OutputT *output_data, const int NR_FREQS,
                                   const int NR_RECEIVERS,
                                   const int DOWNSAMPLE_FACTOR,
                                   cudaStream_t stream) {

  detect_and_average_fft<InputT, OutputT><<<1, 1024, 0, stream>>>(
      cufft_data, output_data, NR_FREQS, NR_RECEIVERS, DOWNSAMPLE_FACTOR);
}

template <typename InputT, typename OutputT>
__global__ void detect_and_downsample_multi_channel_fft(
    const InputT *__restrict__ cufft_data, OutputT *__restrict__ output_data,
    const int NR_CHANNELS, const int NR_POLARIZATIONS, const int NR_FREQS,
    const int NR_RECEIVERS, const int DOWNSAMPLE_FACTOR) {
  const int num_output_freqs = NR_FREQS / DOWNSAMPLE_FACTOR;

  // Flattened 3D Grid: x = output frequencies, y = receivers, z = channels *
  // pols
  const int out_freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rx_idx = blockIdx.y;
  const int chan_pol_idx = blockIdx.z;
  const int chan = chan_pol_idx / NR_POLARIZATIONS;
  const int pol = chan_pol_idx % NR_POLARIZATIONS;

  if (out_freq_idx >= num_output_freqs || rx_idx >= NR_RECEIVERS)
    return;

  float sum = 0.0f;
  int count = 0;

  int start_f = out_freq_idx * DOWNSAMPLE_FACTOR;
  for (int j = 0; j < DOWNSAMPLE_FACTOR; ++j) {
    float2 in = cufft_data[0][chan][pol][rx_idx][start_f + j];
    float val = in.x * in.x + in.y * in.y;

    if (!isnan(val)) {
      sum += val;
      count++;
    }
  }

  float final_val = (count > 0) ? (sum / (float)count) : 0.0f;

  output_data[0][chan][pol][rx_idx][out_freq_idx] = final_val;
}

template <typename InputT, typename OutputT>
void detect_and_downsample_multi_channel_fft_launch(
    const InputT *cufft_data, OutputT *output_data, const int NR_CHANNELS,
    const int NR_POLARIZATIONS, const int NR_FREQS, const int NR_RECEIVERS,
    const int DOWNSAMPLE_FACTOR, cudaStream_t stream) {
  detect_and_downsample_multi_channel_fft<InputT, OutputT>
      <<<dim3((NR_FREQS / DOWNSAMPLE_FACTOR + 255) / 256, NR_RECEIVERS,
              NR_CHANNELS * NR_POLARIZATIONS),
         256, 0, stream>>>(cufft_data, output_data, NR_CHANNELS,
                           NR_POLARIZATIONS, NR_FREQS, NR_RECEIVERS,
                           DOWNSAMPLE_FACTOR);
}

__global__ void detect_and_downsample_fft(
    const float2 *__restrict__ cufft_data, float *__restrict__ output_data,
    const int NR_CHANNELS, const int NR_POLARIZATIONS, const int NR_FREQS,
    const int NR_BEAMS, const int DOWNSAMPLE_FACTOR) {
  const int num_output_freqs = NR_FREQS / DOWNSAMPLE_FACTOR;

  // Flattened 3D Grid: x = output frequencies, y = receivers, z = channels *
  // pols
  const int out_freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int beam_idx = blockIdx.y;
  const int chan_pol_idx = blockIdx.z;
  const int chan = chan_pol_idx / NR_POLARIZATIONS;
  const int pol = chan_pol_idx % NR_POLARIZATIONS;

  if (out_freq_idx >= num_output_freqs || beam_idx >= NR_BEAMS)
    return;

  float sum = 0.0f;
  int count = 0;

  const int start_f = out_freq_idx * DOWNSAMPLE_FACTOR;
  const int base_pointer = chan * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS +
                           pol * NR_BEAMS * NR_FREQS + beam_idx * NR_FREQS;
  for (int j = 0; j < DOWNSAMPLE_FACTOR; ++j) {
    float2 in = cufft_data[base_pointer + start_f + j];
    float val = in.x * in.x + in.y * in.y;

    if (!isnan(val)) {
      sum += val;
      count++;
    }
  }

  float final_val = (count > 0) ? (sum / (float)count) : 0.0f;

  const int output_base_pointer =
      chan * NR_POLARIZATIONS * NR_BEAMS * num_output_freqs +
      pol * NR_BEAMS * num_output_freqs + beam_idx * num_output_freqs;
  output_data[output_base_pointer + out_freq_idx] = final_val;
}

void detect_and_downsample_fft_launch(const float2 *cufft_data,
                                      float *output_data, const int NR_CHANNELS,
                                      const int NR_POLARIZATIONS,
                                      const int NR_FREQS, const int NR_BEAMS,
                                      const int DOWNSAMPLE_FACTOR,
                                      cudaStream_t stream) {

  detect_and_downsample_fft<<<dim3((NR_FREQS / DOWNSAMPLE_FACTOR + 255) / 256,
                                   NR_BEAMS, NR_CHANNELS * NR_POLARIZATIONS),
                              256, 0, stream>>>(
      cufft_data, output_data, NR_CHANNELS, NR_POLARIZATIONS, NR_FREQS,
      NR_BEAMS, DOWNSAMPLE_FACTOR);
}

__global__ void scale_visibilities_kernel(float *__restrict__ data,
                                          const size_t count,
                                          const float scale) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] *= scale;
  }
}

inline void scale_visibilities(float *data, const size_t count,
                               const float scale, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
  scale_visibilities_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(data, count,
                                                                  scale);
}

__global__ void identityMinusMatrixKernel(const __restrict__ float2 *d_A,
                                          __half2 *d_output, const int N,
                                          const int batches) {
  // Calculate the global 1D thread index
  int batch = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = N * N;

  if (idx < total_elements) {
    int lookup_idx = total_elements * batch + idx;
    float2 val = d_A[lookup_idx];
    // If the index falls on the diagonal
    if (idx % (N + 1) == 0) {
      d_output[lookup_idx] =
          __float22half2_rn(make_float2(1.0f - val.x, -val.y));
    } else {
      // If it's an off-diagonal element
      d_output[lookup_idx] = __float22half2_rn(make_float2(-val.x, -val.y));
    }
  }
}

void computeIdentityMinusA(const float2 *d_A, __half2 *d_output, const int N,
                           const int batches, cudaStream_t stream) {
  int total_elements = N * N;

  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  identityMinusMatrixKernel<<<dim3(blocksPerGrid, batches, 1), threadsPerBlock,
                              0, stream>>>(d_A, d_output, N, batches);
}

__global__ void weightsDebugKernel(const __half2 *d_in, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    __half2 val = d_in[idx];
    printf("Idx %i: %f + %f i\n", idx, __half2float(val.x),
           __half2float(val.y));
  }
}

void weightsDebugLaunch(const __half2 *d_in, int N, cudaStream_t stream) {

  int threadsPerBlock = 256;

  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  weightsDebugKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_in, N);
}

__global__ void conjugateMatrixKernel(__half2 *d_in, const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    __half2 val = d_in[idx];
    val.y = -val.y;
    d_in[idx] = val;
  }
}

void conjugateMatrix(__half2 *d_in, const int N, cudaStream_t stream) {

  int threadsPerBlock = 256;

  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  conjugateMatrixKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_in, N);
}

__global__ void incoherent_sum(const __restrict__ float2 *d_input,
                               float *d_output, const size_t nr_channels,
                               const size_t nr_polarizations,
                               const size_t nr_receivers,
                               const size_t nr_fine_channels,
                               const size_t time_bins_per_block) {

  __shared__ float detected_data[1024];
  // This kernel needs to detect and sum over antennas for each channel / pol.

  int time_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int time_in_block_idx = threadIdx.y;
  int antenna_id = threadIdx.x;
  int pol = blockIdx.y;
  int coarse_channel = blockIdx.z / nr_fine_channels;
  int fine_channel = blockIdx.z % nr_fine_channels;
  int linearized_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  int linearized_time_idx = blockIdx.x * blockDim.y + linearized_thread_idx;

  float squared_mag = 0.0f;
  if (time_idx < time_bins_per_block) {
    const size_t base_pointer =
        coarse_channel * nr_polarizations * nr_fine_channels *
            time_bins_per_block * nr_receivers +
        fine_channel * nr_polarizations * time_bins_per_block * nr_receivers +
        pol * time_bins_per_block * nr_receivers + time_idx * nr_receivers +
        antenna_id;

    float2 val = d_input[base_pointer];
    squared_mag = val.x * val.x + val.y * val.y;
  }

  int shared_pointer = time_in_block_idx * nr_receivers + antenna_id;
  detected_data[shared_pointer] = squared_mag;
  __syncthreads();

  // now add the shared data
  int n = nr_receivers;

  while (n > 1) {
    // Stride is ceiling(n / 2). This handles odd lengths properly.
    int stride = (n + 1) / 2;

    // Only the first floor(n / 2) threads do work in this iteration
    if (antenna_id < (n / 2)) {
      detected_data[shared_pointer] += detected_data[shared_pointer + stride];
    }

    // Synchronize to ensure all additions are visible before the next pass
    __syncthreads();

    // The new array size is the stride we just used
    n = stride;
  }

  if (linearized_thread_idx < blockDim.y &&
      linearized_time_idx < time_bins_per_block) {
    d_output[coarse_channel * nr_fine_channels * nr_polarizations *
                 time_bins_per_block +
             fine_channel * nr_polarizations * time_bins_per_block +
             pol * time_bins_per_block + blockIdx.x * blockDim.y +
             linearized_thread_idx] =
        detected_data[linearized_thread_idx * nr_receivers];
  }
}

void incoherent_sum_launch(const float2 *d_input, float *d_output,
                           const size_t nr_channels,
                           const size_t nr_polarizations,
                           const size_t nr_receivers,
                           const size_t nr_fine_channels,
                           const size_t time_bins_per_block,
                           cudaStream_t stream) {

  int nr_time_steps_per_block = 1024 / nr_receivers;
  int nr_time_blocks = (time_bins_per_block + nr_time_steps_per_block - 1) /
                       nr_time_steps_per_block;

  incoherent_sum<<<dim3(nr_time_blocks, nr_polarizations,
                        nr_channels * nr_fine_channels),
                   dim3(nr_receivers, nr_time_steps_per_block, 1), 0, stream>>>(
      d_input, d_output, nr_channels, nr_polarizations, nr_receivers,
      nr_fine_channels, time_bins_per_block);
}

// fold_and_accumulate_kernel
//
// Input  (incoherent_sum): [nr_channels][nr_fine_channels][nr_polarizations]
//                          [nr_time_bins]   (row-major, power floats)
// Output (fold_accumulator): [nr_channels][nr_fine_channels][nr_polarizations]
//                            [n_bins]      (row-major, atomically accumulated)
// dm_delays: [nr_channels * nr_fine_channels]  (int32 sample offsets,
//            positive = channel arrives late relative to reference frequency)
//
// Each thread handles one (channel, fine_channel, polarization, time_bin) cell.
// The absolute sample index for that time bin is:
//   abs_sample = total_samples_elapsed + local_time_bin
// After subtracting the DM delay, phase is mapped to a bin with fmod and
// an atomicAdd accumulates the power.  Negative phase (delayed channel at
// the start of an observation) wraps via the +1 trick rather than a branch.
__global__ void fold_and_accumulate_kernel(
    const float *__restrict__ incoherent_sum,
    float *__restrict__ fold_accumulator, uint32_t *__restrict__ hit_counts,
    const int32_t *__restrict__ dm_delays, int64_t total_samples_elapsed,
    double period_samples, int n_bins, int nr_channels, int nr_fine_channels,
    int nr_polarizations, int nr_time_bins) {
  // Grid: (nr_channels * nr_fine_channels, nr_polarizations, nr_time_bins)
  const int cf_idx =
      blockIdx.x * blockDim.x + threadIdx.x; // flat channel×fine index
  const int pol_idx = blockIdx.y;
  const int t_idx = blockIdx.z * blockDim.z + threadIdx.z;

  if (cf_idx >= nr_channels * nr_fine_channels || t_idx >= nr_time_bins)
    return;

  const int c_idx = cf_idx / nr_fine_channels;
  const int f_idx = cf_idx % nr_fine_channels;

  const int32_t delay = dm_delays[cf_idx];

  // Corrected absolute sample index for this time bin.
  const int64_t abs_sample =
      total_samples_elapsed + static_cast<int64_t>(t_idx) - delay;

  // fmod with period_samples, then map to [0, 1).  Using double throughout
  // to preserve phase accuracy over long observations.
  double phase = fmod(static_cast<double>(abs_sample) / period_samples, 1.0);
  if (phase < 0.0)
    phase += 1.0;

  const int bin = static_cast<int>(phase * n_bins);

  // Input index: [c][f][pol][t]
  const int in_idx =
      ((c_idx * nr_fine_channels + f_idx) * nr_polarizations + pol_idx) *
          nr_time_bins +
      t_idx;

  // Output index: [c][f][pol][bin]
  const int out_idx = bin;

  float sum = incoherent_sum[in_idx];
  if (sum > 0) {

    atomicAdd(&fold_accumulator[out_idx], incoherent_sum[in_idx]);
    atomicAdd(&hit_counts[out_idx], 1u);
  }
}

// Launch wrapper matching the call site in LambdaPulsarFoldPipeline.
//
// Thread layout:
//   blockDim: (32, 1, 8)   — 32 cf cells × 8 time bins per block = 256 threads
//   gridDim:  ceil over (nr_channels * nr_fine_channels), nr_polarizations,
//             ceil over nr_time_bins
//
// nr_polarizations is always small (≤ 4) so it maps directly to gridDim.y
// without needing a thread dimension.
inline void fold_and_accumulate_launch(
    const float *incoherent_sum, float *fold_accumulator, uint32_t *hit_counts,
    const int32_t *dm_delays, int64_t total_samples_elapsed,
    double period_samples, int n_bins, int nr_channels, int nr_fine_channels,
    int nr_polarizations, int nr_time_bins, cudaStream_t stream) {
  constexpr int CF_THREADS = 32;
  constexpr int T_THREADS = 8;

  const dim3 block(CF_THREADS, 1, T_THREADS);
  const dim3 grid((nr_channels * nr_fine_channels + CF_THREADS - 1) /
                      CF_THREADS,
                  nr_polarizations, (nr_time_bins + T_THREADS - 1) / T_THREADS);

  fold_and_accumulate_kernel<<<grid, block, 0, stream>>>(
      incoherent_sum, fold_accumulator, hit_counts, dm_delays,
      total_samples_elapsed, period_samples, n_bins, nr_channels,
      nr_fine_channels, nr_polarizations, nr_time_bins);
}

// normalise_fold_kernel
//
// Divides each element of fold_accumulator by its hit count in-place.
// Bins with zero hits (possible at the very start of an observation when DM
// delays push some channels before sample 0) are left as zero.
__global__ void normalise_fold_kernel(
    const float *__restrict__ fold_accumulator, float *__restrict__ fold_output,
    const uint32_t *__restrict__ hit_counts, int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const uint32_t hits = hit_counts[idx];
  if (hits > 0) {
    fold_output[idx] = fold_accumulator[idx] / static_cast<float>(hits);
  } else {
    fold_output[idx] = 0;
  }
}

inline void normalise_fold_launch(const float *fold_accumulator,
                                  float *fold_output,
                                  const uint32_t *hit_counts,
                                  int total_elements, cudaStream_t stream) {
  constexpr int THREADS = 256;
  const int blocks = (total_elements + THREADS - 1) / THREADS;

  normalise_fold_kernel<<<blocks, THREADS, 0, stream>>>(
      fold_accumulator, fold_output, hit_counts, total_elements);
}

__global__ void detect_and_convert_to_half(const float4 *__restrict__ d_input,
                                           __half *__restrict__ d_output,
                                           const int n) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  while (tid < n) {
    float4 output = d_input[tid];
    float out = sqrtf(output.x * output.x + output.y * output.y +
                      output.w * output.w + output.z * output.z);
    d_output[tid] = __float2half(out);

    tid += stride;
  };
};

inline void detect_and_convert_to_half_launch(const float4 *d_input,
                                              __half *d_output, const int n,
                                              cudaStream_t stream) {
  detect_and_convert_to_half<<<dim3(16, 1, 1), 1024, 0, stream>>>(d_input,
                                                                  d_output, n);
}

__global__ void apply_delays(const __half *__restrict__ d_input,
                             __half *__restrict__ d_output,
                             const int *__restrict__ d_fpga_delays,
                             const size_t input_stride_per_fpga,
                             const size_t nr_time_samples_per_packet,
                             const size_t total_to_copy_per_fpga,
                             const size_t total_to_copy_per_time_step) {
  __shared__ int fpga_delay;

  if (threadIdx.x == 0) {
    fpga_delay = d_fpga_delays[blockIdx.y];
  }
  __syncthreads();

  const int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int fpga_idx = blockIdx.y;

  const int base_pointer =
      fpga_idx * input_stride_per_fpga +
      (nr_time_samples_per_packet + fpga_delay) * total_to_copy_per_time_step +
      thread_idx;
  const int output_base_pointer =
      fpga_idx * total_to_copy_per_fpga + thread_idx;

  if (thread_idx < total_to_copy_per_fpga) {
    d_output[output_base_pointer] = d_input[base_pointer];
  }
};

inline void
apply_delays_launch(const __half *d_input, __half *d_output,
                    const int *d_fpga_delays, const int nr_receivers_per_packet,
                    const int nr_fpgas, const int nr_packets_for_correlation,
                    const int nr_polarizations, const int nr_channels,
                    const int nr_time_samples_per_packet, cudaStream_t stream) {

  const size_t total_to_copy_per_fpga =
      nr_receivers_per_packet * nr_channels * nr_time_samples_per_packet *
      nr_packets_for_correlation * 2 /* complex */ * nr_polarizations;
  const size_t input_stride_per_fpga =
      nr_receivers_per_packet * nr_channels * nr_time_samples_per_packet *
      (nr_packets_for_correlation + 2) * 2 * nr_polarizations;

  const size_t total_to_copy_per_time_step =
      nr_receivers_per_packet * nr_channels * 2 * nr_polarizations;

  const int blocks_needed = (total_to_copy_per_fpga + 1024 - 1) / 1024;

  const dim3 grid(blocks_needed, nr_fpgas, 1);

  apply_delays<<<grid, 1024, 0, stream>>>(
      d_input, d_output, d_fpga_delays, input_stride_per_fpga,
      nr_time_samples_per_packet, total_to_copy_per_fpga,
      total_to_copy_per_time_step);
};
