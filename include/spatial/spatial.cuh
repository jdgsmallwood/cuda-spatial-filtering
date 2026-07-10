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

// Accumulate directly from TCC CorrelatorOutput into TrimmedVisibilities
// accumulator, fusing visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed
// + accumulate_visibilities into one kernel pass.
// corr_out layout: float[n_channels][n_baselines][n_pol][n_pol][2] (CorrelatorOutput)
// accum layout:    float[n_channels][n_unpadded][n_pol][n_pol][2] (TrimmedVisibilities)
// inner_stride = n_pol * n_pol * 2 (= 8 for NR_POL=2)
void accumulate_visibilities_from_corr(const float *corr_out, float *accum,
                                       int n_channels, int n_baselines,
                                       int n_unpadded, int inner_stride,
                                       cudaStream_t stream);

// Fuses visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed into one
// kernel (no accumulation -- accumulate_visibilities stays in post_eigen so
// cuSOLVER staggering prevents concurrent atomic contention on the accumulator).
void corr_to_trimmed(const float *corr_out, float *trimmed, int n_channels,
                     int n_baselines, int n_unpadded, int inner_stride,
                     cudaStream_t stream);

__global__ void convert_float_to_half_kernel(const float *input, __half *output,
                                             const int n);

void convert_float_to_half(const float *d_input, __half *d_output, const int n,
                           cudaStream_t stream);

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_PADDED_RECEIVERS,
          size_t NR_TIMES_PER_BLOCK>
__global__ void aligned_to_corr_input_kernel(const __half *input,
                                             __half *corr_input) {
  constexpr size_t COMPLEX = 2;
  constexpr size_t NR_BLOCKS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;

  const size_t total = NR_CHANNELS * NR_PACKETS_FOR_CORRELATION *
                       NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS *
                       NR_POLARIZATIONS * COMPLEX;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  while (idx < total) {
    size_t rem = idx;
    const size_t z = rem % COMPLEX;
    rem /= COMPLEX;
    const size_t p = rem % NR_POLARIZATIONS;
    rem /= NR_POLARIZATIONS;
    const size_t receiver = rem % NR_RECEIVERS;
    rem /= NR_RECEIVERS;
    const size_t sample = rem % (NR_PACKETS_FOR_CORRELATION *
                                 NR_TIME_STEPS_PER_PACKET);
    const size_t c = rem / (NR_PACKETS_FOR_CORRELATION *
                            NR_TIME_STEPS_PER_PACKET);

    const size_t packet = sample / NR_TIME_STEPS_PER_PACKET;
    const size_t time_in_packet = sample % NR_TIME_STEPS_PER_PACKET;
    const size_t block = sample / NR_TIMES_PER_BLOCK;
    const size_t time_in_block = sample % NR_TIMES_PER_BLOCK;
    const size_t fpga = receiver / NR_RECEIVERS_PER_PACKET;
    const size_t receiver_in_packet = receiver % NR_RECEIVERS_PER_PACKET;

    const size_t input_idx =
        ((((((fpga * NR_PACKETS_FOR_CORRELATION + packet) *
                 NR_TIME_STEPS_PER_PACKET +
             time_in_packet) *
                NR_CHANNELS +
            c) *
               NR_RECEIVERS_PER_PACKET +
           receiver_in_packet) *
              NR_POLARIZATIONS +
          p) *
             COMPLEX +
         z);

    const size_t output_idx =
        ((((((c * NR_BLOCKS_FOR_CORRELATION + block) * NR_PADDED_RECEIVERS +
             receiver) *
                NR_POLARIZATIONS +
            p) *
               NR_TIMES_PER_BLOCK +
           time_in_block) *
              COMPLEX) +
         z);

    corr_input[output_idx] = input[input_idx];
    idx += stride;
  }
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_PADDED_RECEIVERS,
          size_t NR_TIMES_PER_BLOCK>
void aligned_to_corr_input(const __half *input, __half *corr_input,
                           cudaStream_t stream) {
  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");
  static_assert((NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET) %
                        NR_TIMES_PER_BLOCK ==
                    0,
                "correlation samples must divide evenly into blocks");

  constexpr size_t COMPLEX = 2;
  constexpr size_t total = NR_CHANNELS * NR_PACKETS_FOR_CORRELATION *
                           NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS *
                           NR_POLARIZATIONS * COMPLEX;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  aligned_to_corr_input_kernel<
      NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS, NR_RECEIVERS_PER_PACKET,
      NR_TIME_STEPS_PER_PACKET, NR_PACKETS_FOR_CORRELATION, NR_PADDED_RECEIVERS,
      NR_TIMES_PER_BLOCK><<<blocks, threads, 0, stream>>>(input, corr_input);
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION>
__global__ void aligned_to_col_maj_cons_kernel(const __half *input,
                                               __half *output) {
  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t COMPLEX = 2;
  constexpr size_t NR_SAMPLES =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;
  const size_t total = NR_CHANNELS * NR_POLARIZATIONS * COMPLEX * NR_SAMPLES *
                       NR_FPGA_SOURCES * NR_RECEIVERS_PER_PACKET;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  while (idx < total) {
    size_t rem = idx;
    const size_t receiver_in_packet = rem % NR_RECEIVERS_PER_PACKET;
    rem /= NR_RECEIVERS_PER_PACKET;
    const size_t fpga = rem % NR_FPGA_SOURCES;
    rem /= NR_FPGA_SOURCES;
    const size_t sample = rem % NR_SAMPLES;
    rem /= NR_SAMPLES;
    const size_t z = rem % COMPLEX;
    rem /= COMPLEX;
    const size_t p = rem % NR_POLARIZATIONS;
    const size_t c = rem / NR_POLARIZATIONS;

    const size_t packet = sample / NR_TIME_STEPS_PER_PACKET;
    const size_t time_in_packet = sample % NR_TIME_STEPS_PER_PACKET;

    const size_t input_idx =
        ((((((fpga * NR_PACKETS_FOR_CORRELATION + packet) *
                 NR_TIME_STEPS_PER_PACKET +
             time_in_packet) *
                NR_CHANNELS +
            c) *
               NR_RECEIVERS_PER_PACKET +
           receiver_in_packet) *
              NR_POLARIZATIONS +
          p) *
             COMPLEX +
         z);

    output[idx] = input[input_idx];
    idx += stride;
  }
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION>
void aligned_to_col_maj_cons(const __half *input, __half *output,
                             cudaStream_t stream) {
  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");
  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t COMPLEX = 2;
  constexpr size_t total =
      NR_CHANNELS * NR_POLARIZATIONS * COMPLEX *
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET * NR_FPGA_SOURCES *
      NR_RECEIVERS_PER_PACKET;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  aligned_to_col_maj_cons_kernel<NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
                                 NR_RECEIVERS_PER_PACKET,
                                 NR_TIME_STEPS_PER_PACKET,
                                 NR_PACKETS_FOR_CORRELATION>
      <<<blocks, threads, 0, stream>>>(input, output);
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_PADDED_RECEIVERS,
          size_t NR_TIMES_PER_BLOCK>
__global__ void packet_to_corr_input_kernel(const __half *input,
                                            __half *corr_input) {
  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t COMPLEX = 2;
  constexpr size_t NR_BLOCKS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;

  const size_t total = NR_CHANNELS * NR_PACKETS_FOR_CORRELATION *
                       NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS *
                       NR_POLARIZATIONS * COMPLEX;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  while (idx < total) {
    size_t rem = idx;
    const size_t z = rem % COMPLEX;
    rem /= COMPLEX;
    const size_t p = rem % NR_POLARIZATIONS;
    rem /= NR_POLARIZATIONS;
    const size_t receiver = rem % NR_RECEIVERS;
    rem /= NR_RECEIVERS;
    const size_t sample = rem % (NR_PACKETS_FOR_CORRELATION *
                                 NR_TIME_STEPS_PER_PACKET);
    const size_t c = rem / (NR_PACKETS_FOR_CORRELATION *
                            NR_TIME_STEPS_PER_PACKET);

    const size_t packet = sample / NR_TIME_STEPS_PER_PACKET;
    const size_t time_in_packet = sample % NR_TIME_STEPS_PER_PACKET;
    const size_t block = sample / NR_TIMES_PER_BLOCK;
    const size_t time_in_block = sample % NR_TIMES_PER_BLOCK;
    const size_t fpga = receiver / NR_RECEIVERS_PER_PACKET;
    const size_t receiver_in_packet = receiver % NR_RECEIVERS_PER_PACKET;

    const size_t input_idx =
        ((((((c * (NR_PACKETS_FOR_CORRELATION + 2) + packet) *
                 NR_FPGA_SOURCES +
             fpga) *
                NR_TIME_STEPS_PER_PACKET +
            time_in_packet) *
               NR_RECEIVERS_PER_PACKET +
           receiver_in_packet) *
              NR_POLARIZATIONS +
          p) *
             COMPLEX +
         z);

    const size_t output_idx =
        ((((((c * NR_BLOCKS_FOR_CORRELATION + block) * NR_PADDED_RECEIVERS +
             receiver) *
                NR_POLARIZATIONS +
            p) *
               NR_TIMES_PER_BLOCK +
           time_in_block) *
              COMPLEX) +
         z);

    corr_input[output_idx] = input[input_idx];
    idx += stride;
  }
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_PADDED_RECEIVERS,
          size_t NR_TIMES_PER_BLOCK>
void packet_to_corr_input(const __half *input, __half *corr_input,
                          cudaStream_t stream) {
  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");
  static_assert((NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET) %
                        NR_TIMES_PER_BLOCK ==
                    0,
                "correlation samples must divide evenly into blocks");

  constexpr size_t COMPLEX = 2;
  constexpr size_t total = NR_CHANNELS * NR_PACKETS_FOR_CORRELATION *
                           NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS *
                           NR_POLARIZATIONS * COMPLEX;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  packet_to_corr_input_kernel<
      NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS, NR_RECEIVERS_PER_PACKET,
      NR_TIME_STEPS_PER_PACKET, NR_PACKETS_FOR_CORRELATION, NR_PADDED_RECEIVERS,
      NR_TIMES_PER_BLOCK><<<blocks, threads, 0, stream>>>(input, corr_input);
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION>
__global__ void packet_to_col_maj_cons_kernel(const __half *input,
                                              __half *output) {
  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t COMPLEX = 2;
  constexpr size_t NR_SAMPLES =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;
  const size_t total = NR_CHANNELS * NR_POLARIZATIONS * COMPLEX * NR_SAMPLES *
                       NR_FPGA_SOURCES * NR_RECEIVERS_PER_PACKET;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  while (idx < total) {
    size_t rem = idx;
    const size_t receiver_in_packet = rem % NR_RECEIVERS_PER_PACKET;
    rem /= NR_RECEIVERS_PER_PACKET;
    const size_t fpga = rem % NR_FPGA_SOURCES;
    rem /= NR_FPGA_SOURCES;
    const size_t sample = rem % NR_SAMPLES;
    rem /= NR_SAMPLES;
    const size_t z = rem % COMPLEX;
    rem /= COMPLEX;
    const size_t p = rem % NR_POLARIZATIONS;
    const size_t c = rem / NR_POLARIZATIONS;

    const size_t packet = sample / NR_TIME_STEPS_PER_PACKET;
    const size_t time_in_packet = sample % NR_TIME_STEPS_PER_PACKET;

    const size_t input_idx =
        ((((((c * (NR_PACKETS_FOR_CORRELATION + 2) + packet) *
                 NR_FPGA_SOURCES +
             fpga) *
                NR_TIME_STEPS_PER_PACKET +
            time_in_packet) *
               NR_RECEIVERS_PER_PACKET +
           receiver_in_packet) *
              NR_POLARIZATIONS +
          p) *
             COMPLEX +
         z);

    output[idx] = input[input_idx];
    idx += stride;
  }
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS_FOR_CORRELATION>
void packet_to_col_maj_cons(const __half *input, __half *output,
                            cudaStream_t stream) {
  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");
  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t COMPLEX = 2;
  constexpr size_t total =
      NR_CHANNELS * NR_POLARIZATIONS * COMPLEX *
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET * NR_FPGA_SOURCES *
      NR_RECEIVERS_PER_PACKET;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  packet_to_col_maj_cons_kernel<NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
                                NR_RECEIVERS_PER_PACKET,
                                NR_TIME_STEPS_PER_PACKET,
                                NR_PACKETS_FOR_CORRELATION>
      <<<blocks, threads, 0, stream>>>(input, output);
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_BEAMS,
          size_t NR_TIME_STEPS_FOR_CORRELATION>
__global__ void beam_ccglib_to_half_output_kernel(const float *input,
                                                  __half *output) {
  constexpr size_t COMPLEX = 2;
  const size_t total = NR_CHANNELS * NR_POLARIZATIONS * NR_BEAMS *
                       NR_TIME_STEPS_FOR_CORRELATION * COMPLEX;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  while (idx < total) {
    size_t rem = idx;
    const size_t z = rem % COMPLEX;
    rem /= COMPLEX;
    const size_t s = rem % NR_TIME_STEPS_FOR_CORRELATION;
    rem /= NR_TIME_STEPS_FOR_CORRELATION;
    const size_t m = rem % NR_BEAMS;
    rem /= NR_BEAMS;
    const size_t p = rem % NR_POLARIZATIONS;
    const size_t c = rem / NR_POLARIZATIONS;

    const size_t input_idx =
        (((((c * NR_POLARIZATIONS + p) * COMPLEX + z) * NR_BEAMS + m) *
              NR_TIME_STEPS_FOR_CORRELATION) +
         s);
    output[idx] = __float2half(input[input_idx]);
    idx += stride;
  }
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_BEAMS,
          size_t NR_TIME_STEPS_FOR_CORRELATION>
void beam_ccglib_to_half_output(const float *input, __half *output,
                                cudaStream_t stream) {
  constexpr size_t COMPLEX = 2;
  constexpr size_t total = NR_CHANNELS * NR_POLARIZATIONS * NR_BEAMS *
                           NR_TIME_STEPS_FOR_CORRELATION * COMPLEX;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  beam_ccglib_to_half_output_kernel<NR_CHANNELS, NR_POLARIZATIONS, NR_BEAMS,
                                    NR_TIME_STEPS_FOR_CORRELATION>
      <<<blocks, threads, 0, stream>>>(input, output);
}

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS>
__global__ void scale_and_convert_to_half_kernel(
    const char2 *__restrict__ d_input, const int16_t *__restrict__ d_scale,
    const float2 *__restrict__ d_gains, __half2 *__restrict__ d_output) {

  // input format is
  // int8_t[channel][packet][fpga][time][receiver_in_pkt][pol][complex]

  int channel_idx = blockIdx.x % NR_CHANNELS;
  int packet_idx = blockIdx.x / NR_CHANNELS;
  int fpga_idx = blockIdx.y;

  static_assert(NR_RECEIVERS % NR_RECEIVERS_PER_PACKET == 0,
                "NR_RECEIVERS must be divisible by NR_RECEIVERS_PER_PACKET");

  constexpr size_t ELEMS_PER_TIME = NR_POLARIZATIONS * NR_RECEIVERS_PER_PACKET;

  if (threadIdx.x >= ELEMS_PER_TIME) {
    return;
  }

  int pol_idx = threadIdx.x % NR_POLARIZATIONS;
  int recv_in_pkt = threadIdx.x / NR_POLARIZATIONS;
  int receiver_pol_idx = recv_in_pkt * NR_POLARIZATIONS + pol_idx;

  int scale_ptr = channel_idx * NR_PACKETS * NR_RECEIVERS * NR_POLARIZATIONS +
                  packet_idx * NR_RECEIVERS * NR_POLARIZATIONS +
                  receiver_pol_idx;
  int scale_val_int = static_cast<int>(d_scale[scale_ptr]);

  int gain_ptr = channel_idx * NR_RECEIVERS * NR_POLARIZATIONS +
                 fpga_idx * NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS +
                 receiver_pol_idx;
  float2 gain = __ldg(&d_gains[gain_ptr]);

  size_t nr_fpga = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;

  int input_base = channel_idx * NR_PACKETS * nr_fpga *
                       NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS_PER_PACKET *
                       NR_POLARIZATIONS +
                   packet_idx * nr_fpga * NR_TIME_STEPS_PER_PACKET *
                       NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS +
                   fpga_idx * NR_TIME_STEPS_PER_PACKET *
                       NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS +
                   receiver_pol_idx;

#pragma unroll 4
  for (int time_step = 0; time_step < NR_TIME_STEPS_PER_PACKET; ++time_step) {
    int ptr = input_base +
              time_step * NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS;

    char2 sample = d_input[ptr];
    int val_real = static_cast<int>(sample.x) * scale_val_int;
    int val_imag = static_cast<int>(sample.y) * scale_val_int;

    float2 float_val{static_cast<float>(val_real),
                     static_cast<float>(val_imag)};

    float2 gain_applied_val{float_val.x * gain.x - float_val.y * gain.y,
                            float_val.x * gain.y + float_val.y * gain.x};

    d_output[ptr] = __float22half2_rn(gain_applied_val);
  }
};

template <int N> constexpr bool dependent_false = false;

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_RECEIVERS,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_TIME_STEPS_PER_PACKET,
          size_t NR_PACKETS>
void scale_and_convert_to_half(const char2 *d_input, const int16_t *d_scale,
                               const float2 *d_gains, __half2 *d_output,
                               cudaStream_t stream) {

  constexpr size_t NR_FPGA_SOURCES = NR_RECEIVERS / NR_RECEIVERS_PER_PACKET;
  constexpr size_t ELEMS_PER_TIME = NR_POLARIZATIONS * NR_RECEIVERS_PER_PACKET;
  constexpr size_t THREADS = ((ELEMS_PER_TIME + 31) / 32) * 32;

  scale_and_convert_to_half_kernel<
      NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS, NR_RECEIVERS_PER_PACKET,
      NR_TIME_STEPS_PER_PACKET, NR_PACKETS>
      <<<dim3(NR_CHANNELS * NR_PACKETS, NR_FPGA_SOURCES, 1),
         dim3(THREADS, 1, 1), 0, stream>>>(d_input, d_scale, d_gains,
                                           d_output);
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

__device__ inline float
sorted_percentile_linear_device(const float *sorted_values, int count,
                                float quantile) {
  if (count <= 0) {
    return 0.0f;
  }
  if (count == 1) {
    return sorted_values[0];
  }
  const float clamped = fminf(1.0f, fmaxf(0.0f, quantile));
  const float position = clamped * static_cast<float>(count - 1);
  const int lower = static_cast<int>(floorf(position));
  const int upper = static_cast<int>(ceilf(position));
  if (lower == upper) {
    return sorted_values[lower];
  }
  const float fraction = position - static_cast<float>(lower);
  return sorted_values[lower] +
         fraction * (sorted_values[upper] - sorted_values[lower]);
}

__global__ void detectSignalEigenmodeCountsKernel(
    const float *__restrict__ d_eigenvalues, int32_t *__restrict__ d_counts,
    int N, int total_batches, float delta) {
  const int batch = blockIdx.x;
  if (batch >= total_batches || threadIdx.x != 0) {
    return;
  }

  const float *eigs = d_eigenvalues + batch * N;
  const float p20 = sorted_percentile_linear_device(eigs, N, 0.2f);
  const float p50 = sorted_percentile_linear_device(eigs, N, 0.5f);
  const float p80 = sorted_percentile_linear_device(eigs, N, 0.8f);
  const float sigma_noise = (p80 - p20) / (2.0f * 0.8416f);
  const float threshold = p50 + delta * sigma_noise;

  int detected = 0;
  for (int i = 0; i < N; ++i) {
    if (eigs[i] > threshold) {
      ++detected;
    }
  }
  d_counts[batch] = detected;
}

void detectSignalEigenmodeCounts(const float *d_eigenvalues, int32_t *d_counts,
                                 int N, int batches, float delta,
                                 cudaStream_t stream) {
  detectSignalEigenmodeCountsKernel<<<batches, 32, 0, stream>>>(
      d_eigenvalues, d_counts, N, batches, delta);
}

__global__ void computeShrinkScaleFactorsKernel(
    const float *__restrict__ d_eigenvalues,
    const int32_t *__restrict__ d_counts, float *__restrict__ d_scales, int N,
    int total_batches) {
  const int batch = blockIdx.x;
  const int k = threadIdx.x;
  if (batch >= total_batches || k >= N) {
    return;
  }

  const float *eigs = d_eigenvalues + batch * N;
  float *scales = d_scales + batch * N;
  const int detected = max(0, min(N, static_cast<int>(d_counts[batch])));

  __shared__ float lambda_bar;
  if (k == 0) {
    lambda_bar = 0.0f;
    if (detected < N) {
      for (int i = 0; i < N - detected; ++i) {
        lambda_bar += eigs[i];
      }
      lambda_bar /= static_cast<float>(N - detected);
    }
  }
  __syncthreads();

  if (k < N - detected) {
    scales[k] = 1.0f;
  } else {
    const float eig = eigs[k];
    scales[k] = (eig > 0.0f) ? lambda_bar / eig : 0.0f;
  }
}

void computeShrinkScaleFactors(const float *d_eigenvalues,
                               const int32_t *d_counts, float *d_scales, int N,
                               int batches, cudaStream_t stream) {
  computeShrinkScaleFactorsKernel<<<batches, N, 0, stream>>>(
      d_eigenvalues, d_counts, d_scales, N, batches);
}

__global__ void buildIdentityMinusProjectionKernel(
    const float2 *__restrict__ d_eigenvectors,
    const int32_t *__restrict__ d_counts, __half2 *__restrict__ d_output, int N,
    int total_batches) {
  const int batch = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = N * N;
  if (batch >= total_batches || idx >= total_elements) {
    return;
  }

  const int row = idx % N;
  const int col = idx / N;
  const int detected = max(0, min(N, static_cast<int>(d_counts[batch])));
  const int col_offset = N - detected;
  const int base = batch * total_elements;

  float2 sum = make_float2(0.0f, 0.0f);
  for (int k = 0; k < detected; ++k) {
    const int eig_col = col_offset + k;
    const float2 u_row = d_eigenvectors[base + eig_col * N + row];
    const float2 u_col = d_eigenvectors[base + eig_col * N + col];
    sum.x += u_row.x * u_col.x + u_row.y * u_col.y;
    sum.y += u_row.y * u_col.x - u_row.x * u_col.y;
  }

  const float real = ((row == col) ? 1.0f : 0.0f) - sum.x;
  const float imag = -sum.y;
  d_output[base + idx] = __float22half2_rn(make_float2(real, imag));
}

void buildIdentityMinusProjectionFromEigenvectors(
    const float2 *d_eigenvectors, const int32_t *d_counts, __half2 *d_output,
    int N, int batches, cudaStream_t stream) {
  const int total_elements = N * N;
  const int threadsPerBlock = 256;
  const int blocksPerGrid =
      (total_elements + threadsPerBlock - 1) / threadsPerBlock;
  buildIdentityMinusProjectionKernel<<<dim3(blocksPerGrid, batches, 1),
                                       threadsPerBlock, 0, stream>>>(
      d_eigenvectors, d_counts, d_output, N, batches);
}

// Copies V_in → V_out, scaling each column j by d[batch*N + j].
// V is stored column-major (cuSOLVER convention): element (row, col) lives at
// flat index col*N + row within a single NxN batch slice.
__global__ void scaleEigenvectorColumnsKernel(const float2 *__restrict__ V_in,
                                              float2 *__restrict__ V_out,
                                              const float *__restrict__ d,
                                              int N, int total_batches) {
  int batch = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // flat index within N*N
  int total_elements = N * N;
  if (idx < total_elements && batch < total_batches) {
    int col = idx / N;
    float scale = d[batch * N + col];
    int flat_idx = batch * total_elements + idx;
    V_out[flat_idx] = make_float2(V_in[flat_idx].x * scale, V_in[flat_idx].y * scale);
  }
}

void scaleEigenvectorColumns(const float2 *V_in, float2 *V_out,
                              const float *d_scales, int N, int batches,
                              cudaStream_t stream) {
  int total_elements = N * N;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
  scaleEigenvectorColumnsKernel<<<dim3(blocksPerGrid, batches, 1),
                                   threadsPerBlock, 0, stream>>>(
      V_in, V_out, d_scales, N, batches);
}

// Extracts sqrt(diag(V)) per batch into d_sqrt_diag[batch*N + k].
// Guards against zero or negative diagonal with a floor of 1e-10.
// Column-major layout: diagonal element (k,k) lives at batch*N*N + k*N + k.
__global__ void extractDiagonalSqrtKernel(const float2 *__restrict__ V,
                                           float *__restrict__ d_sqrt_diag,
                                           int N, int total_batches) {
  const int batch = blockIdx.x;
  const int k = threadIdx.x;
  if (batch >= total_batches || k >= N)
    return;
  const float diag_re = V[batch * N * N + k * N + k].x;
  d_sqrt_diag[batch * N + k] = sqrtf(fmaxf(diag_re, 1e-10f));
}

// Whitens V in-place: V_ij /= d_sqrt_diag[i] * d_sqrt_diag[j].
__global__ void applyDiagonalWhiteningKernel(float2 *__restrict__ V,
                                              const float *__restrict__ d_sqrt_diag,
                                              int N, int total_batches) {
  const int batch = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = N * N;
  if (batch >= total_batches || idx >= total_elements)
    return;
  const int row = idx % N;
  const int col = idx / N;
  const float denom = d_sqrt_diag[batch * N + row] * d_sqrt_diag[batch * N + col];
  const float2 v = V[batch * total_elements + idx];
  V[batch * total_elements + idx] = make_float2(v.x / denom, v.y / denom);
}

// Transforms V in-place by D^{1/2} then re-normalises each column to unit norm.
// Used after eigendecomposition of the whitened matrix to restore eigenvectors
// to the original (unwhitened) space with correct directions.
__global__ void unwhitenAndNormalizeKernel(float2 *__restrict__ V,
                                            const float *__restrict__ d_sqrt_diag,
                                            int N, int total_batches) {
  extern __shared__ float s_norm_sq[];
  const int col = blockIdx.x;
  const int batch = blockIdx.y;
  const int row = threadIdx.x;
  if (batch >= total_batches || col >= N || row >= N)
    return;

  const float scale = d_sqrt_diag[batch * N + row];
  float2 elem = V[batch * N * N + col * N + row];
  elem.x *= scale;
  elem.y *= scale;

  s_norm_sq[row] = elem.x * elem.x + elem.y * elem.y;
  __syncthreads();

  if (row == 0) {
    float total = 0.0f;
    for (int i = 0; i < N; ++i)
      total += s_norm_sq[i];
    s_norm_sq[0] = total;
  }
  __syncthreads();

  const float inv_norm = (s_norm_sq[0] > 0.0f) ? rsqrtf(s_norm_sq[0]) : 0.0f;
  V[batch * N * N + col * N + row] = make_float2(elem.x * inv_norm, elem.y * inv_norm);
}

// Whiten the batched Hermitian matrix V in-place and store sqrt(diag(V)) to
// d_sqrt_diag for later use by unwhitenEigenvectors.
void diagonalWhiten(float2 *V, float *d_sqrt_diag, int N, int batches,
                    cudaStream_t stream) {
  extractDiagonalSqrtKernel<<<batches, N, 0, stream>>>(V, d_sqrt_diag, N,
                                                        batches);
  const int total_elements = N * N;
  const int threadsPerBlock = 256;
  const int blocksPerGrid =
      (total_elements + threadsPerBlock - 1) / threadsPerBlock;
  applyDiagonalWhiteningKernel<<<dim3(blocksPerGrid, batches), threadsPerBlock,
                                  0, stream>>>(V, d_sqrt_diag, N, batches);
}

// Un-whiten and re-normalise all eigenvector columns of V (output of cuSOLVER
// on the whitened matrix).  Restores correct null directions for null mode.
void unwhitenEigenvectors(float2 *V, const float *d_sqrt_diag, int N,
                           int batches, cudaStream_t stream) {
  unwhitenAndNormalizeKernel<<<dim3(N, batches), N, N * sizeof(float), stream>>>(
      V, d_sqrt_diag, N, batches);
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
  static constexpr size_t HALVES_PER_VECTOR = sizeof(uint4) / sizeof(__half);

  const size_t vector_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t vector_count = total_to_copy_per_fpga / HALVES_PER_VECTOR;
  const int fpga_idx = blockIdx.y;
  const int fpga_delay = d_fpga_delays[fpga_idx];

  const size_t input_base =
      static_cast<size_t>(fpga_idx) * input_stride_per_fpga +
      (nr_time_samples_per_packet + fpga_delay) * total_to_copy_per_time_step;
  const size_t output_base =
      static_cast<size_t>(fpga_idx) * total_to_copy_per_fpga;

  const bool aligned =
      (input_base % HALVES_PER_VECTOR == 0) &&
      (output_base % HALVES_PER_VECTOR == 0);

  if (aligned) {
    const uint4 *__restrict__ input_vec =
        reinterpret_cast<const uint4 *>(d_input + input_base);
    uint4 *__restrict__ output_vec =
        reinterpret_cast<uint4 *>(d_output + output_base);

    if (vector_idx < vector_count) {
      output_vec[vector_idx] = input_vec[vector_idx];
    }

  } else {
    const size_t scalar_start = vector_idx * HALVES_PER_VECTOR;
#pragma unroll
    for (size_t i = 0; i < HALVES_PER_VECTOR; ++i) {
      const size_t scalar_idx = scalar_start + i;
      if (scalar_idx < total_to_copy_per_fpga) {
        d_output[output_base + scalar_idx] = d_input[input_base + scalar_idx];
      }
    }
  }

  if (aligned && vector_idx == 0) {
    for (size_t i = vector_count * HALVES_PER_VECTOR;
         i < total_to_copy_per_fpga; ++i) {
      d_output[output_base + i] = d_input[input_base + i];
    }
  }
};

inline void
apply_delays_launch(const __half *d_input, __half *d_output,
                    const int *d_fpga_delays, const int nr_receivers_per_packet,
                    const int nr_fpgas, const int nr_packets_for_correlation,
                    const int nr_polarizations, const int nr_channels,
                    const int nr_time_samples_per_packet, cudaStream_t stream) {
  /* input format of d_input is __half[FPGA][PACKET +
   * 2][TIME][CHANNEL][RECEIVER][POL][COMPLEX] format of d_output is
   * __half[FPGA][PACKET][TIME][CHANNEL][RECEIVER][POL][COMPLEX]
   */

  const size_t total_to_copy_per_fpga =
      nr_receivers_per_packet * nr_channels * nr_time_samples_per_packet *
      nr_packets_for_correlation * 2 /* complex */ * nr_polarizations;
  const size_t input_stride_per_fpga =
      nr_receivers_per_packet * nr_channels * nr_time_samples_per_packet *
      (nr_packets_for_correlation + 2) * 2 * nr_polarizations;

  const size_t total_to_copy_per_time_step =
      nr_receivers_per_packet * nr_channels * 2 * nr_polarizations;

  static constexpr int THREADS = 256;
  static constexpr size_t HALVES_PER_VECTOR = sizeof(uint4) / sizeof(__half);
  const size_t vector_count =
      (total_to_copy_per_fpga + HALVES_PER_VECTOR - 1) / HALVES_PER_VECTOR;
  const int blocks_needed = (vector_count + THREADS - 1) / THREADS;

  const dim3 grid(blocks_needed, nr_fpgas, 1);

  apply_delays<<<grid, THREADS, 0, stream>>>(
      d_input, d_output, d_fpga_delays, input_stride_per_fpga,
      nr_time_samples_per_packet, total_to_copy_per_fpga,
      total_to_copy_per_time_step);
};

__global__ void
sum_fft_over_packets(const float2 *__restrict__ d_input,
                     float *__restrict__ d_output, const size_t nr_channels,
                     const size_t nr_beams, const size_t nr_polarizations,
                     const size_t nr_packets, const size_t nr_fft_freqs) {

  __shared__ float final_sum[128];

  const size_t linear_thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
  const size_t channel_idx = blockIdx.y % nr_channels;
  const size_t pol_idx = blockIdx.y / nr_channels;
  const size_t beam_idx = blockIdx.z;
  const size_t packet_idx = blockIdx.x * blockDim.y + threadIdx.y;

  const int input_base_pointer =
      channel_idx * nr_polarizations * nr_beams * nr_packets * nr_fft_freqs +
      pol_idx * nr_beams * nr_packets * nr_fft_freqs +
      beam_idx * nr_packets * nr_fft_freqs + packet_idx * nr_fft_freqs +
      threadIdx.x;
  const int output_pointer =
      channel_idx * nr_polarizations * nr_beams * nr_fft_freqs +
      pol_idx * nr_beams * nr_fft_freqs + beam_idx * nr_fft_freqs + threadIdx.x;

  if (linear_thread_idx < 128) {
    final_sum[linear_thread_idx] = 0;
  }

  __syncthreads();

  if (packet_idx < nr_packets) {
    float2 val = d_input[input_base_pointer];
    float mag = sqrtf(val.x * val.x + val.y * val.y);
    atomicAdd(&final_sum[threadIdx.x], mag);
  }
  __syncthreads();

  if (threadIdx.y == 0) {
    atomicAdd(&d_output[output_pointer], final_sum[threadIdx.x]);
  }
};

inline void sum_fft_over_packets_launch(const float2 *d_input, float *d_output,
                                        const size_t nr_beams,
                                        const size_t nr_channels,
                                        const size_t nr_polarizations,
                                        const size_t nr_fft_freqs,
                                        const size_t nr_packets_to_sum,
                                        cudaStream_t stream

) {
  const size_t packets_per_block = 1024 / nr_fft_freqs;
  const size_t number_blocks_required =
      (nr_packets_to_sum + packets_per_block - 1) / packets_per_block;

  dim3 grid(number_blocks_required, nr_channels * nr_polarizations, nr_beams);
  dim3 threads(nr_fft_freqs, packets_per_block, 1);
  sum_fft_over_packets<<<grid, threads, 0, stream>>>(
      d_input, d_output, nr_channels, nr_beams, nr_polarizations,
      nr_packets_to_sum, nr_fft_freqs);
};
