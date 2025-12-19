#pragma once
#include "spatial/packet_formats.hpp"
#include "spatial/spatial.hpp"
#include <chrono>
#include <complex>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <vector>

#include "ccglib/common/precision.h"
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.cuh"
#include "spatial/tensor.hpp"
#include <atomic>
#include <ccglib/ccglib.hpp>
#include <ccglib/common/complex_order.h>
#include <ccglib/transpose/transpose.h>
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <cusolverDn.h>
#include <highfive/highfive.hpp>
#include <iostream>
#include <libtcc/Correlator.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

template <typename T> struct BeamWeightsT {
  std::complex<__half> weights[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
                              [T::NR_RECEIVERS];
};

struct BufferReleaseContext {
  ProcessorStateBase *state;
  size_t buffer_index;
  bool dummy_run;
};

struct OutputTransferCompleteContext {
  std::shared_ptr<Output> output;
  size_t block_index;
};

struct EigenOutputTransferCompleteContext {
  std::shared_ptr<Output> output;
  size_t block_index;
};

// Static function to be called by cudaLaunchHostFunc
static void release_buffer_host_func(void *data) {

  auto *ctx = static_cast<BufferReleaseContext *>(data);
  if (!ctx->dummy_run) {
    // LOG_DEBUG("Releasing buffer #{}", ctx->buffer_index);
    ctx->state->release_buffer(ctx->buffer_index);
  }
  delete ctx;
}

static void output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  LOG_DEBUG("Marking beam data output transfer for block #{} complete",
            ctx->block_index);
  ctx->output->register_beam_data_transfer_complete(ctx->block_index);
  delete ctx;
}

static void output_visibilities_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  LOG_INFO("Marking output transfer for block #{} complete", ctx->block_index);
  ctx->output->register_visibilities_transfer_complete(ctx->block_index);
  delete ctx;
}

static void eigen_output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);

  ctx->output->register_eigendecomposition_data_transfer_complete(
      ctx->block_index);
  delete ctx;
}

static void fft_output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);

  ctx->output->register_fft_transfer_complete(ctx->block_index);
  delete ctx;
}

template <typename T> class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;
  static constexpr int NR_EIGENVALUES =
      T::NR_PADDED_RECEIVERS * T::NR_CHANNELS * T::NR_POLARIZATIONS;
  static constexpr int NR_CORRELATED_BLOCKS_TO_ACCUMULATE =
      T::NR_CORRELATED_BLOCKS_TO_ACCUMULATE;

  inline static const __half alpha = __float2half(1.0f);
  static constexpr float alpha_32 = 1.0f;

  using CorrelatorInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using CorrelatorOutput =
      float[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
           [T::NR_POLARIZATIONS][COMPLEX];

  using Visibilities =
      std::complex<float>[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS];

  using TrimmedVisibilities =
      std::complex<float>[T::NR_CHANNELS][NR_UNPADDED_BASELINES]
                         [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS];

  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];
  using Eigenvalues = float[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                           [T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  using BeamformerInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using HalfBeamformerOutput =
      __half[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using BeamWeights = BeamWeightsT<T>;

  // a = unpadded baselines
  // b = block
  // c = channel
  // d = padded receivers
  // f = fpga
  // l = baseline
  // m = beam
  // n = receivers per packet
  // o = packets for correlation
  // p = polarization
  // q = second polarization
  // r = receiver
  // s = time consolidated <block x time>
  // t = times per block
  // u = time steps per packet
  // z = complex

  inline static const std::vector<int> modePacket{'c', 'o', 'f', 'u',
                                                  'n', 'p', 'z'};
  // o and u need to end up together and will be interpreted as b x t in the
  // next transformation. Similarly f x n = r in next transformation.
  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'l', 'c', 'p', 'q',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'a', 'c', 'p',
                                                                  'q', 'z'};

  inline static const std::vector<int> modeVisCorrTrimmed{'c', 'a', 'p', 'q',
                                                          'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'q', 'a', 'z'};
  // Convert back to interleaved instead of planar output.
  // This is not strictly necessary to do in the pipeline.
  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  inline static const std::vector<int> modeBeamOutput{'c', 'p', 'm', 's', 'z'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm',
                                                         'r'};

  inline static const std::unordered_map<int, int64_t> extent = {

      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'f', T::NR_FPGA_SOURCES},
      {'l', NR_BASELINES},
      {'m', T::NR_BEAMS},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'o', T::NR_PACKETS_FOR_CORRELATION},
      {'p', T::NR_POLARIZATIONS},
      {'q', T::NR_POLARIZATIONS}, // 2nd polarization for baselines
      {'r', T::NR_RECEIVERS},
      {'s', NR_BLOCKS_FOR_CORRELATION *NR_TIMES_PER_BLOCK},
      {'t', NR_TIMES_PER_BLOCK},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'z', 2}, // real, imaginary

  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;
  std::atomic<int> num_integrated_units_processed;
  std::atomic<int> num_correlation_units_integrated;

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry,
      d_samples_scaled;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding, d_samples_cufft_input, d_samples_cufft_output;
  std::vector<typename T::PaddedPacketSamplesType *> d_samples_padded,
      d_samples_reord; // This is not the right type for reord
                       // - but it will do I guess. Size will be correct.
  std::vector<CorrelatorInput *> d_correlator_input;
  std::vector<CorrelatorOutput *> d_correlator_output;

  std::vector<Visibilities *> d_visibilities_converted;
  std::vector<TrimmedVisibilities *> d_visibilities_baseline,
      d_visibilities_trimmed_baseline, d_visibilities_trimmed,
      d_visibilities_permuted;
  std::vector<DecompositionVisibilities *> d_decomposition_visibilities_input;
  std::vector<Eigenvalues *> d_eigenvalues;
  TrimmedVisibilities *d_visibilities_accumulator;
  std::vector<BeamformerInput *> d_beamformer_input;
  std::vector<BeamformerOutput *> d_beamformer_output, d_beamformer_data_output;
  std::vector<HalfBeamformerOutput *> d_beamformer_data_output_half;
  std::vector<__half *> d_samples_consolidated, d_samples_consolidated_col_maj,
      d_weights, d_weights_updated, d_weights_permuted;
  std::vector<typename T::PacketScalesType *> d_scales;

  BeamWeights *h_weights;

  int visibilities_start_seq_num;
  int visibilities_end_seq_num;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;
  std::vector<cufftHandle> fft_plan;
  std::vector<void *> d_cufft_work_area;
  std::vector<int *> d_cusolver_info;
  std::vector<cusolverDnParams_t> cusolver_params;
  std::vector<void *> d_cusolver_work_area;
  std::vector<void *> h_cusolver_work_area;
  std::vector<size_t> d_cusolver_work_area_size;
  std::vector<size_t> h_cusolver_work_area_size;
  cusolverEigMode_t cusolver_jobz;
  cublasFillMode_t cusolver_uplo;
  std::vector<cusolverDnHandle_t> cusolver_handle;
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS;

public:
  static constexpr size_t NR_BENCHMARKING_RUNS = 100;
  size_t benchmark_runs_done = 0;
  cudaEvent_t start_run[NR_BENCHMARKING_RUNS], stop_run[NR_BENCHMARKING_RUNS];
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {
    using clock = std::chrono::high_resolution_clock;

    auto cpu_start_total = clock::now();

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
    }
    visibilities_missing_packets += packet_data->get_num_missing_packets();

    // Record GPU start event
    auto cpu_start = clock::now();
    cudaEventRecord(start_run[benchmark_runs_done], streams[current_buffer]);
    auto cpu_end = clock::now();
    LOG_DEBUG("CPU time for cudaEventRecord start_run: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // d_samples_entry memcpy
    cpu_start = clock::now();
    cudaMemcpyAsync(d_samples_entry[current_buffer],
                    (void *)packet_data->get_samples_ptr(),
                    packet_data->get_samples_elements_size(), cudaMemcpyDefault,
                    streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for d_samples_entry cudaMemcpyAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // d_scales memcpy
    cpu_start = clock::now();
    cudaMemcpyAsync(d_scales[current_buffer],
                    (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for d_scales cudaMemcpyAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // BufferReleaseContext + host function
    cpu_start = clock::now();
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    // do in separate thread - no need to tie up GPU pipeline.
    CUDA_CHECK(cudaLaunchHostFunc(streams[num_buffers + current_buffer],
                                  release_buffer_host_func, ctx));
    cpu_end = clock::now();
    LOG_DEBUG(
        "CPU time for BufferReleaseContext alloc + cudaLaunchHostFunc: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                              cpu_start)
            .count());

    // scale_and_convert_to_half kernel
    cpu_start = clock::now();
    scale_and_convert_to_half<
        typename T::InputPacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfInputPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
        T::NR_TIME_STEPS_PER_PACKET, T::NR_PACKETS_FOR_CORRELATION>(
        (typename T::InputPacketSamplesPlanarType *)
            d_samples_entry[current_buffer],
        d_scales[current_buffer],
        (typename T::HalfInputPacketSamplesPlanarType *)
            d_samples_half[current_buffer],
        streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for scale_and_convert_to_half launch: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // tensor_16.runPermutation "packetToFPGA"
    cpu_start = clock::now();
    tensor_16.runPermutation(
        "packetToPadding", alpha, (__half *)d_samples_half[current_buffer],
        (__half *)d_samples_padding[current_buffer], streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_16.runPermutation packetToFPGA: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    tensor_16.runPermutation("packetToCUFFTInput", alpha,
                             (__half *)d_samples_half[current_buffer],
                             (__half *)d_samples_cufft_input[current_buffer],
                             streams[current_buffer]);

    CUFFT_CHECK(cufftXtExec(
        fft_plan[current_buffer], (void *)d_samples_cufft_input[current_buffer],
        (void *)d_samples_cufft_output[current_buffer], CUFFT_FORWARD));
    // cudaMemcpyAsync for padding
    cpu_start = clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_samples_padded[current_buffer],
                               d_samples_padding[current_buffer],
                               sizeof(typename T::HalfPacketSamplesType),
                               cudaMemcpyDefault, streams[current_buffer]));
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for d_samples_padded cudaMemcpyAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // cudaMemsetAsync
    cpu_start = clock::now();
    CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<char *>(d_samples_padded[current_buffer]) +
            sizeof(typename T::HalfPacketSamplesType),
        0,
        sizeof(typename T::PaddedPacketSamplesType) -
            sizeof(typename T::HalfPacketSamplesType),
        streams[current_buffer]));
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for cudaMemsetAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // tensor_16.runPermutation "paddedToCorrInput"
    cpu_start = clock::now();
    tensor_16.runPermutation(
        "paddedToCorrInput", alpha, (__half *)d_samples_padded[current_buffer],
        (__half *)d_correlator_input[current_buffer], streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_16.runPermutation paddedToCorrInput: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // correlator.launchAsync
    cpu_start = clock::now();
    correlator.launchAsync((CUstream)streams[current_buffer],
                           (CUdeviceptr)d_correlator_output[current_buffer],
                           (CUdeviceptr)d_correlator_input[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for correlator.launchAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    tensor_32.runPermutation("visCorrToBaseline", alpha_32,
                             (float *)d_correlator_output[current_buffer],
                             (float *)d_visibilities_baseline[current_buffer],
                             streams[current_buffer]);
    CUDA_CHECK(cudaMemcpyAsync(d_visibilities_trimmed_baseline[current_buffer],
                               d_visibilities_baseline[current_buffer],
                               sizeof(TrimmedVisibilities), cudaMemcpyDefault,
                               streams[current_buffer]));

    tensor_32.runPermutation(
        "visBaselineTrimmedToTrimmed", alpha_32,
        (float *)d_visibilities_trimmed_baseline[current_buffer],
        (float *)d_visibilities_trimmed[current_buffer],
        streams[current_buffer]);
    // tensor_32.runPermutation "visCorrToDecomp"
    cpu_start = clock::now();
    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)d_visibilities_trimmed[current_buffer],
                             (float *)d_visibilities_permuted[current_buffer],
                             streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_32.runPermutation visCorrToDecomp: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    unpack_triangular_baseline_batch_launch<cuComplex>(
        (cuComplex *)d_visibilities_permuted[current_buffer],
        (cuComplex *)d_decomposition_visibilities_input[current_buffer],
        T::NR_RECEIVERS,
        T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS,
        T::NR_CHANNELS, streams[current_buffer]);

    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        cusolver_handle[current_buffer], cusolver_params[current_buffer],
        cusolver_jobz, cusolver_uplo, T::NR_RECEIVERS, CUDA_C_32F,
        (void *)d_decomposition_visibilities_input[current_buffer],
        T::NR_RECEIVERS, CUDA_R_32F, (void *)d_eigenvalues[current_buffer],
        CUDA_C_32F, d_cusolver_work_area[current_buffer],
        d_cusolver_work_area_size[current_buffer],
        h_cusolver_work_area[current_buffer],
        h_cusolver_work_area_size[current_buffer],
        d_cusolver_info[current_buffer], CUSOLVER_BATCH_SIZE));

    // accumulate_visibilities (CPU wrapper)
    cpu_start = clock::now();
    accumulate_visibilities((float *)d_visibilities_trimmed[current_buffer],
                            (float *)d_visibilities_accumulator,
                            2 * NR_UNPADDED_BASELINES * T::NR_POLARIZATIONS *
                                T::NR_POLARIZATIONS * T::NR_CHANNELS,
                            streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for accumulate_visibilities: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    tensor_16.runPermutation("packetToPlanar", alpha,
                             (__half *)d_samples_half[current_buffer],
                             (__half *)d_samples_consolidated[current_buffer],
                             streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_16.runPermutation packetToPlanar: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    tensor_16.runPermutation(
        "consToColMajCons", alpha,
        (__half *)d_samples_consolidated[current_buffer],
        (__half *)d_samples_consolidated_col_maj[current_buffer],
        streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_16.runPermutation consToColMajCons: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    update_weights(d_weights[current_buffer], d_weights_updated[current_buffer],
                   T::NR_BEAMS, T::NR_RECEIVERS, T::NR_CHANNELS,
                   T::NR_POLARIZATIONS, (float *)d_eigenvalues[current_buffer],
                   (float *)d_visibilities_converted[current_buffer],
                   streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for update_weights: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights_updated[current_buffer],
                             (__half *)d_weights_permuted[current_buffer],
                             streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG(
        "CPU time for tensor_16.runPermutation weightsInputToCCGLIB: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                              cpu_start)
            .count());

    cpu_start = clock::now();
    (*gemm_handles[current_buffer])
        .Run((CUdeviceptr)d_weights_permuted[current_buffer],
             (CUdeviceptr)d_samples_consolidated_col_maj[current_buffer],
             (CUdeviceptr)d_beamformer_output[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for GEMM run: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)d_beamformer_output[current_buffer],
                             (float *)d_beamformer_data_output[current_buffer],
                             streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for tensor_32.runPermutation beamCCGLIBToOutput: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    convert_float_to_half(
        (float *)d_beamformer_data_output[current_buffer],
        (__half *)d_beamformer_data_output_half[current_buffer],
        2 * T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_BEAMS *
            NR_TIME_STEPS_FOR_CORRELATION,
        streams[current_buffer]);

    cpu_start = clock::now();
    cudaEventRecord(stop_run[benchmark_runs_done], streams[current_buffer]);
    cpu_end = clock::now();
    LOG_DEBUG("CPU time for cudaEventRecord stop_run: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Output handling
    if (output_ != nullptr && !dummy_run) {
      cpu_start = clock::now();
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      size_t eigenvalue_block_num =
          output_->register_eigendecomposition_data_block(start_seq_num,
                                                          end_seq_num);
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);
      void *landing_pointer = output_->get_beam_data_landing_pointer(block_num);
      cudaMemcpyAsync(landing_pointer,
                      d_beamformer_data_output_half[current_buffer],
                      sizeof(HalfBeamformerOutput), cudaMemcpyDefault,
                      streams[current_buffer]);
      cpu_end = clock::now();
      LOG_DEBUG("CPU time for output registration + copying: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());
      cpu_start = clock::now();
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};
      cudaLaunchHostFunc(streams[current_buffer],
                         output_transfer_complete_host_func, output_ctx);
      cpu_end = clock::now();
      LOG_DEBUG("CPU time for host function launch: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());

      // memcpy arrivals
      cpu_start = clock::now();
      bool *arrivals_output_pointer =
          (bool *)output_->get_arrivals_data_landing_pointer(block_num);
      std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                  packet_data->get_arrivals_size());
      output_->register_arrivals_transfer_complete(block_num);
      cpu_end = clock::now();
      LOG_DEBUG("CPU time for memcpy arrivals + register completion: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());

      void *eigenvalues_output_pointer =
          (void *)output_->get_eigenvalues_data_landing_pointer(
              eigenvalue_block_num);

      void *eigenvectors_output_pointer =
          (void *)output_->get_eigenvectors_data_landing_pointer(
              eigenvalue_block_num);

      cudaMemcpyAsync(eigenvalues_output_pointer, d_eigenvalues[current_buffer],
                      sizeof(Eigenvalues), cudaMemcpyDefault,
                      streams[current_buffer]);

      cudaMemcpyAsync(eigenvectors_output_pointer,
                      d_decomposition_visibilities_input[current_buffer],
                      sizeof(DecompositionVisibilities), cudaMemcpyDefault,
                      streams[current_buffer]);

      auto *eig_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = eigenvalue_block_num};
      cudaLaunchHostFunc(streams[current_buffer],
                         eigen_output_transfer_complete_host_func,
                         eig_output_ctx);

      auto *fft_output_pointer =
          (void *)output_->get_fft_landing_pointer(fft_block_num);
      cudaMemcpyAsync(fft_output_pointer,
                      d_samples_cufft_output[current_buffer],
                      sizeof(typename T::FFTOutputType), cudaMemcpyDefault,
                      streams[current_buffer]);

      auto *fft_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = fft_block_num};
      cudaLaunchHostFunc(streams[current_buffer],
                         fft_output_transfer_complete_host_func,
                         fft_output_ctx);
      num_correlation_units_integrated += 1;
      if (num_correlation_units_integrated >=
          NR_CORRELATED_BLOCKS_TO_ACCUMULATE) {
        dump_visibilities();
      }
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
      benchmark_runs_done = (benchmark_runs_done + 1) % NR_BENCHMARKING_RUNS;
    }
    auto cpu_end_total = clock::now();
    LOG_DEBUG("Total CPU time for execute_pipeline: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  cpu_end_total - cpu_start_total)
                  .count());
  }
  LambdaGPUPipeline(const int num_buffers, BeamWeightsT<T> *h_weights)

      : num_buffers(num_buffers), h_weights(h_weights),

        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),

        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        cusolver_jobz(CUSOLVER_EIG_MODE_VECTOR),
        cusolver_uplo(CUBLAS_FILL_MODE_UPPER)

  {
    std::cout << "Correlator instantiated with NR_CHANNELS: " << T::NR_CHANNELS
              << ", NR_RECEIVERS: " << T::NR_PADDED_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_CORRELATION: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_RECEIVERS_PER_BLOCK: "
              << T::NR_PADDED_RECEIVERS_PER_BLOCK << std::endl;
    streams.resize(2 * num_buffers);
    d_weights.resize(num_buffers);
    d_weights_updated.resize(num_buffers);
    d_weights_permuted.resize(num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_scaled.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_cufft_input.resize(num_buffers);
    d_samples_cufft_output.resize(num_buffers);
    d_samples_padded.resize(num_buffers);
    d_samples_padding.resize(num_buffers);
    d_samples_consolidated.resize(num_buffers);
    d_samples_consolidated_col_maj.resize(num_buffers);
    d_samples_reord.resize(num_buffers);
    d_correlator_input.resize(num_buffers);
    d_correlator_output.resize(num_buffers);
    d_beamformer_input.resize(num_buffers);
    d_beamformer_output.resize(num_buffers);
    d_beamformer_data_output.resize(num_buffers);
    d_beamformer_data_output_half.resize(num_buffers);
    d_visibilities_converted.resize(num_buffers);
    d_visibilities_permuted.resize(num_buffers);
    d_visibilities_baseline.resize(num_buffers);
    d_visibilities_trimmed_baseline.resize(num_buffers);
    d_visibilities_trimmed.resize(num_buffers);
    d_decomposition_visibilities_input.resize(num_buffers);
    d_eigenvalues.resize(num_buffers);

    fft_plan.resize(num_buffers);
    d_cufft_work_area.resize(num_buffers);
    d_cusolver_info.resize(num_buffers);
    cusolver_params.resize(num_buffers);
    d_cusolver_work_area.resize(num_buffers);
    h_cusolver_work_area.resize(num_buffers);
    d_cusolver_work_area_size.resize(num_buffers);
    h_cusolver_work_area_size.resize(num_buffers);
    cusolver_handle.resize(num_buffers);

    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[num_buffers + i],
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_entry[i],
                            sizeof(typename T::InputPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_scaled[i],
                            sizeof(typename T::InputPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_input[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_output[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated_col_maj[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_padded[i],
                            sizeof(typename T::PaddedPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_padding[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_reord[i],
                            sizeof(typename T::PaddedPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i],
                            sizeof(typename T::PacketScalesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_weights_permuted[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_weights_updated[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_correlator_input[i], sizeof(CorrelatorInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_correlator_output[i],
                            sizeof(CorrelatorOutput)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_beamformer_input[i], sizeof(BeamformerInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_output[i],
                            sizeof(BeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_data_output[i],
                            sizeof(BeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_data_output_half[i],
                            sizeof(HalfBeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_converted[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_permuted[i],
                            sizeof(TrimmedVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_baseline[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_trimmed_baseline[i],
                            sizeof(TrimmedVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_trimmed[i],
                            sizeof(TrimmedVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_eigenvalues[i], sizeof(Eigenvalues)));
      CUDA_CHECK(cudaMalloc((void **)&d_decomposition_visibilities_input[i],
                            sizeof(DecompositionVisibilities)));
    }

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, streams[i], ccglib::ValueType::float16,
          ccglib::mma::basic));
    }

    LOG_DEBUG("Copying weights...");
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemcpy(d_weights[i], h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
    }
    for (auto i = 0; i < NR_BENCHMARKING_RUNS; ++i) {
      cudaEventCreate(&start_run[i]);
      cudaEventCreate(&stop_run[i]);
    }

    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addPermutation("packet", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToPadding");
    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");
    tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
    tensor_32.addPermutation("visCorrTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToDecomp");
    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBToOutput");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visCorrTrimmed",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToTrimmed");

    // set up CUFFT plan for fine-channelization
    const int CUFFT_RANK = 1;
    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const long long CUFFT_ISTRIDE = 1;
    const long long CUFFT_OSTRIDE = 1;
    const long long CUFFT_IDIST = CUFFT_FFT_SIZE;
    const long long CUFFT_ODIST = CUFFT_FFT_SIZE;
    const size_t NUM_TOTAL_BATCHES =
        T::NR_CHANNELS * T::NR_RECEIVERS * T::NR_POLARIZATIONS;
    size_t work_size = 0;
    cudaDataType input_type = CUDA_C_16F;
    cudaDataType output_type = CUDA_C_16F;
    cudaDataType compute_type = CUDA_C_16F;

    for (int i = 0; i < num_buffers; ++i) {
      CUFFT_CHECK(cufftCreate(&fft_plan[i]));
      CUFFT_CHECK(cufftXtMakePlanMany(
          fft_plan[i], CUFFT_RANK, N, NULL, CUFFT_ISTRIDE, CUFFT_IDIST,
          input_type, NULL, CUFFT_OSTRIDE, CUFFT_ODIST, output_type,
          NUM_TOTAL_BATCHES, &work_size, compute_type));

      CUFFT_CHECK(cufftSetStream(fft_plan[i], streams[i]));
      CUDA_CHECK(cudaMalloc(&d_cufft_work_area[i], work_size));
      CUFFT_CHECK(cufftSetWorkArea(fft_plan[i], d_cufft_work_area[i]));
    }
    // set up cuSOLVER for correlation matrix decomposition.
    //
    //
    for (int i = 0; i < num_buffers; ++i) {

      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle[i]));

      CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params[i]));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle[i], streams[i]));
      CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
          cusolver_handle[i], cusolver_params[i], cusolver_jobz, cusolver_uplo,
          T::NR_RECEIVERS, CUDA_C_32F, d_decomposition_visibilities_input[i],
          T::NR_RECEIVERS, // LDA
          CUDA_R_32F,      // Eigenvalue Type (Real)
          d_eigenvalues[i],
          CUDA_C_32F, // Computation Type
          &d_cusolver_work_area_size[i], &h_cusolver_work_area_size[i],
          CUSOLVER_BATCH_SIZE));
      CUDA_CHECK(cudaMalloc((void **)&d_cusolver_work_area[i],
                            d_cusolver_work_area_size[i]));
      h_cusolver_work_area[i] = std::malloc(h_cusolver_work_area_size[i]);
      CUDA_CHECK(cudaMalloc((void **)&d_cusolver_info[i],
                            CUSOLVER_BATCH_SIZE * sizeof(int)));
    }
    // warm up the pipeline.
    // This will JIT the template kernels to avoid having a long startup time
    // Because everything is zeroed it should have negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();
    // these need to be set after the dummy run.
    visibilities_start_seq_num = -1;
    visibilities_end_seq_num = -1;
    visibilities_missing_packets = 0;
  };
  ~LambdaGPUPipeline() {
    // If there are visibilities in the accumulator on the GPU - dump them
    // out to disk. These will get tagged with a -1 end_seq_id currently
    // which is not fully ideal.

    for (auto stream : streams) {
      cudaStreamDestroy(stream);
    }

    for (auto sample : d_samples_entry) {
      cudaFree(sample);
    }

    for (auto scale : d_scales) {
      cudaFree(scale);
    }

    for (auto weight : d_weights) {
      cudaFree(weight);
    }

    for (auto weight : d_weights_updated) {
      cudaFree(weight);
    }

    for (auto weight : d_weights_permuted) {
      cudaFree(weight);
    }

    for (auto correlator_input : d_correlator_input) {
      cudaFree(correlator_input);
    }

    for (auto correlator_output : d_correlator_output) {
      cudaFree(correlator_output);
    }

    for (auto beamformer_input : d_beamformer_input) {
      cudaFree(beamformer_input);
    }
    for (auto beamformer_output : d_beamformer_output) {
      cudaFree(beamformer_output);
    }

    for (auto samples_padding : d_samples_padding) {
      cudaFree(samples_padding);
    }

    for (auto samples_half : d_samples_half) {
      cudaFree(samples_half);
    }

    for (auto samples_cufft : d_samples_cufft_input) {
      cudaFree(samples_cufft);
    }
    for (auto samples_cufft : d_samples_cufft_output) {
      cudaFree(samples_cufft);
    }

    for (auto samples_padded : d_samples_padded) {
      cudaFree(samples_padded);
    }

    for (auto samples_consolidated : d_samples_consolidated) {
      cudaFree(samples_consolidated);
    }

    for (auto samples_consolidated_col_maj : d_samples_consolidated_col_maj) {
      cudaFree(samples_consolidated_col_maj);
    }

    for (auto eigenvalues : d_eigenvalues) {
      cudaFree(eigenvalues);
    }

    for (auto vis : d_visibilities_trimmed_baseline) {
      cudaFree(vis);
    }

    for (auto vis : d_visibilities_baseline) {
      cudaFree(vis);
    }

    for (auto vis : d_visibilities_trimmed) {
      cudaFree(vis);
    }

    for (auto event : start_run) {
      cudaEventDestroy(event);
    }
    for (auto event : stop_run) {
      cudaEventDestroy(event);
    }
  };

  void dump_visibilities(const unsigned long long end_seq_num = 0) override {

    LOG_INFO("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_correlation_units_integrated;
    LOG_INFO("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    cudaDeviceSynchronize();
    const int visibilities_total_packets =
        current_num_integrated_units_processed *
        visibilities_total_packets_per_block;
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num, end_seq_num, visibilities_missing_packets,
        visibilities_total_packets);
    visibilities_start_seq_num = -1;
    visibilities_missing_packets = 0;
    void *landing_pointer =
        output_->get_visibilities_landing_pointer(block_num);
    cudaMemcpyAsync(landing_pointer, d_visibilities_accumulator,
                    sizeof(TrimmedVisibilities), cudaMemcpyDefault, streams[0]);
    auto *output_ctx = new OutputTransferCompleteContext{
        .output = this->output_, .block_index = block_num};

    cudaLaunchHostFunc(streams[0],
                       output_visibilities_transfer_complete_host_func,
                       output_ctx);
    cudaMemsetAsync(d_visibilities_accumulator, 0, sizeof(TrimmedVisibilities),
                    streams[0]);
    num_correlation_units_integrated.store(0);
    cudaDeviceSynchronize();
  };
};
