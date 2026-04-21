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
#include <ccglib/pipeline/pipeline.h>
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

template <typename T> struct CudaDeleter {
  void operator()(T *ptr) { cudaFree(ptr); }
};

template <typename T> using DevicePtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T> DevicePtr<T> make_device_ptr(size_t size = sizeof(T)) {
  T *ptr = nullptr;
  cudaMalloc((void **)&ptr, size);
  return DevicePtr<T>(ptr);
}

struct ManagedCufftPlan {
  cufftHandle handle = 0;
  ManagedCufftPlan() { CUFFT_CHECK(cufftCreate(&handle)); }
  ~ManagedCufftPlan() {
    if (handle)
      cufftDestroy(handle);
  }
  operator cufftHandle() const { return handle; }
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

static void pulsar_fold_output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);

  ctx->output->register_pulsar_fold_transfer_complete(ctx->block_index);
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

  std::vector<std::unique_ptr<ccglib::pipeline::Pipeline>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry,
      d_samples_scaled;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding;
  std::vector<typename T::FFTCUFFTPreprocessingType *>
      d_samples_cufft_preprocessing;
  std::vector<typename T::FFTCUFFTInputType *> d_samples_cufft_input;
  std::vector<typename T::FFTCUFFTOutputType *> d_samples_cufft_output;
  std::vector<typename T::FFTOutputType *> d_cufft_downsampled_output;
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
  int next_fft_channel_pol;

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

    tensor_16.runPermutation(
        "packetToCUFFTInput", alpha, (__half *)d_samples_half[current_buffer],
        (__half *)d_samples_cufft_preprocessing[current_buffer],
        streams[current_buffer]);

    int channel_to_fft = (next_fft_channel_pol) % T::NR_CHANNELS;
    int polarization_to_fft = (next_fft_channel_pol) / T::NR_CHANNELS;

    get_data_for_fft_launch<typename T::FFTCUFFTPreprocessingType,
                            typename T::FFTCUFFTInputType>(
        (typename T::FFTCUFFTPreprocessingType *)
            d_samples_cufft_preprocessing[current_buffer],
        d_samples_cufft_input[current_buffer], T::NR_CHANNELS,
        T::NR_POLARIZATIONS, NR_TIME_STEPS_FOR_CORRELATION, T::NR_RECEIVERS,
        channel_to_fft, polarization_to_fft, streams[current_buffer]);

    CUFFT_CHECK(cufftXtExec(
        fft_plan[current_buffer], (void *)d_samples_cufft_input[current_buffer],
        (void *)d_samples_cufft_output[current_buffer], CUFFT_FORWARD));

    detect_and_average_fft_launch<typename T::FFTCUFFTOutputType,
                                  typename T::FFTOutputType>(
        d_samples_cufft_output[current_buffer],
        d_cufft_downsampled_output[current_buffer],
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_RECEIVERS, T::FFT_DOWNSAMPLE_FACTOR, streams[current_buffer]);
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
      size_t fft_block_num = output_->register_fft_block(
          start_seq_num, end_seq_num, channel_to_fft, polarization_to_fft);
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
                      d_cufft_downsampled_output[current_buffer],
                      sizeof(typename T::FFTOutputType), cudaMemcpyDefault,
                      streams[current_buffer]);

      next_fft_channel_pol =
          (next_fft_channel_pol + 1) % (T::NR_CHANNELS * T::NR_POLARIZATIONS);

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

        correlator(cu::Device(0), 16, T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
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

    next_fft_channel_pol = 0;

    streams.resize(2 * num_buffers);
    d_weights.resize(num_buffers);
    d_weights_updated.resize(num_buffers);
    d_weights_permuted.resize(num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_scaled.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_cufft_input.resize(num_buffers);
    d_samples_cufft_preprocessing.resize(num_buffers);
    d_cufft_downsampled_output.resize(num_buffers);
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
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_preprocessing[i],
                            sizeof(typename T::FFTCUFFTPreprocessingType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_input[i],
                            sizeof(typename T::FFTCUFFTInputType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_output[i],
                            sizeof(typename T::FFTCUFFTOutputType)));
      CUDA_CHECK(cudaMalloc((void **)&d_cufft_downsampled_output[i],
                            sizeof(typename T::FFTOutputType)));
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

    const std::complex<float> alpha_ccglib = {1, 0};
    const std::complex<float> beta_ccglib = {0, 0};
    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, streams[i], ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib));
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
    const size_t NUM_TOTAL_BATCHES = T::NR_RECEIVERS;
    size_t work_size = 0;
    cudaDataType input_type = CUDA_C_32F;
    cudaDataType output_type = CUDA_C_32F;
    cudaDataType compute_type = CUDA_C_32F;

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

    for (auto samples_cufft : d_samples_cufft_preprocessing) {
      cudaFree(samples_cufft);
    }
    for (auto samples_cufft : d_samples_cufft_output) {
      cudaFree(samples_cufft);
    }
    for (auto cufft : d_cufft_downsampled_output) {
      cudaFree(cufft);
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

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

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

template <typename T> class LambdaAntennaSpectraPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  inline static const __half alpha = __float2half(1.0f);

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
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};
  inline static const std::unordered_map<int, int64_t> extent = {

      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
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

  int current_buffer;
  std::atomic<int> last_frame_processed;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry,
      d_samples_scaled;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding;
  std::vector<typename T::FFTCUFFTPreprocessingType *>
      d_samples_cufft_preprocessing;
  std::vector<typename T::MultiChannelFFTCUFFTInputType *>
      d_samples_cufft_input;
  std::vector<typename T::MultiChannelFFTCUFFTOutputType *>
      d_samples_cufft_output;
  std::vector<typename T::MultiChannelAntennaFFTOutputType *>
      d_cufft_downsampled_output;
  std::vector<typename T::PacketScalesType *> d_scales;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;
  std::vector<cufftHandle> fft_plan;
  std::vector<void *> d_cufft_work_area;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    LOG_INFO("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);
    // visibilities_missing_packets += packet_data->get_num_missing_packets();

    // d_samples_entry memcpy
    cudaMemcpyAsync(d_samples_entry[current_buffer],
                    (void *)packet_data->get_samples_ptr(),
                    packet_data->get_samples_elements_size(), cudaMemcpyDefault,
                    streams[current_buffer]);

    // d_scales memcpy
    cudaMemcpyAsync(d_scales[current_buffer],
                    (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    streams[current_buffer]);

    // BufferReleaseContext + host function
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    // do in separate thread - no need to tie up GPU pipeline.
    CUDA_CHECK(cudaLaunchHostFunc(streams[num_buffers + current_buffer],
                                  release_buffer_host_func, ctx));

    // scale_and_convert_to_half kernel
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

    tensor_16.runPermutation(
        "packetToCUFFTInput", alpha, (__half *)d_samples_half[current_buffer],
        (__half *)d_samples_cufft_preprocessing[current_buffer],
        streams[current_buffer]);

    // convert to float
    get_data_for_multi_channel_fft_launch<
        typename T::FFTCUFFTPreprocessingType,
        typename T::MultiChannelFFTCUFFTInputType>(
        (typename T::FFTCUFFTPreprocessingType *)
            d_samples_cufft_preprocessing[current_buffer],
        d_samples_cufft_input[current_buffer], T::NR_CHANNELS,
        T::NR_POLARIZATIONS, NR_TIME_STEPS_FOR_CORRELATION, T::NR_RECEIVERS,
        streams[current_buffer]);

    CUFFT_CHECK(cufftXtExec(
        fft_plan[current_buffer], (void *)d_samples_cufft_input[current_buffer],
        (void *)d_samples_cufft_output[current_buffer], CUFFT_FORWARD));

    detect_and_downsample_multi_channel_fft_launch<
        typename T::MultiChannelFFTCUFFTOutputType,
        typename T::MultiChannelAntennaFFTOutputType>(
        d_samples_cufft_output[current_buffer],
        d_cufft_downsampled_output[current_buffer], T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_RECEIVERS, T::FFT_DOWNSAMPLE_FACTOR, streams[current_buffer]);
    // Output handling
    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num, -1, -1);
      auto *fft_output_pointer =
          (void *)output_->get_fft_landing_pointer(fft_block_num);
      cudaMemcpyAsync(fft_output_pointer,
                      d_cufft_downsampled_output[current_buffer],
                      sizeof(typename T::MultiChannelAntennaFFTOutputType),
                      cudaMemcpyDefault, streams[current_buffer]);

      auto *fft_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = fft_block_num};
      cudaLaunchHostFunc(streams[current_buffer],
                         fft_output_transfer_complete_host_func,
                         fft_output_ctx);
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaAntennaSpectraPipeline(const int num_buffers)

      : num_buffers(num_buffers), tensor_16(extent, CUTENSOR_R_16F, 128)

  {
    std::cout << "Spectra Analyzer instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << std::endl;

    streams.resize(2 * num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_scaled.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_cufft_input.resize(num_buffers);
    d_samples_cufft_preprocessing.resize(num_buffers);
    d_cufft_downsampled_output.resize(num_buffers);
    d_samples_cufft_output.resize(num_buffers);

    fft_plan.resize(num_buffers);
    d_cufft_work_area.resize(num_buffers);

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
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_preprocessing[i],
                            sizeof(typename T::FFTCUFFTPreprocessingType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_input[i],
                            sizeof(typename T::MultiChannelFFTCUFFTInputType)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_samples_cufft_output[i],
                     sizeof(typename T::MultiChannelFFTCUFFTOutputType)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_cufft_downsampled_output[i],
                     sizeof(typename T::MultiChannelAntennaFFTOutputType)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i],
                            sizeof(typename T::PacketScalesType)));
    }

    last_frame_processed = 0;
    current_buffer = 0;
    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");

    // set up CUFFT plan for fine-channelization
    const int CUFFT_RANK = 1;
    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const long long CUFFT_ISTRIDE = 1;
    const long long CUFFT_OSTRIDE = 1;
    const long long CUFFT_IDIST = CUFFT_FFT_SIZE;
    const long long CUFFT_ODIST = CUFFT_FFT_SIZE;
    const size_t NUM_TOTAL_BATCHES =
        T::NR_RECEIVERS * T::NR_CHANNELS * T::NR_POLARIZATIONS;
    LOG_INFO("FFT initialized with {} total batches with a {} FFT each run "
             "(RECEIVERS x CHANNELS x POL)",
             NUM_TOTAL_BATCHES, CUFFT_FFT_SIZE);
    size_t work_size = 0;
    cudaDataType input_type = CUDA_C_32F;
    cudaDataType output_type = CUDA_C_32F;
    cudaDataType compute_type = CUDA_C_32F;

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
  };
  ~LambdaAntennaSpectraPipeline() {
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

    for (auto samples_half : d_samples_half) {
      cudaFree(samples_half);
    }

    for (auto samples_cufft : d_samples_cufft_input) {
      cudaFree(samples_cufft);
    }

    for (auto samples_cufft : d_samples_cufft_preprocessing) {
      cudaFree(samples_cufft);
    }
    for (auto samples_cufft : d_samples_cufft_output) {
      cudaFree(samples_cufft);
    }
    for (auto cufft : d_cufft_downsampled_output) {
      cudaFree(cufft);
    }
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }
};

template <typename T>
class LambdaBeamformedSpectraPipeline : public GPUPipeline {
private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  using FFTCUFFTInputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION /
            T::FFT_DOWNSAMPLE_FACTOR];
  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamWeights = BeamWeightsT<T>;
  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_consolidated, samples_consolidated_col_maj;
    DevicePtr<FFTCUFFTInputType> samples_cufft_input;
    DevicePtr<FFTCUFFTOutputType> samples_cufft_output;
    DevicePtr<FFTOutputType> cufft_downsampled_output;
    DevicePtr<BeamWeights> weights;
    DevicePtr<BeamWeights> weights_permuted;
    DevicePtr<BeamformerOutput> beamformer_output;
    DevicePtr<void> cufft_work_area;

    std::unique_ptr<ccglib::mma::GEMM> gemm_handle;

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_cufft_input(make_device_ptr<FFTCUFFTInputType>()),
          samples_cufft_output(make_device_ptr<FFTCUFFTOutputType>()),
          cufft_downsampled_output(make_device_ptr<FFTOutputType>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          cufft_work_area(make_device_ptr<void>(work_size)) {
      // Stream Creation
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      // GEMM Initialization
      gemm_handle = std::make_unique<ccglib::mma::GEMM>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::ValueType::float16, ccglib::mma::basic);
    }

    ~PipelineResources() {
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
    }
    PipelineResources(PipelineResources &&other) noexcept
        : stream(other.stream), host_stream(other.host_stream),
          fft_plan(std::move(other.fft_plan)),
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_consolidated(std::move(other.samples_consolidated)),
          samples_consolidated_col_maj(
              std::move(other.samples_consolidated_col_maj)),
          samples_cufft_input(std::move(other.samples_cufft_input)),
          samples_cufft_output(std::move(other.samples_cufft_output)),
          cufft_downsampled_output(std::move(other.cufft_downsampled_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
          cufft_work_area(std::move(other.cufft_work_area)),
          gemm_handle(std::move(other.gemm_handle)) {
      other.stream = nullptr;
      other.host_stream = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);

        stream = other.stream;
        host_stream = other.host_stream;
        fft_plan = std::move(other.fft_plan);
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        samples_cufft_input = std::move(other.samples_cufft_input);
        samples_cufft_output = std::move(other.samples_cufft_output);
        cufft_downsampled_output = std::move(other.cufft_downsampled_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        cufft_work_area = std::move(other.cufft_work_area);
        gemm_handle = std::move(other.gemm_handle);

        other.stream = nullptr;
        other.host_stream = nullptr;
      }
      return *this;
    }

    // 3. Explicitly Delete Copying
    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers;
  std::vector<PipelineResources> buffers;

  // We are converting it to fp16 so this should not be changable anymore.

  inline static const __half alpha = __float2half(1.0f);

  static constexpr float alpha_32 = 1.0f;
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
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'm', 's', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm',
                                                         'r'};

  inline static const std::unordered_map<int, int64_t> extent = {

      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
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

  BeamWeights *h_weights;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }
    auto &b = buffers[current_buffer];

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    LOG_INFO("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    cudaMemcpyAsync(
        b.samples_entry.get(), (void *)packet_data->get_samples_ptr(),
        packet_data->get_samples_elements_size(), cudaMemcpyDefault, b.stream);

    cudaMemcpyAsync(b.scales.get(), (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    b.stream);

    // BufferReleaseContext + host function
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    // do in separate thread - no need to tie up GPU pipeline.
    CUDA_CHECK(
        cudaLaunchHostFunc(b.host_stream, release_buffer_host_func, ctx));

    // scale_and_convert_to_half kernel
    scale_and_convert_to_half<
        typename T::InputPacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfInputPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
        T::NR_TIME_STEPS_PER_PACKET, T::NR_PACKETS_FOR_CORRELATION>(
        (typename T::InputPacketSamplesPlanarType *)b.samples_entry.get(),
        b.scales.get(),
        (typename T::HalfInputPacketSamplesPlanarType *)b.samples_half.get(),
        b.stream);

    tensor_16.runPermutation("packetToPlanar", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    // this only needs to be run once.
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)b.weights.get(),
                             (__half *)b.weights_permuted.get(), b.stream);

    b.gemm_handle->Run((CUdeviceptr)b.weights_permuted.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);

    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    detect_and_downsample_fft_launch(
        (float2 *)b.samples_cufft_output.get(),
        (float *)b.cufft_downsampled_output.get(), T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_BEAMS, T::FFT_DOWNSAMPLE_FACTOR, b.stream);
    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num, -1, -1);
      auto *fft_output_pointer =
          (void *)output_->get_fft_landing_pointer(fft_block_num);
      cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                      sizeof(FFTOutputType), cudaMemcpyDefault, b.stream);

      auto *fft_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = fft_block_num};
      cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                         fft_output_ctx);
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaBeamformedSpectraPipeline(const int num_buffers,
                                  BeamWeightsT<T> *h_weights)

      : num_buffers(num_buffers), h_weights(h_weights),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128)

  {
    std::cout << "Beamformed Spectra instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << T::NR_BEAMS << std::endl;

    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES =
        T::NR_BEAMS * T::NR_CHANNELS * T::NR_POLARIZATIONS;

    size_t work_size = 0;
    {
      // Temporary plan to calculate work_size
      cufftHandle temp_plan;
      CUFFT_CHECK(cufftCreate(&temp_plan));
      CUFFT_CHECK(cufftXtMakePlanMany(temp_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      cufftDestroy(temp_plan);
    }

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device, work_size);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));

      // Copy initial weights
      cudaMemcpy(b.weights.get(), h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
    }
    last_frame_processed = 0;
    current_buffer = 0;
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");

    tensor_32.addTensor(modeCUFFTInput, "cufftInput");
    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");
    cudaDeviceSynchronize();
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
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }
};

template <typename T>
class LambdaAdaptiveBeamformedSpectraPipeline : public GPUPipeline {
private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  // cuSOLVER batch size: one (NR_RECEIVERS × NR_RECEIVERS) matrix per
  // channel × pol × pol, matching LambdaGPUPipeline exactly.
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS;

  // -------------------------------------------------------------------------
  // Array-type aliases
  // -------------------------------------------------------------------------
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
                         [T::NR_POLARIZATIONS];

  // Full NR_RECEIVERS × NR_RECEIVERS matrices (one per channel × pol × pol),
  // laid out as a flat batch for cuSOLVER / cuBLAS.
  // Shape: [CUSOLVER_BATCH_SIZE][NR_RECEIVERS][NR_RECEIVERS]
  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  // Eigenvalues: one real vector of length NR_RECEIVERS per batch element.
  using Eigenvalues =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  using FFTCUFFTInputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using BeamTransferSingleChannelPol =
      float2[2 * T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION /
            T::FFT_DOWNSAMPLE_FACTOR];
  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamOutput =
      std::complex<__half>[2 * T::NR_BEAMS][NR_TIME_STEPS_FOR_CORRELATION]
                          [T::NR_CHANNELS][T::NR_POLARIZATIONS];
  using ProjectionMatrix =
      std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                          [T::NR_RECEIVERS];
  using FloatProjectionMatrix =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];
  using BeamWeights = BeamWeightsT<T>;
  struct RFIMitigatedT {
    static constexpr size_t NR_CHANNELS = T::NR_CHANNELS;
    static constexpr size_t NR_POLARIZATIONS = T::NR_POLARIZATIONS;
    static constexpr size_t NR_BEAMS = 2 * T::NR_BEAMS;
    static constexpr size_t NR_RECEIVERS = T::NR_RECEIVERS;
  };

  using RFIMitigatedBeamWeights = BeamWeightsT<RFIMitigatedT>;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
    DevicePtr<FFTCUFFTInputType> samples_cufft_input, beam_shape;
    DevicePtr<BeamOutput> beam_output;
    DevicePtr<FFTCUFFTOutputType> samples_cufft_output;
    DevicePtr<FFTOutputType> cufft_downsampled_output;
    DevicePtr<BeamWeights> weights;
    DevicePtr<BeamWeights> weights_permuted, weights_updated;
    DevicePtr<RFIMitigatedBeamWeights> weights_rfi_mitigated;
    DevicePtr<RFIMitigatedBeamWeights> weights_beamformer;
    DevicePtr<BeamformerOutput> beamformer_output;
    DevicePtr<void> cufft_work_area;

    // Correlator I/O
    DevicePtr<CorrelatorInput> correlator_input;
    DevicePtr<CorrelatorOutput> correlator_output;

    // Intermediate visibility layout buffers
    DevicePtr<Visibilities> visibilities_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed;

    // Per-block correlation matrix (input to cuSOLVER, overwritten in-place
    // with eigenvectors on output).
    DevicePtr<DecompositionVisibilities> decomp_visibilities;
    DevicePtr<ProjectionMatrix> projection_matrix;
    DevicePtr<FloatProjectionMatrix> float_projection_matrix;

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues;

    // cuSOLVER handles / workspace
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverDnParams_t cusolver_params = nullptr;
    DevicePtr<int> cusolver_info; // [CUSOLVER_BATCH_SIZE]
    DevicePtr<void> cusolver_work_device;
    void *cusolver_work_host = nullptr;
    size_t cusolver_work_device_size = 0;
    size_t cusolver_work_host_size = 0;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_handle;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_weight_projection_handle;

    cublasHandle_t cublas_handle = nullptr;

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_cufft_input(make_device_ptr<FFTCUFFTInputType>()),
          beam_shape(make_device_ptr<FFTCUFFTInputType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          samples_cufft_output(make_device_ptr<FFTCUFFTOutputType>()),
          cufft_downsampled_output(make_device_ptr<FFTOutputType>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          weights_rfi_mitigated(make_device_ptr<RFIMitigatedBeamWeights>()),
          weights_beamformer(make_device_ptr<RFIMitigatedBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          samples_padding(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          projection_matrix(make_device_ptr<ProjectionMatrix>()),
          float_projection_matrix(make_device_ptr<FloatProjectionMatrix>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
          cusolver_info(
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))),
          cufft_work_area(make_device_ptr<void>(work_size)) {
      // Stream Creation
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
      CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

      CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
          cusolver_handle, cusolver_params, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER, T::NR_RECEIVERS, CUDA_C_32F,
          reinterpret_cast<void *>(decomp_visibilities.get()), T::NR_RECEIVERS,
          CUDA_R_32F, reinterpret_cast<void *>(eigenvalues.get()), CUDA_C_32F,
          &cusolver_work_device_size, &cusolver_work_host_size,
          CUSOLVER_BATCH_SIZE));

      cusolver_work_device = make_device_ptr<void>(cusolver_work_device_size);
      cusolver_work_host = std::malloc(cusolver_work_host_size);

      const std::complex<float> alpha_ccglib = {1, 0};
      const std::complex<float> beta_ccglib = {0, 0};
      // GEMM Initialization
      gemm_handle = std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, 2 * T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib);

      gemm_weight_projection_handle =
          std::make_unique<ccglib::pipeline::Pipeline>(
              T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
              T::NR_RECEIVERS, T::NR_RECEIVERS, cu_device, stream,
              ccglib::complex_interleaved, ccglib::complex_interleaved,
              ccglib::mma::row_major, ccglib::mma::col_major,
              ccglib::mma::row_major, ccglib::ValueType::float16,
              ccglib::ValueType::float16, ccglib::mma::opt, alpha_ccglib,
              beta_ccglib);
      CUBLAS_CHECK(cublasCreate(&cublas_handle));
      CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    }

    ~PipelineResources() {
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
      if (cusolver_handle)
        cusolverDnDestroy(cusolver_handle);
      if (cusolver_params)
        cusolverDnDestroyParams(cusolver_params);
      if (cusolver_work_host)
        std::free(cusolver_work_host);
      if (cublas_handle)
        cublasDestroy(cublas_handle);
    }
    PipelineResources(PipelineResources &&other) noexcept
        : stream(other.stream), host_stream(other.host_stream),
          fft_plan(std::move(other.fft_plan)),
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_consolidated(std::move(other.samples_consolidated)),
          samples_consolidated_col_maj(
              std::move(other.samples_consolidated_col_maj)),
          beam_output(std::move(other.beam_output)),
          beam_shape(std::move(other.beam_shape)),
          samples_cufft_input(std::move(other.samples_cufft_input)),
          samples_cufft_output(std::move(other.samples_cufft_output)),
          cufft_downsampled_output(std::move(other.cufft_downsampled_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
          cufft_work_area(std::move(other.cufft_work_area)),
          gemm_handle(std::move(other.gemm_handle)),
          samples_padding(std::move(other.samples_padding)),
          samples_padded(std::move(other.samples_padded)),
          correlator_input(std::move(other.correlator_input)),
          correlator_output(std::move(other.correlator_output)),
          float_projection_matrix(std::move(other.float_projection_matrix)),
          projection_matrix(std::move(other.projection_matrix)),
          visibilities_baseline(std::move(other.visibilities_baseline)),
          visibilities_trimmed_baseline(
              std::move(other.visibilities_trimmed_baseline)),
          visibilities_trimmed(std::move(other.visibilities_trimmed)),
          decomp_visibilities(std::move(other.decomp_visibilities)),
          eigenvalues(std::move(other.eigenvalues)),
          cusolver_handle(other.cusolver_handle),
          cusolver_params(other.cusolver_params),
          cusolver_work_device(std::move(other.cusolver_work_device)),
          cusolver_work_host(other.cusolver_work_host),
          cusolver_work_device_size(other.cusolver_work_device_size),
          cusolver_work_host_size(other.cusolver_work_host_size),
          cusolver_info(std::move(other.cusolver_info))

    {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.cusolver_handle = nullptr;
      other.cusolver_params = nullptr;
      other.cusolver_work_host = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);

        stream = other.stream;
        host_stream = other.host_stream;
        fft_plan = std::move(other.fft_plan);
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        beam_output = std::move(other.beam_output);
        beam_shape = std::move(other.beam_shape);
        samples_cufft_input = std::move(other.samples_cufft_input);
        samples_cufft_output = std::move(other.samples_cufft_output);
        cufft_downsampled_output = std::move(other.cufft_downsampled_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        cufft_work_area = std::move(other.cufft_work_area);
        gemm_handle = std::move(other.gemm_handle);

        other.stream = nullptr;

        other.host_stream = nullptr;
      }
      return *this;
    }

    // 3. Explicitly Delete Copying
    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers;
  std::vector<PipelineResources> buffers;

  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;
  tcc::Correlator correlator;
  // We are converting it to fp16 so this should not be changable anymore.

  inline static const __half alpha = __float2half(1.0f);

  inline static const std::vector<size_t> NR_SIGNAL_EIGENVECTORS{1, 1, 1, 1,
                                                                 6, 1, 5, 5};

  static constexpr float alpha_32 = 1.0f;
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
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'e', 's', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'e', 's'};
  inline static const std::vector<int> modeBeamOutput{'e', 's', 'c', 'p', 'z'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsBeamMajor{'m', 'c', 'p', 'r',
                                                            'z'};

  inline static const std::vector<int> modeWeights2xBeamMajor{'e', 'c', 'p',
                                                              'r', 'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'e',
                                                         'r'};

  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'p', 'q', 'l', 'c',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'p', 'a', 'c',
                                                                  'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'a', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
      {'m', T::NR_BEAMS},
      {'e', T::NR_BEAMS * 2}, // rfi mitigated beam + original beam
      {'d', T::NR_PADDED_RECEIVERS},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'l', NR_BASELINES},
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

  BeamWeights *h_weights;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }
    auto &b = buffers[current_buffer];

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    LOG_INFO("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    cudaMemcpyAsync(
        b.samples_entry.get(), (void *)packet_data->get_samples_ptr(),
        packet_data->get_samples_elements_size(), cudaMemcpyDefault, b.stream);

    cudaMemcpyAsync(b.scales.get(), (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    b.stream);

    // BufferReleaseContext + host function
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    // do in separate thread - no need to tie up GPU pipeline.
    CUDA_CHECK(
        cudaLaunchHostFunc(b.host_stream, release_buffer_host_func, ctx));

    // scale_and_convert_to_half kernel
    scale_and_convert_to_half<
        typename T::InputPacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfInputPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
        T::NR_TIME_STEPS_PER_PACKET, T::NR_PACKETS_FOR_CORRELATION>(
        (typename T::InputPacketSamplesPlanarType *)b.samples_entry.get(),
        b.scales.get(),
        (typename T::HalfInputPacketSamplesPlanarType *)b.samples_half.get(),
        b.stream);

    tensor_16.runPermutation("packetToPlanar", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    tensor_16.runPermutation(
        "packetToPadding", alpha,
        reinterpret_cast<__half *>(b.samples_half.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

    // ------------------------------------------------------------------
    // 5. Copy unpadded → padded buffer then zero-fill the padding region
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketSamplesType),
                               cudaMemcpyDefault, b.stream));
    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketSamplesType),
                        b.stream));

    // ------------------------------------------------------------------
    // 6. Permute padded → correlator input layout
    // ------------------------------------------------------------------
    tensor_16.runPermutation(
        "paddedToCorrInput", alpha,
        reinterpret_cast<__half *>(b.samples_padded.get()),
        reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);

    // ------------------------------------------------------------------
    // 7. Cross-correlate with tcc::Correlator
    // ------------------------------------------------------------------
    correlator.launchAsync(
        static_cast<CUstream>(b.stream),
        reinterpret_cast<CUdeviceptr>(b.correlator_output.get()),
        reinterpret_cast<CUdeviceptr>(b.correlator_input.get()));
    // ------------------------------------------------------------------
    // 8. Rearrange correlator output to baseline-major, then trim padding
    // ------------------------------------------------------------------
    tensor_32.runPermutation(
        "visCorrToBaseline", alpha_32,
        reinterpret_cast<float *>(b.correlator_output.get()),
        reinterpret_cast<float *>(b.visibilities_baseline.get()), b.stream);

    CUDA_CHECK(cudaMemcpyAsync(
        b.visibilities_trimmed_baseline.get(), b.visibilities_baseline.get(),
        sizeof(TrimmedVisibilities) / 2, cudaMemcpyDefault, b.stream));

    void *source_pol_1_1 =
        (char *)b.visibilities_baseline.get() + 3 * sizeof(Visibilities) / 4;
    void *dest_pol_1_1 = (char *)b.visibilities_trimmed_baseline.get() +
                         sizeof(TrimmedVisibilities) / 2;

    CUDA_CHECK(cudaMemcpyAsync(dest_pol_1_1, source_pol_1_1,
                               sizeof(TrimmedVisibilities) / 2,
                               cudaMemcpyDefault, b.stream));
    tensor_32.runPermutation(
        "visBaselineTrimmedToDecomp", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed_baseline.get()),
        reinterpret_cast<float *>(b.visibilities_trimmed.get()), b.stream);

    unpack_triangular_baseline_batch_launch<cuComplex>(
        reinterpret_cast<cuComplex *>(b.visibilities_trimmed.get()),
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        T::NR_RECEIVERS, CUSOLVER_BATCH_SIZE, T::NR_CHANNELS, b.stream);

    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

    // ------------------------------------------------------------------
    // 11. Form P_block = U U^H via batched cuBLAS cherk.
    //
    //     After cuSOLVER, decomp_visibilities holds the full eigenvector
    //     matrix V (NR_RECEIVERS × NR_RECEIVERS, column = eigenvector,
    //     ascending order).  The signal subspace U consists of the last
    //     NR_SIGNAL_EIGENVECTORS columns, i.e. the sub-matrix starting at
    //     column offset (NR_RECEIVERS - NR_SIGNAL_EIGENVECTORS).
    //
    //     cuSOLVER stores column-major (Fortran order), so column j starts
    //     at row-offset 0 and the pointer to column j is:
    //       V_ptr + j * NR_RECEIVERS    (in cuComplex elements)
    //
    //     cherk computes:  C ← alpha * A * A^H + beta * C
    //       A = U  (NR_RECEIVERS × NR_SIGNAL_EIGENVECTORS, col-major)
    //       C = P  (NR_RECEIVERS × NR_RECEIVERS, col-major)
    //
    //     We call it once per batch element in a simple loop.  A batched
    //     cherk variant is not available in cuBLAS; the loop is over
    //     CUSOLVER_BATCH_SIZE elements and is negligible CPU overhead
    //     compared with the GPU kernels.
    // ------------------------------------------------------------------
    {
      constexpr int N = T::NR_RECEIVERS;
      const cuComplex herk_alpha{1.0f, 0.0f};
      const cuComplex herk_beta{0.0f, 0.0f}; // overwrite projection_block

      auto *V_base = reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
      auto *P_base =
          reinterpret_cast<cuComplex *>(b.float_projection_matrix.get());
      const size_t CUBLAS_BATCH_SIZE_PER_CHANNEL = T::NR_POLARIZATIONS;

      for (int channel = 0; channel < T::NR_CHANNELS; ++channel) {
        const int K = NR_SIGNAL_EIGENVECTORS[channel];
        const int col_offset = N - K; // first signal-subspace column
        // Pointer to signal-subspace U = last K columns of V_batch.
        for (int batch = 0; batch < CUBLAS_BATCH_SIZE_PER_CHANNEL; batch++) {
          // Pointer to the start of eigenvector matrix for this batch element.
          cuComplex *V_batch =
              V_base +
              (channel * CUBLAS_BATCH_SIZE_PER_CHANNEL + batch) * N * N;
          cuComplex *U = V_batch + col_offset * N;
          // Pointer to output P for this batch element.
          cuComplex *P_batch =
              P_base +
              (channel * CUBLAS_BATCH_SIZE_PER_CHANNEL + batch) * N * N;

          CUBLAS_CHECK(cublasGemmEx(b.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C,
                                    N, N, K, &herk_alpha, U, CUDA_C_32F, N, U,
                                    CUDA_C_32F, N, &herk_beta, P_batch,
                                    CUDA_C_32F, N, CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
      }
    }

    computeIdentityMinusA((float2 *)b.float_projection_matrix.get(),
                          (__half2 *)b.projection_matrix.get(), T::NR_RECEIVERS,
                          T::NR_CHANNELS * T::NR_POLARIZATIONS, b.stream);

    // conjugateMatrix((__half2 *)b.projection_matrix.get(),
    //                 T::NR_RECEIVERS * T::NR_RECEIVERS * T::NR_CHANNELS *
    //                     T::NR_POLARIZATIONS,
    //                 b.stream);

    {
      const cuComplex herk_alpha{1.0f, 0.0f};
      const cuComplex herk_beta{0.0f, 0.0f}; // overwrite projection_block
      const int N = T::NR_RECEIVERS;
      size_t CUBLAS_NUM_BATCHES =
          CUSOLVER_BATCH_SIZE; // T::NR_POLARIZATIONS * T::NR_CHANNELS
      size_t CUBLAS_STRIDE_A = T::NR_RECEIVERS * T::NR_RECEIVERS;
      size_t CUBLAS_STRIDE_B = T::NR_RECEIVERS * T::NR_BEAMS;
      size_t CUBLAS_STRIDE_C = T::NR_RECEIVERS * T::NR_BEAMS;

      // cublasGemmStridedBatchedEx(
      //     b.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, T::NR_BEAMS, N, N,
      //     &herk_alpha, b.weights.get(), CUDA_C_16F, T::NR_BEAMS,
      //     CUBLAS_STRIDE_B, b.projection_matrix.get(), CUDA_C_16F, N,
      //     CUBLAS_STRIDE_A, &herk_beta, b.weights_updated.get(), CUDA_C_16F,
      //     T::NR_BEAMS, CUBLAS_STRIDE_C, CUBLAS_NUM_BATCHES,
      //     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      //
      //
      b.gemm_weight_projection_handle->Run(
          (CUdeviceptr)b.weights.get(), (CUdeviceptr)b.projection_matrix.get(),
          (CUdeviceptr)b.weights_updated.get());
    }

    // weightsDebugLaunch((__half2 *)b.weights_updated.get(),
    //                    T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_RECEIVERS
    //                    *
    //                        T::NR_BEAMS,
    //                    b.stream);

    tensor_16.runPermutation("weightsToBeamMajor", alpha,
                             (__half *)b.weights_updated.get(),
                             (__half *)b.weights_permuted.get(), b.stream);

    void *dest_ptr =
        (char *)b.weights_rfi_mitigated.get() + sizeof(BeamWeights);
    cudaMemcpyAsync(dest_ptr, b.weights_permuted.get(), sizeof(BeamWeights),
                    cudaMemcpyDefault, b.stream);

    tensor_16.runPermutation("weights2xBeamMajorToCCGLIB", alpha,
                             (__half *)b.weights_rfi_mitigated.get(),
                             (__half *)b.weights_beamformer.get(), b.stream);

    b.gemm_handle->Run((CUdeviceptr)b.weights_beamformer.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);

    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.beam_shape.get(), b.stream);
    convert_float_to_half((float *)b.beam_shape.get(),
                          (__half *)b.beam_output.get(),
                          sizeof(BeamOutput) / sizeof(__half));
    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    detect_and_downsample_fft_launch(
        (float2 *)b.samples_cufft_output.get(),
        (float *)b.cufft_downsampled_output.get(), T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        2 * T::NR_BEAMS, T::FFT_DOWNSAMPLE_FACTOR, b.stream);

    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.

      size_t beam_block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      auto *beam_output_pointer =
          (void *)output_->get_beam_data_landing_pointer(beam_block_num);

      cudaMemcpyAsync(beam_output_pointer, b.beam_output.get(),
                      sizeof(BeamOutput), cudaMemcpyDefault, b.stream);

      auto *beam_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = beam_block_num};

      cudaLaunchHostFunc(b.stream, output_transfer_complete_host_func,
                         beam_output_ctx);

      bool *arrivals_output_pointer =
          (bool *)output_->get_arrivals_data_landing_pointer(beam_block_num);
      std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                  packet_data->get_arrivals_size());
      output_->register_arrivals_transfer_complete(beam_block_num);

      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num, -1, -1);
      auto *fft_output_pointer =
          (void *)output_->get_fft_landing_pointer(fft_block_num);
      cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                      sizeof(FFTOutputType), cudaMemcpyDefault, b.stream);

      auto *fft_output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = fft_block_num};
      cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                         fft_output_ctx);

      size_t eig_block_num = output_->register_eigendecomposition_data_block(
          start_seq_num, end_seq_num);

      void *eigval_ptr =
          output_->get_eigenvalues_data_landing_pointer(eig_block_num);
      void *eigvec_ptr =
          output_->get_eigenvectors_data_landing_pointer(eig_block_num);
      CUDA_CHECK(cudaMemcpyAsync(eigvec_ptr, b.decomp_visibilities.get(),
                                 sizeof(DecompositionVisibilities),
                                 cudaMemcpyDefault, b.stream));

      CUDA_CHECK(cudaMemcpyAsync(eigval_ptr, b.eigenvalues.get(),
                                 sizeof(Eigenvalues), cudaMemcpyDefault,
                                 b.stream));

      auto *ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = eig_block_num};
      CUDA_CHECK(cudaLaunchHostFunc(
          b.stream, eigen_output_transfer_complete_host_func, ctx));
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaAdaptiveBeamformedSpectraPipeline(const int num_buffers,
                                          BeamWeightsT<T> *h_weights)

      : num_buffers(num_buffers), h_weights(h_weights),
        correlator(cu::Device(0), 16, T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128)

  {
    std::cout << "Beamformed Spectra instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << 2 * T::NR_BEAMS << std::endl;

    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES =
        2 * T::NR_BEAMS * T::NR_CHANNELS * T::NR_POLARIZATIONS;

    size_t work_size = 0;
    {
      // Temporary plan to calculate work_size
      cufftHandle temp_plan;
      CUFFT_CHECK(cufftCreate(&temp_plan));
      CUFFT_CHECK(cufftXtMakePlanMany(temp_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      cufftDestroy(temp_plan);
    }

    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsBeamMajor, "weightsBeamMajor");
    tensor_16.addTensor(modeWeights2xBeamMajor, "weights2xBeamMajor");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");

    tensor_32.addTensor(modeCUFFTInput, "cufftInput");
    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    // Permutation descriptors
    tensor_16.addPermutation("packet", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToPadding");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToDecomp");

    tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsBeamMajor",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsToBeamMajor");
    tensor_16.addPermutation("weights2xBeamMajor", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F,
                             "weights2xBeamMajorToCCGLIB");
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");
    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBToOutput");

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device, work_size);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      cudaMemcpyAsync(b.weights_rfi_mitigated.get(), b.weights_permuted.get(),
                      sizeof(BeamWeights), cudaMemcpyDefault, b.stream);
    }
    last_frame_processed = 0;
    current_buffer = 0;
    cudaDeviceSynchronize();
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
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }
};

template <typename T> class LambdaCorrBeamOnlyGPUPipeline : public GPUPipeline {

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

  std::vector<std::unique_ptr<ccglib::pipeline::Pipeline>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry,
      d_samples_scaled;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding;
  std::vector<typename T::PaddedPacketSamplesType *> d_samples_padded,
      d_samples_reord; // This is not the right type for reord
                       // - but it will do I guess. Size will be correct.
  std::vector<CorrelatorInput *> d_correlator_input;
  std::vector<CorrelatorOutput *> d_correlator_output;

  std::vector<Visibilities *> d_visibilities_converted;
  std::vector<TrimmedVisibilities *> d_visibilities_baseline,
      d_visibilities_trimmed_baseline, d_visibilities_trimmed;
  TrimmedVisibilities *d_visibilities_accumulator;
  std::vector<BeamformerInput *> d_beamformer_input;
  std::vector<BeamformerOutput *> d_beamformer_output, d_beamformer_data_output;
  std::vector<HalfBeamformerOutput *> d_beamformer_data_output_half;
  std::vector<__half *> d_samples_consolidated, d_samples_consolidated_col_maj,
      d_weights, d_weights_permuted;
  std::vector<typename T::PacketScalesType *> d_scales;

  BeamWeights *h_weights;

  int visibilities_start_seq_num;
  int visibilities_end_seq_num;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;

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
    tensor_16.runPermutation(
        "weightsInputToCCGLIB", alpha, (__half *)d_weights[current_buffer],
        (__half *)d_weights_permuted[current_buffer], streams[current_buffer]);
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
  LambdaCorrBeamOnlyGPUPipeline(const int num_buffers,
                                BeamWeightsT<T> *h_weights)

      : num_buffers(num_buffers), h_weights(h_weights),

        correlator(cu::Device(0),
                   16, // tcc::Format::fp16,
                   T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),

        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128)

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
    d_weights_permuted.resize(num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_scaled.resize(num_buffers);
    d_samples_half.resize(num_buffers);
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
    d_visibilities_baseline.resize(num_buffers);
    d_visibilities_trimmed_baseline.resize(num_buffers);
    d_visibilities_trimmed.resize(num_buffers);

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
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_baseline[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_trimmed_baseline[i],
                            sizeof(TrimmedVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_trimmed[i],
                            sizeof(TrimmedVisibilities)));
    }

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    const std::complex<float> alpha_ccglib = {1, 0};
    const std::complex<float> beta_ccglib = {0, 0};
    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, streams[i], ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib));
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
  ~LambdaCorrBeamOnlyGPUPipeline() {
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

    for (auto samples_padded : d_samples_padded) {
      cudaFree(samples_padded);
    }

    for (auto samples_consolidated : d_samples_consolidated) {
      cudaFree(samples_consolidated);
    }

    for (auto samples_consolidated_col_maj : d_samples_consolidated_col_maj) {
      cudaFree(samples_consolidated_col_maj);
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

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

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

template <typename T, int NR_SIGNAL_EIGENVECTORS, int NR_RUNS_TO_AVERAGE>
class LambdaProjectionPipeline : public GPUPipeline {

  static_assert(NR_SIGNAL_EIGENVECTORS >= 1 &&
                    NR_SIGNAL_EIGENVECTORS <= T::NR_RECEIVERS,
                "NR_SIGNAL_EIGENVECTORS must be in [1, NR_RECEIVERS]");
  static_assert(NR_RUNS_TO_AVERAGE >= 1, "NR_RUNS_TO_AVERAGE must be >= 1");

private:
  // -------------------------------------------------------------------------
  // Compile-time constants
  // -------------------------------------------------------------------------
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // fp16 pipeline
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

  // cuSOLVER batch size: one (NR_RECEIVERS × NR_RECEIVERS) matrix per
  // channel × pol × pol, matching LambdaGPUPipeline exactly.
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS;

  // cuBLAS batch size for the cherk calls (same logical grouping).
  static constexpr int CUBLAS_BATCH_SIZE = CUSOLVER_BATCH_SIZE;

  inline static const __half alpha_16 = __float2half(1.0f);
  static constexpr float alpha_32 = 1.0f;

  // -------------------------------------------------------------------------
  // Array-type aliases
  // -------------------------------------------------------------------------
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

  // Full NR_RECEIVERS × NR_RECEIVERS matrices (one per channel × pol × pol),
  // laid out as a flat batch for cuSOLVER / cuBLAS.
  // Shape: [CUSOLVER_BATCH_SIZE][NR_RECEIVERS][NR_RECEIVERS]
  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  // Eigenvalues: one real vector of length NR_RECEIVERS per batch element.
  using Eigenvalues = float[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                           [T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  // Accumulated projection matrix P_acc = sum_k U_k U_k^H, same shape as the
  // full correlation matrix.
  using ProjectionAccumulator = DecompositionVisibilities;

  // -------------------------------------------------------------------------
  // Tensor-mode labels (single-character axis identifiers)
  //
  // a = unpadded baselines
  // b = block
  // c = channel
  // d = padded receivers
  // f = fpga
  // l = baseline (padded)
  // n = receivers per packet
  // o = packets for correlation
  // p = polarization
  // q = second polarization
  // r = receiver
  // s = time consolidated (block × time)
  // t = times per block
  // u = time steps per packet
  // z = complex (real, imaginary)
  // -------------------------------------------------------------------------
  inline static const std::vector<int> modePacket{'c', 'o', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'l', 'c', 'p', 'q',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'a', 'c', 'p',
                                                                  'q', 'z'};
  inline static const std::vector<int> modeVisCorrTrimmed{'c', 'a', 'p', 'q',
                                                          'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'q', 'a', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'f', T::NR_FPGA_SOURCES},
      {'l', NR_BASELINES},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'o', T::NR_PACKETS_FOR_CORRELATION},
      {'p', T::NR_POLARIZATIONS},
      {'q', T::NR_POLARIZATIONS},
      {'r', T::NR_RECEIVERS},
      {'s', NR_BLOCKS_FOR_CORRELATION *NR_TIMES_PER_BLOCK},
      {'t', NR_TIMES_PER_BLOCK},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'z', 2},
  };

  // -------------------------------------------------------------------------
  // Per-buffer resources (RAII, move-only)
  // -------------------------------------------------------------------------
  struct PipelineResources {
    cudaStream_t stream = nullptr;
    cudaStream_t host_stream = nullptr;

    // Raw samples
    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half;
    DevicePtr<typename T::HalfPacketSamplesType> samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;

    // Correlator I/O
    DevicePtr<CorrelatorInput> correlator_input;
    DevicePtr<CorrelatorOutput> correlator_output;

    // Intermediate visibility layout buffers
    DevicePtr<Visibilities> visibilities_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed;

    // Per-block correlation matrix (input to cuSOLVER, overwritten in-place
    // with eigenvectors on output).
    DevicePtr<DecompositionVisibilities> decomp_visibilities;

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues;

    // cuSOLVER handles / workspace
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverDnParams_t cusolver_params = nullptr;
    DevicePtr<int> cusolver_info; // [CUSOLVER_BATCH_SIZE]
    DevicePtr<void> cusolver_work_device;
    void *cusolver_work_host = nullptr;
    size_t cusolver_work_device_size = 0;
    size_t cusolver_work_host_size = 0;

    // cuBLAS handle for cherk (UU^H accumulation)
    cublasHandle_t cublas_handle = nullptr;

    // Per-block projection matrix P = U U^H, written by cherk before being
    // added to the shared accumulator.
    DevicePtr<DecompositionVisibilities> projection_block;

    // -----------------------------------------------------------------------
    PipelineResources() = default;

    explicit PipelineResources(
        const DecompositionVisibilities *decomp_for_workspace_query)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_padding(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
          cusolver_info(
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))),
          projection_block(make_device_ptr<DecompositionVisibilities>()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      // cuSOLVER
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
      CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

      CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
          cusolver_handle, cusolver_params, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER, T::NR_RECEIVERS, CUDA_C_32F,
          decomp_for_workspace_query, T::NR_RECEIVERS, CUDA_R_32F, nullptr,
          CUDA_C_32F, &cusolver_work_device_size, &cusolver_work_host_size,
          CUSOLVER_BATCH_SIZE));

      cusolver_work_device = make_device_ptr<void>(cusolver_work_device_size);
      cusolver_work_host = std::malloc(cusolver_work_host_size);

      // cuBLAS
      CUBLAS_CHECK(cublasCreate(&cublas_handle));
      CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    }

    ~PipelineResources() {
      if (stream)
        cudaStreamDestroy(stream);
      if (host_stream)
        cudaStreamDestroy(host_stream);
      if (cusolver_handle)
        cusolverDnDestroy(cusolver_handle);
      if (cusolver_params)
        cusolverDnDestroyParams(cusolver_params);
      if (cusolver_work_host)
        std::free(cusolver_work_host);
      if (cublas_handle)
        cublasDestroy(cublas_handle);
    }

    // Move-only
    PipelineResources(PipelineResources &&o) noexcept
        : stream(o.stream), host_stream(o.host_stream),
          samples_entry(std::move(o.samples_entry)),
          scales(std::move(o.scales)), samples_half(std::move(o.samples_half)),
          samples_padding(std::move(o.samples_padding)),
          samples_padded(std::move(o.samples_padded)),
          correlator_input(std::move(o.correlator_input)),
          correlator_output(std::move(o.correlator_output)),
          visibilities_baseline(std::move(o.visibilities_baseline)),
          visibilities_trimmed_baseline(
              std::move(o.visibilities_trimmed_baseline)),
          visibilities_trimmed(std::move(o.visibilities_trimmed)),
          decomp_visibilities(std::move(o.decomp_visibilities)),
          eigenvalues(std::move(o.eigenvalues)),
          cusolver_handle(o.cusolver_handle),
          cusolver_params(o.cusolver_params),
          cusolver_info(std::move(o.cusolver_info)),
          cusolver_work_device(std::move(o.cusolver_work_device)),
          cusolver_work_host(o.cusolver_work_host),
          cusolver_work_device_size(o.cusolver_work_device_size),
          cusolver_work_host_size(o.cusolver_work_host_size),
          cublas_handle(o.cublas_handle),
          projection_block(std::move(o.projection_block)) {
      o.stream = nullptr;
      o.host_stream = nullptr;
      o.cusolver_handle = nullptr;
      o.cusolver_params = nullptr;
      o.cusolver_work_host = nullptr;
      o.cublas_handle = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&o) noexcept {
      if (this != &o) {
        this->~PipelineResources();
        new (this) PipelineResources(std::move(o));
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  int num_buffers;
  int current_buffer = 0;
  std::vector<PipelineResources> buffers;

  tcc::Correlator correlator;
  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  // Shared projection accumulator (lives outside per-buffer resources, exactly
  // like d_visibilities_accumulator in LambdaGPUPipeline).
  DecompositionVisibilities *d_projection_accumulator = nullptr;

  // Final averaged projection matrix. cuSOLVER overwrites this in-place with
  // eigenvectors, which are then transferred to the Output landing pointer.
  DecompositionVisibilities *d_projection_averaged = nullptr;

  // Scratch eigenvalue buffer for the final cuSOLVER call — allocated once
  // since dump_projection is serialised by cudaDeviceSynchronize.
  // These eigenvalues are NOT exported; only the eigenvectors are.
  Eigenvalues *d_projection_eigenvalues_scratch = nullptr;

  std::atomic<int> num_runs_integrated{0};

  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;

public:
  // -------------------------------------------------------------------------
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start_total = clock::now();

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    LOG_INFO("LambdaProjectionPipeline run: start_seq={} end_seq={}",
             start_seq_num, end_seq_num);

    auto &b = buffers[current_buffer];

    // ------------------------------------------------------------------
    // 1. Transfer raw samples and scales to device
    // ------------------------------------------------------------------
    auto cpu_start = clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        b.samples_entry.get(), packet_data->get_samples_ptr(),
        packet_data->get_samples_elements_size(), cudaMemcpyDefault, b.stream));
    LOG_DEBUG("CPU overhead for samples cudaMemcpyAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    cpu_start = clock::now();
    CUDA_CHECK(cudaMemcpyAsync(b.scales.get(), packet_data->get_scales_ptr(),
                               packet_data->get_scales_element_size(),
                               cudaMemcpyDefault, b.stream));
    LOG_DEBUG("CPU overhead for scales cudaMemcpyAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 2. Release the input buffer asynchronously on the host stream so
    //    the GPU pipeline stream is not stalled.
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    CUDA_CHECK(
        cudaLaunchHostFunc(b.host_stream, release_buffer_host_func, ctx));
    LOG_DEBUG("CPU time for BufferReleaseContext + cudaLaunchHostFunc: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 3. Scale integer samples to fp16
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    scale_and_convert_to_half<
        typename T::InputPacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfInputPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
        T::NR_TIME_STEPS_PER_PACKET, T::NR_PACKETS_FOR_CORRELATION>(
        reinterpret_cast<typename T::InputPacketSamplesPlanarType *>(
            b.samples_entry.get()),
        b.scales.get(),
        reinterpret_cast<typename T::HalfInputPacketSamplesPlanarType *>(
            b.samples_half.get()),
        b.stream);
    LOG_DEBUG("CPU time for scale_and_convert_to_half launch: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 4. Permute packet → padding layout
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    tensor_16.runPermutation(
        "packetToPadding", alpha_16,
        reinterpret_cast<__half *>(b.samples_half.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);
    LOG_DEBUG("CPU time for tensor packetToPadding: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 5. Copy unpadded → padded buffer then zero-fill the padding region
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketSamplesType),
                               cudaMemcpyDefault, b.stream));
    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketSamplesType),
                        b.stream));
    LOG_DEBUG("CPU overhead for padding memcpy + memset: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 6. Permute padded → correlator input layout
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    tensor_16.runPermutation(
        "paddedToCorrInput", alpha_16,
        reinterpret_cast<__half *>(b.samples_padded.get()),
        reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);
    LOG_DEBUG("CPU time for tensor paddedToCorrInput: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 7. Cross-correlate with tcc::Correlator
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    correlator.launchAsync(
        static_cast<CUstream>(b.stream),
        reinterpret_cast<CUdeviceptr>(b.correlator_output.get()),
        reinterpret_cast<CUdeviceptr>(b.correlator_input.get()));
    LOG_DEBUG("CPU time for correlator.launchAsync: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 8. Rearrange correlator output to baseline-major, then trim padding
    // ------------------------------------------------------------------
    tensor_32.runPermutation(
        "visCorrToBaseline", alpha_32,
        reinterpret_cast<float *>(b.correlator_output.get()),
        reinterpret_cast<float *>(b.visibilities_baseline.get()), b.stream);

    CUDA_CHECK(cudaMemcpyAsync(
        b.visibilities_trimmed_baseline.get(), b.visibilities_baseline.get(),
        sizeof(TrimmedVisibilities), cudaMemcpyDefault, b.stream));

    tensor_32.runPermutation(
        "visBaselineTrimmedToTrimmed", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed_baseline.get()),
        reinterpret_cast<float *>(b.visibilities_trimmed.get()), b.stream);

    // ------------------------------------------------------------------
    // 9. Expand triangular baselines → full NR_RECEIVERS × NR_RECEIVERS
    //    Hermitian matrices (one per channel × pol × pol).
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    tensor_32.runPermutation(
        "visCorrToDecomp", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed.get()),
        reinterpret_cast<float *>(b.decomp_visibilities.get()), b.stream);
    LOG_DEBUG("CPU time for tensor visCorrToDecomp: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    unpack_triangular_baseline_batch_launch<cuComplex>(
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        T::NR_RECEIVERS, CUSOLVER_BATCH_SIZE, T::NR_CHANNELS, b.stream);

    // ------------------------------------------------------------------
    // 10. Eigen-decompose R per batch element.
    //     cuSOLVER overwrites decomp_visibilities with eigenvectors
    //     (columns, ascending eigenvalue order).
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));
    LOG_DEBUG("CPU time for cusolverDnXsyevBatched launch: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 11. Form P_block = U U^H via batched cuBLAS cherk.
    //
    //     After cuSOLVER, decomp_visibilities holds the full eigenvector
    //     matrix V (NR_RECEIVERS × NR_RECEIVERS, column = eigenvector,
    //     ascending order).  The signal subspace U consists of the last
    //     NR_SIGNAL_EIGENVECTORS columns, i.e. the sub-matrix starting at
    //     column offset (NR_RECEIVERS - NR_SIGNAL_EIGENVECTORS).
    //
    //     cuSOLVER stores column-major (Fortran order), so column j starts
    //     at row-offset 0 and the pointer to column j is:
    //       V_ptr + j * NR_RECEIVERS    (in cuComplex elements)
    //
    //     cherk computes:  C ← alpha * A * A^H + beta * C
    //       A = U  (NR_RECEIVERS × NR_SIGNAL_EIGENVECTORS, col-major)
    //       C = P  (NR_RECEIVERS × NR_RECEIVERS, col-major)
    //
    //     We call it once per batch element in a simple loop.  A batched
    //     cherk variant is not available in cuBLAS; the loop is over
    //     CUSOLVER_BATCH_SIZE elements and is negligible CPU overhead
    //     compared with the GPU kernels.
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    {
      constexpr int N = T::NR_RECEIVERS;
      constexpr int K = NR_SIGNAL_EIGENVECTORS;
      constexpr int col_offset = N - K; // first signal-subspace column
      const float herk_alpha = 1.0f;
      const float herk_beta = 0.0f; // overwrite projection_block

      auto *V_base = reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
      auto *P_base = reinterpret_cast<cuComplex *>(b.projection_block.get());

      for (int batch = 0; batch < CUBLAS_BATCH_SIZE; ++batch) {
        // Pointer to the start of eigenvector matrix for this batch element.
        cuComplex *V_batch = V_base + batch * N * N;
        // Pointer to signal-subspace U = last K columns of V_batch.
        cuComplex *U = V_batch + col_offset * N;
        // Pointer to output P for this batch element.
        cuComplex *P_batch = P_base + batch * N * N;

        CUBLAS_CHECK(
            cublasCherk(b.cublas_handle,
                        CUBLAS_FILL_MODE_UPPER, // fill upper triangle of P
                        CUBLAS_OP_N,            // no transpose on U
                        N, K, &herk_alpha, U, N, &herk_beta, P_batch, N));
      }
    }
    LOG_DEBUG("CPU time for batched cublasCherk (UU^H): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 12. Add P_block to the shared accumulator.
    //     accumulate_visibilities adds src into dst element-wise on the
    //     stream (matching usage in LambdaGPUPipeline).
    // ------------------------------------------------------------------
    cpu_start = clock::now();
    accumulate_visibilities(
        reinterpret_cast<float *>(b.projection_block.get()),
        reinterpret_cast<float *>(d_projection_accumulator),
        // Factor of 2 for complex (real + imag); full square matrix.
        2 * CUSOLVER_BATCH_SIZE * T::NR_RECEIVERS * T::NR_RECEIVERS, b.stream);
    LOG_DEBUG("CPU time for accumulate_visibilities (projection): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start)
                  .count());

    // ------------------------------------------------------------------
    // 13. After NR_RUNS_TO_AVERAGE blocks, average, decompose and export.
    // ------------------------------------------------------------------
    if (!dummy_run) {
      num_runs_integrated.fetch_add(1);
      if (num_runs_integrated.load() >= NR_RUNS_TO_AVERAGE) {
        dump_projection(start_seq_num, end_seq_num);
      }
    }

    // ------------------------------------------------------------------
    // 14. Rotate buffer index.
    // ------------------------------------------------------------------
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }

    LOG_DEBUG("Total CPU time for execute_pipeline: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - cpu_start_total)
                  .count());
  }

  // -------------------------------------------------------------------------
  // dump_visibilities — public override required by GPUPipeline.
  //
  // Called externally (e.g. on shutdown) to flush whatever has accumulated.
  // We delegate to dump_projection with end_seq_num = 0 (unknown) matching
  // the convention in LambdaGPUPipeline.
  // -------------------------------------------------------------------------
  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    if (num_runs_integrated.load() > 0) {
      dump_projection(/* start */ 0, end_seq_num);
    }
  }

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  LambdaProjectionPipeline(const int num_buffers_in)
      : num_buffers(num_buffers_in),

        correlator(cu::Device(0), 16, T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),

        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128) {
    std::cout << "LambdaProjectionPipeline instantiated:"
              << " NR_CHANNELS=" << T::NR_CHANNELS
              << " NR_RECEIVERS=" << T::NR_PADDED_RECEIVERS
              << " NR_POLARIZATIONS=" << T::NR_POLARIZATIONS
              << " NR_SAMPLES_PER_CH="
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << " NR_SIGNAL_EIGENVECTORS=" << NR_SIGNAL_EIGENVECTORS
              << " NR_RUNS_TO_AVERAGE=" << NR_RUNS_TO_AVERAGE << std::endl;

    // Allocate shared accumulator, averaged-projection, and scratch eigenvalue
    // buffers.  The eigenvalue scratch is needed by the final cuSOLVER call
    // but is NOT exported — only eigenvectors are transferred to the Output.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_projection_accumulator),
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_projection_averaged),
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_projection_eigenvalues_scratch),
                   sizeof(Eigenvalues)));
    CUDA_CHECK(cudaMemset(d_projection_accumulator, 0,
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(cudaMemset(d_projection_averaged, 0,
                          sizeof(DecompositionVisibilities)));

    // We need a valid device pointer for the cuSOLVER workspace query inside
    // PipelineResources.  Use the (already allocated) accumulator pointer;
    // the query is read-only with respect to the matrix pointer.
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(d_projection_accumulator);
    }

    cudaDeviceSynchronize();

    // Tensor descriptors
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");

    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    // Permutation descriptors
    tensor_16.addPermutation("packet", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToPadding");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visCorrTrimmed",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToTrimmed");
    tensor_32.addPermutation("visCorrTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToDecomp");

    // Warm-up to JIT all template kernels before first real data arrives.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, /*dummy_run=*/true);
    cudaDeviceSynchronize();

    // Reset accumulator after warm-up (dummy run may have polluted it with
    // zeros, but be explicit for correctness).
    CUDA_CHECK(cudaMemset(d_projection_accumulator, 0,
                          sizeof(DecompositionVisibilities)));
    num_runs_integrated.store(0);
  }

  // -------------------------------------------------------------------------
  // Destructor
  // -------------------------------------------------------------------------
  ~LambdaProjectionPipeline() {
    cudaFree(d_projection_accumulator);
    cudaFree(d_projection_averaged);
    cudaFree(d_projection_eigenvalues_scratch);
    // buffers destructs automatically, cleaning up per-buffer GPU memory,
    // cuSOLVER handles, and cuBLAS handles.
  }

private:
  // -------------------------------------------------------------------------
  // dump_projection
  //
  // 1. Synchronise all streams to ensure all pending cherk / accumulate
  //    work is complete.
  // 2. Divide the accumulator by the number of runs (scale_visibilities)
  //    to form the time-averaged projection matrix P_avg.
  // 3. Eigen-decompose P_avg using the first buffer's cuSOLVER handle.
  //    Eigenvectors are written in-place into d_projection_averaged;
  //    eigenvalues go to d_projection_eigenvalues_scratch and are discarded.
  // 4. Copy eigenvectors only to the Output landing pointer.
  // 5. Reset the accumulator and run counter.
  // -------------------------------------------------------------------------
  void dump_projection(const uint64_t start_seq_num,
                       const uint64_t end_seq_num) {
    LOG_INFO("LambdaProjectionPipeline: dumping averaged projection "
             "(num_runs={})",
             num_runs_integrated.load());

    const int runs = num_runs_integrated.load();

    // Synchronise device so all pending GPU work is complete.
    cudaDeviceSynchronize();

    // Copy accumulator → averaged buffer, then divide by run count.
    // We reuse stream 0 (buffers[0].stream) for these serialised steps.
    auto &b0 = buffers[0];

    CUDA_CHECK(cudaMemcpyAsync(d_projection_averaged, d_projection_accumulator,
                               sizeof(DecompositionVisibilities),
                               cudaMemcpyDefault, b0.stream));

    // Divide each element by 'runs' (scale factor = 1/runs).
    // scale_visibilities is assumed to exist alongside accumulate_visibilities.
    scale_visibilities(reinterpret_cast<float *>(d_projection_averaged),
                       2 * CUSOLVER_BATCH_SIZE * T::NR_RECEIVERS *
                           T::NR_RECEIVERS,
                       1.0f / static_cast<float>(runs), b0.stream);

    // Eigen-decompose the averaged projection matrix.
    // cuSOLVER writes eigenvectors in-place (ascending eigenvalue order).
    // Eigenvalues land in d_projection_eigenvalues_scratch — they are NOT
    // exported.
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b0.cusolver_handle, b0.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(d_projection_averaged), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(d_projection_eigenvalues_scratch),
        CUDA_C_32F, b0.cusolver_work_device.get(), b0.cusolver_work_device_size,
        b0.cusolver_work_host, b0.cusolver_work_host_size,
        b0.cusolver_info.get(), CUSOLVER_BATCH_SIZE));

    cudaDeviceSynchronize();

    // Export eigenvectors only via the Output interface.
    if (output_ != nullptr) {
      size_t block_num = output_->register_eigendecomposition_data_block(
          start_seq_num, end_seq_num);

      // d_projection_averaged now holds the eigenvectors (cuSOLVER in-place).
      //
      void *eigval_ptr =
          output_->get_eigenvalues_data_landing_pointer(block_num);
      void *eigvec_ptr =
          output_->get_eigenvectors_data_landing_pointer(block_num);
      CUDA_CHECK(cudaMemcpyAsync(eigvec_ptr, d_projection_averaged,
                                 sizeof(DecompositionVisibilities),
                                 cudaMemcpyDefault, b0.stream));

      CUDA_CHECK(cudaMemcpyAsync(eigval_ptr, d_projection_eigenvalues_scratch,
                                 sizeof(Eigenvalues), cudaMemcpyDefault,
                                 b0.stream));

      auto *ctx = new OutputTransferCompleteContext{.output = this->output_,
                                                    .block_index = block_num};
      CUDA_CHECK(cudaLaunchHostFunc(
          b0.stream, eigen_output_transfer_complete_host_func, ctx));
    }

    // Reset accumulator and run counter.
    CUDA_CHECK(cudaMemsetAsync(d_projection_accumulator, 0,
                               sizeof(DecompositionVisibilities), b0.stream));
    num_runs_integrated.store(0);

    cudaDeviceSynchronize();
    LOG_INFO("LambdaProjectionPipeline: dump complete.");
  }
};

// All pulsar-specific parameters passed to the constructor.
// period_samples   : pulsar period in downsampled time bins
//                    (i.e. at rate = raw_rate / FFT_DOWNSAMPLE_FACTOR).
// n_bins           : number of phase bins in the fold profile.
// dm               : dispersion measure (pc cm^-3); pass 0 to disable.
// ref_freq_mhz     : reference frequency (MHz) at which DM delay = 0.
// chan_bw_mhz      : bandwidth of a single coarse channel (MHz).
// lowest_chan_freq_mhz : centre frequency of coarse channel 0 (MHz).
struct PulsarFoldParameters {
  double period_samples;
  int n_bins;
  double dm;
  double ref_freq_mhz;
  double chan_bw_mhz;
  double lowest_chan_freq_mhz;
};

// Signal path per execute_pipeline() call:
//  1. H→D transfer of raw samples and scales.
//  2. Release input buffer on host stream.
//  3. scale_and_convert_to_half — int → fp16.
//  4. cuTENSOR permutation → cuFFT-ready axis order.
//  5. get_data_for_multi_channel_fft_launch — fp16 → fp32.
//  6. Batched forward FFT: NR_CHANNELS × NR_POLARIZATIONS × NR_RECEIVERS
//     batches, each of length NR_TIME_STEPS_FOR_CORR (= NR_FINE_CHANNELS).
//  7. detect_and_downsample_multi_channel_fft_launch — |z|², time-average.
//     Output: [NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
//             [NR_TIME_BINS_PER_BLOCK][NR_FINE_CHANNELS]
//  8. incoherent_sum_launch — sum over NR_RECEIVERS.
//     Output: [NR_CHANNELS][NR_FINE_CHANNELS][NR_POLARIZATIONS]
//             [NR_TIME_BINS_PER_BLOCK]
//  9. fold_and_accumulate_launch — phase-bin each time sample (with per-channel
//     DM delay correction) and atomically accumulate into the fold profile.
// 10. After nr_blocks_per_dump blocks, copy D→H and reset the accumulator.
//
// Exported fold profile layout:
//   float[NR_CHANNELS][NR_FINE_CHANNELS][NR_POLARIZATIONS][n_bins]
template <typename T, size_t NR_FINE_CHANNELS = 16>
class LambdaPulsarFoldPipeline : public GPUPipeline {

private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16;

  static constexpr int NR_TIME_STEPS_FOR_CORR =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;

  static constexpr int COMPLEX = 2;
  static constexpr int FFT_SIZE = NR_FINE_CHANNELS;

  static constexpr int NR_SPECTRA_PER_BLOCK = NR_TIME_STEPS_FOR_CORR / FFT_SIZE;

  using IncoherentSumType = float[T::NR_CHANNELS][NR_FINE_CHANNELS]
                                 [T::NR_POLARIZATIONS][NR_SPECTRA_PER_BLOCK];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
           [NR_SPECTRA_PER_BLOCK][FFT_SIZE][COMPLEX];

  inline static const __half alpha = __float2half(1.0f);
  inline static const float alpha_32 = 1.0f;

  inline static const std::vector<int> modePacket{'c', 'o', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};

  inline static const std::vector<int> modeCUFFTOutput{'c', 'p', 'r',
                                                       's', 'e', 'z'};

  inline static const std::vector<int> modeIncoherentSumInput{'c', 'e', 'p',
                                                              's', 'r', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'s', NR_SPECTRA_PER_BLOCK},
      {'c', T::NR_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'r', T::NR_FPGA_SOURCES *T::NR_RECEIVERS_PER_PACKET},
      {'o', T::NR_PACKETS_FOR_CORRELATION},
      {'p', T::NR_POLARIZATIONS},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'e', NR_FINE_CHANNELS},
      {'z', 2},
  };

  struct PipelineResources {
    cudaStream_t stream = nullptr;
    cudaStream_t host_stream = nullptr;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half;
    DevicePtr<typename T::FFTCUFFTPreprocessingType>
        samples_cufft_preprocessing;
    DevicePtr<typename T::MultiChannelFFTCUFFTInputType> samples_cufft_input;
    DevicePtr<FFTOutputType> samples_cufft_output, incoherent_sum_input;
    DevicePtr<IncoherentSumType> incoherent_sum;
    DevicePtr<void> cufft_work_area;

    PipelineResources(size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_cufft_preprocessing(
              make_device_ptr<typename T::FFTCUFFTPreprocessingType>()),
          samples_cufft_input(
              make_device_ptr<typename T::MultiChannelFFTCUFFTInputType>()),
          samples_cufft_output(make_device_ptr<FFTOutputType>()),
          incoherent_sum_input(make_device_ptr<FFTOutputType>()),
          incoherent_sum(make_device_ptr<IncoherentSumType>()),
          cufft_work_area(make_device_ptr<void>(work_size)) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));
    }

    ~PipelineResources() {
      if (stream)
        cudaStreamDestroy(stream);
      if (host_stream)
        cudaStreamDestroy(host_stream);
    }

    PipelineResources(PipelineResources &&o) noexcept
        : stream(o.stream), host_stream(o.host_stream),
          fft_plan(std::move(o.fft_plan)),
          samples_entry(std::move(o.samples_entry)),
          scales(std::move(o.scales)), samples_half(std::move(o.samples_half)),
          samples_cufft_preprocessing(std::move(o.samples_cufft_preprocessing)),
          samples_cufft_input(std::move(o.samples_cufft_input)),
          samples_cufft_output(std::move(o.samples_cufft_output)),
          incoherent_sum(std::move(o.incoherent_sum)),
          incoherent_sum_input(std::move(o.incoherent_sum_input)),
          cufft_work_area(std::move(o.cufft_work_area)) {
      o.stream = nullptr;
      o.host_stream = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&o) noexcept {
      if (this != &o) {
        this->~PipelineResources();
        new (this) PipelineResources(std::move(o));
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers;
  int current_buffer = 0;
  std::vector<PipelineResources> buffers;

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  PulsarFoldParameters pulsar_params_;

  std::vector<int32_t> dm_delays_; // host copy
  int32_t *d_dm_delays_ =
      nullptr; // device copy [NR_CHANNELS * NR_FINE_CHANNELS]

  float *d_fold_accumulator_ = nullptr;
  float *d_fold_output_ = nullptr;
  uint32_t *d_hit_counts_ = nullptr;
  size_t fold_accumulator_elements_ = 0;

  int nr_blocks_per_dump_;

  // Counts downsampled time bins elapsed since pipeline start; never reset,
  // so pulsar phase tracks continuously across dump boundaries.
  std::atomic<int64_t> total_samples_elapsed_{0};
  std::atomic<int> blocks_accumulated_{0};

  uint64_t fold_start_seq_ = 0;
  bool fold_start_set_ = false;

  // Δt [s] = 4.148808 × DM × (1/f² − 1/f_ref²), f in MHz.
  static double dm_delay_seconds(double dm, double f_mhz, double f_ref_mhz) {
    if (dm == 0.0)
      return 0.0;
    constexpr double K_DM_S = 4.148808;
    return K_DM_S * dm *
           (1.0 / (f_mhz * f_mhz) - 1.0 / (f_ref_mhz * f_ref_mhz));
  }

  // Precompute per-(coarse, fine) channel DM delay in downsampled bins and
  // upload to device. sample_rate_hz is the rate after time-averaging
  // (= raw_rate / FFT_DOWNSAMPLE_FACTOR).
  void precompute_dm_delays(double sample_rate_hz) {
    const int n_coarse = T::NR_CHANNELS;
    const int n_fine = NR_FINE_CHANNELS;
    dm_delays_.resize(n_coarse * n_fine);

    const double fine_bw_mhz =
        pulsar_params_.chan_bw_mhz / static_cast<double>(n_fine);

    for (int c = 0; c < n_coarse; ++c) {
      const double coarse_centre_mhz = pulsar_params_.lowest_chan_freq_mhz +
                                       0.5 * pulsar_params_.chan_bw_mhz +
                                       c * pulsar_params_.chan_bw_mhz * 27 / 32;

      std::cout << "Coarse Channel " << c << " has center " << coarse_centre_mhz
                << " MHz.\n";

      for (int f = 0; f < n_fine; ++f) {
        // FFT bin ordering: 0 = DC, 1..N/2-1 = positive, N/2..N-1 = negative.
        const double fine_offset_mhz =
            (f < n_fine / 2) ? f * fine_bw_mhz : (f - n_fine) * fine_bw_mhz;

        const double chan_freq_mhz = coarse_centre_mhz + fine_offset_mhz;
        const double delay_s = dm_delay_seconds(
            pulsar_params_.dm, chan_freq_mhz, pulsar_params_.ref_freq_mhz);

        dm_delays_[c * n_fine + f] =
            static_cast<int32_t>(std::round(delay_s * sample_rate_hz));
        std::cout << "Delay samples for channel " << c << " / " << f << " is "
                  << dm_delays_[c * n_fine + f] << std::endl;
      }
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dm_delays_),
                          dm_delays_.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_dm_delays_, dm_delays_.data(),
                          dm_delays_.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  }

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr)
      throw std::logic_error("State has not been set on GPUPipeline object!");

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    LOG_INFO("LambdaPulsarFoldPipeline: start_seq={} end_seq={}", start_seq_num,
             end_seq_num);

    auto &b = buffers[current_buffer];

    CUDA_CHECK(cudaMemcpyAsync(
        b.samples_entry.get(), packet_data->get_samples_ptr(),
        packet_data->get_samples_elements_size(), cudaMemcpyDefault, b.stream));

    CUDA_CHECK(cudaMemcpyAsync(b.scales.get(), packet_data->get_scales_ptr(),
                               packet_data->get_scales_element_size(),
                               cudaMemcpyDefault, b.stream));
    auto *ctx =
        new BufferReleaseContext{.state = this->state_,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    CUDA_CHECK(
        cudaLaunchHostFunc(b.host_stream, release_buffer_host_func, ctx));

    scale_and_convert_to_half<
        typename T::InputPacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfInputPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
        T::NR_TIME_STEPS_PER_PACKET, T::NR_PACKETS_FOR_CORRELATION>(
        reinterpret_cast<typename T::InputPacketSamplesPlanarType *>(
            b.samples_entry.get()),
        b.scales.get(),
        reinterpret_cast<typename T::HalfInputPacketSamplesPlanarType *>(
            b.samples_half.get()),
        b.stream);

    tensor_16.runPermutation(
        "packetToCUFFTInput", alpha,
        reinterpret_cast<__half *>(b.samples_half.get()),
        reinterpret_cast<__half *>(b.samples_cufft_preprocessing.get()),
        b.stream);

    get_data_for_multi_channel_fft_launch<
        typename T::FFTCUFFTPreprocessingType,
        typename T::MultiChannelFFTCUFFTInputType>(
        reinterpret_cast<typename T::FFTCUFFTPreprocessingType *>(
            b.samples_cufft_preprocessing.get()),
        b.samples_cufft_input.get(), T::NR_CHANNELS, T::NR_POLARIZATIONS,
        NR_TIME_STEPS_FOR_CORR, T::NR_RECEIVERS, b.stream);

    CUFFT_CHECK(cufftXtExec(
        b.fft_plan, reinterpret_cast<void *>(b.samples_cufft_input.get()),
        reinterpret_cast<void *>(b.samples_cufft_output.get()), CUFFT_FORWARD));

    tensor_32.runPermutation(
        "cufftOutputToIncoherentSum", alpha_32,
        reinterpret_cast<float *>(b.samples_cufft_output.get()),
        reinterpret_cast<float *>(b.incoherent_sum_input.get()), b.stream);

    incoherent_sum_launch(
        reinterpret_cast<float2 *>(b.incoherent_sum_input.get()),
        reinterpret_cast<float *>(b.incoherent_sum.get()), T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, NR_FINE_CHANNELS,
        NR_SPECTRA_PER_BLOCK, b.stream);
    // Per time bin: phase = fmod((abs_sample - dm_delay[c][f]) /
    // period_samples, 1.0) then atomically accumulate into the corresponding
    // bin.
    fold_and_accumulate_launch(
        reinterpret_cast<const float *>(b.incoherent_sum.get()),
        d_fold_accumulator_, d_hit_counts_, d_dm_delays_,
        total_samples_elapsed_.load(), pulsar_params_.period_samples,
        pulsar_params_.n_bins, T::NR_CHANNELS, NR_FINE_CHANNELS,
        T::NR_POLARIZATIONS, NR_SPECTRA_PER_BLOCK, b.stream);

    if (!dummy_run) {
      if (!fold_start_set_) {
        fold_start_seq_ = start_seq_num;
        fold_start_set_ = true;
      }
      total_samples_elapsed_.fetch_add(NR_SPECTRA_PER_BLOCK);
      if (blocks_accumulated_.fetch_add(1) + 1 >= nr_blocks_per_dump_)
        dump_fold(fold_start_seq_, end_seq_num, b.stream);

      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    if (blocks_accumulated_.load() > 0)
      dump_fold(fold_start_seq_, end_seq_num, buffers[0].stream);
  }

  LambdaPulsarFoldPipeline(const int num_buffers_in,
                           PulsarFoldParameters params, int nr_blocks_per_dump)
      : num_buffers(num_buffers_in), pulsar_params_(params),
        nr_blocks_per_dump_(nr_blocks_per_dump),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128) {

    std::cout << "LambdaPulsarFoldPipeline instantiated:"
              << " NR_CHANNELS=" << T::NR_CHANNELS
              << " NR_RECEIVERS=" << T::NR_RECEIVERS
              << " NR_POLARIZATIONS=" << T::NR_POLARIZATIONS
              << " NR_FINE_CHANNELS=" << NR_FINE_CHANNELS
              << " n_bins=" << params.n_bins
              << " period_samples=" << params.period_samples
              << " dm=" << params.dm
              << " nr_blocks_per_dump=" << nr_blocks_per_dump << std::endl;

    long long N[] = {FFT_SIZE};
    const size_t NUM_BATCHES = static_cast<size_t>(T::NR_RECEIVERS) *
                               T::NR_CHANNELS * T::NR_POLARIZATIONS *
                               NR_SPECTRA_PER_BLOCK;
    size_t work_size = 0;
    {
      cufftHandle temp_plan;
      CUFFT_CHECK(cufftCreate(&temp_plan));
      CUFFT_CHECK(cufftXtMakePlanMany(temp_plan, 1, N, NULL, 1, FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, FFT_SIZE, CUDA_C_32F,
                                      NUM_BATCHES, &work_size, CUDA_C_32F));
      cufftDestroy(temp_plan);
    }

    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(work_size);
      auto &b = buffers.back();
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, FFT_SIZE, CUDA_C_32F,
                                      NUM_BATCHES, &work_size, CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));
    }

    fold_accumulator_elements_ = params.n_bins;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_fold_accumulator_),
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_fold_output_),
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit_counts_),
                          fold_accumulator_elements_ * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_fold_accumulator_, 0,
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fold_output_, 0,
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hit_counts_, 0,
                          fold_accumulator_elements_ * sizeof(uint32_t)));

    // Downsampled sample rate: raw bandwidth × 1e6 / downsample factor.
    // period_samples must be expressed at this same rate.
    const double downsampled_rate_hz =
        (pulsar_params_.chan_bw_mhz * 1e6) / NR_FINE_CHANNELS;
    precompute_dm_delays(downsampled_rate_hz);

    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");
    tensor_32.addTensor(modeCUFFTOutput, "cufftOutput");
    tensor_32.addTensor(modeIncoherentSumInput, "incoherentSumInput");
    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");
    tensor_32.addPermutation("cufftOutput", "incoherentSumInput",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "cufftOutputToIncoherentSum");

    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, /*dummy_run=*/true);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemset(d_fold_accumulator_, 0,
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fold_output_, 0,
                          fold_accumulator_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hit_counts_, 0,
                          fold_accumulator_elements_ * sizeof(uint32_t)));
    total_samples_elapsed_.store(0);
    blocks_accumulated_.store(0);
    fold_start_set_ = false;
  }

  ~LambdaPulsarFoldPipeline() {
    if (d_fold_accumulator_)
      cudaFree(d_fold_accumulator_);
    if (d_fold_output_)
      cudaFree(d_fold_output_);
    if (d_dm_delays_)
      cudaFree(d_dm_delays_);
    if (d_hit_counts_)
      cudaFree(d_hit_counts_);
  }

private:
  void dump_fold(const uint64_t start_seq_num, const uint64_t end_seq_num,
                 cudaStream_t stream) {
    LOG_INFO("LambdaPulsarFoldPipeline: dumping fold profile "
             "(blocks_accumulated={})",
             blocks_accumulated_.load());

    normalise_fold_launch(d_fold_accumulator_, d_fold_output_, d_hit_counts_,
                          static_cast<int>(fold_accumulator_elements_), stream);

    if (output_ != nullptr) {
      size_t block_num =
          output_->register_pulsar_fold_block(start_seq_num, end_seq_num);
      void *landing = output_->get_pulsar_fold_landing_pointer(block_num);
      CUDA_CHECK(cudaMemcpyAsync(landing, d_fold_output_,
                                 fold_accumulator_elements_ * sizeof(float),
                                 cudaMemcpyDefault, stream));
      auto *ctx = new OutputTransferCompleteContext{.output = this->output_,
                                                    .block_index = block_num};
      CUDA_CHECK(cudaLaunchHostFunc(
          stream, pulsar_fold_output_transfer_complete_host_func, ctx));
    }

    // Async memset runs after the copy above on the same stream.
    // CUDA_CHECK(cudaMemsetAsync(d_fold_accumulator_, 0,
    //                            fold_accumulator_elements_ * sizeof(float),
    //                            stream));
    blocks_accumulated_.store(0);
    fold_start_set_ = false;
    // total_samples_elapsed_ is not reset — phase is continuous across dumps.
  }
};
