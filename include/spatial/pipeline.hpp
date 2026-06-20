#pragma once
#include "spatial/packet_formats.hpp"
#include "spatial/pointing.hpp"
#include "spatial/spatial.hpp"
#include <chrono>
#include <complex>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <limits>
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
#include <ctime>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "ascii_header.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "futils.h"
#include "ipcio.h"
#include "multilog.h"

template <typename T> struct BeamWeightsT {
  std::complex<__half> weights[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
                              [T::NR_RECEIVERS];
};

// Vacuum speed of light, m/s -- used to convert a geometric path-length
// difference (direction . antenna_position, in metres) into a phase via
// phase = -2*pi*f/c * path_length.
inline constexpr double kSpeedOfLightMetresPerSecond = 299792458.0;

// Synthesizes per-beam steering weights that point each beam at its target
// (resolved to ENU direction cosines via zenith_direction()/
// topocentric_direction(), see pointing.hpp), optionally folding in an
// existing calibration solution as a multiplicative factor.
//
// For each beam/channel/antenna, computes the geometric steering phasor
// exp(i*phase) with phase = -2*pi*f_channel/c * (l*east + m*north + n*up) --
// the phase aligning that antenna's signal with a plane wave arriving from
// the target direction -- then combines it with the optional calibration gain
// G[chan][pol][receiver]: `final_weight = (1/NR_RECEIVERS) * G * exp(i*phase)`.
// Because both factors are per-antenna multiplicative scalars and
// multiplication commutes with the beamforming sum, this is exactly
// equivalent to applying calibration and steering separately, so an existing
// calibration solution plugs straight in. Pass `calibration_gains = nullptr`
// for pure geometric steering (gain 1+0i).
//
// `antenna_mapping` (receiver index -> absolute antenna ID) is used to look
// `antenna_positions` up by absolute ID while writing
// `weights[...][receiver_idx]` in receiver-index order, matching
// get_gains_structure()'s convention. Receivers absent from either map fall
// back to ENUPosition{0,0,0} (zero path-length difference, i.e. zero geometric
// phase).
//
// Receivers mapped to a *negative* antenna ID (AntennaMapRegistry uses -100
// for FPGA streams with no antenna connected, e.g. FPGA 0 streams 0/2/4/6)
// are null inputs: their weights are left exactly zero for every
// beam/channel/pol, so the disconnected stream's noise is never summed into
// a beam.
template <typename T>
inline BeamWeightsT<T> compute_steering_weights(
    const std::vector<BeamTarget> &targets,
    const std::unordered_map<int, ENUPosition> &antenna_positions,
    const std::unordered_map<int, int> &antenna_mapping,
    const FrequencyPlan &frequency_plan, int min_freq_channel,
    const ArrayLocation &array_location,
    std::chrono::system_clock::time_point utc_time,
    const typename T::AntennaGains *calibration_gains = nullptr) {
  BeamWeightsT<T> result{};

  for (size_t b = 0; b < T::NR_BEAMS; ++b) {
    static const BeamTarget kDefaultZenithTarget{};
    const BeamTarget &target =
        (b < targets.size()) ? targets[b] : kDefaultZenithTarget;

    DirectionCosines dc =
        (target.mode == "zenith")
            ? zenith_direction()
            : topocentric_direction(target.ra_deg, target.dec_deg, utc_time,
                                    array_location.latitude_deg,
                                    array_location.longitude_deg,
                                    array_location.height_m);

    for (size_t chan = 0; chan < T::NR_CHANNELS; ++chan) {
      double frequency_hz = channel_to_frequency_hz(
          min_freq_channel + static_cast<int>(chan), frequency_plan);
      double phase_scale =
          -2.0 * M_PI * frequency_hz / kSpeedOfLightMetresPerSecond;

      for (size_t receiver_idx = 0; receiver_idx < T::NR_RECEIVERS;
           ++receiver_idx) {
        ENUPosition enu{};
        auto mapping_it = antenna_mapping.find(static_cast<int>(receiver_idx));
        if (mapping_it != antenna_mapping.end()) {
          // Negative antenna ID = null input (no antenna on this FPGA
          // stream). Leave its weights at the zero `result{}` was
          // initialized with -- the ENUPosition{0,0,0} fallback below would
          // instead give it a full-amplitude 1/NR_RECEIVERS weight and sum
          // the disconnected stream's noise into every beam.
          if (mapping_it->second < 0)
            continue;
          auto position_it = antenna_positions.find(mapping_it->second);
          if (position_it != antenna_positions.end()) {
            enu = position_it->second;
          }
        }

        double phase =
            phase_scale * (dc.l * enu.east + dc.m * enu.north + dc.n * enu.up);
        std::complex<double> steering_phasor(std::cos(phase), std::sin(phase));

        for (size_t pol = 0; pol < T::NR_POLARIZATIONS; ++pol) {
          std::complex<double> calibration_gain =
              calibration_gains
                  ? std::complex<double>(
                        (*calibration_gains)[chan][pol][receiver_idx])
                  : std::complex<double>(1.0, 0.0);

          std::complex<double> final_weight =
              (1.0 / static_cast<double>(T::NR_RECEIVERS)) * calibration_gain *
              steering_phasor;

          result.weights[chan][pol][b][receiver_idx] = std::complex<__half>(
              __float2half(static_cast<float>(final_weight.real())),
              __float2half(static_cast<float>(final_weight.imag())));

          std::cout << "Weight for channel " << chan << " pol " << pol
                    << " and receiver " << receiver_idx << " is "
                    << final_weight.real() << " + " << final_weight.imag()
                    << "j.\n";
        }
      }
    }
  }

  return result;
}

// Periodically refreshes a pipeline's device-side beam weights so each beam
// stays pointed at its target as the sky rotates (host-side counterpart to
// the per-buffer `b.weights`/`b.stream` GPU machinery in every
// Lambda*Pipeline).
//
// Construct one per pipeline with the parsed targets and array geometry
// (CommonArgs::beam_targets/antenna_positions/antenna_mapping/frequency_plan/
// min_freq_channel/array_location/steering_update_interval_seconds). With no
// targets (no --targets-filename), it is permanently inert: maybe_refresh()
// always returns false and the pipeline's static h_weights are left
// untouched -- steering is opt-in.
//
// The pipeline registers every buffer's device-weights pointer + stream once
// at construction (register_buffer(), before the warmup run). When a refresh
// is due, maybe_refresh() recomputes `current_weights_` host-side at most
// once per `update_interval` (pure CPU work -- no GPU sync) and enqueues the
// `cudaMemcpyAsync(device_weights, ..., stream)` onto *every* registered
// buffer's stream in that same call -- all buffers always beamform with
// identical weights; no buffer is ever left running on weights computed at a
// different time than its peers'.
//
// *** Call maybe_refresh() at the very top of execute_pipeline ***, before
// any kernel that reads the weights is enqueued, and only from
// pipeline_feeder's single dedicated thread (which calls execute_pipeline
// strictly sequentially). That's what makes this safe with no extra
// synchronization: every refresh copy and every kernel that reads a buffer's
// weights are enqueued from that one thread, and CUDA streams execute
// enqueued work FIFO -- so on each buffer's stream the copy lands after any
// still-in-flight run (which completes on the old weights) and before the
// buffer's next run (which reads the new ones). A separate timer thread
// issuing the copies concurrently would NOT have this guarantee (enqueue
// order between two host threads is unspecified) and could corrupt a run by
// landing new weights mid-kernel-chain.
template <typename T> struct BeamSteering {
  using BeamWeights = BeamWeightsT<T>;

  BeamSteering(std::vector<BeamTarget> targets,
               std::unordered_map<int, ENUPosition> antenna_positions,
               std::unordered_map<int, int> antenna_mapping,
               FrequencyPlan frequency_plan, int min_freq_channel,
               ArrayLocation array_location, double update_interval_seconds,
               int num_buffers,
               const typename T::AntennaGains *calibration_gains = nullptr)
      : targets_(std::move(targets)),
        antenna_positions_(std::move(antenna_positions)),
        antenna_mapping_(std::move(antenna_mapping)),
        frequency_plan_(frequency_plan), min_freq_channel_(min_freq_channel),
        array_location_(array_location),
        update_interval_(update_interval_seconds),
        calibration_gains_(calibration_gains) {
    buffers_.reserve(num_buffers);
  }

  // True once real targets have been supplied (vs. permanently inert).
  bool active() const { return !targets_.empty(); }

  // Register one buffer's device weights + the stream its kernels run on.
  // Call once per buffer from the pipeline constructor, before the warmup
  // run, so the first (always-overdue) maybe_refresh() reaches every buffer.
  void register_buffer(BeamWeights *device_weights, cudaStream_t stream) {
    buffers_.push_back({device_weights, stream});
  }

  // Returns true if a refresh was recomputed and the copies were enqueued
  // (informational only).
  bool maybe_refresh() {
    if (!active() || buffers_.empty())
      return false;

    // last_update_ starts at the epoch, so the very first call -- during the
    // constructor's warmup run -- is immediately overdue and synthesizes real
    // weights right away rather than running on placeholder h_weights.
    const auto now = std::chrono::system_clock::now();
    if ((now - last_update_) < update_interval_)
      return false;

    current_weights_ = compute_steering_weights<T>(
        targets_, antenna_positions_, antenna_mapping_, frequency_plan_,
        min_freq_channel_, array_location_, now, calibration_gains_);
    last_update_ = now;

    // One recompute, every buffer, one call: all copies are enqueued here so
    // no buffer beamforms with older (or newer) weights than its peers.
    // current_weights_ stays untouched until the next recompute -- minutes
    // away -- so every async copy reads it intact.
    for (const auto &buf : buffers_) {
      cudaMemcpyAsync(buf.device_weights, &current_weights_,
                      sizeof(BeamWeights), cudaMemcpyDefault, buf.stream);
    }
    return true;
  }

private:
  struct RegisteredBuffer {
    BeamWeights *device_weights;
    cudaStream_t stream;
  };

  std::vector<BeamTarget> targets_;
  std::unordered_map<int, ENUPosition> antenna_positions_;
  std::unordered_map<int, int> antenna_mapping_;
  FrequencyPlan frequency_plan_;
  int min_freq_channel_;
  ArrayLocation array_location_;
  // seconds-as-double, comparable directly against system_clock durations.
  std::chrono::duration<double> update_interval_;
  const typename T::AntennaGains *calibration_gains_;

  std::vector<RegisteredBuffer> buffers_;
  BeamWeights current_weights_{};
  // Epoch-initialized so the first maybe_refresh() call is always overdue.
  std::chrono::system_clock::time_point last_update_{};
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
    // DEBUG_LOG("Releasing buffer #{}", ctx->buffer_index);
    ctx->state->release_buffer(ctx->buffer_index);
  }
  delete ctx;
}

static void output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  DEBUG_LOG("Marking beam data output transfer for block #{} complete",
            ctx->block_index);
  ctx->output->register_beam_data_transfer_complete(ctx->block_index);
  delete ctx;
}

static void output_visibilities_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  INFO_LOG("Marking output transfer for block #{} complete", ctx->block_index);
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

template <size_t NR_CHANNELS, size_t NR_RECEIVERS, size_t NR_POLARIZATIONS>
auto get_default_gains() {
  std::array<std::complex<float>, NR_CHANNELS * NR_RECEIVERS * NR_POLARIZATIONS>
      output;
  output.fill({1.0f, 0.0f});
  return output;
};

template <typename T> struct LambdaPipelineIngest {

  static void ingest_and_scale(ProcessorStateBase *state,
                               FinalPacketData *packet_data,
                               cudaStream_t stream, cudaStream_t host_stream,
                               void *d_samples_entry, void *d_scales,
                               void *d_gains, void *d_samples_half,
                               bool dummy_run) {
    if (!dummy_run && state == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    cudaMemcpyAsync(d_samples_entry, (void *)packet_data->get_samples_ptr(),
                    packet_data->get_samples_elements_size(), cudaMemcpyDefault,
                    stream);
    cudaMemcpyAsync(d_scales, (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    stream);

    auto *ctx =
        new BufferReleaseContext{.state = state,
                                 .buffer_index = packet_data->buffer_index,
                                 .dummy_run = dummy_run};
    CUDA_CHECK(cudaLaunchHostFunc(host_stream, release_buffer_host_func, ctx));

    scale_and_convert_to_half<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                              T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
                              T::NR_TIME_STEPS_PER_PACKET,
                              T::NR_PACKETS_FOR_CORRELATION + 2>(
        (char2 *)d_samples_entry, (int16_t *)d_scales, (float2 *)d_gains,
        (__half2 *)d_samples_half, stream);
  }
};

template <typename T> class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;

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

  inline static const std::vector<int> modePacket{'c', 'g', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePacketPreAlign{'f', 'g', 'u', 'c',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
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
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'm', 's', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};
  // Flattened view of samples_aligned: o*u merged into s for direct fusion
  // to planarColMajCons layout (alignedToColMajCons permutation).
  inline static const std::vector<int> modePacketAlignedFlat{'f', 's', 'c',
                                                              'n', 'p', 'z'};
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
      {'g', T::NR_PACKETS_FOR_CORRELATION + 2},
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

  tcc::Correlator correlator;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry; // y
    DevicePtr<typename T::PacketScalesType> scales;              // y
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align; // y
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_padding, samples_consolidated,
        samples_consolidated_col_maj;                                      // y
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;         // y
    DevicePtr<typename T::FFTCUFFTInputType> samples_cufft_input;          // y
    DevicePtr<typename T::FFTCUFFTOutputType> samples_cufft_output;        // y
    DevicePtr<typename T::FFTOutputType> cufft_downsampled_output;         // y
    DevicePtr<BeamWeights> weights, weights_permuted, weights_updated;     // y
    DevicePtr<BeamformerOutput> beamformer_output, beamformer_data_output; // y
    DevicePtr<HalfBeamformerOutput> beamformer_data_output_half;           // y
    DevicePtr<void> cufft_work_area;                                       // y

    // Correlator I/O
    DevicePtr<CorrelatorInput> correlator_input;   // y
    DevicePtr<CorrelatorOutput> correlator_output; // y

    // Intermediate visibility layout buffers
    DevicePtr<Visibilities> visibilities_baseline; // y
    DevicePtr<TrimmedVisibilities> visibilities_trimmed_baseline,
        visibilities_trimmed, visibilities_permuted; // y

    // Per-block correlation matrix (input to cuSOLVER, overwritten in-place
    // with eigenvectors on output).
    DevicePtr<DecompositionVisibilities> decomp_visibilities; // y

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues; // y

    // cuSOLVER handles / workspace
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverDnParams_t cusolver_params = nullptr;
    DevicePtr<int> cusolver_info; // [CUSOLVER_BATCH_SIZE]
    DevicePtr<void> cusolver_work_device;
    void *cusolver_work_host = nullptr;
    size_t cusolver_work_device_size = 0;
    size_t cusolver_work_host_size = 0;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_handle;

    // Instantiated CUDA graphs of the two static mid-pipeline sections
    // (pre- and post-eigendecomposition).  All device pointers in those
    // sections are fixed per PipelineResources, so the capture stays valid
    // for the buffer's lifetime.  nullptr -> run the section eagerly
    // (capture failed or disabled via SPATIAL_DISABLE_CUDA_GRAPH).
    cudaGraphExec_t graph_pre = nullptr;
    cudaGraphExec_t graph_post = nullptr;

    // Recorded on `stream` right after the post-eigendecomposition section
    // (which includes the accumulate_visibilities add into
    // d_visibilities_accumulator) on every execute_pipeline call.
    // dump_visibilities waits on this from all buffers before reading the
    // accumulator, instead of a full cudaDeviceSynchronize().
    cudaEvent_t accumulate_done = nullptr;

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_cufft_input(make_device_ptr<typename T::FFTCUFFTInputType>()),
          samples_cufft_output(
              make_device_ptr<typename T::FFTCUFFTOutputType>()),
          cufft_downsampled_output(
              make_device_ptr<typename T::FFTOutputType>()),
          beamformer_data_output_half(make_device_ptr<HalfBeamformerOutput>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          beamformer_data_output(make_device_ptr<BeamformerOutput>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          visibilities_permuted(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
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
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib);
    }

    ~PipelineResources() {
      if (graph_pre)
        cudaGraphExecDestroy(graph_pre);
      if (graph_post)
        cudaGraphExecDestroy(graph_post);
      if (accumulate_done)
        cudaEventDestroy(accumulate_done);
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
      if (cusolver_handle)
        cusolverDnDestroy(cusolver_handle);
      if (cusolver_params)
        cusolverDnDestroyParams(cusolver_params);
      if (cusolver_work_host)
        std::free(cusolver_work_host);
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
          gemm_handle(std::move(other.gemm_handle)), graph_pre(other.graph_pre),
          graph_post(other.graph_post), accumulate_done(other.accumulate_done) {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.graph_pre = nullptr;
      other.graph_post = nullptr;
      other.accumulate_done = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);
        if (graph_pre)
          cudaGraphExecDestroy(graph_pre);
        if (graph_post)
          cudaGraphExecDestroy(graph_post);
        if (accumulate_done)
          cudaEventDestroy(accumulate_done);
        graph_pre = other.graph_pre;
        graph_post = other.graph_post;
        accumulate_done = other.accumulate_done;
        other.graph_pre = nullptr;
        other.graph_post = nullptr;
        other.accumulate_done = nullptr;

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

  BeamWeights *h_weights;
  // Periodically refreshes b.weights to track this pipeline's beam targets
  // (see compute_steering_weights()/BeamSteering above pipeline.hpp:46+ for
  // why and how) -- inert (a permanent no-op) when no --targets-filename was
  // supplied, in which case h_weights above remains the sole source of truth,
  // exactly as before this feature existed.
  BeamSteering<T> beam_steering_;
  TrimmedVisibilities *d_visibilities_accumulator;

  // Recorded on buffers[0].stream after dump_visibilities resets
  // d_visibilities_accumulator. Each buffer's post-eigen section waits on
  // this before its next accumulate_visibilities, so the reset can never
  // race with an in-flight accumulation -- without a cudaDeviceSynchronize().
  cudaEvent_t visibilities_reset_done = nullptr;

  typename T::AntennaGains *d_gains;
  std::vector<PipelineResources> buffers;
  int *d_subpacket_delays;
  int visibilities_start_seq_num;
  int visibilities_end_seq_num;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;
  cusolverEigMode_t cusolver_jobz;
  cublasFillMode_t cusolver_uplo;
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS;

public:
  static constexpr size_t NR_BENCHMARKING_RUNS = 100;
  size_t benchmark_runs_done = 0;
  cudaEvent_t start_run[NR_BENCHMARKING_RUNS], stop_run[NR_BENCHMARKING_RUNS];
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    auto &b = buffers[current_buffer];
    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call, so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // b.weights, and only from this single-threaded pipeline_feeder context
    // -- see the BeamSteering<T> comment block (pipeline.hpp above) for why
    // that ordering is what makes this safe without extra synchronization.
    beam_steering_.maybe_refresh();

    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
    }
    visibilities_missing_packets += packet_data->get_num_missing_packets();

    // Record GPU start event
    cudaEventRecord(start_run[benchmark_runs_done], b.stream);

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

    // Pre-eigendecomposition section: a single graph launch replaces ~12
    // individual launches when capture succeeded at construction (see
    // enqueue_pre_eigen / capture_graph).
    if (b.graph_pre != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_pre, b.stream));
    } else {
      enqueue_pre_eigen(b);
    }

    // Eager: cuSOLVER may do host-side work per call that a captured graph
    // would not replay, so it stays out of the graphs.
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

    // The post-eigen section's first op (accumulate_visibilities) adds into
    // d_visibilities_accumulator. Wait for dump_visibilities' last reset of
    // that accumulator (on buffers[0].stream) to land first -- a GPU-side
    // wait, not a host-blocking sync. No-op until the first dump happens.
    cudaStreamWaitEvent(b.stream, visibilities_reset_done, 0);

    // Post-eigendecomposition section: accumulation, beamforming GEMM and
    // output-layout permutations — again one graph launch when available.
    if (b.graph_post != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_post, b.stream));
    } else {
      enqueue_post_eigen(b);
    }

    // Lets dump_visibilities (on buffers[0].stream) wait for this buffer's
    // accumulate_visibilities to land before reading
    // d_visibilities_accumulator, without a cudaDeviceSynchronize().
    cudaEventRecord(b.accumulate_done, b.stream);

    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    detect_and_downsample_fft_launch(
        (float2 *)b.samples_cufft_output.get(),
        (float *)b.cufft_downsampled_output.get(), T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_BEAMS, T::FFT_DOWNSAMPLE_FACTOR, b.stream);
    cudaEventRecord(stop_run[benchmark_runs_done], b.stream);

    // Output handling
    if (output_ != nullptr && !dummy_run) {
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      size_t eigenvalue_block_num =
          output_->register_eigendecomposition_data_block(start_seq_num,
                                                          end_seq_num);
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);

      if (block_num != std::numeric_limits<size_t>::max()) {
        void *landing_pointer =
            output_->get_beam_data_landing_pointer(block_num);
        cudaMemcpyAsync(landing_pointer, b.beamformer_data_output_half.get(),
                        sizeof(HalfBeamformerOutput), cudaMemcpyDefault,
                        b.stream);
        auto *output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = block_num};
        cudaLaunchHostFunc(b.stream, output_transfer_complete_host_func,
                           output_ctx);
        bool *arrivals_output_pointer =
            (bool *)output_->get_arrivals_data_landing_pointer(block_num);
        std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                    packet_data->get_arrivals_size());
        output_->register_arrivals_transfer_complete(block_num);
      }

      if (eigenvalue_block_num != std::numeric_limits<size_t>::max()) {

        void *eigenvalues_output_pointer =
            (void *)output_->get_eigenvalues_data_landing_pointer(
                eigenvalue_block_num);

        void *eigenvectors_output_pointer =
            (void *)output_->get_eigenvectors_data_landing_pointer(
                eigenvalue_block_num);

        cudaMemcpyAsync(eigenvalues_output_pointer, b.eigenvalues.get(),
                        sizeof(Eigenvalues), cudaMemcpyDefault, b.stream);

        cudaMemcpyAsync(
            eigenvectors_output_pointer, b.decomp_visibilities.get(),
            sizeof(DecompositionVisibilities), cudaMemcpyDefault, b.stream);

        auto *eig_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = eigenvalue_block_num};
        cudaLaunchHostFunc(b.stream, eigen_output_transfer_complete_host_func,
                           eig_output_ctx);
      }

      if (fft_block_num != std::numeric_limits<size_t>::max()) {
        auto *fft_output_pointer =
            (void *)output_->get_fft_landing_pointer(fft_block_num);
        cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                        sizeof(typename T::FFTOutputType), cudaMemcpyDefault,
                        b.stream);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }
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
  }

  // ---- Static mid-pipeline sections -----------------------------------
  // Everything in these two methods operates on device pointers that are
  // fixed for the lifetime of a PipelineResources, which is what allows
  // them to be captured into CUDA graphs (capture_graph) and replayed as a
  // single launch each instead of ~20 individual launches.  Any op added
  // here must use only per-buffer device pointers — no per-run host
  // pointers — to keep the capture valid.

  // From half-converted samples up to the unpacked hermitian matrices that
  // feed the eigendecomposition.
  void enqueue_pre_eigen(PipelineResources &b) {
    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    tensor_16.runPermutation("alignedToPadding", alpha,
                             (__half *)b.samples_aligned.get(),
                             (__half *)b.samples_padding.get(), b.stream);

    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketAlignedSamplesType),
                               cudaMemcpyDefault, b.stream));

    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                        b.stream));

    tensor_16.runPermutation("paddedToCorrInput", alpha,
                             (__half *)b.samples_padded.get(),
                             (__half *)b.correlator_input.get(), b.stream);

    correlator.launchAsync((CUstream)b.stream,
                           (CUdeviceptr)b.correlator_output.get(),
                           (CUdeviceptr)b.correlator_input.get());

    // Fuses visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed into
    // one kernel pass (saves 3 cuTensor launches + 1 D2D copy).  No atomic
    // accumulation here -- accumulate_visibilities stays in enqueue_post_eigen
    // where cuSOLVER naturally staggers concurrent buffer access to the shared
    // accumulator and avoids contention.
    corr_to_trimmed((float *)b.correlator_output.get(),
                    (float *)b.visibilities_trimmed.get(), T::NR_CHANNELS,
                    NR_BASELINES, NR_UNPADDED_BASELINES,
                    T::NR_POLARIZATIONS * T::NR_POLARIZATIONS * 2, b.stream);

    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)b.visibilities_trimmed.get(),
                             (float *)b.visibilities_permuted.get(), b.stream);
    unpack_triangular_baseline_batch_launch<cuComplex>(
        (cuComplex *)b.visibilities_permuted.get(),
        (cuComplex *)b.decomp_visibilities.get(), T::NR_RECEIVERS,
        T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS,
        T::NR_CHANNELS, b.stream);
  }

  // From visibility accumulation through beamforming to the cuFFT input
  // layout.  The cuSOLVER eigendecomposition sits between the two sections
  // and stays eager (it may do host-side work per call that stream capture
  // would not replay); cuFFT + downsample stay eager after this section for
  // the same caution.
  void enqueue_post_eigen(PipelineResources &b) {
    accumulate_visibilities((float *)b.visibilities_trimmed.get(),
                            (float *)d_visibilities_accumulator,
                            2 * NR_UNPADDED_BASELINES * T::NR_POLARIZATIONS *
                                T::NR_POLARIZATIONS * T::NR_CHANNELS,
                            b.stream);

    // Fuses alignedToPlanar + consToColMajCons into a single cuTensor launch
    // using the flat mode alias (o*u → s) for the source descriptor.
    tensor_16.runPermutation("alignedToColMajCons", alpha,
                             (__half *)b.samples_aligned.get(),
                             (__half *)b.samples_consolidated_col_maj.get(),
                             b.stream);

    update_weights((__half *)b.weights.get(), (__half *)b.weights_updated.get(),
                   T::NR_BEAMS, T::NR_RECEIVERS, T::NR_CHANNELS,
                   T::NR_POLARIZATIONS, (float *)b.eigenvalues.get(),
                   (float *)b.visibilities_trimmed.get(), b.stream);

    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)b.weights_updated.get(),
                             (__half *)b.weights_permuted.get(), b.stream);

    b.gemm_handle->Run((CUdeviceptr)b.weights_permuted.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());
    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.beamformer_data_output.get(), b.stream);

    convert_float_to_half((float *)b.beamformer_data_output.get(),
                          (__half *)b.beamformer_data_output_half.get(),
                          2 * T::NR_CHANNELS * T::NR_POLARIZATIONS *
                              T::NR_BEAMS * NR_TIME_STEPS_FOR_CORRELATION,
                          b.stream);

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);
  }

  // Capture one section into an instantiated graph.  Raw CUDA calls (not
  // CUDA_CHECK, which exits the process) so a capture-incompatible op
  // degrades to eager execution instead of aborting startup.
  template <typename EnqueueFn>
  bool capture_graph(PipelineResources &b, EnqueueFn &&enqueue,
                     cudaGraphExec_t &exec_out) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    bool enqueue_ok = true;
    try {
      enqueue(b);
    } catch (...) {
      // ccglib/cudawrappers throw on capture-incompatible calls.
      enqueue_ok = false;
    }
    const cudaError_t end_err = cudaStreamEndCapture(b.stream, &graph);
    if (!enqueue_ok || end_err != cudaSuccess || graph == nullptr) {
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
      cudaGetLastError();
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess ||
        exec == nullptr) {
      cudaGraphDestroy(graph);
      cudaGetLastError();
      return false;
    }
    cudaGraphDestroy(graph);
    exec_out = exec;
    return true;
  }

  LambdaGPUPipeline(const int num_buffers, BeamWeightsT<T> *h_weights,
                    BeamSteering<T> beam_steering)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
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

    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES =
        T::NR_BEAMS * T::NR_CHANNELS * T::NR_POLARIZATIONS;

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    CUDA_CHECK(cudaEventCreateWithFlags(&visibilities_reset_done,
                                        cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

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
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addTensor(modePacketAligned, "aligned");
    tensor_16.addTensor(modePacketAlignedFlat, "alignedFlat");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");
    tensor_32.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addPermutation("aligned", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToPadding");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");
    tensor_16.addPermutation("aligned", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "alignedToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("alignedFlat", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToColMajCons");
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
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");

    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device, work_size);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();
      CUDA_CHECK(
          cudaEventCreateWithFlags(&b.accumulate_done, cudaEventDisableTiming));
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
    }
    for (auto i = 0; i < NR_BENCHMARKING_RUNS; ++i) {
      cudaEventCreate(&start_run[i]);
      cudaEventCreate(&stop_run[i]);
    }

    cudaDeviceSynchronize();

    // Warm up the pipeline *before* attempting graph capture. This is the
    // first-ever call to most cuTENSOR permutations, the TCC correlator, and
    // the ccglib GEMM (only "weightsInputToCCGLIB" has run so far, above).
    // Each of these libraries may do one-time lazy initialization on first
    // use -- NVRTC JIT compilation, module loading, on-disk JIT-cache
    // writes -- which is not safe to perform while cudaStreamBeginCapture is
    // active (can hang or segfault, especially with a cold cache). Running
    // this eagerly first ensures capture below only ever records
    // already-loaded kernels. Because everything is zeroed it should have
    // negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();

    // Capture the two static mid-pipeline sections of each buffer into CUDA
    // graphs (~21 launches collapse into 2 per run).  Any failure falls back
    // to eager execution for all buffers — functionally identical, just more
    // launch overhead.  SPATIAL_DISABLE_CUDA_GRAPH=1 forces the eager path.
    if (std::getenv("SPATIAL_DISABLE_CUDA_GRAPH") == nullptr) {
      bool all_ok = true;
      for (auto &b : buffers) {
        if (!capture_graph(
                b, [this](PipelineResources &r) { enqueue_pre_eigen(r); },
                b.graph_pre) ||
            !capture_graph(
                b, [this](PipelineResources &r) { enqueue_post_eigen(r); },
                b.graph_post)) {
          all_ok = false;
          break;
        }
      }
      if (!all_ok) {
        for (auto &b : buffers) {
          if (b.graph_pre) {
            cudaGraphExecDestroy(b.graph_pre);
            b.graph_pre = nullptr;
          }
          if (b.graph_post) {
            cudaGraphExecDestroy(b.graph_post);
            b.graph_post = nullptr;
          }
        }
        WARN_LOG("CUDA graph capture failed — pipeline will run eagerly");
      } else {
        INFO_LOG("CUDA graphs captured for {} pipeline buffers",
                 buffers.size());
        // Exercise the captured graphs (cudaGraphLaunch replay) once before
        // real traffic arrives, so a replay problem surfaces here rather
        // than on the first live buffer.
        execute_pipeline(&warmup_packet, true);
      }
      cudaDeviceSynchronize();
    }

    // these need to be set after the dummy run(s).
    visibilities_start_seq_num = -1;
    visibilities_end_seq_num = -1;
    visibilities_missing_packets = 0;
  };
  ~LambdaGPUPipeline() {
    // If there are visibilities in the accumulator on the GPU - dump them
    // out to disk. These will get tagged with a -1 end_seq_id currently
    // which is not fully ideal.

    if (visibilities_reset_done)
      cudaEventDestroy(visibilities_reset_done);
  };
  virtual void set_subpacket_delays(int *delays_subpacket) override {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }

  virtual void set_antenna_gains(std::complex<float> *gains) override {
    std::cout << "setting antenna gains on LambdaGPUPipeline...\n";
    gains_ = gains;
    CUDA_CHECK(cudaMemcpy(d_gains, gains, sizeof(typename T::AntennaGains),
                          cudaMemcpyDefault));

    std::cout << "Loaded gains are:\n";
    for (auto i = 0; i < T::NR_CHANNELS; ++i) {
      for (auto j = 0; j < T::NR_POLARIZATIONS; ++j) {
        for (auto k = 0; k < T::NR_RECEIVERS; ++k) {
          std::cout << "channel " << i << " pol " << j << " receiver " << k
                    << " val "
                    << gains[i * T::NR_POLARIZATIONS * T::NR_RECEIVERS +
                             j * T::NR_RECEIVERS + k]
                           .real()
                    << " + "
                    << gains[i * T::NR_POLARIZATIONS * T::NR_RECEIVERS +
                             j * T::NR_RECEIVERS + k]
                           .imag()
                    << "j.\n";
        }
      }
    }
    cudaDeviceSynchronize();
    std::cout << "gains uploaded successfully...\n";
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

    INFO_LOG("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_correlation_units_integrated;
    INFO_LOG("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    // GPU-side wait (not a host-blocking sync): make buffers[0].stream's
    // upcoming memcpy of d_visibilities_accumulator wait for every buffer's
    // most recent accumulate_visibilities to land first. accumulate_done is a
    // no-op wait until it has been recorded at least once.
    for (auto &buf : buffers) {
      cudaStreamWaitEvent(buffers[0].stream, buf.accumulate_done, 0);
    }
    const int visibilities_total_packets =
        current_num_integrated_units_processed *
        visibilities_total_packets_per_block;
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num, end_seq_num, visibilities_missing_packets,
        visibilities_total_packets);
    visibilities_start_seq_num = -1;
    visibilities_missing_packets = 0;
    if (block_num != std::numeric_limits<size_t>::max()) {
      void *landing_pointer =
          output_->get_visibilities_landing_pointer(block_num);
      cudaMemcpyAsync(landing_pointer, d_visibilities_accumulator,
                      sizeof(TrimmedVisibilities), cudaMemcpyDefault,
                      buffers[0].stream);
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};

      cudaLaunchHostFunc(buffers[0].stream,
                         output_visibilities_transfer_complete_host_func,
                         output_ctx);
    }
    cudaMemsetAsync(d_visibilities_accumulator, 0, sizeof(TrimmedVisibilities),
                    buffers[0].stream);
    num_correlation_units_integrated.store(0);
    // Record the reset so every buffer's next accumulate_visibilities (via
    // the wait above, in execute_pipeline) is ordered after it.
    cudaEventRecord(visibilities_reset_done, buffers[0].stream);
  };
};

template <typename T> class LambdaAntennaSpectraPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  typename T::AntennaGains *d_gains;
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

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry;
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

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    // dummy_run must be forwarded so the warmup run's release_buffer host
    // func skips the release (state_ is unset during construction).
    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, streams[current_buffer],
        streams[current_buffer], d_samples_entry[current_buffer],
        d_scales[current_buffer], d_gains, d_samples_half[current_buffer],
        dummy_run);

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
          output_->register_fft_block(start_seq_num, end_seq_num);
      // size_t::max means no FFT writer attached -- the landing pointer
      // would be nullptr.
      if (fft_block_num != std::numeric_limits<size_t>::max()) {
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
    INFO_LOG("FFT initialized with {} total batches with a {} FFT each run "
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

  typename T::AntennaGains *d_gains;
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
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    auto &b = buffers[current_buffer];
    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call, so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // b.weights, and only from this single-threaded pipeline_feeder context
    // -- see the BeamSteering<T> comment block (pipeline.hpp above) for why
    // that ordering is what makes this safe without extra synchronization.
    beam_steering_.maybe_refresh();

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

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
          output_->register_fft_block(start_seq_num, end_seq_num);
      // size_t::max means no FFT writer attached -- the landing pointer
      // would be nullptr.
      if (fft_block_num != std::numeric_limits<size_t>::max()) {
        auto *fft_output_pointer =
            (void *)output_->get_fft_landing_pointer(fft_block_num);
        cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                        sizeof(FFTOutputType), cudaMemcpyDefault, b.stream);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaBeamformedSpectraPipeline(const int num_buffers,
                                  BeamWeightsT<T> *h_weights,
                                  BeamSteering<T> beam_steering)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
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

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

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
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
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
  static constexpr int NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE = 5;

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
            [T::NR_PACKETS_FOR_CORRELATION][T::NR_TIME_STEPS_PER_PACKET];
  using FineChannelRemovedType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [T::NR_TIME_STEPS_PER_PACKET -
             2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using FineChannelCopyType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [(T::NR_TIME_STEPS_PER_PACKET -
              2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE) /
             2];
  using FineChannelSeekType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [T::NR_TIME_STEPS_PER_PACKET / 2 +
             NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_PACKET -
            2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamOutput =
      __half[2 * T::NR_BEAMS][T::NR_PACKETS_FOR_CORRELATION]
            [T::NR_CHANNELS * (T::NR_TIME_STEPS_PER_PACKET -
                               2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE)];
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
    ManagedCufftPlan fft_plan, fft_plan_fine_channel;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align;
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
    DevicePtr<FFTCUFFTInputType> samples_cufft_input;
    DevicePtr<BeamOutput> beam_output;
    DevicePtr<FFTCUFFTOutputType> samples_cufft_output,
        samples_cufft_output_fine_channel;
    DevicePtr<FineChannelRemovedType> samples_fine_channel_removed, beam_shape,
        cufft_downsampled_input;
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
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_cufft_input(make_device_ptr<FFTCUFFTInputType>()),
          beam_shape(make_device_ptr<FineChannelRemovedType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          samples_cufft_output(make_device_ptr<FFTCUFFTOutputType>()),
          samples_cufft_output_fine_channel(
              make_device_ptr<FFTCUFFTOutputType>()),
          samples_fine_channel_removed(
              make_device_ptr<FineChannelRemovedType>()),
          cufft_downsampled_output(make_device_ptr<FFTOutputType>()),
          cufft_downsampled_input(make_device_ptr<FineChannelRemovedType>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          weights_rfi_mitigated(make_device_ptr<RFIMitigatedBeamWeights>()),
          weights_beamformer(make_device_ptr<RFIMitigatedBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
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

  std::unordered_map<int, int> NR_SIGNAL_EIGENVECTORS;
  int min_freq_channel;

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

  inline static const std::vector<int> modePacket{'c', 'y', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};

  inline static const std::vector<int> modePacketPreAlign{'f', 'y', 'u', 'c',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
                                                         'n', 'p', 'z'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'e', 's', 'z'};
  inline static const std::vector<int> modeCUFFTOutput{'c', 'p', 'e',
                                                       'o', 'u', 'z'};
  inline static const std::vector<int> modeFineChannelRemove{'u', 'c', 'p',
                                                             'e', 'o', 'z'};
  inline static const std::vector<int> modeFineChannelRemoved{'g', 'c', 'p',
                                                              'e', 'o', 'z'};
  inline static const std::vector<int> modeBeamFFTDownsample{'c', 'p', 'e',
                                                             'o', 'g', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'e', 's'};
  inline static const std::vector<int> modeBeamOutput{'e', 'o', 'c',
                                                      'g', 'p', 'z'};
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
      {'d', T::NR_PADDED_RECEIVERS},
      {'e', T::NR_BEAMS * 2}, // rfi mitigated beam + original beam
      {'f', T::NR_FPGA_SOURCES},
      {'g',
       T::NR_TIME_STEPS_PER_PACKET - 2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE},
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
      {'y', T::NR_PACKETS_FOR_CORRELATION + 2},
      {'z', 2}, // real, imaginary
  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;
  int *d_subpacket_delays;
  typename T::AntennaGains *d_gains;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    auto &b = buffers[current_buffer];
    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call, so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // b.weights, and only from this single-threaded pipeline_feeder context
    // -- see the BeamSteering<T> comment block (pipeline.hpp above) for why
    // that ordering is what makes this safe without extra synchronization.
    beam_steering_.maybe_refresh();

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    tensor_16.runPermutation("alignedToPlanar", alpha,
                             (__half *)b.samples_aligned.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    tensor_16.runPermutation(
        "alignedToPadding", alpha,
        reinterpret_cast<__half *>(b.samples_aligned.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

    // ------------------------------------------------------------------
    // 5. Copy unpadded → padded buffer then zero-fill the padding region
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketAlignedSamplesType),
                               cudaMemcpyDefault, b.stream));
    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketAlignedSamplesType),
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
        const int K = NR_SIGNAL_EIGENVECTORS[min_freq_channel + channel];
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
      size_t CUBLAS_STRIDE_A = T::NR_RECEIVERS * T::NR_RECEIVERS;
      size_t CUBLAS_STRIDE_B = T::NR_RECEIVERS * T::NR_BEAMS;
      size_t CUBLAS_STRIDE_C = T::NR_RECEIVERS * T::NR_BEAMS;

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

    // convert_float_to_half((float *)b.beam_shape.get(),
    //                       (__half *)b.beam_output.get(),
    //                       sizeof(BeamOutput) / sizeof(__half), b.stream);
    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    tensor_32.runPermutation("cufftOutputToFineChannelRemove", alpha_32,
                             (float *)b.samples_cufft_output.get(),
                             (float *)b.samples_cufft_output_fine_channel.get(),
                             b.stream);

    // First half goes in second half of destination
    void *src_ptr = (char *)b.samples_cufft_output_fine_channel.get();
    size_t src_size = sizeof(FineChannelCopyType);

    dest_ptr = (char *)b.samples_fine_channel_removed.get() +
               sizeof(FineChannelCopyType);

    cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, b.stream);

    src_ptr = (char *)b.samples_cufft_output_fine_channel.get() +
              sizeof(FineChannelSeekType);
    dest_ptr = (char *)b.samples_fine_channel_removed.get();
    cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, b.stream);

    tensor_32.runPermutation("fineChannelRemovedToBeamOutput", alpha_32,
                             (float *)b.samples_fine_channel_removed.get(),
                             (float *)b.beam_shape.get(), b.stream);

    tensor_32.runPermutation("fineChannelRemovedToBeamFFTDownsample", alpha_32,
                             (float *)b.samples_fine_channel_removed.get(),
                             (float *)b.cufft_downsampled_input.get(),
                             b.stream);

    cudaMemsetAsync((float *)b.cufft_downsampled_output.get(), 0,
                    sizeof(FFTOutputType), b.stream);

    sum_fft_over_packets_launch(
        (float2 *)b.cufft_downsampled_input.get(),
        (float *)b.cufft_downsampled_output.get(), 2 * T::NR_BEAMS,
        T::NR_CHANNELS, T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET - 2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE,
        T::NR_PACKETS_FOR_CORRELATION, b.stream);

    detect_and_convert_to_half_launch(
        (float4 *)b.beam_shape.get(), (__half *)b.beam_output.get(),
        sizeof(BeamOutput) / sizeof(__half), b.stream);

    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.

      size_t beam_block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      if (beam_block_num != std::numeric_limits<size_t>::max()) {
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
      }

      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);
      if (fft_block_num != std::numeric_limits<size_t>::max()) {
        auto *fft_output_pointer =
            (void *)output_->get_fft_landing_pointer(fft_block_num);
        cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                        sizeof(FFTOutputType), cudaMemcpyDefault, b.stream);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }

      size_t eig_block_num = output_->register_eigendecomposition_data_block(
          start_seq_num, end_seq_num);

      if (eig_block_num != std::numeric_limits<size_t>::max()) {
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
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaAdaptiveBeamformedSpectraPipeline(
      const int num_buffers, BeamWeightsT<T> *h_weights,
      const std::unordered_map<int, int> nr_signal_eigenvectors,
      const int min_freq_channel, BeamSteering<T> beam_steering)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        correlator(cu::Device(0), 16, T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        NR_SIGNAL_EIGENVECTORS(nr_signal_eigenvectors),
        min_freq_channel(min_freq_channel) {
    std::cout << "Beamformed Spectra instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << 2 * T::NR_BEAMS << std::endl;

    const long long CUFFT_FFT_SIZE = 64;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES = 2 * T::NR_BEAMS * T::NR_CHANNELS *
                                     T::NR_POLARIZATIONS *
                                     T::NR_PACKETS_FOR_CORRELATION;

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
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketAligned, "aligned");
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
    tensor_32.addTensor(modeCUFFTOutput, "cufftOutput");
    tensor_32.addTensor(modeFineChannelRemove, "fineChannelRemove");
    tensor_32.addTensor(modeFineChannelRemoved, "fineChannelRemoved");
    tensor_32.addTensor(modeBeamFFTDownsample, "beamFFTDownsample");

    // Permutation descriptors
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_16.addPermutation("aligned", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToPadding");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("aligned", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "alignedToPlanar");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToDecomp");

    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsBeamMajor",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsToBeamMajor");
    tensor_16.addPermutation("weights2xBeamMajor", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F,
                             "weights2xBeamMajorToCCGLIB");
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");
    tensor_32.addPermutation("cufftOutput", "fineChannelRemove",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "cufftOutputToFineChannelRemove");

    tensor_32.addPermutation("fineChannelRemoved", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "fineChannelRemovedToBeamOutput");
    tensor_32.addPermutation("fineChannelRemoved", "beamFFTDownsample",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "fineChannelRemovedToBeamFFTDownsample");

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));

    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

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
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
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

  virtual void set_subpacket_delays(int *delays_subpacket) {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
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

  typename T::AntennaGains *d_gains;
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
  // Fused: packet → col-maj-cons in one step (replaces packetToPlanar +
  // consToColMajCons). o,u adjacent in output with u inner gives stride(s)=
  // stride(u), same layout as ['c','p','z','s','f','n'] with s=o*u.
  inline static const std::vector<int> modePacketDirectColMajCons = {
      'c', 'p', 'z', 'o', 'u', 'f', 'n'};
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

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry;

  std::vector<typename T::HalfPacketSamplesType *> d_samples_half;
  std::vector<typename T::PaddedPacketSamplesType *> d_samples_padded;
  std::vector<CorrelatorInput *> d_correlator_input;
  std::vector<CorrelatorOutput *> d_correlator_output;

  TrimmedVisibilities *d_visibilities_accumulator;

  // Captured replay of the static per-buffer permutation/correlator/gemm
  // sequence (see enqueue_main); a nullptr entry falls back to the eager
  // enqueue_main(i) call.
  std::vector<cudaGraphExec_t> graph_main;

  // Recorded on streams[i] right after enqueue_main's
  // accumulate_visibilities lands. dump_visibilities waits on all of these
  // before reading d_visibilities_accumulator, instead of a
  // cudaDeviceSynchronize().
  std::vector<cudaEvent_t> accumulate_done;

  // Recorded on streams[0] after dump_visibilities resets
  // d_visibilities_accumulator. Each buffer's enqueue_main waits on this
  // before its next accumulate_visibilities, so the reset can never race
  // with an in-flight accumulation -- without a cudaDeviceSynchronize().
  cudaEvent_t visibilities_reset_done = nullptr;

  std::vector<BeamformerInput *> d_beamformer_input;
  std::vector<BeamformerOutput *> d_beamformer_output, d_beamformer_data_output;
  std::vector<HalfBeamformerOutput *> d_beamformer_data_output_half;
  std::vector<__half *> d_samples_consolidated_col_maj, d_weights,
      d_weights_permuted;
  std::vector<typename T::PacketScalesType *> d_scales;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract. Note this pipeline
  // keeps its weights in d_weights[buffer_index]/streams[buffer_index]
  // (rather than buffers[i].weights/.stream); register_buffer() doesn't care
  // -- it just needs each buffer's device pointer and stream.
  BeamSteering<T> beam_steering_;

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

    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
    }
    visibilities_missing_packets += packet_data->get_num_missing_packets();

    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call (the d_weights[i]/streams[i]
    // pairs registered in the constructor below), so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // d_weights[current_buffer], and only from this single-threaded
    // pipeline_feeder context -- see the BeamSteering<T> comment block
    // (pipeline.hpp above) for why that ordering is what makes this safe
    // without extra synchronization.
    beam_steering_.maybe_refresh();

    // Record GPU start event
    cudaEventRecord(start_run[benchmark_runs_done], streams[current_buffer]);

    // dummy_run must be forwarded: on the constructor's warmup run state_
    // is not set yet and packet_data->buffer_index is meaningless, so the
    // release_buffer host func has to skip the release (a hardcoded false
    // here made the warmup segfault in release_buffer_host_func).
    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, streams[current_buffer],
        streams[current_buffer], d_samples_entry[current_buffer],
        d_scales[current_buffer], d_gains, d_samples_half[current_buffer],
        dummy_run);

    // enqueue_main's first op (accumulate_visibilities) adds into
    // d_visibilities_accumulator. Wait for dump_visibilities' last reset of
    // that accumulator (on streams[0]) to land first -- a GPU-side wait,
    // not a host-blocking sync. No-op until the first dump happens.
    cudaStreamWaitEvent(streams[current_buffer], visibilities_reset_done, 0);

    // Static mid-pipeline section: a single graph launch replaces ~15
    // individual launches when capture succeeded at construction (see
    // enqueue_main / capture_graph).
    if (graph_main[current_buffer] != nullptr) {
      CUDA_CHECK(
          cudaGraphLaunch(graph_main[current_buffer], streams[current_buffer]));
    } else {
      enqueue_main(current_buffer);
    }

    // Lets dump_visibilities (on streams[0]) wait for this buffer's
    // accumulate_visibilities to land before reading
    // d_visibilities_accumulator, without a cudaDeviceSynchronize().
    cudaEventRecord(accumulate_done[current_buffer], streams[current_buffer]);

    cudaEventRecord(stop_run[benchmark_runs_done], streams[current_buffer]);

    // Output handling
    if (output_ != nullptr && !dummy_run) {
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      // size_t::max means no beam writer is attached (e.g. observe.cu's
      // beam_writer = nullptr): the landing pointers would be nullptr, so
      // skip the beam/arrivals copies entirely.
      if (block_num != std::numeric_limits<size_t>::max()) {
        void *landing_pointer =
            output_->get_beam_data_landing_pointer(block_num);
        cudaMemcpyAsync(landing_pointer,
                        d_beamformer_data_output_half[current_buffer],
                        sizeof(HalfBeamformerOutput), cudaMemcpyDefault,
                        streams[current_buffer]);
        auto *output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = block_num};
        cudaLaunchHostFunc(streams[current_buffer],
                           output_transfer_complete_host_func, output_ctx);

        // memcpy arrivals
        bool *arrivals_output_pointer =
            (bool *)output_->get_arrivals_data_landing_pointer(block_num);
        std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                    packet_data->get_arrivals_size());
        output_->register_arrivals_transfer_complete(block_num);
      }
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
  }

  // ---- Static mid-pipeline section -------------------------------------
  // Everything here operates on device pointers that are fixed for the
  // lifetime of buffer i, which is what allows it to be captured into a CUDA
  // graph (capture_graph) and replayed as a single launch instead of ~15
  // individual launches. Any op added here must use only per-buffer device
  // pointers -- no per-run host pointers -- to keep the capture valid.
  void enqueue_main(int i) {
    tensor_16.runPermutation("packetToPadding", alpha,
                             (__half *)d_samples_half[i],
                             (__half *)d_samples_padded[i], streams[i]);

    tensor_16.runPermutation("paddedToCorrInput", alpha,
                             (__half *)d_samples_padded[i],
                             (__half *)d_correlator_input[i], streams[i]);

    correlator.launchAsync((CUstream)streams[i],
                           (CUdeviceptr)d_correlator_output[i],
                           (CUdeviceptr)d_correlator_input[i]);

    // Fused: visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed +
    // accumulate_visibilities in one kernel pass (no intermediate buffers).
    accumulate_visibilities_from_corr(
        (float *)d_correlator_output[i], (float *)d_visibilities_accumulator,
        T::NR_CHANNELS, NR_BASELINES, NR_UNPADDED_BASELINES,
        T::NR_POLARIZATIONS * T::NR_POLARIZATIONS * COMPLEX, streams[i]);

    tensor_16.runPermutation("packetToColMajCons", alpha,
                             (__half *)d_samples_half[i],
                             (__half *)d_samples_consolidated_col_maj[i],
                             streams[i]);

    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights[i],
                             (__half *)d_weights_permuted[i], streams[i]);
    (*gemm_handles[i])
        .Run((CUdeviceptr)d_weights_permuted[i],
             (CUdeviceptr)d_samples_consolidated_col_maj[i],
             (CUdeviceptr)d_beamformer_output[i]);

    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)d_beamformer_output[i],
                             (float *)d_beamformer_data_output[i], streams[i]);

    convert_float_to_half((float *)d_beamformer_data_output[i],
                          (__half *)d_beamformer_data_output_half[i],
                          2 * T::NR_CHANNELS * T::NR_POLARIZATIONS *
                              T::NR_BEAMS * NR_TIME_STEPS_FOR_CORRELATION,
                          streams[i]);
  }

  // Capture one section into an instantiated graph. Raw CUDA calls (not
  // CUDA_CHECK, which exits the process) so a capture-incompatible op
  // degrades to eager execution instead of aborting startup.
  template <typename EnqueueFn>
  bool capture_graph(cudaStream_t stream, EnqueueFn &&enqueue,
                     cudaGraphExec_t &exec_out) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    bool enqueue_ok = true;
    try {
      enqueue();
    } catch (...) {
      // ccglib/cudawrappers throw on capture-incompatible calls.
      enqueue_ok = false;
    }
    const cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
    if (!enqueue_ok || end_err != cudaSuccess || graph == nullptr) {
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
      cudaGetLastError();
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess ||
        exec == nullptr) {
      cudaGraphDestroy(graph);
      cudaGetLastError();
      return false;
    }
    cudaGraphDestroy(graph);
    exec_out = exec;
    return true;
  }

  LambdaCorrBeamOnlyGPUPipeline(const int num_buffers,
                                BeamWeightsT<T> *h_weights,
                                BeamSteering<T> beam_steering)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),

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
    d_samples_half.resize(num_buffers);
    d_samples_padded.resize(num_buffers);
    d_samples_consolidated_col_maj.resize(num_buffers);
    d_correlator_input.resize(num_buffers);
    d_correlator_output.resize(num_buffers);
    d_beamformer_input.resize(num_buffers);
    d_beamformer_output.resize(num_buffers);
    d_beamformer_data_output.resize(num_buffers);
    d_beamformer_data_output_half.resize(num_buffers);
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[num_buffers + i],
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_entry[i],
                            sizeof(typename T::InputPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated_col_maj[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_padded[i],
                            sizeof(typename T::PaddedPacketSamplesType)));
      // The tail beyond HalfPacketSamplesType must be zero for TCC (padding
      // receivers). The D2D copy in enqueue_main never touches this tail, so
      // one upfront zero is sufficient — no per-run cudaMemset needed.
      CUDA_CHECK(cudaMemset(d_samples_padded[i], 0,
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
    }

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    CUDA_CHECK(cudaEventCreateWithFlags(&visibilities_reset_done,
                                        cudaEventDisableTiming));
    graph_main.assign(num_buffers, nullptr);
    accumulate_done.resize(num_buffers);
    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaEventCreateWithFlags(&accumulate_done[i],
                                          cudaEventDisableTiming));
    }
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

    DEBUG_LOG("Copying weights...");
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemcpy(d_weights[i], h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
      // d_weights[i] is a __half* but is allocated and copied against
      // sizeof(BeamWeights) (above), exactly like every other pipeline's
      // b.weights -- the reinterpret_cast mirrors that existing convention.
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(
          reinterpret_cast<BeamWeights *>(d_weights[i]), streams[i]);
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
    tensor_16.addTensor(modePacketDirectColMajCons, "packetColMajCons");
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
    tensor_16.addPermutation("packet", "packetColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToColMajCons");
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

    // Warm up the pipeline *before* attempting graph capture. This is the
    // first-ever call to most cuTENSOR permutations, the TCC correlator, and
    // the ccglib GEMM. Each of these libraries may do one-time lazy
    // initialization on first use -- NVRTC JIT compilation, module loading,
    // on-disk JIT-cache writes -- which is not safe to perform while
    // cudaStreamBeginCapture is active (can hang or segfault, especially
    // with a cold cache). Running this eagerly first ensures capture below
    // only ever records already-loaded kernels. Because everything is
    // zeroed it should have negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();

    // Capture the static mid-pipeline section of each buffer into a CUDA
    // graph (~15 launches collapse into 1 per run). Any failure falls back
    // to eager execution for all buffers -- functionally identical, just
    // more launch overhead. SPATIAL_DISABLE_CUDA_GRAPH=1 forces the eager
    // path.
    if (std::getenv("SPATIAL_DISABLE_CUDA_GRAPH") == nullptr) {
      bool all_ok = true;
      for (auto i = 0; i < num_buffers; ++i) {
        if (!capture_graph(
                streams[i], [this, i]() { enqueue_main(i); }, graph_main[i])) {
          all_ok = false;
          break;
        }
      }
      if (!all_ok) {
        for (auto &g : graph_main) {
          if (g) {
            cudaGraphExecDestroy(g);
            g = nullptr;
          }
        }
        WARN_LOG("CUDA graph capture failed — pipeline will run eagerly");
      } else {
        INFO_LOG("CUDA graphs captured for {} pipeline buffers", num_buffers);
        // Exercise the captured graphs (cudaGraphLaunch replay) once before
        // real traffic arrives, so a replay problem surfaces here rather
        // than on the first live buffer.
        execute_pipeline(&warmup_packet, true);
      }
      cudaDeviceSynchronize();
    }

    // these need to be set after the dummy run(s).
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

    for (auto samples_half : d_samples_half) {
      cudaFree(samples_half);
    }

    for (auto samples_padded : d_samples_padded) {
      cudaFree(samples_padded);
    }

    for (auto samples_consolidated_col_maj : d_samples_consolidated_col_maj) {
      cudaFree(samples_consolidated_col_maj);
    }

    for (auto event : start_run) {
      cudaEventDestroy(event);
    }
    for (auto event : stop_run) {
      cudaEventDestroy(event);
    }

    for (auto g : graph_main) {
      if (g)
        cudaGraphExecDestroy(g);
    }
    for (auto event : accumulate_done) {
      if (event)
        cudaEventDestroy(event);
    }
    if (visibilities_reset_done)
      cudaEventDestroy(visibilities_reset_done);
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

    INFO_LOG("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_correlation_units_integrated;
    INFO_LOG("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    // GPU-side wait (not a host-blocking sync): make streams[0]'s upcoming
    // memcpy of d_visibilities_accumulator wait for every buffer's most
    // recent accumulate_visibilities to land first. accumulate_done is a
    // no-op wait until it has been recorded at least once.
    for (auto &event : accumulate_done) {
      cudaStreamWaitEvent(streams[0], event, 0);
    }
    const int visibilities_total_packets =
        current_num_integrated_units_processed *
        visibilities_total_packets_per_block;
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num, end_seq_num, visibilities_missing_packets,
        visibilities_total_packets);
    visibilities_start_seq_num = -1;
    visibilities_missing_packets = 0;
    if (block_num != std::numeric_limits<size_t>::max()) {
      void *landing_pointer =
          output_->get_visibilities_landing_pointer(block_num);
      cudaMemcpyAsync(landing_pointer, d_visibilities_accumulator,
                      sizeof(TrimmedVisibilities), cudaMemcpyDefault,
                      streams[0]);
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};

      cudaLaunchHostFunc(streams[0],
                         output_visibilities_transfer_complete_host_func,
                         output_ctx);
    }
    cudaMemsetAsync(d_visibilities_accumulator, 0, sizeof(TrimmedVisibilities),
                    streams[0]);
    num_correlation_units_integrated.store(0);
    // Record the reset so every buffer's next accumulate_visibilities (via
    // the wait above, in execute_pipeline) is ordered after it.
    cudaEventRecord(visibilities_reset_done, streams[0]);
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

  typename T::AntennaGains *d_gains;
  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;

public:
  // -------------------------------------------------------------------------
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("LambdaProjectionPipeline run: start_seq={} end_seq={}",
             start_seq_num, end_seq_num);

    auto &b = buffers[current_buffer];

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);
    tensor_16.runPermutation(
        "packetToPadding", alpha_16,
        reinterpret_cast<__half *>(b.samples_half.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

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

    tensor_16.runPermutation(
        "paddedToCorrInput", alpha_16,
        reinterpret_cast<__half *>(b.samples_padded.get()),
        reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);
    correlator.launchAsync(
        static_cast<CUstream>(b.stream),
        reinterpret_cast<CUdeviceptr>(b.correlator_output.get()),
        reinterpret_cast<CUdeviceptr>(b.correlator_input.get()));

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
    tensor_32.runPermutation(
        "visCorrToDecomp", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed.get()),
        reinterpret_cast<float *>(b.decomp_visibilities.get()), b.stream);

    unpack_triangular_baseline_batch_launch<cuComplex>(
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        T::NR_RECEIVERS, CUSOLVER_BATCH_SIZE, T::NR_CHANNELS, b.stream);

    // ------------------------------------------------------------------
    // 10. Eigen-decompose R per batch element.
    //     cuSOLVER overwrites decomp_visibilities with eigenvectors
    //     (columns, ascending eigenvalue order).
    // ------------------------------------------------------------------
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

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

    // ------------------------------------------------------------------
    // 12. Add P_block to the shared accumulator.
    //     accumulate_visibilities adds src into dst element-wise on the
    //     stream (matching usage in LambdaGPUPipeline).
    // ------------------------------------------------------------------
    accumulate_visibilities(
        reinterpret_cast<float *>(b.projection_block.get()),
        reinterpret_cast<float *>(d_projection_accumulator),
        // Factor of 2 for complex (real + imag); full square matrix.
        2 * CUSOLVER_BATCH_SIZE * T::NR_RECEIVERS * T::NR_RECEIVERS, b.stream);

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

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));
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
  void dump_projection(const uint64_t start_seq_num,
                       const uint64_t end_seq_num) {
    INFO_LOG("LambdaProjectionPipeline: dumping averaged projection "
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
      // size_t::max means no eigen writer attached -- the landing pointers
      // would be nullptr.
      if (block_num != std::numeric_limits<size_t>::max()) {
        // d_projection_averaged now holds the eigenvectors (cuSOLVER
        // in-place).
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
    }

    // Reset accumulator and run counter.
    CUDA_CHECK(cudaMemsetAsync(d_projection_accumulator, 0,
                               sizeof(DecompositionVisibilities), b0.stream));
    num_runs_integrated.store(0);

    cudaDeviceSynchronize();
    INFO_LOG("LambdaProjectionPipeline: dump complete.");
  }
};

template <typename T, bool RFI_MITIGATE = false>
class LambdaPulsarFoldPipeline : public GPUPipeline {
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

  using ProjectionMatrix =
      std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                          [T::NR_RECEIVERS];
  using FloatProjectionMatrix =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  struct RFIMitigatedT {
    static constexpr size_t NR_CHANNELS = T::NR_CHANNELS;
    static constexpr size_t NR_POLARIZATIONS = T::NR_POLARIZATIONS;
    static constexpr size_t NR_BEAMS = 2 * T::NR_BEAMS;
    static constexpr size_t NR_RECEIVERS = T::NR_RECEIVERS;
  };

  using RFIMitigatedBeamWeights = BeamWeightsT<RFIMitigatedT>;

  static constexpr int num_beams = T::NR_BEAMS * (RFI_MITIGATE ? 2 : 1);
  using BeamformerOutput = float[T::NR_CHANNELS][T::NR_POLARIZATIONS][num_beams]
                                [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamOutput = float[num_beams][NR_TIME_STEPS_FOR_CORRELATION]
                          [T::NR_CHANNELS][T::NR_POLARIZATIONS][COMPLEX];

  using BeamWeights = BeamWeightsT<T>;
  using ChosenBeamWeights =
      std::conditional_t<RFI_MITIGATE, RFIMitigatedBeamWeights, BeamWeights>;
  bool header_written;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align;
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
    DevicePtr<BeamOutput> beam_output;
    DevicePtr<BeamWeights> weights, weights_permuted;
    DevicePtr<BeamWeights> weights_updated;
    DevicePtr<RFIMitigatedBeamWeights> weights_rfi_mitigated;
    DevicePtr<ChosenBeamWeights> weights_beamformer;
    DevicePtr<BeamformerOutput> beamformer_output;

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

    PipelineResources(CUdevice cu_device)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          weights_rfi_mitigated(make_device_ptr<RFIMitigatedBeamWeights>()),
          weights_beamformer(make_device_ptr<ChosenBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
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
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))) {
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
      // if RFI mitigate - we will have double the number of beams.
      gemm_handle = std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, num_beams,
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
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_consolidated(std::move(other.samples_consolidated)),
          samples_consolidated_col_maj(
              std::move(other.samples_consolidated_col_maj)),
          beam_output(std::move(other.beam_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
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
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        beam_output = std::move(other.beam_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        gemm_handle = std::move(other.gemm_handle);

        other.stream = nullptr;

        other.host_stream = nullptr;
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers{1};
  std::vector<PipelineResources> buffers;

  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;
  tcc::Correlator correlator;
  // We are converting it to fp16 so this should not be changable anymore.

  inline static const __half alpha = __float2half(1.0f);

  std::unordered_map<int, int> NR_SIGNAL_EIGENVECTORS;
  int min_freq_channel;

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

  inline static const std::vector<int> modePacket{'c', 'y', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};

  inline static const std::vector<int> modePacketPreAlign{'f', 'y', 'u', 'c',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
                                                         'n', 'p', 'z'};
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
      {'d', T::NR_PADDED_RECEIVERS},
      {'e', num_beams}, // rfi mitigated beam + original beam
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
      {'y', T::NR_PACKETS_FOR_CORRELATION + 2},
      {'z', 2}, // real, imaginary
  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;

  multilog_t *log;
  key_t dada_key;
  dada_hdu_t *hdu;
  key_t rfi_dada_key;
  dada_hdu_t *rfi_hdu;

  char *obs_header;
  char *d_obs_header;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;
  int *d_subpacket_delays;
  typename T::AntennaGains *d_gains;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    auto &b = buffers[current_buffer];
    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call, so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // b.weights, and only from this single-threaded pipeline_feeder context
    // -- see the BeamSteering<T> comment block (pipeline.hpp above) for why
    // that ordering is what makes this safe without extra synchronization.
    beam_steering_.maybe_refresh();

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    tensor_16.runPermutation("alignedToPlanar", alpha,
                             (__half *)b.samples_aligned.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    if (RFI_MITIGATE) {
      tensor_16.runPermutation(
          "alignedToPadding", alpha,
          reinterpret_cast<__half *>(b.samples_aligned.get()),
          reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

      // ------------------------------------------------------------------
      // 5. Copy unpadded → padded buffer then zero-fill the padding region
      // ------------------------------------------------------------------
      CUDA_CHECK(
          cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                          sizeof(typename T::HalfPacketAlignedSamplesType),
                          cudaMemcpyDefault, b.stream));
      CUDA_CHECK(
          cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                              sizeof(typename T::HalfPacketAlignedSamplesType),
                          0,
                          sizeof(typename T::PaddedPacketSamplesType) -
                              sizeof(typename T::HalfPacketAlignedSamplesType),
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
          reinterpret_cast<void *>(b.decomp_visibilities.get()),
          T::NR_RECEIVERS, CUDA_R_32F,
          reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
          b.cusolver_work_device.get(), b.cusolver_work_device_size,
          b.cusolver_work_host, b.cusolver_work_host_size,
          b.cusolver_info.get(), CUSOLVER_BATCH_SIZE));

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

        auto *V_base =
            reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
        auto *P_base =
            reinterpret_cast<cuComplex *>(b.float_projection_matrix.get());
        const size_t CUBLAS_BATCH_SIZE_PER_CHANNEL = T::NR_POLARIZATIONS;

        for (int channel = 0; channel < T::NR_CHANNELS; ++channel) {
          const int K = NR_SIGNAL_EIGENVECTORS[min_freq_channel + channel];
          const int col_offset = N - K; // first signal-subspace column
          // Pointer to signal-subspace U = last K columns of V_batch.
          for (int batch = 0; batch < CUBLAS_BATCH_SIZE_PER_CHANNEL; batch++) {
            // Pointer to the start of eigenvector matrix for this batch
            // element.
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
                            (__half2 *)b.projection_matrix.get(),
                            T::NR_RECEIVERS,
                            T::NR_CHANNELS * T::NR_POLARIZATIONS, b.stream);

      // conjugateMatrix((__half2 *)b.projection_matrix.get(),
      //                 T::NR_RECEIVERS * T::NR_RECEIVERS * T::NR_CHANNELS *
      //                     T::NR_POLARIZATIONS,
      //                 b.stream);

      {
        size_t CUBLAS_STRIDE_A = T::NR_RECEIVERS * T::NR_RECEIVERS;
        size_t CUBLAS_STRIDE_B = T::NR_RECEIVERS * T::NR_BEAMS;
        size_t CUBLAS_STRIDE_C = T::NR_RECEIVERS * T::NR_BEAMS;

        b.gemm_weight_projection_handle->Run(
            (CUdeviceptr)b.weights.get(),
            (CUdeviceptr)b.projection_matrix.get(),
            (CUdeviceptr)b.weights_updated.get());
      }

      // weightsDebugLaunch((__half2 *)b.weights_updated.get(),
      //                    T::NR_CHANNELS * T::NR_POLARIZATIONS *
      //                    T::NR_RECEIVERS
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
    } else {
      // Re-permute b.weights -> b.weights_permuted on every execute_pipeline
      // call so that tracking updates written by maybe_refresh() (which lands
      // in b.weights) are reflected in the GEMM.  Without this step, the GEMM
      // always ran on the weights_permuted that was initialised once in the
      // constructor, ignoring every steering update after that.
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      tensor_16.runPermutation("weights2xBeamMajorToCCGLIB", alpha,
                               (__half *)b.weights_permuted.get(),
                               (__half *)b.weights_beamformer.get(), b.stream);
    }

    b.gemm_handle->Run((CUdeviceptr)b.weights_beamformer.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamCCGLIBtoOutput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.beam_output.get(), b.stream);

    if (output_ != nullptr && !dummy_run) {

      // PSRDADA beam streaming -- the header/data block writes below all
      // dereference `hdu`, which is null when the sink is disabled
      // (dada_key == 0). Skip them in that case; the eigen-output block that
      // follows uses `output_` (not PSRDADA) and is intentionally left
      // outside this guard.
      if (dada_key != 0) {
        if (!header_written) {
          std::cout << "writing header...\n";
          uint64_t rfi_header_size = 0;
          uint64_t header_size = ipcbuf_get_bufsz(hdu->header_block);
          char *header = ipcbuf_get_next_write(hdu->header_block);
          cudaMemcpyAsync(header, d_obs_header, header_size, cudaMemcpyDefault,
                          b.stream);

          if constexpr (RFI_MITIGATE) {

            rfi_header_size = ipcbuf_get_bufsz(rfi_hdu->header_block);
            char *rfi_header = ipcbuf_get_next_write(rfi_hdu->header_block);

            cudaMemcpyAsync(rfi_header, d_obs_header, header_size,
                            cudaMemcpyDefault, b.stream);
          }

          // // Enable EOD so that subsequent transfers will move to the next
          // buffer
          // // in the header block
          // if (ipcbuf_enable_eod(hdu->header_block) < 0) {
          //   multilog(log, LOG_ERR, "Could not enable EOD on Header Block\n");
          // }

          cudaDeviceSynchronize();
          // flag the header block for this "observation" as filled
          if (ipcbuf_mark_filled(hdu->header_block, header_size) < 0) {
            multilog(log, LOG_ERR, "could not mark filled Header Block\n");
            std::cout << "could not mark filled header block...\n";
          }

          if constexpr (RFI_MITIGATE) {
            if (ipcbuf_mark_filled(rfi_hdu->header_block, rfi_header_size) <
                0) {
              multilog(log, LOG_ERR, "could not mark filled Header Block\n");
              std::cout << "could not mark filled header block...\n";
            }
          }
          header_written = true;
        }

        uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
        // write 1 block worth of data block via the "block" method
        {
          uint64_t rfi_block_size = 0;
          uint64_t block_id;
          char *block = ipcio_open_block_write(hdu->data_block, &block_id);
          if (!block) {
            multilog(log, LOG_ERR, "ipcio_open_block_write failed\n");
            std::cout << "open block write failed\n";
          }

          // control how much gets written using the block_size.
          // can toggle polarization from outer to inner dimensions
          // in order to control how many polarizations get written out.
          cudaMemcpyAsync(block, (char *)b.beam_output.get(), block_size,
                          cudaMemcpyDefault, b.stream);

          if constexpr (RFI_MITIGATE) {

            rfi_block_size = ipcbuf_get_bufsz((ipcbuf_t *)rfi_hdu->data_block);
            uint64_t rfi_block_id;
            char *rfi_block =
                ipcio_open_block_write(rfi_hdu->data_block, &rfi_block_id);
            if (!rfi_block) {
              multilog(log, LOG_ERR, "ipcio_open_block_write failed\n");
              std::cout << "open block write failed\n";
            }
            // This is a big hack it will only take the X pol right now.
            cudaMemcpyAsync(rfi_block, (char *)b.beam_output.get() + block_size,
                            rfi_block_size, cudaMemcpyDefault, b.stream);
          }

          cudaDeviceSynchronize();

          if (ipcio_close_block_write(hdu->data_block, block_size) < 0) {
            multilog(log, LOG_ERR, "ipcio_close_block_write failed\n");
          }
          if constexpr (RFI_MITIGATE) {
            if (ipcio_close_block_write(rfi_hdu->data_block, rfi_block_size) <
                0) {
              multilog(log, LOG_ERR, "ipcio_close_block_write failed\n");
            }
          }
        }
      } // end PSRDADA beam streaming (dada_key != 0)

      if constexpr (RFI_MITIGATE) {
        size_t eig_block_num = output_->register_eigendecomposition_data_block(
            start_seq_num, end_seq_num);

        if (eig_block_num != std::numeric_limits<size_t>::max()) {
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
      }
    }

    // Rotate buffer indices
  }
  LambdaPulsarFoldPipeline(
      BeamWeightsT<T> *h_weights,
      const std::unordered_map<int, int> nr_signal_eigenvectors,
      const int min_freq_channel, key_t dada_key, std::string header_filename,
      key_t rfi_dada_key, BeamSteering<T> beam_steering)

      : num_buffers(1), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        correlator(cu::Device(0), 16, T::NR_PADDED_RECEIVERS, T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        NR_SIGNAL_EIGENVECTORS(nr_signal_eigenvectors), header_written(false),
        min_freq_channel(min_freq_channel), dada_key(dada_key),
        rfi_dada_key(rfi_dada_key) {
    std::cout << "Pulsar Fold instantiated with NR_CHANNELS: " << T::NR_CHANNELS
              << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << num_beams << std::endl;
    std::cout << "[PulsarFoldPipeline] beam output size is "
              << sizeof(BeamOutput) << " bytes." << std::endl;

    const size_t NUM_TOTAL_BATCHES = num_beams * T::NR_CHANNELS *
                                     T::NR_POLARIZATIONS *
                                     T::NR_PACKETS_FOR_CORRELATION;

    // set up PSRDADA ring buffer. A zero dada_key disables the PSRDADA sink
    // entirely (no connect/lock, no header read, no block writes) -- the
    // pipeline then computes beams into b.beam_output but never streams them
    // out. Production always passes a real key (DADA_DEFAULT_BLOCK_KEY); the
    // zero-key path exists so tests can drive the full GPU compute without a
    // running ring buffer. Kept in lockstep with the same guard in
    // execute_pipeline's output block and in the destructor.
    log = nullptr;
    hdu = nullptr;
    rfi_hdu = nullptr;
    obs_header = nullptr;
    d_obs_header = nullptr;
    if (dada_key != 0) {
      log = multilog_open("pulsar_fold_writer", 0);
      multilog_add(log, stderr);
      hdu = dada_hdu_create(log);
      dada_hdu_set_key(hdu, dada_key);
      // connect to HDU
      if (dada_hdu_connect(hdu) < 0) {
        multilog(log, LOG_ERR, "could not connect to HDU\n");
      }

      // lock as writer on the HDU
      if (dada_hdu_lock_write(hdu) < 0) {
        multilog(log, LOG_ERR, "could not lock write on HDU\n");
      }

      if constexpr (RFI_MITIGATE) {
        rfi_hdu = dada_hdu_create(log);
        dada_hdu_set_key(rfi_hdu, rfi_dada_key);

        if (dada_hdu_connect(rfi_hdu) < 0) {
          multilog(log, LOG_ERR, "could not connect to RFI HDU\n");
        }

        // lock as writer on the HDU
        if (dada_hdu_lock_write(rfi_hdu) < 0) {
          multilog(log, LOG_ERR, "could not lock write on RFI HDU\n");
        }
      }

      obs_header = (char *)malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);

      if (fileread(header_filename.c_str(), obs_header,
                   DADA_DEFAULT_HEADER_SIZE) < 0) {
        free(obs_header);
        fprintf(stderr, "ERROR: could not read ASCII header from %s\n",
                header_filename);
      }

      // Overwrite UTC_START with the current wall-clock UTC so DSPSR folds
      // at the correct pulsar phase. The header file contains a placeholder
      // that is only valid for the exact instant it was written.
      {
        time_t now = time(nullptr);
        struct tm utc_tm{};
        gmtime_r(&now, &utc_tm);
        char utc_start[32];
        strftime(utc_start, sizeof(utc_start), "%Y-%m-%d-%H:%M:%S", &utc_tm);
        if (ascii_header_set(obs_header, "UTC_START", "%s", utc_start) < 0)
          fprintf(stderr, "WARNING: could not set UTC_START in PSRDADA header\n");
        else
          fprintf(stderr, "INFO: PSRDADA header UTC_START set to %s\n", utc_start);
      }

      cudaMalloc(&d_obs_header, DADA_DEFAULT_HEADER_SIZE);
      cudaMemcpy(d_obs_header, obs_header, DADA_DEFAULT_HEADER_SIZE,
                 cudaMemcpyDefault);
    }
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketAligned, "aligned");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsBeamMajor, "weightsBeamMajor");
    tensor_16.addTensor(modeWeights2xBeamMajor, "weights2xBeamMajor");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    // Permutation descriptors
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_16.addPermutation("aligned", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToPadding");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("aligned", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "alignedToPlanar");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToDecomp");

    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsBeamMajor",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsToBeamMajor");

    tensor_16.addPermutation("weights2xBeamMajor", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F,
                             "weights2xBeamMajorToCCGLIB");

    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBtoOutput");

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));

    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      if constexpr (RFI_MITIGATE) {
        cudaMemcpyAsync(b.weights_rfi_mitigated.get(), b.weights_permuted.get(),
                        sizeof(BeamWeights), cudaMemcpyDefault, b.stream);
      }
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
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

  ~LambdaPulsarFoldPipeline() {
    // Mirror the constructor: nothing to tear down when the PSRDADA sink was
    // disabled (dada_key == 0), and hdu/log are left null in that case.
    if (dada_key == 0)
      return;

    // Signal end-of-data on the data block before releasing the write lock.
    // Without this, DSPSR's reader loop blocks indefinitely on the next
    // ipcio_open_block_read() call -- it sees a broken writer connection
    // rather than a clean observation end, and does not finalize the folded
    // profile.
    if (ipcbuf_enable_eod((ipcbuf_t *)hdu->data_block) < 0)
      multilog(log, LOG_ERR, "ipcbuf_enable_eod failed on data block\n");

    if (dada_hdu_unlock_write(hdu) < 0) {
      multilog(log, LOG_ERR, "dada_hdu_unlock_write failed\n");
    }

    // disconnect from HDU
    if (dada_hdu_disconnect(hdu) < 0)
      multilog(log, LOG_ERR, "could not disconnect from hdu\n");

    if constexpr (RFI_MITIGATE) {
      if (ipcbuf_enable_eod((ipcbuf_t *)rfi_hdu->data_block) < 0)
        multilog(log, LOG_ERR, "ipcbuf_enable_eod failed on rfi data block\n");

      if (dada_hdu_unlock_write(rfi_hdu) < 0) {
        multilog(log, LOG_ERR, "dada_hdu_unlock_write failed\n");
      }

      // disconnect from HDU
      if (dada_hdu_disconnect(rfi_hdu) < 0)
        multilog(log, LOG_ERR, "could not disconnect from rfi hdu\n");
    }
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }

  virtual void set_subpacket_delays(int *delays_subpacket) {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }

  // Test/debug hook. After execute_pipeline() completes, copies the most
  // recently computed beamformer output -- the exact device-side
  // `b.beam_output` buffer that gets streamed to PSRDADA -- to a host
  // destination (`dst` must have room for beam_output_size_bytes()). Lets
  // tests inspect the beams without a PSRDADA sink (see the dada_key == 0
  // path); not used on the production hot path.
  void copy_latest_beam_output_to_host(void *dst) {
    auto &b = buffers[current_buffer];
    CUDA_CHECK(cudaStreamSynchronize(b.stream));
    CUDA_CHECK(cudaMemcpy(dst, b.beam_output.get(), sizeof(BeamOutput),
                          cudaMemcpyDefault));
  }
  static constexpr size_t beam_output_size_bytes() {
    return sizeof(BeamOutput);
  }
  // Beam-output shape: [NUM_BEAMS][NR_TIMES][NR_CHANNELS][NR_POL][COMPLEX].
  static constexpr int beam_output_num_beams() { return num_beams; }
  static constexpr int beam_output_num_times() {
    return NR_TIME_STEPS_FOR_CORRELATION;
  }
};
