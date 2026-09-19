#pragma once
#include <optional>
#include "spatial/packet_formats.hpp"
#include "spatial/pointing.hpp"
#include "spatial/spatial.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <limits>
#include <mutex>
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
#include <libfilter/Filter.h>
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

// Set BEAM_DEBUG=1 in the environment to enable verbose beam-tracking and
// beam-output diagnostics.  Checked once at first call; zero cost thereafter.
inline bool beam_debug() {
  static bool result = std::getenv("BEAM_DEBUG") != nullptr;
  return result;
}

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

    if (beam_debug()) {
      std::cout << "[BeamSteering] beam " << b
                << " target=(" << target.ra_deg << "," << target.dec_deg
                << ") mode=" << target.mode
                << " direction_cosines l=" << dc.l
                << " m=" << dc.m << " n=" << dc.n << "\n";
    }

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

        // calibration_gains is loaded per coarse/FPGA channel (see get_gains_structure) and
        // applied uniformly across every fine channel within it -- chan here ranges over the
        // widened T::NR_CHANNELS (coarse*NR_EFFECTIVE_FINE_CHANNELS + effective_fine, where
        // NR_EFFECTIVE_FINE_CHANNELS accounts for the edge-trimmed fine channels dropped by
        // channelizer_output_to_corr_input), so map back down to the coarse index before indexing
        // the NR_FPGA_CHANNELS-sized AntennaGains array.
        const size_t coarse_chan = chan / T::NR_EFFECTIVE_FINE_CHANNELS;
        for (size_t pol = 0; pol < T::NR_POLARIZATIONS; ++pol) {
          std::complex<double> calibration_gain =
              calibration_gains
                  ? std::complex<double>(
                        (*calibration_gains)[coarse_chan][pol][receiver_idx])
                  : std::complex<double>(1.0, 0.0);

          std::complex<double> final_weight =
              (1.0 / static_cast<double>(T::NR_RECEIVERS)) * calibration_gain *
              steering_phasor;

          result.weights[chan][pol][b][receiver_idx] = std::complex<__half>(
              __float2half(static_cast<float>(final_weight.real())),
              __float2half(static_cast<float>(final_weight.imag())));

          if (beam_debug()) {
            std::cout << "[BeamSteering] weight beam=" << b
                      << " chan=" << chan << " pol=" << pol
                      << " recv=" << receiver_idx << " = "
                      << final_weight.real() << "+" << final_weight.imag()
                      << "j\n";
          }
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
    if (!targets_.empty()) {
      INFO_LOG("BeamSteering: tracking {} beam target(s) with {:.1f}s update interval",
               targets_.size(), update_interval_seconds);
      for (size_t i = 0; i < targets_.size(); ++i) {
        INFO_LOG("  beam {}: mode={} ra={:.4f} dec={:.4f}",
                 i, targets_[i].mode, targets_[i].ra_deg, targets_[i].dec_deg);
      }
    } else {
      INFO_LOG("BeamSteering: no targets supplied -- steering is disabled (inert)");
    }
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
    if (!active() || buffers_.empty()) {
      if (beam_debug())
        std::cout << "[BeamSteering] maybe_refresh: inactive (targets="
                  << targets_.size() << " buffers=" << buffers_.size() << ")\n";
      return false;
    }

    // last_update_ starts at the epoch, so the very first call -- during the
    // constructor's warmup run -- is immediately overdue and synthesizes real
    // weights right away rather than running on placeholder h_weights.
    const auto now = std::chrono::system_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - last_update_).count();

    if ((now - last_update_) < update_interval_) {
      if (beam_debug())
        std::cout << "[BeamSteering] maybe_refresh: not due ("
                  << elapsed_s << "s elapsed, interval="
                  << update_interval_.count() << "s)\n";
      return false;
    }

    INFO_LOG("BeamSteering: refreshing beam weights ({:.1f}s since last update)",
             elapsed_s);

    current_weights_ = compute_steering_weights<T>(
        targets_, antenna_positions_, antenna_mapping_, frequency_plan_,
        min_freq_channel_, array_location_, now, calibration_gains_);
    last_update_ = now;

    INFO_LOG("BeamSteering: weights recomputed, enqueuing to {} buffer(s)",
             buffers_.size());

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

// Allocates only when Condition is true at compile time, otherwise returns a
// null DevicePtr (no cudaMalloc) -- for buffers a pipeline only reads/writes
// inside an `if constexpr (Condition)` branch elsewhere, so allocating them
// unconditionally would just waste device memory in the disabled branch.
template <typename T, bool Condition>
DevicePtr<T> make_device_ptr_if(size_t size = sizeof(T)) {
  if constexpr (Condition) {
    return make_device_ptr<T>(size);
  } else {
    return DevicePtr<T>();
  }
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

struct EigenOutputTransferWithCountsContext {
  std::shared_ptr<Output> output;
  size_t block_index;
  void *counts_dst;
  const int32_t *counts_src;
  size_t counts_size_bytes;
  std::mutex *stats_mutex = nullptr;
  std::vector<int32_t> *stats_history = nullptr;
};

struct BeamCountsTransferCompleteContext {
  std::shared_ptr<Output> output;
  size_t block_index;
  void *counts_dst;
  const int32_t *counts_src;
  size_t counts_size_bytes;
  std::mutex *stats_mutex = nullptr;
  std::vector<int32_t> *stats_history = nullptr;
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

static void eigen_output_transfer_with_counts_host_func(void *data) {
  auto *ctx = static_cast<EigenOutputTransferWithCountsContext *>(data);
  if (ctx->counts_dst != nullptr) {
    std::memcpy(ctx->counts_dst, ctx->counts_src, ctx->counts_size_bytes);
  }
  if (ctx->stats_mutex != nullptr && ctx->stats_history != nullptr) {
    std::lock_guard<std::mutex> lock(*ctx->stats_mutex);
    const size_t count_elements = ctx->counts_size_bytes / sizeof(int32_t);
    ctx->stats_history->insert(ctx->stats_history->end(), ctx->counts_src,
                               ctx->counts_src + count_elements);
  }
  if (ctx->output != nullptr) {
    ctx->output->register_eigendecomposition_data_transfer_complete(
        ctx->block_index);
  }
  delete ctx;
}

static void beam_counts_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<BeamCountsTransferCompleteContext *>(data);
  if (ctx->counts_dst != nullptr) {
    std::memcpy(ctx->counts_dst, ctx->counts_src, ctx->counts_size_bytes);
  }
  if (ctx->stats_mutex != nullptr && ctx->stats_history != nullptr) {
    std::lock_guard<std::mutex> lock(*ctx->stats_mutex);
    const size_t n = ctx->counts_size_bytes / sizeof(int32_t);
    ctx->stats_history->insert(ctx->stats_history->end(), ctx->counts_src,
                               ctx->counts_src + n);
  }
  ctx->output->register_beam_counts_transfer_complete(ctx->block_index);
  delete ctx;
}

static void fft_output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);

  ctx->output->register_fft_transfer_complete(ctx->block_index);
  delete ctx;
}

inline float sorted_percentile_linear(const float *sorted_values, int count,
                                      float quantile) {
  if (count <= 0) {
    return 0.0f;
  }
  if (count == 1) {
    return sorted_values[0];
  }
  const float clamped = std::max(0.0f, std::min(1.0f, quantile));
  const float position = clamped * static_cast<float>(count - 1);
  const int lower = static_cast<int>(std::floor(position));
  const int upper = static_cast<int>(std::ceil(position));
  if (lower == upper) {
    return sorted_values[lower];
  }
  const float fraction = position - static_cast<float>(lower);
  return sorted_values[lower] +
         fraction * (sorted_values[upper] - sorted_values[lower]);
}

inline int detect_signal_eigenmode_count(const float *sorted_eigenvalues, int n,
                                         float delta) {
  const float p20 = sorted_percentile_linear(sorted_eigenvalues, n, 0.2f);
  const float p50 = sorted_percentile_linear(sorted_eigenvalues, n, 0.5f);
  const float p80 = sorted_percentile_linear(sorted_eigenvalues, n, 0.8f);
  const float sigma_noise = (p80 - p20) / (2.0f * 0.8416f);
  const float threshold = p50 + delta * sigma_noise;
  int detected = 0;
  for (int i = 0; i < n; ++i) {
    if (sorted_eigenvalues[i] > threshold) {
      ++detected;
    }
  }
  return detected;
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

    // Pre-channelization: operates on the raw FPGA/coarse channel count, not the
    // (possibly fine-channelized) widened T::NR_CHANNELS.
    scale_and_convert_to_half<T::NR_FPGA_CHANNELS, T::NR_POLARIZATIONS,
                              T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
                              T::NR_TIME_STEPS_PER_PACKET,
                              T::NR_PACKETS_FOR_CORRELATION + 2>(
        (char2 *)d_samples_entry, (int16_t *)d_scales, (float2 *)d_gains,
        (__half2 *)d_samples_half, stream);
  }
};

// Pre-correlation fine channelization via ASTRON's gpu-filter (ppf::Filter), a GPU polyphase
// filterbank. One ppf::Filter instance per coarse/FPGA channel -- FilterArgs has no "channel of
// channels" axis (confirmed against libfilter/Filter.cc's launch grid, which is
// (nrPolarizations, nrReceivers, nrSamplesPerChannel/16) with no coarse-channel dimension), and
// each instance's FIR stage carries PFB history that must not mix data from different coarse
// channels.
//
// Ring-buffer mode is intentionally left off (FilterArgs::ringBufferSize stays nullopt): with it
// off, Filter::launchAsync carries zero cross-call state (confirmed by reading Filter.cc -- it
// only builds a kernel-parameter list and calls stream.launchKernel(), no member read/write), so
// each call is a pure function of the input buffer handed to it. gpu-filter's FIR kernel
// (readInputAndDoFIRfiltering in FilterAndCorrect.cu) looks *forward* from each output tick, so
// that input must include NR_TAPS-1 trailing look-ahead samples past the "real" data;
// reorder_to_filter_input (spatial.cuh) zero-pads them at the end of every buffer rather than
// carrying continuity across buffers -- a known, accepted per-buffer edge transient for v1.
// Because there's no cross-call state, this is also what makes concurrent multi-buffer use of the
// same Filter instances (LambdaGPUPipeline's normal mode of operation) and CUDA graph capture
// both safe.
//
// Native per-antenna delay compensation (FilterArgs::delays, applied during channelization in the
// frequency domain) is intentionally left disabled here -- wiring real delay values through is
// deferred to the call site that retires the old apply_fine_delay_correction() path.
template <typename T> struct FineChannelizer {
  static constexpr size_t NR_FINE = T::NR_FINE_CHANNELS;
  static constexpr size_t NR_COARSE = T::NR_FPGA_CHANNELS;
  static constexpr unsigned NR_TAPS = 16;
  static_assert(T::NR_TIME_STEPS_FOR_CORRELATION % NR_FINE == 0,
               "NR_TIME_STEPS_FOR_CORRELATION must divide evenly by NR_FINE_CHANNELS");
  static constexpr size_t NR_SAMPLES_PER_FINE_CHANNEL =
      T::NR_TIME_STEPS_FOR_CORRELATION / NR_FINE;
  // These two constraints only matter when the channelizer is actually active (NR_FINE > 1) --
  // gated so that PipelineResources' *unconditional* FineChannelizer<T>::FilterInputType /
  // FilterOutputType member declarations (needed since C++ has no conditional member types)
  // don't break existing small/synthetic NR_FINE_CHANNELS==1 configs whose
  // NR_TIME_STEPS_FOR_CORRELATION was never designed around gpu-filter's internal batching size.
  static_assert(NR_FINE == 1 || NR_SAMPLES_PER_FINE_CHANNEL % 16 == 0,
               "gpu-filter's internal FFT batching (NR_TIMES_PER_ITERATION=16, fixed inside the "
               "library) requires NR_TIME_STEPS_FOR_CORRELATION / NR_FINE_CHANNELS to be a "
               "multiple of 16");
  // Matches TCC's NR_TIMES_PER_BLOCK for fp16 (128 bits / 16-bit half = 8) -- confirmed identical
  // formula in gpu-filter's own kernel (NR_TIMES_PER_OUTPUT_BLOCK = 128/bits(output format)), so
  // gpu-filter's output needs no block-regrouping to feed TCC's CorrelatorInput.
  static constexpr size_t NR_TIMES_PER_OUTPUT_BLOCK = 8;
  static_assert(NR_FINE == 1 || NR_SAMPLES_PER_FINE_CHANNEL % NR_TIMES_PER_OUTPUT_BLOCK == 0,
               "NR_SAMPLES_PER_FINE_CHANNEL must be a multiple of NR_TIMES_PER_OUTPUT_BLOCK");

  // Per-coarse-channel input: contiguous [receiver][pol][time] complex-float time series, with
  // NR_TAPS-1 trailing look-ahead samples after the NR_SAMPLES_PER_FINE_CHANNEL*NR_FINE "real"
  // samples (see reorder_to_filter_input in spatial.cuh). float, not half: gpu-filter's
  // InputSample only implements i16/i8/fp32 in its kernel -- fp16 hits a compile-time #error.
  using FilterInputType =
      float2[NR_COARSE][T::NR_RECEIVERS][T::NR_POLARIZATIONS]
            [(NR_SAMPLES_PER_FINE_CHANNEL + NR_TAPS - 1) * NR_FINE];

  // Per-coarse-channel output: gpu-filter's native layout,
  // [fine_channel][block][receiver][pol][time_in_block], complex-half -- bit-identical to TCC's
  // CorrelatorInput's trailing [COMPLEX] axis (cuda::std::complex<__half> == __half[2]).
  using FilterOutputType =
      __half2[NR_COARSE][NR_FINE][NR_SAMPLES_PER_FINE_CHANNEL / NR_TIMES_PER_OUTPUT_BLOCK]
             [T::NR_RECEIVERS][T::NR_POLARIZATIONS][NR_TIMES_PER_OUTPUT_BLOCK];

  std::vector<std::unique_ptr<ppf::Filter>> filters;

  explicit FineChannelizer(CUdevice cu_device) {
    cu::Device device(cu_device);

    ppf::FilterArgs args;
    args.nrReceivers = T::NR_RECEIVERS;
    args.nrChannels = NR_FINE;
    args.nrSamplesPerChannel = NR_SAMPLES_PER_FINE_CHANNEL;
    args.nrPolarizations = T::NR_POLARIZATIONS;
    args.input.sampleFormat = ppf::FilterArgs::fp32;
    args.input.isPurelyReal = false;
    args.firFilter = ppf::FilterArgs::FIR_Filter{NR_TAPS, ppf::FilterArgs::fp32};
    args.fft.sampleFormat = ppf::FilterArgs::fp32;
    args.fft.shift = true;  // ascending-frequency channel order -- confirmed against the kernel
    args.fft.mirror = false;
    args.output.sampleFormat = ppf::FilterArgs::fp16;

    INFO_LOG("FineChannelizer: constructing {} gpu-filter instances ({} fine channels each, {} "
             "samples/fine-channel, {} taps) -- this JIT-compiles {} nearly-identical NVRTC "
             "kernels and may take a while",
             NR_COARSE, NR_FINE, NR_SAMPLES_PER_FINE_CHANNEL, NR_TAPS, NR_COARSE);
    filters.reserve(NR_COARSE);
    for (size_t c = 0; c < NR_COARSE; ++c) {
      filters.push_back(std::make_unique<ppf::Filter>(device, args));
    }
  }

  // Launches one Filter per coarse channel on the given stream. d_filter_input/d_filter_output
  // must point at FilterInputType/FilterOutputType-shaped device buffers.
  //
  // Deliberately calls gpu-filter's cu::Stream-based launchAsync overload with our own
  // non-owning cu::Stream/cu::DeviceMemory wrappers, NOT its raw CUstream/CUdeviceptr overload
  // (Filter::launchAsync(CUstream, CUdeviceptr, CUdeviceptr, ...), libfilter/Filter.cc:258-275).
  // That raw-pointer overload was found, via direct isolated reproduction (bypassing this
  // project's pipeline entirely -- confirmed with nothing else, not even TCC, involved) to hang
  // indefinitely inside cuLaunchKernel every time, regardless of CUDA context model (tested both
  // the implicit runtime-API primary context this whole pipeline already relies on, and a fresh
  // explicit driver-API context). Manually constructing the same wrapper objects and calling the
  // cu::Stream-based overload directly (identical underlying stream/buffers, identical
  // parameters) instead works reliably and returns immediately. The exact mechanism inside
  // gpu-filter's own wrapping call wasn't isolated further; this is a confirmed, reproducible
  // workaround, not a guess -- see docs/architecture.md Section 6 for the full evidence trail.
  void launchAsync(cudaStream_t stream, void *d_filter_input, void *d_filter_output) {
    constexpr size_t in_bytes_per_channel = sizeof(FilterInputType) / NR_COARSE;
    constexpr size_t out_bytes_per_channel = sizeof(FilterOutputType) / NR_COARSE;
    cu::Stream cuStream((CUstream)stream);
    for (size_t c = 0; c < NR_COARSE; ++c) {
      cu::DeviceMemory in((CUdeviceptr)((char *)d_filter_input + c * in_bytes_per_channel));
      cu::DeviceMemory out((CUdeviceptr)((char *)d_filter_output + c * out_bytes_per_channel));
      filters[c]->launchAsync(cuStream, out, in);
    }
  }
};

static constexpr unsigned TCC_THREAD_BLOCKS_PER_SM = 4;
