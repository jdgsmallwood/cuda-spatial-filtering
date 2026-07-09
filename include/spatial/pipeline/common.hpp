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

struct EigenOutputTransferWithCountsContext {
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

    scale_and_convert_to_half<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                              T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
                              T::NR_TIME_STEPS_PER_PACKET,
                              T::NR_PACKETS_FOR_CORRELATION + 2>(
        (char2 *)d_samples_entry, (int16_t *)d_scales, (float2 *)d_gains,
        (__half2 *)d_samples_half, stream);
  }
};

static constexpr unsigned TCC_THREAD_BLOCKS_PER_SM = 2;
