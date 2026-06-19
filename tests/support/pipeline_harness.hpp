#pragma once

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "synthetic_packets.hpp"

#include <array>
#include <complex>
#include <cuda_fp16.h>
#include <memory>
#include <unordered_map>
#include <utility>

// The shared pipeline-test driver: wires a real `ProcessorState` to a real
// `GPUPipeline` via the production "synchronous pipeline" seam
// (ProcessorState::handle_buffer_completion calls pipeline_->execute_pipeline
// directly when synchronous_pipeline = true -- see spatial.hpp), feeds it
// synthetic wire-format packets, and lets callers assert on whatever the
// pipeline produced in a real `Output`.
//
// This intentionally drives the *real* ProcessorState ingestion path (not
// DummyFinalPacketData's shortcut of poking LambdaConfig array layouts
// directly) -- that's what "test at a high level" means here: the seam under
// test is PacketInput -> ProcessorState -> Pipeline -> Output, the same one
// production code uses end to end.
namespace test_support {

// Builds beam weights of unit magnitude / zero phase for every
// channel/polarization/beam/receiver -- the "do nothing to the signal"
// baseline used by test_pipeline.cu::Ex1 and friends.
template <typename Config>
BeamWeightsT<Config> make_unity_beam_weights() {
  BeamWeightsT<Config> weights{};
  const std::complex<__half> unity(__float2half(1.0f), __float2half(0.0f));
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t r = 0; r < Config::NR_RECEIVERS; ++r)
          weights.weights[c][p][b][r] = unity;
  return weights;
}

// Builds antenna calibration gains of unit magnitude / zero phase -- the
// "apply no calibration" baseline.
template <typename Config>
typename Config::AntennaGains make_unity_antenna_gains() {
  typename Config::AntennaGains gains{};
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t r = 0; r < Config::NR_RECEIVERS; ++r)
        gains[c][p][r] = std::complex<float>(1.0f, 0.0f);
  return gains;
}

// Per-variant pipeline construction. The 7 `Lambda*Pipeline` classes take
// genuinely different constructor arguments (e.g. LambdaGPUPipeline(int,
// BeamWeightsT<T>*) vs LambdaAntennaSpectraPipeline(int) vs
// LambdaProjectionPipeline<T, NR_EIG, NR_RUNS>(int)), so a single generic
// factory isn't realistic -- these small per-variant helpers are it. Only the
// one needed by this phase's self-test is implemented; the rest are added
// incrementally as later phases need them.
namespace pipeline_factories {

template <typename Config>
std::unique_ptr<LambdaGPUPipeline<Config>>
make_gpu_pipeline(int num_buffers, BeamWeightsT<Config> *weights,
                  int min_freq_channel = 0) {
  // Inert beam steering (empty targets => maybe_refresh() is a permanent
  // no-op, leaving the static `weights` untouched).
  BeamSteering<Config> beam_steering(/*targets=*/{}, /*antenna_positions=*/{},
                                     /*antenna_mapping=*/{}, FrequencyPlan{},
                                     min_freq_channel, ArrayLocation{},
                                     /*update_interval_seconds=*/1.0,
                                     num_buffers);
  return std::make_unique<LambdaGPUPipeline<Config>>(num_buffers, weights,
                                                     std::move(beam_steering));
}

// Constructs a LambdaPulsarFoldPipeline with its PSRDADA sink disabled
// (dada_key == 0), so the full GPU compute (ingest/scale -> permute -> ccglib
// beamform) runs without a running ring buffer or a header.hdr file. The
// computed beams stay in the pipeline's device buffer and are read back via
// LambdaPulsarFoldPipeline::copy_latest_beam_output_to_host(). Steering is
// inert (empty targets); RFI mitigation is off, so the eigen path never runs.
template <typename Config>
std::unique_ptr<LambdaPulsarFoldPipeline<Config, false>>
make_pulsar_fold_pipeline(BeamWeightsT<Config> *weights,
                          int min_freq_channel = 0) {
  BeamSteering<Config> beam_steering(/*targets=*/{}, /*antenna_positions=*/{},
                                     /*antenna_mapping=*/{}, FrequencyPlan{},
                                     min_freq_channel, ArrayLocation{},
                                     /*update_interval_seconds=*/1.0,
                                     /*num_buffers=*/1);
  return std::make_unique<LambdaPulsarFoldPipeline<Config, false>>(
      weights, /*nr_signal_eigenvectors=*/std::unordered_map<int, int>{},
      min_freq_channel, /*dada_key=*/0, /*header_filename=*/"",
      /*rfi_dada_key=*/0, std::move(beam_steering));
}

// Constructs a LambdaPulsarFoldPipeline with active beam steering (targets is
// non-empty so maybe_refresh() actually fires) but with the PSRDADA sink
// disabled (dada_key == 0). Used to test that the non-RFI-mitigate path
// re-applies updated steering weights each pipeline run.
//
// `update_interval_seconds` is set large so only the constructor's warmup run
// triggers a refresh -- the test's single real run then exercises whatever
// state maybe_refresh() left behind.
template <typename Config>
std::unique_ptr<LambdaPulsarFoldPipeline<Config, false>>
make_tracked_pulsar_fold_pipeline(BeamWeightsT<Config> *weights,
                                   std::vector<BeamTarget> targets,
                                   int min_freq_channel = 0) {
  BeamSteering<Config> beam_steering(
      std::move(targets), /*antenna_positions=*/{}, /*antenna_mapping=*/{},
      FrequencyPlan{}, min_freq_channel, ArrayLocation{},
      /*update_interval_seconds=*/3600.0, /*num_buffers=*/1);
  return std::make_unique<LambdaPulsarFoldPipeline<Config, false>>(
      weights, /*nr_signal_eigenvectors=*/std::unordered_map<int, int>{},
      min_freq_channel, /*dada_key=*/0, /*header_filename=*/"",
      /*rfi_dada_key=*/0, std::move(beam_steering));
}

} // namespace pipeline_factories

// Drives a complete synthetic correlation buffer through a real
// ProcessorState -> GPUPipeline -> Output, synchronously.
template <typename Config, size_t NR_BUFFERS = 3>
class SyntheticPipelineRun {
public:
  explicit SyntheticPipelineRun(
      GPUPipeline &pipeline, std::shared_ptr<Output> output,
      std::array<int64_t, Config::NR_FPGA_SOURCES> fpga_delays = {},
      std::unordered_map<uint32_t, int> fpga_id_map = {{0, 0}},
      size_t min_freq_channel = 0)
      : pipeline_(pipeline), output_(std::move(output)),
        state_(Config::NR_PACKETS_FOR_CORRELATION,
               Config::NR_TIME_STEPS_PER_PACKET, min_freq_channel, fpga_delays,
               std::move(fpga_id_map)) {
    pipeline_.set_state(&state_);
    pipeline_.set_output(output_);
    pipeline_.set_subpacket_delays(subpacket_delays_.data());

    state_.set_pipeline(&pipeline_);
    state_.synchronous_pipeline = true;
  }

  // Fills exactly one correlation buffer with synthetic packets -- mirroring
  // ProcessorStateTest::FillOneBufferTest's proven fill order (packets
  // 0..NR_PACKETS_FOR_CORRELATION for each channel/FPGA, followed by the "-1"
  // initialization packet, since "sample num initialization is done off the
  // first packet received") -- then drives the pipeline synchronously and
  // waits for the GPU to finish.
  //
  //   sample_fn(channel, fpga, packet_index, time, receiver, polarization)
  //       -> std::complex<int8_t>
  //   scale_fn(channel, fpga, packet_index, receiver, polarization)
  //       -> int16_t
  //
  // `packet_index` ranges over [-1, NR_PACKETS_FOR_CORRELATION].
  template <typename SampleFn, typename ScaleFn>
  void run(SampleFn &&sample_fn, ScaleFn &&scale_fn, uint64_t start_sample = 1000) {
    const int nr_packets_for_correlation =
        static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION);

    for (size_t channel = 0; channel < Config::NR_CHANNELS; ++channel) {
      for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES; ++fpga) {
        for (int pkt = 0; pkt <= nr_packets_for_correlation; ++pkt) {
          feed_packet(channel, fpga, pkt, start_sample, sample_fn, scale_fn);
        }
        feed_packet(channel, fpga, -1, start_sample, sample_fn, scale_fn);
      }
    }

    state_.process_all_available_packets();
    state_.handle_buffer_completion(true);
    cudaDeviceSynchronize();
  }

  // Convenience overload for synthetic data that doesn't depend on
  // channel/FPGA/packet index (the common case -- e.g. a constant value or a
  // tone defined purely in terms of (time, receiver, polarization)).
  template <typename SampleFn, typename ScaleFn>
  void run_uniform(SampleFn &&sample_fn, ScaleFn &&scale_fn,
                   uint64_t start_sample = 1000) {
    run(
        [&](size_t, size_t, int, int t, int r, int p) { return sample_fn(t, r, p); },
        [&](size_t, size_t, int, int r, int p) { return scale_fn(r, p); },
        start_sample);
  }

  ProcessorState<Config, NR_BUFFERS> &processor_state() { return state_; }

private:
  template <typename SampleFn, typename ScaleFn>
  void feed_packet(size_t channel, size_t fpga, int pkt, uint64_t start_sample,
                   SampleFn &sample_fn, ScaleFn &scale_fn) {
    // Matches FillOneBufferTest's `start_sample + pkt *
    // NR_TIME_STEPS_PER_PACKET` exactly, including its int/size_t/uint64_t mix
    // -- for pkt == -1 this relies on unsigned wraparound to land on
    // `start_sample - NR_TIME_STEPS_PER_PACKET`, which is the intended "one
    // packet before the start" sample count.
    const uint64_t sample_count =
        start_sample + pkt * Config::NR_TIME_STEPS_PER_PACKET;

    feed_lambda_packet<Config>(
        state_, sample_count, static_cast<uint32_t>(fpga),
        static_cast<uint16_t>(channel),
        [&](int t, int r, int p) { return sample_fn(channel, fpga, pkt, t, r, p); },
        [&](int r, int p) { return scale_fn(channel, fpga, pkt, r, p); });
  }

  GPUPipeline &pipeline_;
  std::shared_ptr<Output> output_;
  ProcessorState<Config, NR_BUFFERS> state_;
  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays_{};
};

} // namespace test_support
