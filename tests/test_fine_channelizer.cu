// Exercises pre-correlation fine channelization (FineChannelizer<T>, ASTRON's gpu-filter PFB)
// end to end through LambdaGPUPipeline, via the same real
// PacketInput -> ProcessorState -> Pipeline -> Output seam PipelineHarnessSelfTest proves out for
// the NR_FINE_CHANNELS == 1 path (see support/test_harness_selftest.cu).
//
// This is deliberately a property/invariant-based test, not an exact-value golden: verifying the
// PFB's numerical correctness bin-for-bin against a reference implementation is tracked as
// follow-up work (see the plan's Section 9 note on a numpy/scipy reference PFB). What this test
// proves is that the whole channelized data path -- reorder_to_filter_input -> gpu-filter's
// Filter::launchAsync (x NR_FPGA_CHANNELS) -> channelizer_output_to_corr_input -> TCC ->
// eigendecomposition -> corr_input_to_col_maj_cons -> beamforming -- runs without CUDA errors and
// produces physically sane (finite, Hermitian-autocorrelation) output at the widened
// (coarse * fine) channel count.
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using Config = test_support::SmallFineChannelizedConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 32, "test assumes 32 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 2, "test assumes a 2-per-side edge trim");
static_assert(Config::NR_CHANNELS == 28,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

class FineChannelizerPipelineTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// Constant (2, -2) sample / scale-1 input across every FPGA channel, packet slot, time step,
// receiver and polarization -- same stimulus test_harness_selftest.cu's Ex1-equivalent run uses.
// Not a tone (no expected fine-channel bin to check against), just enough signal to exercise the
// full FIR+FFT+correction chain without degenerate all-zero input.
std::complex<int8_t> constant_sample(size_t /*channel*/, size_t /*fpga*/,
                                     int /*packet*/, int /*time*/,
                                     int /*receiver*/, int /*polarization*/) {
  return std::complex<int8_t>(2, -2);
}

int16_t constant_scale(size_t /*channel*/, size_t /*fpga*/, int /*packet*/,
                       int /*receiver*/, int /*polarization*/) {
  return 1;
}

struct HarnessRun {
  std::shared_ptr<SingleHostMemoryOutput<Config>> output;
  BeamWeightsT<Config> weights;
  std::unique_ptr<LambdaGPUPipeline<Config>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<Config>> driver;
};

template <typename SampleFn>
HarnessRun run_channelized_pipeline(SampleFn &&sample_fn) {
  HarnessRun run;
  run.output = std::make_shared<SingleHostMemoryOutput<Config>>();
  run.weights = test_support::make_unity_beam_weights<Config>();
  run.pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION, &run.weights);
  run.driver = std::make_unique<test_support::SyntheticPipelineRun<Config>>(
      *run.pipeline, run.output);

  run.driver->run(std::forward<SampleFn>(sample_fn), constant_scale);
  run.pipeline->dump_visibilities();
  cudaDeviceSynchronize();
  return run;
}

} // namespace

TEST_F(FineChannelizerPipelineTest, SatisfiesPhysicalInvariants) {
  HarnessRun run = run_channelized_pipeline(constant_sample);

  test_support::assert_all_finite(*run.output->beam_data, "beam_data");
  test_support::assert_all_finite(*run.output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<Config>(*run.output->visibilities);
}

namespace {

// Single complex tone at exactly one cycle-per-fine-channel-bin frequency (k0=8 of 32), identical
// across every receiver/polarization. Only assert_all_finite/assert_autocorrelation_invariants
// (above) can't distinguish a correctly-channelized signal from a badly time-shifted/corrupted
// one -- both can produce finite, Hermitian-PSD output. A real tone can: gpu-filter's PFB is
// frequency-selective, so a correctly-fed tone must land almost all its power in one fine-channel
// bin, while a corrupted feed (e.g. the reorder_to_filter_input history-placement bug this test
// was added to catch -- see CLAUDE.md's "gpu-filter's FIR kernel looks forward, not backward" note
// -- which fed gpu-filter mostly zero-padded/partial-window data) does not preserve that
// concentration.
constexpr int kToneBin = 8;
constexpr float kToneAmplitude = 100.0f; // headroom under int8_t's +/-127 range
constexpr float kTwoPi = 6.283185307179586f;

std::complex<int8_t> tone_sample(size_t /*channel*/, size_t /*fpga*/,
                                 int /*packet*/, int time, int /*receiver*/,
                                 int /*polarization*/) {
  // Periodicity is over the untrimmed NR_FINE_CHANNELS -- gpu-filter still channelizes the full
  // band; only channelizer_output_to_corr_input drops the edge bins downstream.
  const float phase = kTwoPi * static_cast<float>(kToneBin) *
                      static_cast<float>(time) /
                      static_cast<float>(Config::NR_FINE_CHANNELS);
  return std::complex<int8_t>(
      static_cast<int8_t>(std::lround(kToneAmplitude * std::cos(phase))),
      static_cast<int8_t>(std::lround(kToneAmplitude * std::sin(phase))));
}

} // namespace

TEST_F(FineChannelizerPipelineTest, ToneConcentratesInOneFineChannel) {
  HarnessRun run = run_channelized_pipeline(tone_sample);

  const size_t b0 = test_support::baseline_index(0, 0);
  std::array<float, Config::NR_CHANNELS> power{};
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c) {
    power[c] = (*run.output->visibilities)[c][b0][0][0][0];
  }

  const size_t peak_channel =
      std::max_element(power.begin(), power.end()) - power.begin();
  const float peak_power = power[peak_channel];
  ASSERT_GT(peak_power, 0.0f) << "tone should produce nonzero autocorrelation power somewhere";

  double total_power = 0.0;
  for (float p : power)
    total_power += p;

  // A correctly-channelized single tone should put the large majority of the total power into
  // one bin (some leakage into immediate neighbors is expected from the Kaiser-windowed FIR).
  EXPECT_GT(peak_power / total_power, 0.5f)
      << "peak channel " << peak_channel << " holds only "
      << (peak_power / total_power) * 100.0 << "% of total power across "
      << Config::NR_CHANNELS << " channels -- tone is not concentrated, "
      << "suggesting gpu-filter is being fed corrupted/misaligned input";

  // Channels far from the peak (and its immediate neighbors) should carry only a small fraction
  // of the peak's power -- true frequency selectivity, not noise spread roughly evenly everywhere.
  size_t far_channels_checked = 0;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c) {
    const size_t dist = std::min(
        (c > peak_channel) ? c - peak_channel : peak_channel - c,
        Config::NR_CHANNELS - ((c > peak_channel) ? c - peak_channel : peak_channel - c));
    if (dist <= 2)
      continue;
    far_channels_checked++;
    EXPECT_LT(power[c], 0.1f * peak_power)
        << "channel " << c << " (distance " << dist << " from peak channel "
        << peak_channel << ") holds " << power[c]
        << " power vs. peak " << peak_power << " -- expected it to be suppressed";
  }
  ASSERT_GT(far_channels_checked, 0u);
}

// Direct, low-level test of reorder_to_filter_input's buffer layout -- deliberately bypasses the
// full pipeline (LambdaGPUPipeline/TCC/cuSOLVER/beamforming all sit between this kernel's output
// and anything host-observable, and empirically do enough mixing/windowing that a single-tone
// spectral-concentration check through the full pipeline does NOT reliably distinguish correct
// output from the history-placement bug this test targets -- verified by temporarily reintroducing
// that bug and confirming ToneConcentratesInOneFineChannel above still passed). This test instead
// asserts the exact, deterministic property that bug violated: real samples must land at buffer
// offset 0 (not shifted by HISTORY), and the *trailing* HISTORY region -- not the leading one --
// must be zero-padded (see CLAUDE.md's "gpu-filter's FIR kernel looks forward, not backward" note).
TEST(ReorderToFilterInputTest, RealSamplesAtFrontZeroPadAtBack) {
  constexpr size_t NR_CHANNELS = 1, NR_POLARIZATIONS = 2, NR_RECEIVERS = 1,
                   NR_RECEIVERS_PER_PACKET = 1, NR_TIME_STEPS_PER_PACKET = 8,
                   NR_PACKETS_FOR_CORRELATION = 1, NR_TAPS = 4, NR_FINE_CHANNELS = 2;
  constexpr size_t NR_TIME_STEPS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;
  constexpr size_t HISTORY = (NR_TAPS - 1) * NR_FINE_CHANNELS;
  constexpr size_t DST_TIME_LEN = NR_TIME_STEPS_FOR_CORRELATION + HISTORY;

  // samples_reordered layout: [FPGA][PACKET][TIME_IN_PACKET][CHANNEL][RECEIVER_IN_PACKET][POL]
  // [COMPLEX] -- with all dims 1 except TIME/POL here, this is just [TIME][POL][COMPLEX].
  std::vector<__half> h_input(NR_TIME_STEPS_FOR_CORRELATION * NR_POLARIZATIONS * 2);
  for (size_t t = 0; t < NR_TIME_STEPS_FOR_CORRELATION; ++t) {
    for (size_t p = 0; p < NR_POLARIZATIONS; ++p) {
      const size_t base = (t * NR_POLARIZATIONS + p) * 2;
      // Distinct, recognizable real/imag value per (t, p) so any misplacement is obvious.
      h_input[base] = __float2half(static_cast<float>(t * 10 + p + 1));
      h_input[base + 1] = __float2half(-static_cast<float>(t * 10 + p + 1));
    }
  }

  __half *d_input = nullptr;
  float2 *d_output = nullptr;
  ASSERT_EQ(cudaMalloc(&d_input, h_input.size() * sizeof(__half)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output,
                       NR_CHANNELS * NR_RECEIVERS * NR_POLARIZATIONS * DST_TIME_LEN *
                           sizeof(float2)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  reorder_to_filter_input<NR_CHANNELS, NR_POLARIZATIONS, NR_RECEIVERS,
                          NR_RECEIVERS_PER_PACKET, NR_TIME_STEPS_PER_PACKET,
                          NR_PACKETS_FOR_CORRELATION, NR_TAPS, NR_FINE_CHANNELS>(
      d_input, d_output, /*stream=*/0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float2> h_output(NR_CHANNELS * NR_RECEIVERS * NR_POLARIZATIONS * DST_TIME_LEN);
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float2),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_input);
  cudaFree(d_output);

  for (size_t p = 0; p < NR_POLARIZATIONS; ++p) {
    // Real samples must occupy [0, NR_TIME_STEPS_FOR_CORRELATION) -- the front of the buffer.
    for (size_t t = 0; t < NR_TIME_STEPS_FOR_CORRELATION; ++t) {
      const float2 v = h_output[p * DST_TIME_LEN + t];
      const float expected = static_cast<float>(t * 10 + p + 1);
      EXPECT_FLOAT_EQ(v.x, expected)
          << "pol " << p << " t_dst " << t << ": expected real sample " << expected
          << " at the front of the buffer, got " << v.x
          << " -- real data is not starting at offset 0";
      EXPECT_FLOAT_EQ(v.y, -expected) << "pol " << p << " t_dst " << t;
    }
    // The trailing HISTORY region [NR_TIME_STEPS_FOR_CORRELATION, DST_TIME_LEN) must be
    // zero-padded look-ahead, not more real data and not leading history moved to the back.
    for (size_t t = NR_TIME_STEPS_FOR_CORRELATION; t < DST_TIME_LEN; ++t) {
      const float2 v = h_output[p * DST_TIME_LEN + t];
      EXPECT_FLOAT_EQ(v.x, 0.0f)
          << "pol " << p << " t_dst " << t
          << ": trailing look-ahead region must be zero, got " << v.x;
      EXPECT_FLOAT_EQ(v.y, 0.0f);
    }
  }
}

// Direct, low-level test of channelizer_output_to_corr_input's edge-channel trim -- same rationale
// as ReorderToFilterInputTest above: the full-pipeline tone-concentration test can't reliably
// distinguish "trims the correct edge indices" from "trims the wrong ones" (a wrong-but-plausible
// trim just shifts which few channels are silently dropped, without breaking the general
// finite/concentrated-power invariants). This asserts exactly which (coarse, fine) pairs survive
// into corr_input and at which C offset, and that dropped edge channels are truly absent (not just
// unused) from the output.
TEST(ChannelizerOutputToCorrInputTest, DropsEdgeChannelsAndPacksSurvivors) {
  constexpr size_t NR_FPGA_CHANNELS = 2, NR_FINE_CHANNELS = 6, NR_EDGE_TRIM = 1,
                   NR_POLARIZATIONS = 1, NR_RECEIVERS = 2, NR_PADDED_RECEIVERS = 2,
                   NR_BLOCKS_FOR_CORRELATION = 1, NR_TIMES_PER_BLOCK = 1;
  constexpr size_t NR_EFFECTIVE_FINE_CHANNELS = NR_FINE_CHANNELS - 2 * NR_EDGE_TRIM; // 4

  // FineChannelizer::FilterOutputType layout: [coarse][fine][block][receiver][pol][time_in_block].
  // Fill every (coarse, fine) slot with a distinct, recognizable value so misplacement or a
  // surviving "dropped" value is obvious.
  std::vector<__half2> h_input(NR_FPGA_CHANNELS * NR_FINE_CHANNELS *
                               NR_BLOCKS_FOR_CORRELATION * NR_RECEIVERS *
                               NR_POLARIZATIONS * NR_TIMES_PER_BLOCK);
  for (size_t coarse = 0; coarse < NR_FPGA_CHANNELS; ++coarse) {
    for (size_t fine = 0; fine < NR_FINE_CHANNELS; ++fine) {
      const float value = static_cast<float>(coarse * 100 + fine + 1);
      for (size_t r = 0; r < NR_RECEIVERS; ++r) {
        const size_t idx = ((coarse * NR_FINE_CHANNELS + fine) * NR_RECEIVERS + r);
        h_input[idx] = __floats2half2_rn(value, -value);
      }
    }
  }

  __half2 *d_input = nullptr;
  __half *d_output = nullptr;
  constexpr size_t NR_C = NR_FPGA_CHANNELS * NR_EFFECTIVE_FINE_CHANNELS;
  constexpr size_t output_elems =
      NR_C * NR_BLOCKS_FOR_CORRELATION * NR_PADDED_RECEIVERS * NR_POLARIZATIONS *
      NR_TIMES_PER_BLOCK * 2 /* COMPLEX */;
  ASSERT_EQ(cudaMalloc(&d_input, h_input.size() * sizeof(__half2)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, output_elems * sizeof(__half)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, output_elems * sizeof(__half)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half2),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  channelizer_output_to_corr_input<NR_FPGA_CHANNELS, NR_FINE_CHANNELS, NR_EDGE_TRIM,
                                   NR_POLARIZATIONS, NR_RECEIVERS, NR_PADDED_RECEIVERS,
                                   NR_BLOCKS_FOR_CORRELATION, NR_TIMES_PER_BLOCK>(
      d_input, d_output, /*stream=*/0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> h_output(output_elems);
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, output_elems * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_input);
  cudaFree(d_output);

  for (size_t coarse = 0; coarse < NR_FPGA_CHANNELS; ++coarse) {
    for (size_t effective_fine = 0; effective_fine < NR_EFFECTIVE_FINE_CHANNELS;
        ++effective_fine) {
      const size_t fine = effective_fine + NR_EDGE_TRIM;
      const float expected = static_cast<float>(coarse * 100 + fine + 1);
      const size_t c = coarse * NR_EFFECTIVE_FINE_CHANNELS + effective_fine;
      for (size_t r = 0; r < NR_RECEIVERS; ++r) {
        const size_t base = (c * NR_PADDED_RECEIVERS + r) * 2;
        EXPECT_FLOAT_EQ(__half2float(h_output[base]), expected)
            << "C=" << c << " (coarse=" << coarse << ", fine=" << fine
            << ") receiver " << r << ": wrong survivor value -- expected the sample "
            << "from untrimmed fine index " << fine;
        EXPECT_FLOAT_EQ(__half2float(h_output[base + 1]), -expected);
      }
    }
    // Every dropped edge fine index (< NR_EDGE_TRIM or >= NR_FINE_CHANNELS - NR_EDGE_TRIM) must be
    // truly absent -- confirm its recognizable value doesn't appear anywhere in the output at all
    // (not just "unused at its old position"), i.e. no C slot was mistakenly filled from it.
    for (size_t dropped_fine : {size_t{0}, NR_FINE_CHANNELS - 1}) {
      const float dropped_value = static_cast<float>(coarse * 100 + dropped_fine + 1);
      for (__half v : h_output) {
        EXPECT_NE(__half2float(v), dropped_value)
            << "dropped fine channel " << dropped_fine << " (coarse=" << coarse
            << ")'s value leaked into corr_input";
      }
    }
  }
}

// Direct, low-level test of channelizer_output_to_col_maj_cons's edge-channel trim and output
// layout -- same rationale as ChannelizerOutputToCorrInputTest above, for the sibling kernel used
// by pipelines that beamform directly from channelizer output (LambdaBeamformedSpectraPipeline,
// LambdaPulsarFoldPipeline's non-RFI-mitigate path) with no TCC correlation step in between.
TEST(ChannelizerOutputToColMajConsTest, DropsEdgeChannelsAndPacksSurvivors) {
  constexpr size_t NR_FPGA_CHANNELS = 2, NR_FINE_CHANNELS = 6, NR_EDGE_TRIM = 1,
                   NR_POLARIZATIONS = 1, NR_RECEIVERS = 2, NR_RECEIVERS_PER_PACKET = 2,
                   NR_SAMPLES_PER_FINE_CHANNEL = 1, NR_TIMES_PER_OUTPUT_BLOCK = 1;
  constexpr size_t NR_EFFECTIVE_FINE_CHANNELS = NR_FINE_CHANNELS - 2 * NR_EDGE_TRIM; // 4
  constexpr size_t NR_C = NR_FPGA_CHANNELS * NR_EFFECTIVE_FINE_CHANNELS;

  // FineChannelizer::FilterOutputType layout: [coarse][fine][block][receiver][pol][time_in_block]
  // -- block/pol/time_in_block all size 1 here, so the flat index collapses to (coarse,fine,r).
  std::vector<__half2> h_input(NR_FPGA_CHANNELS * NR_FINE_CHANNELS * NR_RECEIVERS);
  for (size_t coarse = 0; coarse < NR_FPGA_CHANNELS; ++coarse) {
    for (size_t fine = 0; fine < NR_FINE_CHANNELS; ++fine) {
      const float value = static_cast<float>(coarse * 100 + fine + 1);
      for (size_t r = 0; r < NR_RECEIVERS; ++r) {
        const size_t idx = (coarse * NR_FINE_CHANNELS + fine) * NR_RECEIVERS + r;
        h_input[idx] = __floats2half2_rn(value, -value);
      }
    }
  }

  __half2 *d_input = nullptr;
  __half *d_output = nullptr;
  constexpr size_t output_elems = NR_C * NR_POLARIZATIONS * 2 /* COMPLEX */ *
                                  NR_SAMPLES_PER_FINE_CHANNEL * NR_RECEIVERS;
  ASSERT_EQ(cudaMalloc(&d_input, h_input.size() * sizeof(__half2)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, output_elems * sizeof(__half)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half2),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  channelizer_output_to_col_maj_cons<NR_FPGA_CHANNELS, NR_FINE_CHANNELS, NR_EDGE_TRIM,
                                     NR_POLARIZATIONS, NR_RECEIVERS,
                                     NR_RECEIVERS_PER_PACKET, NR_SAMPLES_PER_FINE_CHANNEL,
                                     NR_TIMES_PER_OUTPUT_BLOCK>(d_input, d_output,
                                                                /*stream=*/0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> h_output(output_elems);
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, output_elems * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_input);
  cudaFree(d_output);

  // Output layout [C][POL][COMPLEX][S][FPGA][RECEIVER_IN_PACKET], with POL=S=FPGA=1 here:
  // idx = c*4 + z*2 + receiver_in_packet.
  for (size_t coarse = 0; coarse < NR_FPGA_CHANNELS; ++coarse) {
    for (size_t effective_fine = 0; effective_fine < NR_EFFECTIVE_FINE_CHANNELS;
        ++effective_fine) {
      const size_t fine = effective_fine + NR_EDGE_TRIM;
      const float expected = static_cast<float>(coarse * 100 + fine + 1);
      const size_t c = coarse * NR_EFFECTIVE_FINE_CHANNELS + effective_fine;
      for (size_t receiver_in_packet = 0; receiver_in_packet < NR_RECEIVERS_PER_PACKET;
          ++receiver_in_packet) {
        const size_t real_idx = c * 4 + 0 * 2 + receiver_in_packet;
        const size_t imag_idx = c * 4 + 1 * 2 + receiver_in_packet;
        EXPECT_FLOAT_EQ(__half2float(h_output[real_idx]), expected)
            << "C=" << c << " (coarse=" << coarse << ", fine=" << fine
            << ") receiver " << receiver_in_packet
            << ": wrong survivor value -- expected the sample from untrimmed fine index "
            << fine;
        EXPECT_FLOAT_EQ(__half2float(h_output[imag_idx]), -expected);
      }
    }
    for (size_t dropped_fine : {size_t{0}, NR_FINE_CHANNELS - 1}) {
      const float dropped_value = static_cast<float>(coarse * 100 + dropped_fine + 1);
      for (__half v : h_output) {
        EXPECT_NE(__half2float(v), dropped_value)
            << "dropped fine channel " << dropped_fine << " (coarse=" << coarse
            << ")'s value leaked into col_maj_cons output";
      }
    }
  }
}

// set_fine_delays() is retired for the channelized path in favour of gpu-filter's native delay
// compensation (not yet wired through, see FineChannelizer's class comment) -- it must fail
// loudly rather than silently no-op, so a caller can't believe fine delays are active when
// they're actually being ignored.
TEST_F(FineChannelizerPipelineTest, SetFineDelaysThrows) {
  BeamWeightsT<Config> weights = test_support::make_unity_beam_weights<Config>();
  auto pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION, &weights);

  std::array<float, Config::NR_RECEIVERS> delays_ns{};
  EXPECT_THROW(
      pipeline->set_fine_delays(delays_ns.data(), /*base_freq_hz=*/1e9,
                                /*channel_bw_hz=*/1e6, /*min_freq_ch=*/0),
      std::runtime_error);
}
