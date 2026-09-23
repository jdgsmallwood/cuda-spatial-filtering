// Exercises pre-correlation fine channelization wired into LambdaStarweavePipeline
// (the correlation-only, visibilities-only pipeline). Unlike LambdaCorrBeamOnlyGPUPipeline,
// LambdaStarweavePipeline always runs the same delay/reorder -> (optional channelize) ->
// correlate sequence -- NR_FINE_CHANNELS == 1 and > 1 are the same code shape at a different
// width (see enqueue_corr's if constexpr branch), so this test just confirms both widths
// satisfy the same physical invariants.
#include "spatial/output.hpp"
#include "spatial/deripple.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace {

using Config = test_support::SmallFineChannelizedCorrBeamConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 8, "test assumes 8 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 1, "test assumes a 1-per-side edge trim");
static_assert(Config::NR_CHANNELS == 6,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

class FineChannelizedStarweaveTest : public ::testing::Test {
protected:
  void TearDown() override {
    coarse_pfb_deripple_config_path().clear();
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

std::complex<int8_t> constant_sample(size_t, size_t, int, int, int, int) {
  return {2, -2};
}
int16_t constant_scale(size_t, size_t, int, int, int) { return 1; }

template <typename ConfigT> void run_and_check_physical_invariants() {
  auto output = std::make_shared<SingleHostMemoryOutput<ConfigT>>();
  auto pipeline =
      test_support::pipeline_factories::make_starweave_pipeline<ConfigT>(
          /*num_buffers=*/ConfigT::NR_PACKETS_FOR_CORRELATION);
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, output);

  driver.run(constant_sample, constant_scale);
  // SmallFineChannelizedCorrBeamConfig sets NR_CORRELATED_BLOCKS_TO_ACCUMULATE=1, so
  // execute_pipeline's single run already auto-dumps internally -- calling
  // dump_visibilities() again here would clobber SingleHostMemoryOutput's single fixed
  // landing slot with a second, empty (zeroed-accumulator) dump. Only configs whose
  // threshold isn't reached by one run (e.g. SmallSingleFPGAConfig, 10000) need the
  // explicit flush.
  if constexpr (ConfigT::NR_CORRELATED_BLOCKS_TO_ACCUMULATE > 1) {
    pipeline->dump_visibilities();
  }
  cudaDeviceSynchronize();

  test_support::assert_all_finite(*output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<ConfigT>(*output->visibilities);

  using Element = std::remove_all_extents_t<
      typename SingleHostMemoryOutput<ConfigT>::Visibilities>;
  constexpr size_t n =
      sizeof(typename SingleHostMemoryOutput<ConfigT>::Visibilities) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(output->visibilities);
  bool any_nonzero = false;
  for (size_t i = 0; i < n; ++i)
    if (flat[i] != 0.0f)
      any_nonzero = true;
  EXPECT_TRUE(any_nonzero) << "constant input should produce nonzero visibilities somewhere";
  // Pipeline reclaimer/callbacks must finish while SyntheticPipelineRun's
  // ProcessorState is still alive.
  pipeline.reset();
}

template <typename ConfigT>
struct SequenceTrackingOutput : SingleHostMemoryOutput<ConfigT> {
  uint64_t observed_start = 0;
  uint64_t observed_end = 0;
  size_t register_visibilities_block(const size_t start_seq_num,
                                     const size_t end_seq_num,
                                     const int missing,
                                     const int total) override {
    observed_start = start_seq_num;
    observed_end = end_seq_num;
    return SingleHostMemoryOutput<ConfigT>::register_visibilities_block(
        start_seq_num, end_seq_num, missing, total);
  }
};

} // namespace

TEST_F(FineChannelizedStarweaveTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

TEST_F(FineChannelizedStarweaveTest, RuntimeDerippleProducesFiniteVisibilities) {
  const auto path = std::filesystem::temp_directory_path() /
      ("starweave-deripple-" + std::to_string(getpid()) + ".json");
  const std::array<float, Config::NR_EFFECTIVE_FINE_CHANNELS> gains{
      1.0f, 1.02f, 1.04f, 1.1f, 1.06f, 0.98f};
  { std::ofstream out(path);
    out << nlohmann::json{{"schema_version", 1},
                          {"kind", "coarse_pfb_amplitude_deripple"},
                          {"fine_channels", Config::NR_FINE_CHANNELS},
                          {"edge_trim", Config::NR_FINE_CHANNEL_EDGE_TRIM},
                          {"normalization", "coarse_bin_centre"},
                          {"response_id", "synthetic-starweave"},
                          {"voltage_gains", gains}}; }
  coarse_pfb_deripple_config_path() = path.string();
  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();
  auto pipeline = test_support::pipeline_factories::make_starweave_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION);
  test_support::SyntheticPipelineRun<Config> driver(*pipeline, output);
  driver.run(constant_sample, constant_scale);
  cudaDeviceSynchronize();
  test_support::assert_all_finite(*output->visibilities, "de-rippled visibilities");
  test_support::assert_autocorrelation_invariants<Config>(*output->visibilities);
  size_t nonzero = 0;
  for (size_t chan = 0; chan < Config::NR_CHANNELS; ++chan) {
    if (std::abs((*output->visibilities)[chan][0][0][0][0]) > 1e-3f)
      ++nonzero;
  }
  EXPECT_GT(nonzero, 0u);
  pipeline.reset();
  std::filesystem::remove(path);
}

// Regression guard: the disabled (non-channelized) path must keep working unmodified
// alongside the channelizer wiring.
TEST_F(FineChannelizedStarweaveTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallSingleFPGAConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}

// Independent CPU oracle for the complete wire-packet -> ProcessorState ->
// Starweave -> packed-visibility path.  Distinct complex values and scales on
// both FPGA sources make an FPGA-local scale lookup, channel crossover, hand
// swap, or baseline/conjugation reversal visible in the output.
TEST_F(FineChannelizedStarweaveTest, FourFpgaComplexVisibilitiesMatchCpuOracle) {
  using MC = LambdaConfig<2, 4, 8, 8, 2, 2, 1, 1, 32, 32, 1>;
  auto output = std::make_shared<SequenceTrackingOutput<MC>>();
  auto pipeline = test_support::pipeline_factories::make_starweave_pipeline<MC>(1);
  test_support::SyntheticPipelineRun<MC> driver(
      *pipeline, output, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});

  auto sample = [](size_t ch, size_t fpga, int, int, int recv, int pol) {
    const int flat_recv = static_cast<int>(fpga) * 2 + recv;
    return std::complex<int8_t>{
        static_cast<int8_t>(1 + static_cast<int>(ch) + flat_recv + pol),
        static_cast<int8_t>(flat_recv - 2 * pol - static_cast<int>(ch))};
  };
  auto scale = [](size_t ch, size_t fpga, int, int recv, int pol) {
    return static_cast<int16_t>(1 + static_cast<int>(ch) +
                                2 * static_cast<int>(fpga) + recv + pol);
  };
  constexpr uint64_t kStartSample = (uint64_t{1} << 33) + 1000;
  driver.run(sample, scale, kStartSample);

  EXPECT_EQ(output->observed_start, kStartSample);
  EXPECT_EQ(output->observed_end, kStartSample);

  const auto &vis = *output->visibilities;
  for (size_t ch = 0; ch < MC::NR_CHANNELS; ++ch) {
    for (size_t j = 0; j < MC::NR_RECEIVERS; ++j) {
      for (size_t i = 0; i <= j; ++i) {
        const size_t baseline = test_support::baseline_index(i, j);
        for (size_t p = 0; p < MC::NR_POLARIZATIONS; ++p) {
          for (size_t q = 0; q < MC::NR_POLARIZATIONS; ++q) {
            const auto ai = sample(ch, i / 2, 0, 0, i % 2, p);
            const auto aj = sample(ch, j / 2, 0, 0, j % 2, q);
            const float si = scale(ch, i / 2, 0, i % 2, p);
            const float sj = scale(ch, j / 2, 0, j % 2, q);
            const std::complex<float> vi(ai.real() * si, ai.imag() * si);
            const std::complex<float> vj(aj.real() * sj, aj.imag() * sj);
            // HDF5 baseline (i,j) is E_i * conj(E_j). The commissioning
            // exporter explicitly conjugates this repository convention
            // when writing CASA Measurement Set visibilities.
            const auto expected = static_cast<float>(MC::NR_TIME_STEPS_PER_PACKET) *
                                  vi * std::conj(vj);
            EXPECT_NEAR(vis[ch][baseline][p][q][0], expected.real(), 0.1f)
                << "channel=" << ch << " baseline=(" << i << "," << j
                << ") hands=(" << p << "," << q << ")";
            EXPECT_NEAR(vis[ch][baseline][p][q][1], expected.imag(), 0.1f)
                << "channel=" << ch << " baseline=(" << i << "," << j
                << ") hands=(" << p << "," << q << ")";
          }
        }
      }
    }
  }
  pipeline.reset();
}

// The production path channelizes each coarse FPGA channel before TCC. Test
// both axes at once: independent exact tones select different fine bins, while
// four FPGA sources carry known complex phase and packet-scale relationships.
TEST_F(FineChannelizedStarweaveTest, FourFpgaFineBinsAndCrossPhases) {
  using FC = LambdaConfig<2, 4, 128, 8, 2, 2, 1, 1, 32, 32, 1,
                          false, 128, 8, 1>;
  static_assert(FC::NR_CHANNELS == 12);
  auto output = std::make_shared<SingleHostMemoryOutput<FC>>();
  auto pipeline = test_support::pipeline_factories::make_starweave_pipeline<FC>(1);
  test_support::SyntheticPipelineRun<FC> driver(
      *pipeline, output, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});

  auto phasor = [](int flat_recv, int pol) {
    switch ((flat_recv + pol) % 4) {
    case 0: return std::complex<int>{1, 0};
    case 1: return std::complex<int>{0, 1};
    case 2: return std::complex<int>{-1, 0};
    default: return std::complex<int>{0, -1};
    }
  };
  auto sample = [&](size_t ch, size_t fpga, int, int time, int recv, int pol) {
    // Coarse 0 is DC; coarse 1 is +2 cycles per eight raw samples.
    const int phase = ch == 0 ? 0 : time % 4;
    const std::complex<int> tone =
        phase == 0 ? std::complex<int>{8, 0} :
        phase == 1 ? std::complex<int>{0, 8} :
        phase == 2 ? std::complex<int>{-8, 0} :
                     std::complex<int>{0, -8};
    const auto value = tone * phasor(static_cast<int>(fpga) * 2 + recv, pol);
    return std::complex<int8_t>{static_cast<int8_t>(value.real()),
                                static_cast<int8_t>(value.imag())};
  };
  auto scale = [](size_t, size_t fpga, int, int, int pol) {
    return static_cast<int16_t>(1 + static_cast<int>(fpga) + pol);
  };
  driver.run(sample, scale);

  const auto &vis = *output->visibilities;
  for (size_t coarse = 0; coarse < FC::NR_FPGA_CHANNELS; ++coarse) {
    const size_t first = coarse * FC::NR_EFFECTIVE_FINE_CHANNELS;
    size_t peak = first;
    for (size_t fine = 1; fine < FC::NR_EFFECTIVE_FINE_CHANNELS; ++fine)
      if (vis[first + fine][0][0][0][0] > vis[peak][0][0][0][0])
        peak = first + fine;
    // fft.shift=true maps DC to untrimmed bin 4 and +2 to bin 6;
    // trimming one bin per edge yields local retained bins 3 and 5.
    EXPECT_EQ(peak - first, coarse == 0 ? 3u : 5u);
    const float reference_power = vis[peak][0][0][0][0];
    ASSERT_GT(reference_power, 0.0f);
    for (size_t receiver = 0; receiver < FC::NR_RECEIVERS; ++receiver) {
      const size_t baseline = test_support::baseline_index(0, receiver);
      for (size_t pol = 0; pol < FC::NR_POLARIZATIONS; ++pol) {
        const auto a = phasor(static_cast<int>(receiver), static_cast<int>(pol));
        const int receiver_scale =
            1 + static_cast<int>(receiver / 2) + static_cast<int>(pol);
        const std::complex<float> vj(
            static_cast<float>(a.real() * receiver_scale),
            static_cast<float>(a.imag() * receiver_scale));
        const auto expected = reference_power * std::conj(vj);
        EXPECT_NEAR(vis[peak][baseline][0][pol][0], expected.real(),
                    0.02f * reference_power)
            << "coarse=" << coarse << " receiver=" << receiver
            << " pol=" << pol;
        EXPECT_NEAR(vis[peak][baseline][0][pol][1], expected.imag(),
                    0.02f * reference_power)
            << "coarse=" << coarse << " receiver=" << receiver
            << " pol=" << pol;
      }
    }
  }
  pipeline.reset();
}
