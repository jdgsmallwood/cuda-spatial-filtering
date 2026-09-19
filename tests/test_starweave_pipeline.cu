// Tests for LambdaStarweavePipeline.
//
// Correlation-only, visibilities-only pipeline: no beamforming, no
// eigendecomposition/RFI mitigation, no spectral output. Tests drive it through
// the real ProcessorState -> GPUPipeline -> Output seam (SyntheticPipelineRun),
// not via DummyFinalPacketData -- same approach as test_corr_beam_pipeline.cu,
// with every beam-related assertion/test case dropped (there's no beam output
// to check here).
//
// Because dump_visibilities() fires automatically only after
// NR_CORRELATED_BLOCKS_TO_ACCUMULATE runs (10 000 for SmallSingleFPGAConfig),
// single-run tests call pipeline->dump_visibilities() manually then sync.
//
// ---- Exact-value derivation (constant (2,-2) input, scale=1)
//
//   fp16 sample x = (2.0, -2.0) for all (channel, time, receiver, pol)
//
//   Visibility:
//     V[i,j][p][q] = Sum_t conj(x[i,p,t]) * x[j,q,t]
//                  = NR_TIME_STEPS * conj((2,-2))*(2,-2)
//                  = 8 * (2+2j)*(2-2j)
//                  = 8 * (4+4) = 64 + 0j   (same for all i,j,p,q)
//     NR_TIME_STEPS = NR_PACKETS_FOR_CORRELATION x NR_TIME_STEPS_PER_PACKET
//                   = 1 x 8 = 8

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>

namespace {

using Config = test_support::SmallSingleFPGAConfig;

constexpr size_t NR_SAMPLES =
    Config::NR_PACKETS_FOR_CORRELATION * Config::NR_TIME_STEPS_PER_PACKET;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class StarweavePipelineTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// ---------------------------------------------------------------------------
// Sample / scale generators
// ---------------------------------------------------------------------------
std::complex<int8_t> constant_sample(size_t, size_t, int, int, int, int) {
  return {2, -2};
}
int16_t constant_scale(size_t, size_t, int, int, int) { return 1; }

std::complex<int8_t> zero_sample(size_t, size_t, int, int, int, int) {
  return {0, 0};
}

// ---------------------------------------------------------------------------
// Run helper
// ---------------------------------------------------------------------------
struct StarweaveRun {
  std::shared_ptr<SingleHostMemoryOutput<Config>> output;
  std::unique_ptr<LambdaStarweavePipeline<Config>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<Config>> driver;
};

template <typename SampleFn, typename ScaleFn>
StarweaveRun do_run(SampleFn sample_fn, ScaleFn scale_fn) {
  StarweaveRun r;
  r.output = std::make_shared<SingleHostMemoryOutput<Config>>();
  r.pipeline = test_support::pipeline_factories::make_starweave_pipeline<Config>(
      /*num_buffers=*/Config::NR_PACKETS_FOR_CORRELATION);
  r.driver = std::make_unique<test_support::SyntheticPipelineRun<Config>>(
      *r.pipeline, r.output);

  r.driver->run(sample_fn, scale_fn);
  // Visibilities accumulate internally; flush them to output now.
  r.pipeline->dump_visibilities();
  cudaDeviceSynchronize();
  return r;
}

} // namespace

// ---------------------------------------------------------------------------
// Exact visibility values
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, VisibilityExactValues) {
  auto r = do_run(constant_sample, constant_scale);
  const auto &vis = *r.output->visibilities;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t bl = 0; bl < Config::NR_BASELINES_UNPADDED; ++bl)
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
        for (size_t q = 0; q < Config::NR_POLARIZATIONS; ++q) {
          EXPECT_EQ(vis[c][bl][p][q][0], 64.0f)
              << "real  ch=" << c << " bl=" << bl << " pol=(" << p << "," << q << ")";
          EXPECT_EQ(vis[c][bl][p][q][1], 0.0f)
              << "imag  ch=" << c << " bl=" << bl << " pol=(" << p << "," << q << ")";
        }
}

// ---------------------------------------------------------------------------
// Physical invariants (finite, autocorrelation PSD / Hermitian)
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, PhysicalInvariants) {
  auto r = do_run(constant_sample, constant_scale);
  test_support::assert_all_finite(*r.output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<Config>(*r.output->visibilities);
}

// ---------------------------------------------------------------------------
// Zero input -> zero output
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, ZeroInputProducesZeroOutput) {
  auto r = do_run(zero_sample, constant_scale);
  const auto &vis = *r.output->visibilities;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t bl = 0; bl < Config::NR_BASELINES_UNPADDED; ++bl)
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
        for (size_t q = 0; q < Config::NR_POLARIZATIONS; ++q) {
          EXPECT_EQ(vis[c][bl][p][q][0], 0.0f);
          EXPECT_EQ(vis[c][bl][p][q][1], 0.0f);
        }
}

// ---------------------------------------------------------------------------
// Two-channel config: channels carry different constant samples, so the
// correlator input path must use the real channel stride of the half sample
// buffer.
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, TwoChannelConfigProducesExpectedOutput) {
  using Cfg = test_support::SmallTwoChannelConfig;

  auto output = std::make_shared<SingleHostMemoryOutput<Cfg>>();
  auto pipeline =
      test_support::pipeline_factories::make_starweave_pipeline<Cfg>(
          Cfg::NR_PACKETS_FOR_CORRELATION);
  test_support::SyntheticPipelineRun<Cfg> driver(*pipeline, output);

  driver.run(
      [](size_t channel, size_t, int, int, int, int) -> std::complex<int8_t> {
        return channel == 0 ? std::complex<int8_t>{2, -2}
                            : std::complex<int8_t>{3, 1};
      },
      [](size_t, size_t, int, int, int) -> int16_t { return 1; });

  pipeline->dump_visibilities();
  cudaDeviceSynchronize();

  const auto &vis = *output->visibilities;
  const float expected_auto_power[Cfg::NR_CHANNELS] = {64.0f, 80.0f};

  for (size_t c = 0; c < Cfg::NR_CHANNELS; ++c)
    for (size_t rx = 0; rx < Cfg::NR_RECEIVERS; ++rx) {
      const size_t bl = test_support::baseline_index(rx, rx);
      for (size_t p = 0; p < Cfg::NR_POLARIZATIONS; ++p)
        EXPECT_EQ(vis[c][bl][p][p][0], expected_auto_power[c])
            << "ch=" << c << " autocorr rx=" << rx << " pol=" << p;
    }
}

// ---------------------------------------------------------------------------
// Non-unit ingest scales must survive through the pipeline. With real unit
// samples, autocorrelation power is N * scale^2.
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, NonUnitScalesAffectVisibilities) {
  auto r = do_run(
      [](size_t, size_t, int, int, int, int) -> std::complex<int8_t> {
        return {1, 0};
      },
      [](size_t, size_t, int, int receiver, int pol) -> int16_t {
        return static_cast<int16_t>((receiver + 1) * (pol + 1));
      });
  const auto &vis = *r.output->visibilities;

  for (size_t rx = 0; rx < Config::NR_RECEIVERS; ++rx) {
    const size_t bl = test_support::baseline_index(rx, rx);
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
      const float scale = static_cast<float>((rx + 1) * (p + 1));
      EXPECT_EQ(vis[0][bl][p][p][0], NR_SAMPLES * scale * scale)
          << "autocorr rx=" << rx << " pol=" << p;
      EXPECT_EQ(vis[0][bl][p][p][1], 0.0f)
          << "autocorr imag rx=" << rx << " pol=" << p;
    }
  }
}

// ---------------------------------------------------------------------------
// Multi-packet: with NR_PACKETS_FOR_CORRELATION=2 the correlator integrates
// over 2*8=16 time steps, giving autocorrelation power 128 -- exactly double
// the single-packet value (64), proving accumulation across packets.
// ---------------------------------------------------------------------------
TEST_F(StarweavePipelineTest, MultiPacketAccumulatesVisibilityPower) {
  using Cfg = test_support::SmallTwoPacketConfig;

  auto output = std::make_shared<SingleHostMemoryOutput<Cfg>>();
  auto pipeline =
      test_support::pipeline_factories::make_starweave_pipeline<Cfg>(
          Cfg::NR_PACKETS_FOR_CORRELATION);
  test_support::SyntheticPipelineRun<Cfg> driver(*pipeline, output);

  driver.run(
      [](size_t, size_t, int, int, int, int) -> std::complex<int8_t> {
        return {2, -2};
      },
      [](size_t, size_t, int, int, int) -> int16_t { return 1; });

  pipeline->dump_visibilities();
  cudaDeviceSynchronize();

  const auto &vis = *output->visibilities;

  for (size_t rx = 0; rx < Cfg::NR_RECEIVERS; ++rx) {
    const size_t bl = test_support::baseline_index(rx, rx);
    for (size_t p = 0; p < Cfg::NR_POLARIZATIONS; ++p) {
      EXPECT_EQ(vis[0][bl][p][p][0], 128.0f)
          << "autocorr rx=" << rx << " pol=" << p;
      EXPECT_NEAR(vis[0][bl][p][p][1], 0.0f, 1e-3f)
          << "autocorr imag must be ~0  rx=" << rx << " pol=" << p;
    }
  }
}
