// Tests for LambdaCorrBeamOnlyGPUPipeline.
//
// The pipeline runs both the TCC correlator (→ visibilities) and the ccglib
// beamformer (→ beam time-series) in a single pass.  Tests drive it through
// the real ProcessorState → GPUPipeline → Output seam (SyntheticPipelineRun),
// not via DummyFinalPacketData.
//
// Because dump_visibilities() fires automatically only after
// NR_CORRELATED_BLOCKS_TO_ACCUMULATE runs (10 000 for SmallSingleFPGAConfig),
// single-run tests call pipeline->dump_visibilities() manually then sync.
//
// ---- Exact-value derivation (constant (2,−2) input, scale=1, unity weights)
//
//   fp16 sample x = (2.0, -2.0) for all (channel, time, receiver, pol)
//   unity weight  w = (1, 0) for all (channel, pol, beam, receiver)
//
//   Beam:
//     beam[c][p][b][t] = Σ_r w[c][p][b][r] · x[r,p,t]
//                      = NR_RECEIVERS · (2, -2) = 4·(2,−2) = (8, −8)
//     Output layout: __half [ch][pol][beam][time_sample][re/im]
//
//   Visibility:
//     V[i,j][p][q] = Σ_t conj(x[i,p,t]) · x[j,q,t]
//                  = NR_TIME_STEPS · conj((2,−2))·(2,−2)
//                  = 8 · (2+2j)·(2−2j)
//                  = 8 · (4+4) = 64 + 0j   (same for all i,j,p,q)
//     NR_TIME_STEPS = NR_PACKETS_FOR_CORRELATION × NR_TIME_STEPS_PER_PACKET
//                   = 1 × 8 = 8

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
class CorrBeamOnlyPipelineTest : public ::testing::Test {
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
struct CorrBeamOnlyRun {
  std::shared_ptr<SingleHostMemoryOutput<Config>> output;
  BeamWeightsT<Config> weights;
  std::unique_ptr<LambdaCorrBeamOnlyGPUPipeline<Config>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<Config>> driver;
};

template <typename SampleFn, typename ScaleFn>
CorrBeamOnlyRun do_run(BeamWeightsT<Config> weights, SampleFn sample_fn,
                       ScaleFn scale_fn) {
  CorrBeamOnlyRun r;
  r.output = std::make_shared<SingleHostMemoryOutput<Config>>();
  r.weights = std::move(weights);
  r.pipeline =
      test_support::pipeline_factories::make_corr_beam_only_pipeline<Config>(
          /*num_buffers=*/Config::NR_PACKETS_FOR_CORRELATION, &r.weights);
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
// Exact beam values
// ---------------------------------------------------------------------------
TEST_F(CorrBeamOnlyPipelineTest, BeamOutputExactValues) {
  auto r = do_run(test_support::make_unity_beam_weights<Config>(),
                  constant_sample, constant_scale);
  const auto &beam = *r.output->beam_data;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t t = 0; t < NR_SAMPLES; ++t) {
          EXPECT_EQ(__half2float(beam[c][p][b][t][0]), 8.0f)
              << "real  ch=" << c << " pol=" << p << " beam=" << b << " t=" << t;
          EXPECT_EQ(__half2float(beam[c][p][b][t][1]), -8.0f)
              << "imag  ch=" << c << " pol=" << p << " beam=" << b << " t=" << t;
        }
}

// ---------------------------------------------------------------------------
// Exact visibility values
// ---------------------------------------------------------------------------
TEST_F(CorrBeamOnlyPipelineTest, VisibilityExactValues) {
  auto r = do_run(test_support::make_unity_beam_weights<Config>(),
                  constant_sample, constant_scale);
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
TEST_F(CorrBeamOnlyPipelineTest, PhysicalInvariants) {
  auto r = do_run(test_support::make_unity_beam_weights<Config>(),
                  constant_sample, constant_scale);
  test_support::assert_all_finite(*r.output->beam_data, "beam_data");
  test_support::assert_all_finite(*r.output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<Config>(*r.output->visibilities);
}

// ---------------------------------------------------------------------------
// Zero input → zero output
// ---------------------------------------------------------------------------
TEST_F(CorrBeamOnlyPipelineTest, ZeroInputProducesZeroOutput) {
  auto r = do_run(test_support::make_unity_beam_weights<Config>(), zero_sample,
                  constant_scale);
  const auto &beam = *r.output->beam_data;
  const auto &vis = *r.output->visibilities;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t t = 0; t < NR_SAMPLES; ++t) {
          EXPECT_EQ(__half2float(beam[c][p][b][t][0]), 0.0f);
          EXPECT_EQ(__half2float(beam[c][p][b][t][1]), 0.0f);
        }

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t bl = 0; bl < Config::NR_BASELINES_UNPADDED; ++bl)
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
        for (size_t q = 0; q < Config::NR_POLARIZATIONS; ++q) {
          EXPECT_EQ(vis[c][bl][p][q][0], 0.0f);
          EXPECT_EQ(vis[c][bl][p][q][1], 0.0f);
        }
}

// ---------------------------------------------------------------------------
// Zero beamforming weights silence beams but don't affect visibilities.
// The correlator operates on raw samples, independent of beam weights.
// ---------------------------------------------------------------------------
TEST_F(CorrBeamOnlyPipelineTest, ZeroWeightsZeroBeamsNonzeroVisibilities) {
  BeamWeightsT<Config> zero_weights{};
  auto r = do_run(zero_weights, constant_sample, constant_scale);
  const auto &beam = *r.output->beam_data;
  const auto &vis = *r.output->visibilities;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t t = 0; t < NR_SAMPLES; ++t) {
          EXPECT_EQ(__half2float(beam[c][p][b][t][0]), 0.0f);
          EXPECT_EQ(__half2float(beam[c][p][b][t][1]), 0.0f);
        }

  // Autocorrelation power must be strictly positive regardless of weights.
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t rx = 0; rx < Config::NR_RECEIVERS; ++rx) {
      const size_t bl = test_support::baseline_index(rx, rx);
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
        EXPECT_GT(vis[c][bl][p][p][0], 0.0f)
            << "ch=" << c << " rx=" << rx << " pol=" << p
            << ": autocorrelation power must be > 0";
    }
}

// ---------------------------------------------------------------------------
// Beam amplitude scales linearly with weight magnitude.
// Doubling all weights from 1 to 2 should double all beam samples.
// Visibilities are unchanged (they come from the correlator, not the GEMM).
// ---------------------------------------------------------------------------
TEST_F(CorrBeamOnlyPipelineTest, BeamAmplitudeScalesWithWeightMagnitude) {
  BeamWeightsT<Config> double_weights;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t rx = 0; rx < Config::NR_RECEIVERS; ++rx)
          double_weights.weights[c][p][b][rx] =
              std::complex<__half>(__float2half(2.0f), __float2half(0.0f));

  auto r = do_run(double_weights, constant_sample, constant_scale);
  const auto &beam = *r.output->beam_data;
  const auto &vis = *r.output->visibilities;

  // Beams = 2 × (unity-weight result) = (16, -16)
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t t = 0; t < NR_SAMPLES; ++t) {
          EXPECT_EQ(__half2float(beam[c][p][b][t][0]), 16.0f)
              << "real  ch=" << c << " pol=" << p << " beam=" << b << " t=" << t;
          EXPECT_EQ(__half2float(beam[c][p][b][t][1]), -16.0f)
              << "imag  ch=" << c << " pol=" << p << " beam=" << b << " t=" << t;
        }

  // Visibilities must be identical to the unity-weight case (64, 0).
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t bl = 0; bl < Config::NR_BASELINES_UNPADDED; ++bl)
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
        for (size_t q = 0; q < Config::NR_POLARIZATIONS; ++q) {
          EXPECT_EQ(vis[c][bl][p][q][0], 64.0f)
              << "vis real  ch=" << c << " bl=" << bl;
          EXPECT_EQ(vis[c][bl][p][q][1], 0.0f)
              << "vis imag  ch=" << c << " bl=" << bl;
        }
}
