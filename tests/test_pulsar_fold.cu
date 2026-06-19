// Tests for LambdaPulsarFoldPipeline's beam output.
//
// Motivated by a real "all zeros out the other side" symptom: beams coming out
// of the pulsar-fold pipeline were uniformly zero. These tests drive the *real*
// LambdaPulsarFoldPipeline end to end through the shared harness
// (PacketInput -> ProcessorState -> pipeline -> b.beam_output) with its PSRDADA
// sink disabled (dada_key == 0, see make_pulsar_fold_pipeline), and assert that
// non-zero, finite beams actually make it through the beamformer.
//
// They also pin down a specific hypothesis raised during debugging: that the
// tiny (~1e-3) complex beam weights are being flushed to zero by __half
// storage. They are not -- 1e-3 is ~22x the smallest *normal* half (6.1e-5) and
// nowhere near the subnormal floor (~6e-8), so it round-trips with only ~0.05%
// relative error. HalfWeightsSurviveRoundTrip asserts exactly that, and
// SmallWeightsProduceNonZeroBeams proves the small weights still beamform to
// non-zero output through the actual GPU path.
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"

#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using Config = test_support::SmallSingleFPGAConfig;
using Pipeline = LambdaPulsarFoldPipeline<Config, false>;

class PulsarFoldPipelineTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Force CUDA runtime+driver initialization before constructing the
    // pipeline. LambdaPulsarFoldPipeline's first init-list member is
    // `correlator(cu::Device(0), ...)`, a *driver-API* call (cudawrappers)
    // that throws CUDA_ERROR_NOT_INITIALIZED ("initialization error") if the
    // CUDA context hasn't been created yet. In production the context already
    // exists because ProcessorState/Output allocate device memory before the
    // pipeline is built; here the pipeline (with a null Output) is the first
    // CUDA activity, so we initialize explicitly. cudaFree(0) is the idiomatic
    // way to trigger lazy CUDA init.
    cudaFree(0);
  }

  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// Constant non-zero (2, -2) sample / scale-1 input across every index -- the
// same proven non-zero stimulus PipelineHarnessSelfTest feeds.
std::complex<int8_t> constant_sample(int /*time*/, int /*receiver*/,
                                     int /*polarization*/) {
  return std::complex<int8_t>(2, -2);
}
int16_t unit_scale(int /*receiver*/, int /*polarization*/) { return 1; }

// Beam weights at the magnitudes the user observed in the field (~1e-3, varying
// per receiver/pol so the result isn't a degenerate all-equal sum). The exact
// values don't matter; what matters is that weights this small still produce
// non-zero beams.
BeamWeightsT<Config> make_small_beam_weights() {
  BeamWeightsT<Config> weights{};
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
          // ~1e-3 magnitude, sign/phase varied across receivers and pols.
          const float re = (1.0f + 0.3f * r + 0.5f * p) * 1.0e-3f *
                           ((r % 2 == 0) ? 1.0f : -1.0f);
          const float im = (2.0f - 0.2f * r + 0.4f * p) * 1.0e-3f *
                           ((p == 0) ? 1.0f : -1.0f);
          weights.weights[c][p][b][r] =
              std::complex<__half>(__float2half(re), __float2half(im));
        }
  return weights;
}

// Beam weights that are exactly zero -- the control that proves the non-zero
// assertions below are meaningful (this run must come out all zeros).
BeamWeightsT<Config> make_zero_beam_weights() {
  return BeamWeightsT<Config>{};
}

// Runs one synthetic correlation buffer through a real pulsar-fold pipeline and
// returns the device beam_output copied to host. `output = nullptr` keeps
// output_ null so the (already PSRDADA-guarded) streaming block is skipped; the
// beams are read straight from the device buffer instead.
std::vector<float> run_and_capture_beams(BeamWeightsT<Config> &weights) {
  auto pipeline = test_support::pipeline_factories::make_pulsar_fold_pipeline<
      Config>(&weights);
  test_support::SyntheticPipelineRun<Config> driver(*pipeline,
                                                    /*output=*/nullptr);
  driver.run_uniform(constant_sample, unit_scale);

  std::vector<float> beams(Pipeline::beam_output_size_bytes() / sizeof(float));
  pipeline->copy_latest_beam_output_to_host(beams.data());
  return beams;
}

bool any_nonzero(const std::vector<float> &v) {
  for (float x : v)
    if (x != 0.0f)
      return true;
  return false;
}

bool all_finite(const std::vector<float> &v) {
  for (float x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

} // namespace

// The core regression test for the reported "all zeros" symptom: with the
// field-realistic ~1e-3 weights, non-zero finite beams must come through.
TEST_F(PulsarFoldPipelineTest, SmallWeightsProduceNonZeroBeams) {
  auto weights = make_small_beam_weights();
  std::vector<float> beams = run_and_capture_beams(weights);

  EXPECT_TRUE(all_finite(beams)) << "beam_output contains NaN/Inf";
  EXPECT_TRUE(any_nonzero(beams))
      << "beam_output is entirely zero despite non-zero ~1e-3 weights and "
         "non-zero samples -- the beams did not make it through the pipeline";
}

// Unity weights are a second, simpler non-zero case (no possibility of any
// small-weight effect), to localize a failure: if this passes but the small
// weights fail, the problem is weight-magnitude related; if both fail, it's
// upstream of the weights (ingest/scale/permute).
TEST_F(PulsarFoldPipelineTest, UnityWeightsProduceNonZeroBeams) {
  auto weights = test_support::make_unity_beam_weights<Config>();
  std::vector<float> beams = run_and_capture_beams(weights);

  EXPECT_TRUE(all_finite(beams)) << "beam_output contains NaN/Inf";
  EXPECT_TRUE(any_nonzero(beams)) << "beam_output is entirely zero";
}

// Control: zero weights must give zero beams. Proves the non-zero assertions
// above aren't trivially satisfied by leftover/uninitialized device memory.
TEST_F(PulsarFoldPipelineTest, ZeroWeightsProduceZeroBeams) {
  auto weights = make_zero_beam_weights();
  std::vector<float> beams = run_and_capture_beams(weights);

  EXPECT_TRUE(all_finite(beams)) << "beam_output contains NaN/Inf";
  EXPECT_FALSE(any_nonzero(beams))
      << "beam_output is non-zero with zero weights -- the beam buffer is not "
         "actually being written by this run (stale memory?)";
}

// Directly refutes the "__half flushes the weights to zero" hypothesis: the
// observed ~1e-3 weight magnitudes survive float -> __half -> float storage
// with only small rounding error, nowhere near being zeroed. (This is a pure
// numeric check; no GPU/pipeline involved.)
TEST(PulsarFoldHalfWeights, SurviveRoundTrip) {
  // Representative values straight from the user's weight dump.
  const float observed[] = {0.00136937f, 0.00194905f, 0.00242769f,
                            0.0001777f,  -0.000766621f, 0.00261356f,
                            -0.00458948f, 0.000291622f, 0.00399739f};
  for (float w : observed) {
    const float back = __half2float(__float2half(w));
    // Not flushed to zero...
    EXPECT_NE(back, 0.0f) << "weight " << w << " flushed to zero by __half";
    // ...and preserved to within half's ~2^-11 relative precision.
    EXPECT_NEAR(back, w, std::fabs(w) * 1.0e-3f + 1.0e-9f)
        << "weight " << w << " lost too much precision in __half";
  }

  // For contrast, show where half *does* start losing values: the smallest
  // normal half is ~6.1e-5; values well below the subnormal floor (~6e-8) do
  // flush to zero. The observed weights are far above this regime.
  EXPECT_EQ(__half2float(__float2half(1.0e-9f)), 0.0f);
}
