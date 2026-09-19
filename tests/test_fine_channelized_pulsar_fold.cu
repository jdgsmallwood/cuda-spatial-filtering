// Exercises pre-correlation fine channelization wired into LambdaPulsarFoldPipeline. The
// non-RFI-mitigate path reuses channelizer_output_to_col_maj_cons (same as
// LambdaBeamformedSpectraPipeline); the RFI-mitigate path additionally reuses
// channelizer_output_to_corr_input (same as LambdaGPUPipeline) for correlation/eigendecomposition,
// then beamforms from the same channelizer-sourced samples_consolidated_col_maj. No FFT existed
// here to delete -- folding is external (DSPSR reads the PSRDADA ring buffer).
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

using Config = test_support::SmallFineChannelizedPulsarFoldConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 8, "test assumes 8 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 1, "test assumes a 1-per-side edge trim");
static_assert(Config::NR_CHANNELS == 6,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

class FineChannelizedPulsarFoldTest : public ::testing::Test {
protected:
  void SetUp() override { cudaFree(0); }
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

std::complex<int8_t> constant_sample(int /*time*/, int /*receiver*/, int /*polarization*/) {
  return std::complex<int8_t>(2, -2);
}
int16_t unit_scale(int /*receiver*/, int /*polarization*/) { return 1; }

template <typename ConfigT, bool RFI_MITIGATE>
std::vector<float> run_and_capture_beams(BeamWeightsT<ConfigT> &weights) {
  using Pipeline = LambdaPulsarFoldPipeline<ConfigT, RFI_MITIGATE>;
  std::unique_ptr<Pipeline> pipeline;
  if constexpr (RFI_MITIGATE) {
    pipeline = test_support::pipeline_factories::
        make_pulsar_fold_pipeline_rfi_mitigate<ConfigT>(&weights);
  } else {
    pipeline =
        test_support::pipeline_factories::make_pulsar_fold_pipeline<ConfigT>(&weights);
  }
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, /*output=*/nullptr);
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

template <typename ConfigT, bool RFI_MITIGATE> void run_and_check_physical_invariants() {
  auto weights = test_support::make_unity_beam_weights<ConfigT>();
  std::vector<float> beams = run_and_capture_beams<ConfigT, RFI_MITIGATE>(weights);

  EXPECT_TRUE(all_finite(beams)) << "beam_output contains NaN/Inf";
  EXPECT_TRUE(any_nonzero(beams)) << "beam_output is entirely zero";
}

} // namespace

TEST_F(FineChannelizedPulsarFoldTest, SatisfiesPhysicalInvariantsWithoutRFIMitigate) {
  run_and_check_physical_invariants<Config, false>();
}

TEST_F(FineChannelizedPulsarFoldTest, SatisfiesPhysicalInvariantsWithRFIMitigate) {
  run_and_check_physical_invariants<Config, true>();
}

// Regression guard: the original (non-channelized) delay/reorder -> beamform path must keep
// working unmodified alongside the new channelizer wiring added in the same pipeline.
TEST_F(FineChannelizedPulsarFoldTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallSingleFPGAConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig, false>();
}
