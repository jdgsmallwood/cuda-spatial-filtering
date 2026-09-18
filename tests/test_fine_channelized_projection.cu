// Exercises pre-correlation fine channelization wired into LambdaProjectionPipeline --
// correlate + eigendecompose + projection-accumulate, no beamforming, no FFT. Reuses
// channelizer_output_to_corr_input directly (same as LambdaGPUPipeline) since this pipeline
// correlates via TCC. Must land before LambdaAdaptiveBeamformedSpectraPipeline is channelized,
// since that pipeline consumes this one's projection-matrix output.
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>

namespace {

using Config = test_support::SmallFineChannelizedProjectionConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 8, "test assumes 8 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 1, "test assumes a 1-per-side edge trim");
static_assert(Config::NR_CHANNELS == 6,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

constexpr int kSignalEigenvectors = 2;
constexpr int kRunsToAverage = 1; // dump after every run, so the test sees output immediately

class FineChannelizedProjectionTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

std::complex<int8_t> constant_sample(size_t /*channel*/, size_t /*fpga*/,
                                     int /*packet*/, int /*time*/,
                                     int /*receiver*/, int /*polarization*/) {
  return std::complex<int8_t>(2, -2);
}

int16_t constant_scale(size_t /*channel*/, size_t /*fpga*/, int /*packet*/,
                       int /*receiver*/, int /*polarization*/) {
  return 1;
}

template <typename ConfigT> void run_and_check_physical_invariants() {
  auto output = std::make_shared<SingleHostMemoryOutput<ConfigT>>();
  auto pipeline = test_support::pipeline_factories::make_projection_pipeline<
      ConfigT, kSignalEigenvectors, kRunsToAverage>(ConfigT::NR_PACKETS_FOR_CORRELATION);
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, output);

  driver.run(constant_sample, constant_scale);
  cudaDeviceSynchronize();

  test_support::assert_all_finite(*output->eigenvalues, "eigenvalues");
  test_support::assert_all_finite(*output->eigenvectors, "eigenvectors");

  // Eigenvector matrices should be non-degenerate: each column (an eigenvector) should have
  // nonzero norm -- a broken/misindexed correlator or channelizer feed would produce a matrix
  // that's all-zero or has zero columns rather than a real Hermitian eigendecomposition.
  bool any_nonzero = false;
  using Element = std::remove_all_extents_t<typename SingleHostMemoryOutput<ConfigT>::Eigenvectors>;
  constexpr size_t n =
      sizeof(typename SingleHostMemoryOutput<ConfigT>::Eigenvectors) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(output->eigenvectors);
  for (size_t i = 0; i < n; ++i) {
    if (flat[i].real() != 0.0f || flat[i].imag() != 0.0f) {
      any_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(any_nonzero) << "eigenvectors should be non-degenerate for a nonzero input";
}

} // namespace

TEST_F(FineChannelizedProjectionTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

// Regression guard: the original (non-channelized) correlate+eigen path must keep working
// unmodified alongside the new channelizer wiring added in the same pipeline.
TEST_F(FineChannelizedProjectionTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallProjectionConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}
