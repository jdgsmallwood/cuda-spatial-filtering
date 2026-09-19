// Exercises pre-correlation fine channelization wired into LambdaStarweavePipeline
// (the correlation-only, visibilities-only pipeline). Unlike LambdaCorrBeamOnlyGPUPipeline,
// LambdaStarweavePipeline always runs the same delay/reorder -> (optional channelize) ->
// correlate sequence -- NR_FINE_CHANNELS == 1 and > 1 are the same code shape at a different
// width (see enqueue_corr's if constexpr branch), so this test just confirms both widths
// satisfy the same physical invariants.
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>

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
}

} // namespace

TEST_F(FineChannelizedStarweaveTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

// Regression guard: the disabled (non-channelized) path must keep working unmodified
// alongside the channelizer wiring.
TEST_F(FineChannelizedStarweaveTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallSingleFPGAConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}
