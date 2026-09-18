// Exercises pre-correlation fine channelization wired into LambdaCorrBeamOnlyGPUPipeline (the
// benchmark-only correlate+beamform pipeline used by gpu_benchmark/bench_gpu). Unlike every other
// pipeline, NR_FINE_CHANNELS == 1 and > 1 take genuinely different code shapes: the disabled path
// keeps today's fused packet_to_corr_input/packet_to_col_maj_cons single-kernel fast path
// unchanged (this is a benchmark harness -- its whole point is measuring that steady-state
// overhead), while the channelized path reconstructs the delay/reorder -> channelize -> correlate
// -> beamform sequence every other pipeline uses, reusing channelizer_output_to_corr_input and
// corr_input_to_col_maj_cons (same as LambdaGPUPipeline/LambdaAdaptiveBeamformedSpectraPipeline).
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

class FineChannelizedCorrBeamTest : public ::testing::Test {
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
  auto weights = test_support::make_unity_beam_weights<ConfigT>();
  auto pipeline =
      test_support::pipeline_factories::make_corr_beam_only_pipeline<ConfigT>(
          /*num_buffers=*/ConfigT::NR_PACKETS_FOR_CORRELATION, &weights);
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, output);

  driver.run(constant_sample, constant_scale);
  pipeline->dump_visibilities();
  cudaDeviceSynchronize();

  test_support::assert_all_finite(*output->beam_data, "beam_data");
  test_support::assert_all_finite(*output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<ConfigT>(*output->visibilities);

  using Element = std::remove_all_extents_t<
      typename SingleHostMemoryOutput<ConfigT>::BeamOutput>;
  constexpr size_t n =
      sizeof(typename SingleHostMemoryOutput<ConfigT>::BeamOutput) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(output->beam_data);
  bool any_nonzero = false;
  for (size_t i = 0; i < n; ++i)
    if (__half2float(flat[i]) != 0.0f)
      any_nonzero = true;
  EXPECT_TRUE(any_nonzero) << "constant input should produce nonzero beam output somewhere";
}

} // namespace

TEST_F(FineChannelizedCorrBeamTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

// Regression guard: the original fused (non-channelized) fast path must keep working unmodified
// alongside the new channelizer wiring added to the same pipeline.
TEST_F(FineChannelizedCorrBeamTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallSingleFPGAConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}
