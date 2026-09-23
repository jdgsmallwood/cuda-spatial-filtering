// Exercises pre-correlation fine channelization wired into LambdaPulsarFoldPipeline. The
// non-RFI-mitigate path reuses channelizer_output_to_col_maj_cons (same as
// LambdaBeamformedSpectraPipeline); the RFI-mitigate path additionally reuses
// channelizer_output_to_corr_input (same as LambdaGPUPipeline) for correlation/eigendecomposition,
// then beamforms from the same channelizer-sourced samples_consolidated_col_maj. No FFT existed
// here to delete -- folding is external (DSPSR reads the PSRDADA ring buffer).
#include "spatial/output.hpp"
#include "spatial/deripple.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"

#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <unistd.h>
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
    coarse_pfb_deripple_config_path().clear();
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
  // The pipeline's host-buffer reclaimer holds a raw pointer to the driver's
  // ProcessorState. Stop it before the driver (and its state) is destroyed.
  pipeline.reset();
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

TEST_F(FineChannelizedPulsarFoldTest, RuntimeDerippleProducesFiniteBeam) {
  const auto path = std::filesystem::temp_directory_path() /
      ("pulsar-deripple-" + std::to_string(getpid()) + ".json");
  const std::array<float, Config::NR_EFFECTIVE_FINE_CHANNELS> gains{
      1.0f, 1.02f, 1.04f, 1.1f, 1.06f, 0.98f};
  { std::ofstream out(path);
    out << nlohmann::json{{"schema_version", 1},
                          {"kind", "coarse_pfb_amplitude_deripple"},
                          {"fine_channels", Config::NR_FINE_CHANNELS},
                          {"edge_trim", Config::NR_FINE_CHANNEL_EDGE_TRIM},
                          {"normalization", "coarse_bin_centre"},
                          {"response_id", "synthetic-pulsar"},
                          {"voltage_gains", gains}}; }
  auto weights = test_support::make_unity_beam_weights<Config>();
  coarse_pfb_deripple_config_path() = path.string();
  const auto corrected = run_and_capture_beams<Config, false>(weights);
  EXPECT_TRUE(all_finite(corrected));
  EXPECT_TRUE(any_nonzero(corrected));
  std::filesystem::remove(path);
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

TEST_F(FineChannelizedPulsarFoldTest,
       DadaCompactionDropsOnlyRepeatedCoarseBoundary) {
  // Two coarse channels with three retained fine bins each:
  // input frequencies [a,b,c,c,d,e] -> unique DADA axis [a,b,c,d,e].
  constexpr size_t input_channels = 6;
  constexpr size_t output_channels = 5;
  constexpr size_t effective_fine = 3;
  std::vector<float> input(input_channels * 2);
  for (size_t channel = 0; channel < input_channels; ++channel) {
    input[channel * 2] = static_cast<float>(channel);
    input[channel * 2 + 1] = -static_cast<float>(channel);
  }
  std::vector<float> output(output_channels * 2, 99.0f);
  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  compact_shared_fine_boundaries_kernel<
      input_channels, output_channels, effective_fine,
      /*NR_TIMES=*/1, /*NR_POLS=*/1, /*NR_BEAMS=*/1><<<1, 32>>>(
      device_input, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  const std::vector<size_t> retained_input_channels{0, 1, 2, 4, 5};
  for (size_t output_channel = 0; output_channel < output_channels;
       ++output_channel) {
    const float expected =
        static_cast<float>(retained_input_channels[output_channel]);
    EXPECT_EQ(output[output_channel * 2], expected);
    EXPECT_EQ(output[output_channel * 2 + 1], -expected);
  }
}

TEST_F(FineChannelizedPulsarFoldTest,
       DetectedIntensitySumsBothPolarizationsAndCompactsBoundaries) {
  constexpr size_t input_channels = 6;
  constexpr size_t output_channels = 5;
  constexpr size_t effective_fine = 3;
  constexpr size_t polarizations = 2;
  std::vector<float> input(input_channels * polarizations * 2);
  for (size_t channel = 0; channel < input_channels; ++channel) {
    // X=(channel+1)+2i, Y=3+(channel+1)i.
    input[(channel * polarizations + 0) * 2 + 0] =
        static_cast<float>(channel + 1);
    input[(channel * polarizations + 0) * 2 + 1] = 2.0f;
    input[(channel * polarizations + 1) * 2 + 0] = 3.0f;
    input[(channel * polarizations + 1) * 2 + 1] =
        static_cast<float>(channel + 1);
  }
  std::vector<float> output(output_channels, -1.0f);
  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  detect_intensity_and_compact_shared_fine_boundaries_kernel<
      input_channels, output_channels, effective_fine,
      /*NR_TIMES=*/1, polarizations, /*NR_BEAMS=*/1><<<1, 32>>>(
      device_input, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  const std::vector<size_t> retained_input_channels{0, 1, 2, 4, 5};
  for (size_t output_channel = 0; output_channel < output_channels;
       ++output_channel) {
    const float x =
        static_cast<float>(retained_input_channels[output_channel] + 1);
    EXPECT_FLOAT_EQ(output[output_channel], 2.0f * x * x + 13.0f);
  }
}
