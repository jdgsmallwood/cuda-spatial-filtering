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

// Distinct values on every axis catch time/polarization/beam transpositions
// that the single-time, single-beam boundary tests cannot detect.
TEST_F(FineChannelizedPulsarFoldTest,
       ProductionChannelCompactionPreservesTimePolarizationAndBeam) {
  constexpr size_t coarse = 40, fine = 28, channels = coarse * fine;
  constexpr size_t times = 3, pols = 2, beams = 2;
  constexpr size_t unique = coarse * (fine - 1);
  std::vector<float> input(beams * times * channels * pols * 2);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 997) - 498);
  std::vector<float> voltage(beams * times * unique * pols * 2);
  std::vector<float> intensity(beams * times * unique);
  float *din = nullptr, *dv = nullptr, *di = nullptr;
  CUDA_CHECK(cudaMalloc(&din, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, voltage.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&di, intensity.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(din, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  compact_shared_fine_boundaries_kernel<channels, unique, fine, times, pols,
                                        beams, true>
      <<<(voltage.size() + 255) / 256, 256>>>(din, dv);
  CUDA_CHECK(cudaGetLastError());
  detect_intensity_and_compact_shared_fine_boundaries_kernel<
      channels, unique, fine, times, pols, beams, true>
      <<<(intensity.size() + 255) / 256, 256>>>(din, di);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(voltage.data(), dv, voltage.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(intensity.data(), di, intensity.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(din));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(di));

  // Build the oracle by walking coarse/fine bins, independently of the
  // output-index division/modulo formula used by the kernels.
  size_t out = 0;
  for (size_t b = 0; b < beams; ++b)
    for (size_t t = 0; t < times; ++t)
      for (size_t c = 0; c < coarse; ++c)
        for (size_t f = 1; f < fine; ++f, ++out) {
          float power = 0;
          for (size_t p = 0; p < pols; ++p)
            for (size_t iq = 0; iq < 2; ++iq) {
              const float expected = input[
                  ((((b * times + t) * channels + c * fine + f) * pols + p) * 2) + iq];
              EXPECT_EQ(voltage[(out * pols + p) * 2 + iq], expected)
                  << "beam=" << b << " time=" << t << " coarse=" << c
                  << " fine=" << f << " pol=" << p << " component=" << iq;
              power += expected * expected;
            }
          EXPECT_FLOAT_EQ(intensity[out], power);
        }
  EXPECT_EQ(out, intensity.size());
}

TEST_F(FineChannelizedPulsarFoldTest,
       FourFpgaBroadbandDelayCorrectionMatchesUndelayedReference) {
  using FC = LambdaConfig<2, 4, 64, 8, 2, 2, 4, 1, 32, 32, 1,
                          false, 128, 8, 1>;
  constexpr std::array<int64_t, 4> offsets{-23, 0, -16, 31};
  const std::array<int64_t, 4> zero{};
  auto run = [&](const std::array<int64_t, 4> &injected,
                 const std::array<int64_t, 4> &configured) {
    auto weights = test_support::make_unity_beam_weights<FC>();
    auto pipeline = test_support::pipeline_factories::make_pulsar_fold_pipeline<FC>(&weights);
    test_support::SyntheticPipelineRun<FC> driver(
        *pipeline, nullptr, configured, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    auto noise = [](int time, int salt) {
      uint32_t x = static_cast<uint32_t>(time + 4096) * 747796405u +
                   static_cast<uint32_t>(salt + 1) * 2891336453u;
      x ^= x >> 16;
      x *= 2246822519u;
      x ^= x >> 13;
      return static_cast<int8_t>(static_cast<int>(x & 15u) - 8);
    };
    auto sample = [&](size_t ch, size_t fpga, int packet, int tick, int r, int p) {
      const int time = packet * 64 + tick - static_cast<int>(injected[fpga]);
      const int salt = static_cast<int>(ch) * 8 + r * 4 + p * 2;
      return std::complex<int8_t>{noise(time, salt), noise(time, salt + 1)};
    };
    auto scale = [](size_t ch, size_t fpga, int, int r, int p) {
      return static_cast<int16_t>(1 + ch + 2 * fpga + r + p);
    };
    driver.run(sample, scale);
    std::vector<float> result(pipeline->beam_output_size_bytes() / sizeof(float));
    pipeline->copy_latest_beam_output_to_host(result.data());
    pipeline.reset(); // stop the reclaimer before destroying its ProcessorState
    return result;
  };
  const auto reference = run(zero, zero);
  const auto corrected = run(offsets, offsets);
  const auto uncorrected = run(offsets, zero);
  ASSERT_EQ(reference.size(), corrected.size());
  ASSERT_EQ(reference.size(), uncorrected.size());
  double signal_power = 0, wrong_delay_error = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    ASSERT_TRUE(std::isfinite(corrected[i])) << "sample=" << i;
    EXPECT_NEAR(corrected[i], reference[i], 0.001f + 0.001f * std::abs(reference[i]))
        << "sample=" << i;
    signal_power += static_cast<double>(reference[i]) * reference[i];
    const double error = uncorrected[i] - reference[i];
    wrong_delay_error += error * error;
  }
  EXPECT_GT(signal_power, 0);
  EXPECT_GT(wrong_delay_error, 0.1 * signal_power);
}

TEST_F(FineChannelizedPulsarFoldTest,
       StrongSingleReceptorRemainsFiniteAndPreservesVoltageScale) {
  // Four FPGA input layouts and the production 32-bin PFB. One receptor
  // becomes 128 times stronger while the other polarization stays fixed.
  // Raw 100*128 samples fit in FP16; the unnormalised PFB tone does not.
  using FC = LambdaConfig<2, 4, 64, 8, 2, 2, 32, 1, 32, 32, 1,
                          false, 128, 32, 2>;
  auto run = [&](int strong_scale) {
    auto weights = test_support::make_unity_beam_weights<FC>();
    auto pipeline =
        test_support::pipeline_factories::make_pulsar_fold_pipeline<FC>(&weights);
    test_support::SyntheticPipelineRun<FC> driver(
        *pipeline, nullptr, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    auto sample = [](size_t ch, size_t fpga, int packet, int tick, int r, int p) {
      if (p == 1 && (fpga != 1 || r != 0))
        return std::complex<int8_t>{0, 0};
      const double phase = 6.283185307179586 * (3 + ch + p) *
                           (packet * 64 + tick) / 32.0;
      return std::complex<int8_t>{
          static_cast<int8_t>(std::lround(100 * std::cos(phase))),
          static_cast<int8_t>(std::lround(100 * std::sin(phase)))};
    };
    auto scale = [&](size_t, size_t, int, int, int p) {
      return static_cast<int16_t>(p == 1 ? strong_scale : 1);
    };
    driver.run(sample, scale);
    std::vector<float> result(pipeline->beam_output_size_bytes() / sizeof(float));
    pipeline->copy_latest_beam_output_to_host(result.data());
    pipeline.reset();
    return result;
  };
  const auto reference = run(1);
  const auto strong = run(128);
  ASSERT_EQ(reference.size(), strong.size());
  double signal_power = 0, error_power = 0;
  float peak = 0;
  for (size_t i = 0; i < strong.size(); ++i) {
    ASSERT_TRUE(std::isfinite(strong[i])) << "scalar=" << i;
    const size_t pol = (i / 2) % 2; // beam/time/channel/polarization/complex
    const double expected = reference[i] * (pol == 1 ? 128.0 : 1.0);
    signal_power += expected * expected;
    const double error = strong[i] - expected;
    error_power += error * error;
    peak = std::max(peak, std::abs(strong[i]));
  }
  EXPECT_GT(peak, 65504.0f) << "restored float output must retain over-range voltages";
  ASSERT_GT(signal_power, 0);
  EXPECT_LT(error_power / signal_power, 1e-6)
      << "normalisation must preserve both polarization amplitudes";
}
