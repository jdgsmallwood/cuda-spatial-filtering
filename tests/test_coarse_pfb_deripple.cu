#include "spatial/deripple.hpp"
#include "spatial/output.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/pipeline/lambda_pulsar_fold_pipeline.hpp"
#include "spatial/spatial.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <unistd.h>

namespace {
constexpr size_t kFine = 6;
constexpr size_t kCoarse = 2;
constexpr size_t kRetained = 4;
constexpr size_t kPols = 2;

class DerippleTest : public ::testing::Test {
protected:
  std::filesystem::path path;
  void SetUp() override {
    path = std::filesystem::temp_directory_path() /
           ("deripple-test-" + std::to_string(getpid()) + ".json");
  }
  void TearDown() override { std::filesystem::remove(path); }
  void write(const nlohmann::json &value) {
    std::ofstream out(path);
    out << value;
  }
  nlohmann::json valid() {
    return {{"schema_version", 1},
            {"kind", "coarse_pfb_amplitude_deripple"},
            {"fine_channels", 6},
            {"edge_trim", 1},
            {"normalization", "coarse_bin_centre"},
            {"response_id", "synthetic-test-response"},
            {"voltage_gains", {1.0, 1.05, 1.1, 0.95}}};
  }
};

TEST_F(DerippleTest, ValidatesRuntimeTableAndCalibrationProvenance) {
  write(valid());
  auto result = load_coarse_pfb_deripple(path.string(), 6, 1);
  EXPECT_EQ(result.voltage_gains.size(), 4);
  EXPECT_FLOAT_EQ(result.voltage_gains[2], 1.1f);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string(), 8, 1), std::runtime_error);
  nlohmann::json calibrated = {
      {"provenance", {{"deripple_response_id", result.response_id},
                       {"deripple_export_state",
                        "measured_from_derippled_visibilities"},
                       {"deripple_voltage_gains", result.voltage_gains}}}};
  EXPECT_NO_THROW(validate_deripple_calibration(calibrated, result));
  calibrated["provenance"]["deripple_response_id"] = "wrong-response";
  EXPECT_THROW(validate_deripple_calibration(calibrated, result), std::runtime_error);
  calibrated["provenance"]["deripple_response_id"] = result.response_id;
  calibrated["provenance"]["deripple_voltage_gains"][0] = 1.01f;
  EXPECT_THROW(validate_deripple_calibration(calibrated, result), std::runtime_error);
  EXPECT_THROW(validate_deripple_calibration(nlohmann::json::object(), result),
               std::runtime_error);
}

TEST_F(DerippleTest, RejectsMalformedTables) {
  auto document = valid();
  document["voltage_gains"] = {1.0, 1.1};
  write(document);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string(), 6, 1), std::runtime_error);
  document = valid();
  document["voltage_gains"][2] = 1.2;
  write(document);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string(), 6, 1), std::runtime_error);
  document = valid();
  document["voltage_gains"][2] = nullptr;
  write(document);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string(), 6, 1), std::runtime_error);
  document = valid();
  document["response_id"] = "";
  write(document);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string(), 6, 1), std::runtime_error);
  EXPECT_THROW(load_coarse_pfb_deripple(path.string() + ".missing", 6, 1),
               std::runtime_error);
}

TEST_F(DerippleTest, FusedGathersScaleCorrelationBeamAndAntennaPower) {
  // Native filter layout: [coarse][fine][block=1][receiver=1][pol][time=2].
  std::vector<__half2> input(kCoarse * kFine * kPols * 2);
  for (size_t coarse = 0; coarse < kCoarse; ++coarse)
    for (size_t fine = 0; fine < kFine; ++fine)
      for (size_t pol = 0; pol < kPols; ++pol)
        for (size_t t = 0; t < 2; ++t) {
          const float value = float(coarse * 20 + fine * 3 + pol * 2 + t + 1);
          input[((coarse * kFine + fine) * kPols + pol) * 2 + t] =
              __halves2half2(__float2half(value), __float2half(-value));
        }
  const float gains[kRetained] = {1.0f, 1.05f, 1.1f, 0.95f};
  __half2 *d_input = nullptr;
  __half *d_corr = nullptr, *d_beam = nullptr;
  float *d_power = nullptr, *d_gains = nullptr;
  constexpr size_t scalars = kCoarse * kRetained * kPols * 2 * 2;
  constexpr size_t powers = kCoarse * kRetained * kPols;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_input, input.size() * sizeof(__half2)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_corr, scalars * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_beam, scalars * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_power, powers * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_gains, sizeof(gains)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_input, input.data(), input.size() * sizeof(__half2),
                                   cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_gains, gains, sizeof(gains),
                                   cudaMemcpyHostToDevice));
  std::vector<__half> corr(scalars), beam(scalars);
  std::vector<float> power(powers);
  for (bool enabled : {false, true}) {
    const float *active_gains = enabled ? d_gains : nullptr;
    channelizer_output_to_corr_input<kCoarse, kFine, 1, kPols, 1, 1, 1, 2>(
        d_input, d_corr, nullptr, active_gains);
    channelizer_output_to_col_maj_cons<kCoarse, kFine, 1, kPols, 1, 1, 2, 2>(
        d_input, d_beam, nullptr, active_gains);
    channelizer_output_to_antenna_power<kCoarse, kFine, 1, kPols, 1, 2, 2>(
        d_input, d_power, 2, nullptr, active_gains);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(corr.data(), d_corr,
                                     corr.size() * sizeof(__half),
                                     cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(beam.data(), d_beam,
                                     beam.size() * sizeof(__half),
                                     cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(power.data(), d_power,
                                     power.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost));
    for (size_t c = 0; c < kCoarse * kRetained; ++c) {
      const size_t coarse = c / kRetained;
      const size_t effective = c % kRetained;
      const size_t fine = effective + 1; // trimmed raw fine bin
      const float gain = enabled ? gains[effective] : 1.0f;
      for (size_t pol = 0; pol < kPols; ++pol) {
        float expected_power = 0.0f;
        for (size_t t = 0; t < 2; ++t) {
          const float raw = float(coarse * 20 + fine * 3 + pol * 2 + t + 1);
          const size_t corr_base = ((c * kPols + pol) * 2 + t) * 2;
          const size_t beam_re = ((c * kPols + pol) * 2 + 0) * 2 + t;
          const size_t beam_im = ((c * kPols + pol) * 2 + 1) * 2 + t;
          EXPECT_NEAR(__half2float(corr[corr_base]), raw * gain, 0.04f);
          EXPECT_NEAR(__half2float(corr[corr_base + 1]), -raw * gain, 0.04f);
          EXPECT_NEAR(__half2float(beam[beam_re]), raw * gain, 0.04f);
          EXPECT_NEAR(__half2float(beam[beam_im]), -raw * gain, 0.04f);
          expected_power += raw * raw * 2;
        }
        expected_power *= 0.5f * gain * gain;
        EXPECT_NEAR(power[c * kPols + pol], expected_power, 0.02f);
        // Correlator autocorrelation is a power: its real diagonal scales D².
        const float measured_re = __half2float(corr[((c * kPols + pol) * 2) * 2]);
        EXPECT_NEAR(measured_re * measured_re,
                    float(coarse * 20 + fine * 3 + pol * 2 + 1) *
                        float(coarse * 20 + fine * 3 + pol * 2 + 1) *
                        gain * gain,
                    2.5f);
      }
    }
  }
  cudaFree(d_gains);
  cudaFree(d_power);
  cudaFree(d_beam);
  cudaFree(d_corr);
  cudaFree(d_input);
}
} // namespace
