// Exercises pre-correlation fine channelization wired into
// LambdaAdaptiveBeamformedSpectraPipeline -- correlate + eigendecompose + RFI-mitigate (null or
// shrink mode) + dual-beam beamform, reusing channelizer_output_to_corr_input and
// corr_input_to_col_maj_cons (same as LambdaGPUPipeline). The whole-band post-beam FFT/edge-trim/
// downsample is deleted once channelized, same treatment LambdaGPUPipeline's own FFT got.
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
#include <limits>
#include <memory>

namespace {

using Config = test_support::SmallFineChannelizedAdaptiveConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 8, "test assumes 8 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 1, "test assumes a 1-per-side edge trim");
static_assert(Config::NR_CHANNELS == 6,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

// Sized to match LambdaAdaptiveBeamformedSpectraPipeline's own private BeamOutput/Eigenvalues/
// Eigenvectors/EigenmodeCounts types (2*NR_BEAMS for original+RFI-mitigated beams,
// NR_TIME_STEPS_PER_FINE_CHANNEL for the time axis) -- SingleHostMemoryOutput<T> assumes
// single-beam output and doesn't fit this pipeline.
template <typename T> class AdaptiveFineChannelTestOutput : public Output {
public:
  using BeamOut = std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
                                      [T::NR_TIME_STEPS_PER_FINE_CHANNEL];
  using EigVals = float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS];
  using EigVecs = std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                                     [T::NR_RECEIVERS];
  using NullCounts = int32_t[T::NR_CHANNELS][T::NR_POLARIZATIONS];
  using ArrivalsT = bool[T::NR_FPGA_CHANNELS][T::NR_PACKETS_FOR_CORRELATION + 2]
                        [T::NR_FPGA_SOURCES];

  BeamOut *beam_data;
  EigVals *eigenvalues;
  EigVecs *eigenvectors;
  NullCounts *nulled_eigenmode_counts;
  ArrivalsT *arrivals;

  AdaptiveFineChannelTestOutput() {
    beam_data = new BeamOut[1]{};
    eigenvalues = new EigVals[1]{};
    eigenvectors = new EigVecs[1]{};
    nulled_eigenmode_counts = new NullCounts[1]{};
    arrivals = new ArrivalsT[1]{};
  }
  ~AdaptiveFineChannelTestOutput() {
    delete[] beam_data;
    delete[] eigenvalues;
    delete[] eigenvectors;
    delete[] nulled_eigenmode_counts;
    delete[] arrivals;
  }

  size_t register_beam_data_block(size_t, size_t) override { return 1; }
  size_t register_visibilities_block(size_t, size_t, int, int) override {
    return std::numeric_limits<size_t>::max();
  }
  size_t register_eigendecomposition_data_block(size_t, size_t) override { return 1; }
  size_t register_fft_block(size_t, size_t) override {
    return std::numeric_limits<size_t>::max();
  }

  void *get_beam_data_landing_pointer(size_t) override { return beam_data; }
  void *get_visibilities_landing_pointer(size_t) override { return nullptr; }
  void *get_arrivals_data_landing_pointer(size_t) override { return arrivals; }
  void *get_eigenvalues_data_landing_pointer(size_t) override { return eigenvalues; }
  void *get_eigenvectors_data_landing_pointer(size_t) override { return eigenvectors; }
  void *get_nulled_eigenmode_counts_landing_pointer(size_t) override {
    return nulled_eigenmode_counts;
  }
  void *get_beam_counts_landing_pointer(size_t) override { return nulled_eigenmode_counts; }
  void *get_fft_landing_pointer(size_t) override { return nullptr; }

  void register_beam_data_transfer_complete(size_t) override {}
  void register_visibilities_transfer_complete(size_t) override {}
  void register_arrivals_transfer_complete(size_t) override {}
  void register_eigendecomposition_data_transfer_complete(size_t) override {}
  void register_beam_counts_transfer_complete(size_t) override {}
  void register_fft_transfer_complete(size_t) override {}
};

class FineChannelizedAdaptiveTest : public ::testing::Test {
protected:
  void SetUp() override { cudaFree(0); }
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

std::complex<int8_t> constant_sample(size_t, size_t, int, int, int, int) { return {2, -2}; }
int16_t constant_scale(size_t, size_t, int, int, int) { return 1; }

template <typename ConfigT> void run_and_check_physical_invariants() {
  auto output = std::make_shared<AdaptiveFineChannelTestOutput<ConfigT>>();
  BeamWeightsT<ConfigT> weights = test_support::make_unity_beam_weights<ConfigT>();
  auto pipeline = test_support::pipeline_factories::make_adaptive_beamformed_spectra_pipeline<
      ConfigT>(ConfigT::NR_PACKETS_FOR_CORRELATION, &weights);
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, output);

  driver.run(constant_sample, constant_scale);
  cudaDeviceSynchronize();

  test_support::assert_all_finite(*output->eigenvalues, "eigenvalues");
  test_support::assert_all_finite(*output->eigenvectors, "eigenvectors");

  using Element = std::remove_all_extents_t<
      typename AdaptiveFineChannelTestOutput<ConfigT>::BeamOut>;
  constexpr size_t n =
      sizeof(typename AdaptiveFineChannelTestOutput<ConfigT>::BeamOut) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(output->beam_data);
  bool any_nonzero = false;
  for (size_t i = 0; i < n; ++i) {
    const float re = __half2float(flat[i].real());
    const float im = __half2float(flat[i].imag());
    ASSERT_TRUE(std::isfinite(re)) << "beam_data element " << i << " (real) is not finite";
    ASSERT_TRUE(std::isfinite(im)) << "beam_data element " << i << " (imag) is not finite";
    if (re != 0.0f || im != 0.0f)
      any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero) << "constant input should produce nonzero beam output somewhere";
}

} // namespace

TEST_F(FineChannelizedAdaptiveTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

TEST_F(FineChannelizedAdaptiveTest, CoarseRankZeroMatchesWeightsOnlyFineVoltages) {
  auto reference = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto coarse = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::WeightsOnly> pipeline(
        1, &weights, {{0, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, reference);
    driver.run(constant_sample, constant_scale);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::Coarse> pipeline(
        1, &weights, {{0, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, coarse);
    driver.run(constant_sample, constant_scale);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  double power = 0;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (int p = 0; p < 2; ++p)
      for (int b = 0; b < 2; ++b)
        for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_FINE_CHANNEL; ++t) {
          const auto expected = (*reference->beam_data)[c][p][b][t];
          const auto actual = (*coarse->beam_data)[c][p][b][t];
          EXPECT_NEAR(__half2float(actual.real()), __half2float(expected.real()), 0.01f);
          EXPECT_NEAR(__half2float(actual.imag()), __half2float(expected.imag()), 0.01f);
          power += std::norm(std::complex<float>(__half2float(expected.real()),
                                                __half2float(expected.imag())));
        }
  EXPECT_GT(power, 0);
}

TEST_F(FineChannelizedAdaptiveTest, FineAndWeightsOnlyPreserveScaledDelayedAndPermutedVoltages) {
  auto reference = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto fine = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto coarse = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (int p = 0; p < 2; ++p)
      for (int r = 0; r < 4; ++r)
        weights.weights[c][p][0][r] = {__float2half((r + 1) * .125f), __float2half((p - r) * .125f)};
  const auto samples = [](size_t, size_t, int packet, int time, int receiver, int pol) {
    return std::complex<int8_t>((time + 3 * receiver + 2 * pol + packet + 23) % 23 - 11,
                               (2 * time + receiver + 5 * pol + packet + 19) % 19 - 9);
  };
  const auto scales = [](size_t, size_t, int packet, int receiver, int pol) {
    return int16_t(2 + packet + receiver + pol);
  };
  {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::WeightsOnly> pipeline(
        1, &weights, {{0, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, reference);
    int delay[] = {7};
    pipeline.set_subpacket_delays(delay);
    pipeline.set_stream_permutation({3, 3, 2, 2, -1, -1, 0, 0}, {1, 0, 1, 0, 0, 1, 1, 0});
    driver.run(samples, scales);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::Coarse> pipeline(
        1, &weights, {{0, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, coarse);
    int delay[] = {7};
    pipeline.set_subpacket_delays(delay);
    pipeline.set_stream_permutation({3, 3, 2, 2, -1, -1, 0, 0}, {1, 0, 1, 0, 0, 1, 1, 0});
    driver.run(samples, scales);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::Fine> pipeline(
        1, &weights, {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, fine);
    int delay[] = {7};
    pipeline.set_subpacket_delays(delay);
    pipeline.set_stream_permutation({3, 3, 2, 2, -1, -1, 0, 0}, {1, 0, 1, 0, 0, 1, 1, 0});
    driver.run(samples, scales);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  double power = 0;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (int p = 0; p < 2; ++p)
      for (int b = 0; b < 2; ++b)
        for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_FINE_CHANNEL; ++t) {
          const auto expected = (*reference->beam_data)[c][p][b][t];
          const auto actual = (*coarse->beam_data)[c][p][b][t];
          EXPECT_NEAR(__half2float(actual.real()), __half2float(expected.real()), 0.01f);
          EXPECT_NEAR(__half2float(actual.imag()), __half2float(expected.imag()), 0.01f);
          const auto fine_value = (*fine->beam_data)[c][p][b][t];
          EXPECT_NEAR(__half2float(fine_value.real()), __half2float(expected.real()), 0.01f);
          EXPECT_NEAR(__half2float(fine_value.imag()), __half2float(expected.imag()), 0.01f);
          power += std::norm(std::complex<float>(__half2float(expected.real()),
                                                __half2float(expected.imag())));
        }
  EXPECT_GT(power, 0);
}

// Regression guard: the original (non-channelized) correlate+eigen+beamform+FFT path must keep
// working unmodified alongside the new channelizer wiring added in the same pipeline.
// Compare public beam/eigenspectrum outputs with the retained staged path.
// Rank zero avoids comparing the two paths' different whitening/filter rules.
TEST_F(FineChannelizedAdaptiveTest, CoarsePreparationPreservesStagedEigenspectraAndVoltages) {
  auto staged = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto fused = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  const auto samples = [](size_t, size_t, int packet, int time, int receiver, int pol) {
    return std::complex<int8_t>((time + 3 * receiver + pol + packet + 23) % 23 - 11,
                               (2 * time + receiver + 5 * pol + packet + 19) % 19 - 9);
  };
  const auto scales = [](size_t, size_t, int packet, int receiver, int pol) {
    return int16_t(2 + packet + receiver + pol);
  };
  const auto run = [&](auto legacy, auto output) {
    BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1.0, 1);
    AdaptiveBeamPipeline<Config, BeamformingMode::Coarse, decltype(legacy)::value> pipeline(
        1, &weights, {{0, 0}}, 0, std::move(steering));
    test_support::SyntheticPipelineRun<Config> driver(pipeline, output);
    int delay[] = {-7};
    pipeline.set_subpacket_delays(delay);
    pipeline.set_stream_permutation({3, 3, 2, 2, -1, -1, 0, 0}, {1, 0, 1, 0, 0, 1, 1, 0});
    driver.run(samples, scales);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  };
  run(std::true_type{}, staged);
  run(std::false_type{}, fused);
  double eigenvalue_sum = 0;
  for (size_t c = 0; c < Config::NR_FPGA_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
        const float expected = (*staged->eigenvalues)[c][p][r];
        EXPECT_NEAR((*fused->eigenvalues)[c][p][r], expected,
                    1e-5f * std::max(1.f, std::abs(expected)));
        eigenvalue_sum += expected;
      }
  EXPECT_GT(eigenvalue_sum, 0);
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (int p = 0; p < 2; ++p)
      for (int b = 0; b < 2; ++b)
        for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_FINE_CHANNEL; ++t) {
          const auto expected = (*staged->beam_data)[c][p][b][t];
          const auto actual = (*fused->beam_data)[c][p][b][t];
          EXPECT_NEAR(__half2float(actual.real()), __half2float(expected.real()), 0.01f);
          EXPECT_NEAR(__half2float(actual.imag()), __half2float(expected.imag()), 0.01f);
        }
}

TEST_F(FineChannelizedAdaptiveTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallAdaptiveConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}

namespace {
template <BeamformingMode Mode>
auto protected_complex_source() {
  auto output = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1, 1);
  constexpr int channels = Mode == BeamformingMode::Coarse ? 1 : Config::NR_CHANNELS;
  std::unordered_map<int, int> ranks;
  std::vector<std::complex<float>> signatures;
  for (int c = 0; c < channels; ++c) {
    ranks[c] = 3;
    for (int p = 0; p < 2; ++p)
      for (const auto z : std::array<std::complex<float>, 4>{{{1, 0}, {0, 2}, {-3, 0}, {0, -4}}})
        signatures.push_back(z);
  }
  AdaptiveBeamPipeline<Config, Mode> pipeline(1, &weights, ranks, 0, std::move(steering));
  pipeline.set_protected_signatures(signatures);
  test_support::SyntheticPipelineRun<Config> driver(pipeline, output);
  driver.run([](size_t, size_t, int, int, int receiver, int) {
    const std::complex<int8_t> signal[] = {{2, -2}, {4, 4}, {-6, 6}, {-8, -8}};
    return signal[receiver];
  }, constant_scale);
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}
}

TEST_F(FineChannelizedAdaptiveTest, CoarseAndFinePreserveComplexSourceAcrossFineBins) {
  auto coarse = protected_complex_source<BeamformingMode::Coarse>();
  auto fine = protected_complex_source<BeamformingMode::Fine>();
  double power = 0;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (int p = 0; p < 2; ++p)
      for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_FINE_CHANNEL; ++t) {
        const auto reference = (*fine->beam_data)[c][p][0][t];
        const auto expected = std::complex<float>(__half2float(reference.real()), __half2float(reference.imag()));
        power += std::norm(expected);
        for (const auto &output : {coarse, fine}) {
          const auto value = (*output->beam_data)[c][p][1][t];
          EXPECT_NEAR(__half2float(value.real()), expected.real(), 0.02 + 0.003 * std::abs(expected));
          EXPECT_NEAR(__half2float(value.imag()), expected.imag(), 0.02 + 0.003 * std::abs(expected));
        }
      }
  EXPECT_GT(power, 0);
}

namespace {
template <BeamformingMode Mode>
auto synthetic_interference_trial() {
  auto output = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1, 1);
  constexpr int channels = Mode == BeamformingMode::Coarse ? 1 : Config::NR_CHANNELS;
  std::unordered_map<int, int> ranks;
  std::vector<std::complex<float>> signature(channels * 2 * Config::NR_RECEIVERS);
  for (int c = 0; c < channels; ++c) {
    ranks[c] = 1;
    for (int p = 0; p < 2; ++p) signature[(c * 2 + p) * Config::NR_RECEIVERS] = {1, 0};
  }
  AdaptiveBeamPipeline<Config, Mode> pipeline(1, &weights, ranks, 0, std::move(steering));
  if constexpr (Mode != BeamformingMode::WeightsOnly) pipeline.set_protected_signatures(signature);
  test_support::SyntheticPipelineRun<Config> driver(pipeline, output);
  driver.run([](size_t, size_t, int, int time, int receiver, int) {
    if (receiver == 0) return std::complex<int8_t>(2, -2);
    if (Mode != BeamformingMode::WeightsOnly && receiver == 1)
      return std::complex<int8_t>(time % 4 < 2 ? 20 : -20, 0);
    return std::complex<int8_t>(0, 0);
  }, constant_scale);
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}
}

TEST_F(FineChannelizedAdaptiveTest, CoarseAndFineRemoveInterferenceAndRetainSource) {
  const auto reference = synthetic_interference_trial<BeamformingMode::WeightsOnly>();
  for (const auto &output : {synthetic_interference_trial<BeamformingMode::Coarse>(),
                             synthetic_interference_trial<BeamformingMode::Fine>()}) {
    double before_error = 0, after_error = 0, source_power = 0;
    for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
      for (int p = 0; p < 2; ++p)
        for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_FINE_CHANNEL; ++t) {
          auto as_float = [](const std::complex<__half> &v) {
            return std::complex<float>(__half2float(v.real()), __half2float(v.imag()));
          };
          const auto expected = as_float((*reference->beam_data)[c][p][0][t]);
          const auto actual = as_float((*output->beam_data)[c][p][1][t]);
          before_error += std::norm(as_float((*output->beam_data)[c][p][0][t]) - expected);
          after_error += std::norm(actual - expected);
          source_power += std::norm(expected);
          EXPECT_NEAR(actual.real(), expected.real(), 0.03 + 0.003 * std::abs(expected));
          EXPECT_NEAR(actual.imag(), expected.imag(), 0.03 + 0.003 * std::abs(expected));
        }
    EXPECT_GT(before_error, source_power);
    EXPECT_LT(after_error, before_error * 1e-5);
  }
}

// Public pipeline seam: graph execution must consume fresh input and preserve
// protected/unprotected beam voltages, not merely complete without CUDA errors.
TEST_F(FineChannelizedAdaptiveTest, AdaptiveGraphsMatchEagerVoltages) {
  struct EnvRestore {
    std::string name, value;
    bool present;
    EnvRestore(const char *n) : name(n), present(std::getenv(n) != nullptr) {
      if (present) value = std::getenv(n);
    }
    ~EnvRestore() { if (present) setenv(name.c_str(), value.c_str(), 1); else unsetenv(name.c_str()); }
  } restore_enable("SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH"),
    restore_disable("SPATIAL_DISABLE_CUDA_GRAPH");
  setenv("SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH", "1", 1);
  auto check = [&](auto mode_tag, bool protect) {
    constexpr auto mode = decltype(mode_tag)::value;
    std::vector<float> reference;
    for (bool graph : {false, true}) {
      if (graph) unsetenv("SPATIAL_DISABLE_CUDA_GRAPH");
      else setenv("SPATIAL_DISABLE_CUDA_GRAPH", "1", 1);
      auto output = std::make_shared<AdaptiveFineChannelTestOutput<Config>>();
      auto weights = test_support::make_unity_beam_weights<Config>();
      BeamSteering<Config> steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{}, 1, 1);
      constexpr size_t channels = mode == BeamformingMode::Coarse ? Config::NR_FPGA_CHANNELS : Config::NR_CHANNELS;
      std::unordered_map<int, int> ranks;
      for (size_t c = 0; c < channels; ++c) ranks[c] = 1;
      AdaptiveBeamPipeline<Config, mode> pipeline(1, &weights, ranks, 0, std::move(steering));
      std::vector<std::complex<float>> signature(channels * 2 * Config::NR_RECEIVERS, {1, 0});
      if (protect) pipeline.set_protected_signatures(signature);
      test_support::SyntheticPipelineRun<Config> driver(pipeline, output);
      pipeline.prepare_cuda_graphs();
      ASSERT_EQ(pipeline.cuda_graph_count(), graph ? 1u : 0u);
      // Replacing a mask changes a captured pointer and must invalidate it.
      pipeline.set_enabled_inputs(std::vector<uint8_t>(channels * 2 * Config::NR_RECEIVERS, 1));
      ASSERT_EQ(pipeline.cuda_graph_count(), 0u);
      pipeline.prepare_cuda_graphs();
      ASSERT_EQ(pipeline.cuda_graph_count(), graph ? 1u : 0u);
      if (protect) {
        for (size_t i = 0; i < signature.size(); ++i) signature[i] = {float(1 + i % 3), float(i % 2)};
        pipeline.set_protected_signatures(signature);
      }
      int delay[] = {3};
      pipeline.set_subpacket_delays(delay);
      pipeline.set_stream_permutation({3,3,2,2,1,1,0,0}, {1,0,1,0,1,0,1,0});
      for (int frame = 0; frame < 2; ++frame) {
        driver.run([frame](size_t, size_t, int packet, int time, int receiver, int pol) {
          return std::complex<int8_t>((time + receiver * 3 + packet + frame * 7) % 17 - 8,
                                     (time * 2 + pol + receiver + frame * 5) % 13 - 6);
        }, constant_scale, 1000 + frame * Config::NR_PACKETS_FOR_CORRELATION * Config::NR_TIME_STEPS_PER_PACKET);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto *data = reinterpret_cast<const __half *>(output->beam_data);
        const size_t count = sizeof(typename AdaptiveFineChannelTestOutput<Config>::BeamOut) / sizeof(__half);
        for (size_t i = 0; i < count; ++i) {
          const float value = __half2float(data[i]);
          ASSERT_TRUE(std::isfinite(value));
          if (!graph) reference.push_back(value);
          else EXPECT_NEAR(value, reference[frame * count + i], 0.002f * std::max(1.f, std::abs(reference[frame * count + i])));
        }
      }
      EXPECT_EQ(pipeline.eigensolver_failures(), 0u);
      EXPECT_GE(pipeline.processed_blocks(), 2u);
    }
  };
  for (bool protect : {false, true}) {
    check(std::integral_constant<BeamformingMode, BeamformingMode::Coarse>{}, protect);
    check(std::integral_constant<BeamformingMode, BeamformingMode::Fine>{}, protect);
  }
}
