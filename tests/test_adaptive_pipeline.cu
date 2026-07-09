// Tests for LambdaAdaptiveBeamformedSpectraPipeline — covering both nulling
// (the existing behaviour) and the new eigenvalue-shrinkage mode.
//
// The pipeline hard-codes CUFFT_FFT_SIZE=64, so NR_TIME_STEPS_PER_PACKET must
// be 64 (with NR_PACKETS_FOR_CORRELATION=1 → 64 total time steps for the FFT).
//
// ---- Config constraints ----
//   NR_TIME_STEPS_PER_PACKET = 64  (matches CUFFT_FFT_SIZE=64)
//   NR_RECEIVERS             = 4   (TCC: must be a multiple of 32 padded)
//   NR_PADDED_RECEIVERS      = 32  (TCC constraint)
//   NR_POLARIZATIONS         = 2   (TCC constraint)
//   NR_PACKETS_FOR_CORRELATION = 1
//   NR_BEAMS                 = 1   (output has 2×: original + RFI-mitigated)

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/writers.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cuda_fp16.h>
#include <filesystem>
#include <functional>
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace {

// LambdaConfig for the adaptive pipeline: 64 time steps to match CUFFT_FFT_SIZE.
using AdaptiveConfig =
    LambdaConfig<1,   // NR_CHANNELS
                 1,   // NR_FPGA_SOURCES
                 64,  // NR_TIME_STEPS_PER_PACKET (= CUFFT_FFT_SIZE)
                 4,   // NR_RECEIVERS
                 2,   // NR_POLARIZATIONS
                 4,   // NR_RECEIVERS_PER_PACKET
                 1,   // NR_PACKETS_FOR_CORRELATION
                 1,   // NR_BEAMS
                 32,  // NR_PADDED_RECEIVERS
                 32,  // NR_PADDED_RECEIVERS_PER_BLOCK
                 1    // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

using MultiChannelAdaptiveConfig =
    LambdaConfig<2,   // NR_CHANNELS
                 1,   // NR_FPGA_SOURCES
                 64,  // NR_TIME_STEPS_PER_PACKET (= CUFFT_FFT_SIZE)
                 4,   // NR_RECEIVERS
                 2,   // NR_POLARIZATIONS
                 4,   // NR_RECEIVERS_PER_PACKET
                 1,   // NR_PACKETS_FOR_CORRELATION
                 1,   // NR_BEAMS
                 32,  // NR_PADDED_RECEIVERS
                 32,  // NR_PADDED_RECEIVERS_PER_BLOCK
                 1    // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;

// ---------------------------------------------------------------------------
// Test output: pinned buffers sized to match what the adaptive pipeline writes.
// SingleHostMemoryOutput<T> allocates NR_BEAMS beams, but the adaptive pipeline
// writes 2*NR_BEAMS (original + RFI-mitigated), so we need a custom output.
// ---------------------------------------------------------------------------
template <typename T>
class AdaptiveTestOutput : public Output {
  static constexpr int NR_FINE_REMOVE = 5;
  static constexpr int NR_TIMES =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int NR_FINE_CHANS =
      T::NR_TIME_STEPS_PER_PACKET - 2 * NR_FINE_REMOVE;

public:
  using BeamOut = std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                                      [2 * T::NR_BEAMS][NR_TIMES];
  using EigVals =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS];
  using EigVecs = std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                                     [T::NR_RECEIVERS][T::NR_RECEIVERS];
  using NullCounts = int32_t[T::NR_CHANNELS][T::NR_POLARIZATIONS];
  using FFTOut = float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
                      [NR_FINE_CHANS];
  using ArrivalsT = bool[T::NR_CHANNELS][T::NR_PACKETS_FOR_CORRELATION + 2]
                        [T::NR_FPGA_SOURCES];

  BeamOut    *beam_data;
  EigVals    *eigenvalues;
  EigVecs    *eigenvectors;
  NullCounts *nulled_eigenmode_counts;
  FFTOut     *fft_output;
  ArrivalsT  *arrivals;

  AdaptiveTestOutput() {
    beam_data = new BeamOut[1]{};
    eigenvalues = new EigVals[1]{};
    eigenvectors = new EigVecs[1]{};
    nulled_eigenmode_counts = new NullCounts[1]{};
    fft_output = new FFTOut[1]{};
    arrivals = new ArrivalsT[1]{};
  }
  ~AdaptiveTestOutput() {
    delete[] beam_data;
    delete[] eigenvalues;
    delete[] eigenvectors;
    delete[] nulled_eigenmode_counts;
    delete[] fft_output;
    delete[] arrivals;
  }

  size_t register_beam_data_block(size_t, size_t) override { return 1; }
  size_t register_visibilities_block(size_t, size_t, int, int) override {
    return std::numeric_limits<size_t>::max();
  }
  size_t register_eigendecomposition_data_block(size_t, size_t) override {
    return 1;
  }
  size_t register_fft_block(size_t, size_t) override { return 1; }

  void *get_beam_data_landing_pointer(size_t) override { return beam_data; }
  void *get_visibilities_landing_pointer(size_t) override { return nullptr; }
  void *get_arrivals_data_landing_pointer(size_t) override { return arrivals; }
  void *get_eigenvalues_data_landing_pointer(size_t) override {
    return eigenvalues;
  }
  void *get_eigenvectors_data_landing_pointer(size_t) override {
    return eigenvectors;
  }
  void *get_nulled_eigenmode_counts_landing_pointer(size_t) override {
    return nulled_eigenmode_counts;
  }
  void *get_fft_landing_pointer(size_t) override { return fft_output; }

  void register_beam_data_transfer_complete(size_t) override {}
  void register_visibilities_transfer_complete(size_t) override {}
  void register_arrivals_transfer_complete(size_t) override {}
  void register_eigendecomposition_data_transfer_complete(size_t) override {}
  void register_fft_transfer_complete(size_t) override {}
};

// ---------------------------------------------------------------------------
// Helper: total squared magnitude of the RFI-mitigated beam (beam index 1).
// ---------------------------------------------------------------------------
template <typename T>
double rfi_beam_total_power(const AdaptiveTestOutput<T> &out) {
  double total = 0.0;
  for (size_t c = 0; c < T::NR_CHANNELS; ++c)
    for (size_t p = 0; p < T::NR_POLARIZATIONS; ++p)
      for (size_t t = 0; t < T::NR_PACKETS_FOR_CORRELATION *
                                  T::NR_TIME_STEPS_PER_PACKET;
           ++t) {
        // beam index 1 = the RFI-mitigated beam
        const auto &s = out.beam_data[0][c][p][1][t];
        float re = __half2float(s.real());
        float im = __half2float(s.imag());
        total += static_cast<double>(re * re + im * im);
      }
  return total;
}

std::string make_temp_hdf5_file() {
  namespace fs = std::filesystem;
  auto tmpl = fs::temp_directory_path() / "adaptive_beam_output_XXXXXX.h5";
  std::string s = tmpl.string();
  int fd = mkstemps(s.data(), 3);
  if (fd >= 0)
    close(fd);
  return s;
}

template <typename BeamT>
std::vector<uint16_t> beam_bits(const BeamT &beam_data) {
  const auto *ptr = reinterpret_cast<const uint16_t *>(&beam_data);
  return std::vector<uint16_t>(ptr, ptr + sizeof(BeamT) / sizeof(uint16_t));
}

template <typename Config>
std::complex<int8_t> tagged_single_receiver_sample(size_t channel, size_t,
                                                   int, int time,
                                                   int receiver, int pol) {
  if (receiver != 0) {
    return {0, 0};
  }
  const int8_t real = static_cast<int8_t>(time + 1);
  const int8_t imag =
      static_cast<int8_t>(-(1 + 10 * pol + 20 * static_cast<int>(channel)));
  return {real, imag};
}

template <typename Config>
std::complex<float> expected_tagged_beam_value(size_t channel, int pol,
                                               int time) {
  const auto sample =
      tagged_single_receiver_sample<Config>(channel, 0, 0, time, 0, pol);
  return {static_cast<float>(sample.real()), static_cast<float>(sample.imag())};
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class AdaptivePipelineTest : public ::testing::Test {
protected:
  void SetUp() override { cudaFree(0); }

  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// ---------------------------------------------------------------------------
// Sample generators
// ---------------------------------------------------------------------------
// Constant signal on all receivers — produces a rank-1 covariance.
std::complex<int8_t> constant_sample(size_t, size_t, int, int, int, int) {
  return {2, -2};
}
int16_t constant_scale(size_t, size_t, int, int, int) { return 1; }

// Time-varying signal: first half all-receivers (2,-2), second half only
// receiver 0 gets (4,0). Produces a rank-2 covariance so that the mean of
// non-RFI eigenvalues λ̄ > 0, making shrink != null.
std::complex<int8_t> rank2_sample(size_t /*ch*/, size_t /*fpga*/, int /*pkt*/,
                                   int time, int receiver, int /*pol*/) {
  if (time < AdaptiveConfig::NR_TIME_STEPS_PER_PACKET / 2)
    return {2, -2}; // all receivers, first half
  return (receiver == 0) ? std::complex<int8_t>{4, 0}
                         : std::complex<int8_t>{0, 0};
}

std::complex<int8_t> multimode_detect_sample(size_t, size_t, int, int time,
                                             int receiver, int pol) {
  if (pol == 1) {
    return {0, 0};
  }
  const int mode_a = ((time / 8) % 2 == 0) ? 20 : -20;
  const int mode_b = ((time / 4) % 2 == 0) ? 18 : -18;
  const int receiver_sign = (receiver % 2 == 0) ? 1 : -1;
  return {static_cast<int8_t>(mode_a + receiver_sign * mode_b), 0};
}

template <typename Config> struct SampleDump {
  int8_t data[Config::NR_CHANNELS][Config::NR_POLARIZATIONS]
            [Config::NR_RECEIVERS]
            [Config::NR_PACKETS_FOR_CORRELATION *
             Config::NR_TIME_STEPS_PER_PACKET][2];
};

template <typename Config>
SampleDump<Config> make_sample_dump(
    const std::function<std::complex<int8_t>(size_t, size_t, int, int, int,
                                             int)> &sample_fn) {
  SampleDump<Config> samples{};
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t r = 0; r < Config::NR_RECEIVERS; ++r)
        for (size_t t = 0;
             t < Config::NR_PACKETS_FOR_CORRELATION *
                     Config::NR_TIME_STEPS_PER_PACKET;
             ++t) {
          const auto sample = sample_fn(c, 0, 0, static_cast<int>(t),
                                        static_cast<int>(r), static_cast<int>(p));
          samples.data[c][p][r][t][0] = sample.real();
          samples.data[c][p][r][t][1] = sample.imag();
        }
  return samples;
}

template <typename Config> struct WeightDump {
  float data[Config::NR_CHANNELS][Config::NR_POLARIZATIONS][Config::NR_BEAMS]
            [Config::NR_RECEIVERS][2];
};

template <typename Config>
WeightDump<Config> make_weight_dump(const BeamWeightsT<Config> &weights) {
  WeightDump<Config> dump{};
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p)
      for (size_t b = 0; b < Config::NR_BEAMS; ++b)
        for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
          dump.data[c][p][b][r][0] =
              __half2float(weights.weights[c][p][b][r].real());
          dump.data[c][p][b][r][1] =
              __half2float(weights.weights[c][p][b][r].imag());
        }
  return dump;
}

template <typename Config>
double rfi_beam_power_for_pol(const AdaptiveTestOutput<Config> &out, int pol) {
  double total = 0.0;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t t = 0; t < Config::NR_PACKETS_FOR_CORRELATION *
                                Config::NR_TIME_STEPS_PER_PACKET;
         ++t) {
      const auto &s = out.beam_data[0][c][pol][1][t];
      const float re = __half2float(s.real());
      const float im = __half2float(s.imag());
      total += static_cast<double>(re * re + im * im);
    }
  return total;
}

// ---------------------------------------------------------------------------
// Run helper
// ---------------------------------------------------------------------------
struct AdaptiveRun {
  std::shared_ptr<AdaptiveTestOutput<AdaptiveConfig>> output;
  BeamWeightsT<AdaptiveConfig> weights;
  std::unique_ptr<LambdaAdaptiveBeamformedSpectraPipeline<AdaptiveConfig>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<AdaptiveConfig>> driver;

  explicit AdaptiveRun(bool shrink, int K = 1, bool detect = false,
                       float detection_delta = 3.0f) {
    output = std::make_shared<AdaptiveTestOutput<AdaptiveConfig>>();
    weights = test_support::make_unity_beam_weights<AdaptiveConfig>();
    std::unordered_map<int, int> eig_map{{0, K}};
    pipeline = test_support::pipeline_factories::make_adaptive_beamformed_spectra_pipeline<AdaptiveConfig>(
        /*num_buffers=*/3, &weights, eig_map, shrink, /*min_freq_channel=*/0,
        detect, detection_delta);
    driver = std::make_unique<test_support::SyntheticPipelineRun<AdaptiveConfig>>(
        *pipeline, output);
  }

  void run_constant() {
    driver->run(constant_sample,
                [](size_t, size_t, int, int, int) -> int16_t { return 1; });
    cudaDeviceSynchronize();
  }

  void run_rank2() {
    driver->run(rank2_sample,
                [](size_t, size_t, int, int, int) -> int16_t { return 1; });
    cudaDeviceSynchronize();
  }
};

// ---------------------------------------------------------------------------
// Tests: nulling mode (existing behaviour)
// ---------------------------------------------------------------------------

// Basic sanity: the pipeline runs without crash and eigenvalues satisfy the
// cuSOLVER contract (ascending, non-negative for same-pol blocks).
TEST_F(AdaptivePipelineTest, NullingModePhysicalInvariants) {
  AdaptiveRun r(/*shrink=*/false);
  r.run_constant();

  const auto &eigs = *r.output->eigenvalues;
  for (size_t c = 0; c < AdaptiveConfig::NR_CHANNELS; ++c)
    for (size_t p = 0; p < AdaptiveConfig::NR_POLARIZATIONS; ++p) {
      float prev = -std::numeric_limits<float>::infinity();
      for (size_t k = 0; k < AdaptiveConfig::NR_RECEIVERS; ++k) {
        EXPECT_GE(eigs[c][p][k], prev - 1e-3f) << "eigenvalue not ascending";
        EXPECT_GE(eigs[c][p][k], -1e-3f) << "eigenvalue negative";
        prev = eigs[c][p][k];
      }
    }

  // FFT output must be finite
  test_support::assert_all_finite(*r.output->fft_output, "fft_output");
}

// With K=0 (suppress nothing), the filter M = I and the adapted weights equal
// the input weights; the RFI-mitigated beam must have non-zero power.
TEST_F(AdaptivePipelineTest, NullingModeK0NoSuppression) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/0);
  r.run_constant();
  EXPECT_GT(rfi_beam_total_power(*r.output), 0.0)
      << "K=0 should leave weights unchanged — RFI beam must have power";
}

// With K=0 (projection = I) both beams must be bit-identical at every element:
// they use the same weights (adapted_weights = weights * I = weights) on the
// same samples. This is the K=0 invariant and a direct regression test for the
// "stale original-beam weights" bug: if slot 0 of weights_rfi_mitigated were
// ever allowed to drift from the current b.weights (e.g. because a BeamSteering
// refresh fired after construction but slot 0 was only initialised once), then
// beam[0] would use a different set of weights than beam[1] and this test would
// fail even under K=0.
TEST_F(AdaptivePipelineTest, NullingModeK0OriginalAndMitigatedBeamsAreIdentical) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/0);
  r.run_constant();

  const auto &beam = *r.output->beam_data;
  for (size_t c = 0; c < AdaptiveConfig::NR_CHANNELS; ++c)
    for (size_t p = 0; p < AdaptiveConfig::NR_POLARIZATIONS; ++p)
      for (size_t t = 0; t < AdaptiveConfig::NR_PACKETS_FOR_CORRELATION *
                                 AdaptiveConfig::NR_TIME_STEPS_PER_PACKET;
           ++t) {
        EXPECT_EQ(__half2float(beam[c][p][0][t].real()),
                  __half2float(beam[c][p][1][t].real()))
            << "beam[0].real != beam[1].real at c=" << c << " p=" << p
            << " t=" << t;
        EXPECT_EQ(__half2float(beam[c][p][0][t].imag()),
                  __half2float(beam[c][p][1][t].imag()))
            << "beam[0].imag != beam[1].imag at c=" << c << " p=" << p
            << " t=" << t;
      }
}

// With K=1 and constant (rank-1) input plus unity weights:
// w = (1,...,1) is parallel to the dominant eigenvector (1,...,1)/√N.
// Nulling zeroes that direction → adapted weights ≈ 0 → RFI beam power ≈ 0.
TEST_F(AdaptivePipelineTest, NullingModeZerosRFIBeamWithRankOneInput) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/1);
  r.run_constant();

  const double power = rfi_beam_total_power(*r.output);
  // The theoretical result is exactly 0; allow floating-point tolerance.
  EXPECT_LT(power, 1.0)
      << "Null K=1 + rank-1 input: RFI beam should be near zero (power="
      << power << ")";
}

TEST_F(AdaptivePipelineTest, NullingModePreservesTaggedBeamLayoutWhenKIsZero) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/0);
  r.driver->run(tagged_single_receiver_sample<AdaptiveConfig>,
                [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  const auto &beam = *r.output->beam_data;
  for (int pol = 0; pol < static_cast<int>(AdaptiveConfig::NR_POLARIZATIONS);
       ++pol) {
    for (int time = 0;
         time < static_cast<int>(AdaptiveConfig::NR_TIME_STEPS_PER_PACKET);
         ++time) {
      const auto expected =
          expected_tagged_beam_value<AdaptiveConfig>(/*channel=*/0, pol, time);
      for (int beam_idx = 0; beam_idx < 2 * static_cast<int>(AdaptiveConfig::NR_BEAMS);
           ++beam_idx) {
        EXPECT_EQ(__half2float(beam[0][pol][beam_idx][time].real()),
                  expected.real())
            << "beam real mismatch at pol=" << pol << " beam=" << beam_idx
            << " time=" << time;
        EXPECT_EQ(__half2float(beam[0][pol][beam_idx][time].imag()),
                  expected.imag())
            << "beam imag mismatch at pol=" << pol << " beam=" << beam_idx
            << " time=" << time;
      }
    }
  }
}

TEST_F(AdaptivePipelineTest, NullingModePreservesTaggedChannelLayoutWhenKIsZero) {
  using Config = MultiChannelAdaptiveConfig;
  auto output = std::make_shared<AdaptiveTestOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  std::unordered_map<int, int> eig_map{{0, 0}, {1, 0}};
  auto pipeline =
      test_support::pipeline_factories::make_adaptive_beamformed_spectra_pipeline<Config>(
          /*num_buffers=*/3, &weights, eig_map, /*shrink_eigenvalues=*/false);
  test_support::SyntheticPipelineRun<Config> driver(*pipeline, output);
  driver.run(tagged_single_receiver_sample<Config>,
             [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  const auto &beam = *output->beam_data;
  for (int channel = 0; channel < static_cast<int>(Config::NR_CHANNELS);
       ++channel) {
    for (int pol = 0; pol < static_cast<int>(Config::NR_POLARIZATIONS); ++pol) {
      for (int time = 0; time < static_cast<int>(Config::NR_TIME_STEPS_PER_PACKET);
           ++time) {
        const auto expected =
            expected_tagged_beam_value<Config>(channel, pol, time);
        for (int beam_idx = 0; beam_idx < 2 * static_cast<int>(Config::NR_BEAMS);
             ++beam_idx) {
          EXPECT_EQ(__half2float(beam[channel][pol][beam_idx][time].real()),
                    expected.real())
              << "channel-mapped beam real mismatch at channel=" << channel
              << " pol=" << pol << " beam=" << beam_idx << " time=" << time;
          EXPECT_EQ(__half2float(beam[channel][pol][beam_idx][time].imag()),
                    expected.imag())
              << "channel-mapped beam imag mismatch at channel=" << channel
              << " pol=" << pol << " beam=" << beam_idx << " time=" << time;
        }
      }
    }
  }
}

TEST_F(AdaptivePipelineTest, NullingModeKeepsOriginalBeamAndZerosMitigatedBeam) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/1);
  r.run_constant();

  const auto &beam = *r.output->beam_data;
  for (int pol = 0; pol < static_cast<int>(AdaptiveConfig::NR_POLARIZATIONS);
       ++pol) {
    for (int time = 0;
         time < static_cast<int>(AdaptiveConfig::NR_TIME_STEPS_PER_PACKET);
         ++time) {
      EXPECT_EQ(__half2float(beam[0][pol][0][time].real()), 8.0f)
          << "original beam real mismatch at pol=" << pol << " time=" << time;
      EXPECT_EQ(__half2float(beam[0][pol][0][time].imag()), -8.0f)
          << "original beam imag mismatch at pol=" << pol << " time=" << time;
      EXPECT_NEAR(__half2float(beam[0][pol][1][time].real()), 0.0f, 0.25f)
          << "mitigated beam real mismatch at pol=" << pol << " time=" << time;
      EXPECT_NEAR(__half2float(beam[0][pol][1][time].imag()), 0.0f, 0.25f)
          << "mitigated beam imag mismatch at pol=" << pol << " time=" << time;
    }
  }
}

// ---------------------------------------------------------------------------
// Tests: shrinking mode
// ---------------------------------------------------------------------------

TEST_F(AdaptivePipelineTest, ShrinkingModePhysicalInvariants) {
  AdaptiveRun r(/*shrink=*/true);
  r.run_constant();

  const auto &eigs = *r.output->eigenvalues;
  for (size_t c = 0; c < AdaptiveConfig::NR_CHANNELS; ++c)
    for (size_t p = 0; p < AdaptiveConfig::NR_POLARIZATIONS; ++p) {
      float prev = -std::numeric_limits<float>::infinity();
      for (size_t k = 0; k < AdaptiveConfig::NR_RECEIVERS; ++k) {
        EXPECT_GE(eigs[c][p][k], prev - 1e-3f) << "eigenvalue not ascending";
        EXPECT_GE(eigs[c][p][k], -1e-3f) << "eigenvalue negative";
        prev = eigs[c][p][k];
      }
    }

  test_support::assert_all_finite(*r.output->fft_output, "fft_output");
}

// With K=0 in shrink mode: N-K = N, so lambda_bar = mean of all N eigenvalues
// and d[k] = 1.0 for every k.  V_scaled = V → M = V * V^H = I →
// projection_matrix = I → weights unchanged.  Both beams must match the
// unmodified beamformed value, identical to the null-mode K=0 result.
TEST_F(AdaptivePipelineTest, ShrinkingModeK0PreservesTaggedBeamLayout) {
  AdaptiveRun r(/*shrink=*/true, /*K=*/0);
  r.driver->run(tagged_single_receiver_sample<AdaptiveConfig>,
                [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  const auto &beam = *r.output->beam_data;
  for (int pol = 0; pol < static_cast<int>(AdaptiveConfig::NR_POLARIZATIONS);
       ++pol) {
    for (int time = 0;
         time < static_cast<int>(AdaptiveConfig::NR_TIME_STEPS_PER_PACKET);
         ++time) {
      const auto expected =
          expected_tagged_beam_value<AdaptiveConfig>(/*channel=*/0, pol, time);
      for (int beam_idx = 0;
           beam_idx < 2 * static_cast<int>(AdaptiveConfig::NR_BEAMS);
           ++beam_idx) {
        EXPECT_EQ(__half2float(beam[0][pol][beam_idx][time].real()),
                  expected.real())
            << "shrink K=0 beam real mismatch at pol=" << pol
            << " beam=" << beam_idx << " time=" << time;
        EXPECT_EQ(__half2float(beam[0][pol][beam_idx][time].imag()),
                  expected.imag())
            << "shrink K=0 beam imag mismatch at pol=" << pol
            << " beam=" << beam_idx << " time=" << time;
      }
    }
  }
}

TEST_F(AdaptivePipelineTest, WritesAdaptiveBeamOutputToHDF5WithSameLogicalLayout) {
  using BeamOut = AdaptiveTestOutput<AdaptiveConfig>::BeamOut;
  using ArrivalsT = AdaptiveConfig::ArrivalsOutputType;

  AdaptiveRun reference(/*shrink=*/false, /*K=*/0);
  reference.driver->run(tagged_single_receiver_sample<AdaptiveConfig>,
                        [](size_t, size_t, int, int, int) -> int16_t {
                          return 1;
                        });
  cudaDeviceSynchronize();

  const std::string filename = make_temp_hdf5_file();
  {
    HighFive::File file(filename, HighFive::File::Truncate);
    auto beam_writer = std::make_unique<HDF5BeamWriter<BeamOut, ArrivalsT>>(file);
    auto output = std::make_shared<
        BufferedOutput<AdaptiveConfig, typename AdaptiveConfig::FFTOutputType,
                       typename AdaptiveConfig::EigenvalueOutputType,
                       typename AdaptiveConfig::EigenvectorOutputType, BeamOut>>(
        std::move(beam_writer), nullptr, nullptr, nullptr);
    output->start_writer_loop();

    auto weights = test_support::make_unity_beam_weights<AdaptiveConfig>();
    std::unordered_map<int, int> eig_map{{0, 0}};
    auto pipeline =
        test_support::pipeline_factories::make_adaptive_beamformed_spectra_pipeline<AdaptiveConfig>(
            /*num_buffers=*/3, &weights, eig_map, /*shrink_eigenvalues=*/false);
    test_support::SyntheticPipelineRun<AdaptiveConfig> driver(*pipeline, output);
    driver.run(tagged_single_receiver_sample<AdaptiveConfig>,
               [](size_t, size_t, int, int, int) -> int16_t { return 1; });
    cudaDeviceSynchronize();
  }

  HighFive::File verify_file(filename, HighFive::File::ReadOnly);
  auto beam_ds = verify_file.getDataSet("beam_data");
  const auto dims = beam_ds.getDimensions();
  ASSERT_EQ(dims.size(), 6u);
  EXPECT_EQ(dims[0], 1u);
  EXPECT_EQ(dims[1], AdaptiveConfig::NR_CHANNELS);
  EXPECT_EQ(dims[2], AdaptiveConfig::NR_POLARIZATIONS);
  EXPECT_EQ(dims[3], 2 * AdaptiveConfig::NR_BEAMS);
  EXPECT_EQ(dims[4], AdaptiveConfig::NR_PACKETS_FOR_CORRELATION *
                         AdaptiveConfig::NR_TIME_STEPS_PER_PACKET);
  EXPECT_EQ(dims[5], 2u);

  std::vector<uint16_t> stored_bits(
      sizeof(BeamOut) / sizeof(uint16_t));
  beam_ds.read_raw(stored_bits.data());
  EXPECT_EQ(stored_bits, beam_bits(*reference.output->beam_data));
}

// With rank-1 input λ̄ = mean(λ_0..λ_{N-2}) = 0, so d[N-1] = 0/λ = 0.
// Shrinking degenerates to nulling → RFI beam power ≈ 0, same as null mode.
TEST_F(AdaptivePipelineTest, ShrinkingWithRankOneInputMatchesNulling) {
  AdaptiveRun r_shrink(/*shrink=*/true, /*K=*/1);
  r_shrink.run_constant();

  const double power = rfi_beam_total_power(*r_shrink.output);
  EXPECT_LT(power, 1.0)
      << "Shrink K=1 + rank-1 input: λ̄=0 → scale=0 → should behave like "
         "nulling (power=" << power << ")";
}

// With rank-2 input, λ̄ = mean(λ_0, λ_1, λ_2) > 0 (at least λ_2 > 0), so
// d[3] = λ̄/λ_3 > 0.  The RFI-mitigated beam is attenuated but NOT zeroed.
// Null mode with the same input zeroes the dominant direction completely;
// shrink mode must produce strictly MORE power than null mode.
TEST_F(AdaptivePipelineTest, ShrinkingWithRankTwoInputDiffersFromNulling) {
  // Run null and shrink modes sequentially in the same test (no device reset
  // between them — the two AdaptiveRun objects are fully independent and both
  // are destroyed before TearDown's cudaDeviceReset).
  double null_power = 0.0;
  {
    AdaptiveRun r_null(/*shrink=*/false, /*K=*/1);
    r_null.run_rank2();
    null_power = rfi_beam_total_power(*r_null.output);
  }

  double shrink_power = 0.0;
  {
    AdaptiveRun r_shrink(/*shrink=*/true, /*K=*/1);
    r_shrink.run_rank2();
    shrink_power = rfi_beam_total_power(*r_shrink.output);
  }

  EXPECT_GT(shrink_power, null_power + 1e-6)
      << "Shrink mode should produce more power than null mode with rank-2 "
         "input (null=" << null_power << ", shrink=" << shrink_power << ")";
}

TEST_F(AdaptivePipelineTest, DetectionModeTracksPerPolarizationCounts) {
  AdaptiveRun r(/*shrink=*/false, /*K=*/0, /*detect=*/true,
                /*detection_delta=*/0.5f);
  r.driver->run(multimode_detect_sample,
                [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  const auto &counts = *r.output->nulled_eigenmode_counts;
  EXPECT_EQ(counts[0][0], 2);
  EXPECT_EQ(counts[0][1], 0);

  EXPECT_LT(rfi_beam_power_for_pol(*r.output, /*pol=*/0), 1.0)
      << "nulling should suppress the interfered polarization";
  EXPECT_EQ(rfi_beam_power_for_pol(*r.output, /*pol=*/1), 0.0)
      << "clean zero-input polarization should remain zero";
}

TEST_F(AdaptivePipelineTest, DetectionModeWritesParityFixtureToHDF5) {
  const char *path = std::getenv("ADAPTIVE_PARITY_OUTPUT");
  if (path == nullptr || path[0] == '\0') {
    GTEST_SKIP() << "ADAPTIVE_PARITY_OUTPUT not set";
  }

  const float detection_delta = 0.5f;
  AdaptiveRun run(/*shrink=*/false, /*K=*/0, /*detect=*/true,
                  detection_delta);
  auto sample_dump = make_sample_dump<AdaptiveConfig>(multimode_detect_sample);
  auto weight_dump = make_weight_dump<AdaptiveConfig>(run.weights);
  run.driver->run(multimode_detect_sample,
                  [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  {
    HighFive::File file(path, HighFive::File::Truncate);
    file.createAttribute<std::string>("adaptive_projection_mode", "null");
    file.createAttribute<std::string>("adaptive_detection_mode", "threshold");
    file.createAttribute<float>("adaptive_detection_delta", detection_delta);
    file.createAttribute<std::string>("synthetic_case",
                                      "rank2_pol0_zero_pol1");
    auto eigenvalues_ds = file.createDataSet<float>(
        "eigenvalues", HighFive::DataSpace::From(*run.output->eigenvalues));
    eigenvalues_ds.write(*run.output->eigenvalues);

    std::vector<size_t> eigenvector_dims = {
        AdaptiveConfig::NR_CHANNELS, AdaptiveConfig::NR_POLARIZATIONS,
        AdaptiveConfig::NR_RECEIVERS, AdaptiveConfig::NR_RECEIVERS, 2};
    auto eigenvectors_ds =
        file.createDataSet<float>("eigenvectors", HighFive::DataSpace(eigenvector_dims));
    eigenvectors_ds.write_raw(
        reinterpret_cast<const float *>(run.output->eigenvectors));

    auto counts_ds = file.createDataSet<int32_t>(
        "nulled_eigenmode_counts",
        HighFive::DataSpace::From(*run.output->nulled_eigenmode_counts));
    counts_ds.write(*run.output->nulled_eigenmode_counts);

    std::vector<size_t> beam_dims = {
        AdaptiveConfig::NR_CHANNELS, AdaptiveConfig::NR_POLARIZATIONS,
        2 * AdaptiveConfig::NR_BEAMS,
        AdaptiveConfig::NR_PACKETS_FOR_CORRELATION *
            AdaptiveConfig::NR_TIME_STEPS_PER_PACKET,
        2};
    auto beam_ds =
        file.createDataSet<uint16_t>("beam_data", HighFive::DataSpace(beam_dims));
    beam_ds.write_raw(reinterpret_cast<const uint16_t *>(run.output->beam_data));
    auto input_ds = file.createDataSet<int8_t>(
        "input_samples", HighFive::DataSpace::From(sample_dump.data));
    input_ds.write(sample_dump.data);
    auto weights_ds = file.createDataSet<float>(
        "beam_weights", HighFive::DataSpace::From(weight_dump.data));
    weights_ds.write(weight_dump.data);
  }

  HighFive::File verify_file(path, HighFive::File::ReadOnly);
  auto counts_ds = verify_file.getDataSet("nulled_eigenmode_counts");
  const auto dims = counts_ds.getDimensions();
  ASSERT_EQ(dims.size(), 2u);
  EXPECT_EQ(dims[0], AdaptiveConfig::NR_CHANNELS);
  EXPECT_EQ(dims[1], AdaptiveConfig::NR_POLARIZATIONS);
}

// ---------------------------------------------------------------------------
// Unit test: scaleEigenvectorColumnsKernel correctness
// ---------------------------------------------------------------------------
TEST_F(AdaptivePipelineTest, ScaleKernelScalesColumnsCorrectly) {
  // 4×4 complex matrix, 1 batch. Verify each column is scaled by d[col].
  constexpr int N = 4;
  constexpr int BATCHES = 1;

  // Host matrices
  std::vector<float2> h_V(BATCHES * N * N);
  std::vector<float2> h_Vs(BATCHES * N * N, {0.0f, 0.0f});
  std::vector<float> h_d(BATCHES * N);

  // Fill V with (col+1, -(row+1)) and d[col] = (col+1)*0.5
  for (int batch = 0; batch < BATCHES; ++batch)
    for (int col = 0; col < N; ++col)
      for (int row = 0; row < N; ++row) {
        // column-major: element (row, col) at flat index col*N + row
        int idx = batch * N * N + col * N + row;
        h_V[idx] = {static_cast<float>(col + 1),
                    -static_cast<float>(row + 1)};
      }
  for (int b = 0; b < BATCHES; ++b)
    for (int k = 0; k < N; ++k)
      h_d[b * N + k] = (k + 1) * 0.5f;

  // Allocate device buffers
  float2 *d_V = nullptr, *d_Vs = nullptr;
  float *d_d = nullptr;
  CUDA_CHECK(cudaMalloc(&d_V,  sizeof(float2) * BATCHES * N * N));
  CUDA_CHECK(cudaMalloc(&d_Vs, sizeof(float2) * BATCHES * N * N));
  CUDA_CHECK(cudaMalloc(&d_d,  sizeof(float)  * BATCHES * N));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), sizeof(float2) * BATCHES * N * N,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d, h_d.data(), sizeof(float) * BATCHES * N,
                        cudaMemcpyHostToDevice));

  scaleEigenvectorColumns(d_V, d_Vs, d_d, N, BATCHES, /*stream=*/nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_Vs.data(), d_Vs, sizeof(float2) * BATCHES * N * N,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_V);
  cudaFree(d_Vs);
  cudaFree(d_d);

  // Verify
  for (int batch = 0; batch < BATCHES; ++batch)
    for (int col = 0; col < N; ++col)
      for (int row = 0; row < N; ++row) {
        int idx = batch * N * N + col * N + row;
        float scale = h_d[batch * N + col];
        EXPECT_NEAR(h_Vs[idx].x, h_V[idx].x * scale, 1e-5f)
            << "real part mismatch at batch=" << batch
            << " col=" << col << " row=" << row;
        EXPECT_NEAR(h_Vs[idx].y, h_V[idx].y * scale, 1e-5f)
            << "imag part mismatch at batch=" << batch
            << " col=" << col << " row=" << row;
      }
}

// Multi-batch variant: each batch has distinct inputs and scales so that any
// bug in the batch-stride computation (flat_idx = batch*N*N + idx) or the
// scale lookup (d[batch*N + col]) would produce wrong values in at least one
// batch and be caught by the per-element checks below.
TEST_F(AdaptivePipelineTest, ScaleKernelMultiBatchIndexingIsCorrect) {
  constexpr int N = 4;
  constexpr int BATCHES = 3;

  std::vector<float2> h_V(BATCHES * N * N);
  std::vector<float2> h_Vs(BATCHES * N * N, {0.0f, 0.0f});
  std::vector<float>  h_d(BATCHES * N);

  // Give each batch a distinguishably different input and scale vector
  for (int batch = 0; batch < BATCHES; ++batch)
    for (int col = 0; col < N; ++col)
      for (int row = 0; row < N; ++row) {
        int idx = batch * N * N + col * N + row;
        h_V[idx] = {static_cast<float>(batch * 100 + col + 1),
                    -static_cast<float>(row + 1)};
      }
  for (int b = 0; b < BATCHES; ++b)
    for (int k = 0; k < N; ++k)
      h_d[b * N + k] = static_cast<float>(b + 1) * (k + 1) * 0.5f;

  float2 *d_V = nullptr, *d_Vs = nullptr;
  float  *d_d = nullptr;
  CUDA_CHECK(cudaMalloc(&d_V,  sizeof(float2) * BATCHES * N * N));
  CUDA_CHECK(cudaMalloc(&d_Vs, sizeof(float2) * BATCHES * N * N));
  CUDA_CHECK(cudaMalloc(&d_d,  sizeof(float)  * BATCHES * N));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), sizeof(float2) * BATCHES * N * N,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d, h_d.data(), sizeof(float)  * BATCHES * N,
                        cudaMemcpyHostToDevice));

  scaleEigenvectorColumns(d_V, d_Vs, d_d, N, BATCHES, /*stream=*/nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_Vs.data(), d_Vs, sizeof(float2) * BATCHES * N * N,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_V);
  cudaFree(d_Vs);
  cudaFree(d_d);

  for (int batch = 0; batch < BATCHES; ++batch)
    for (int col = 0; col < N; ++col)
      for (int row = 0; row < N; ++row) {
        int idx = batch * N * N + col * N + row;
        float scale = h_d[batch * N + col];
        EXPECT_NEAR(h_Vs[idx].x, h_V[idx].x * scale, 1e-5f)
            << "real mismatch at batch=" << batch
            << " col=" << col << " row=" << row;
        EXPECT_NEAR(h_Vs[idx].y, h_V[idx].y * scale, 1e-5f)
            << "imag mismatch at batch=" << batch
            << " col=" << col << " row=" << row;
      }
}

// ---------------------------------------------------------------------------
// Edge case: all-zero input → all-zero covariance → all eigenvalues ≈ 0.
// Both modes must produce finite (not NaN/Inf) output and not crash.
// ---------------------------------------------------------------------------
// Helper lambda used by both sub-cases to avoid duplication.
static void check_zero_input_finite(bool shrink) {
  AdaptiveRun r(shrink, /*K=*/1);
  r.driver->run(
      [](size_t, size_t, int, int, int, int) -> std::complex<int8_t> {
        return {0, 0};
      },
      [](size_t, size_t, int, int, int) -> int16_t { return 1; });
  cudaDeviceSynchronize();

  test_support::assert_all_finite(*r.output->fft_output, "fft_output");
  for (size_t c = 0; c < AdaptiveConfig::NR_CHANNELS; ++c)
    for (size_t p = 0; p < AdaptiveConfig::NR_POLARIZATIONS; ++p)
      for (size_t k = 0; k < AdaptiveConfig::NR_RECEIVERS; ++k)
        EXPECT_TRUE(std::isfinite((*r.output->eigenvalues)[c][p][k]))
            << "eigenvalue NaN/Inf with zero input (shrink=" << shrink << ")";
}

TEST_F(AdaptivePipelineTest, ZeroInputNullModeFinite) {
  check_zero_input_finite(false);
}

TEST_F(AdaptivePipelineTest, ZeroInputShrinkModeFinite) {
  check_zero_input_finite(true);
}

} // namespace
