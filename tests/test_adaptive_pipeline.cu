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

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
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
  using FFTOut = float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
                      [NR_FINE_CHANS];
  using ArrivalsT = bool[T::NR_CHANNELS][T::NR_PACKETS_FOR_CORRELATION + 2]
                        [T::NR_FPGA_SOURCES];

  BeamOut    *beam_data;
  EigVals    *eigenvalues;
  EigVecs    *eigenvectors;
  FFTOut     *fft_output;
  ArrivalsT  *arrivals;

  AdaptiveTestOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOut)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvalues, sizeof(EigVals)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvectors, sizeof(EigVecs)));
    CUDA_CHECK(cudaMallocHost((void **)&fft_output, sizeof(FFTOut)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(ArrivalsT)));
  }
  ~AdaptiveTestOutput() {
    cudaFreeHost(beam_data);
    cudaFreeHost(eigenvalues);
    cudaFreeHost(eigenvectors);
    cudaFreeHost(fft_output);
    cudaFreeHost(arrivals);
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

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class AdaptivePipelineTest : public ::testing::Test {
protected:
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

// ---------------------------------------------------------------------------
// Run helper
// ---------------------------------------------------------------------------
struct AdaptiveRun {
  std::shared_ptr<AdaptiveTestOutput<AdaptiveConfig>> output;
  BeamWeightsT<AdaptiveConfig> weights;
  std::unique_ptr<LambdaAdaptiveBeamformedSpectraPipeline<AdaptiveConfig>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<AdaptiveConfig>> driver;

  explicit AdaptiveRun(bool shrink, int K = 1) {
    output = std::make_shared<AdaptiveTestOutput<AdaptiveConfig>>();
    weights = test_support::make_unity_beam_weights<AdaptiveConfig>();
    std::unordered_map<int, int> eig_map{{0, K}};
    pipeline = test_support::pipeline_factories::make_adaptive_beamformed_spectra_pipeline<AdaptiveConfig>(
        /*num_buffers=*/3, &weights, eig_map, shrink);
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
