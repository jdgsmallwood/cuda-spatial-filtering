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

// Regression guard: the original (non-channelized) correlate+eigen+beamform+FFT path must keep
// working unmodified alongside the new channelizer wiring added in the same pipeline.
TEST_F(FineChannelizedAdaptiveTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallAdaptiveConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}
