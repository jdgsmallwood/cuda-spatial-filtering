// Exercises pre-correlation fine channelization wired into LambdaBeamformedSpectraPipeline --
// unlike LambdaAntennaSpectraPipeline (test_fine_channelized_antenna_spectra.cu), this pipeline
// beamforms first and keeps its FFT/downsample output path (retargeted, not deleted): once
// channelized, gpu-filter's fine channels replace the whole-band cuFFT's frequency decomposition,
// so cufftXtExec is skipped and detect_and_downsample_fft_launch runs directly on the
// (now per-fine-channel) beamformer output.
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

using Config = test_support::SmallFineChannelizedBeamformedSpectraConfig;

static_assert(Config::NR_FPGA_CHANNELS == 1, "test assumes 1 coarse channel");
static_assert(Config::NR_FINE_CHANNELS == 8, "test assumes 8 fine channels");
static_assert(Config::NR_FINE_CHANNEL_EDGE_TRIM == 1, "test assumes a 1-per-side edge trim");
static_assert(Config::NR_CHANNELS == 6,
             "widened channel count should be coarse * (fine - 2*edge_trim)");

// LambdaBeamformedSpectraPipeline's FFTOutputType is a private class member type, so this
// mirrors its shape from public Config-level constants only (the same pattern
// SingleHostMemoryOutput<T>::BeamOutput already uses for LambdaGPUPipeline's beam output) --
// float[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS][NR_TIME_STEPS_PER_FINE_CHANNEL/FFT_DOWNSAMPLE_FACTOR].
template <typename T> class BeamformedFFTHostOutput : public Output {
public:
  using FFTOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_FINE_CHANNEL / T::FFT_DOWNSAMPLE_FACTOR];
  FFTOutput *fft_output;

  BeamformedFFTHostOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&fft_output, sizeof(*fft_output)));
  }
  ~BeamformedFFTHostOutput() { CUDA_CHECK(cudaFreeHost(fft_output)); }

  size_t register_beam_data_block(size_t, size_t) override { return 1; }
  size_t register_visibilities_block(size_t, size_t, int, int) override { return 1; }
  size_t register_eigendecomposition_data_block(size_t, size_t) override { return 1; }
  size_t register_fft_block(size_t, size_t) override { return 1; }

  void *get_beam_data_landing_pointer(size_t) override { return nullptr; }
  void *get_visibilities_landing_pointer(size_t) override { return nullptr; }
  void *get_arrivals_data_landing_pointer(size_t) override { return nullptr; }
  void *get_eigenvalues_data_landing_pointer(size_t) override { return nullptr; }
  void *get_eigenvectors_data_landing_pointer(size_t) override { return nullptr; }
  void *get_nulled_eigenmode_counts_landing_pointer(size_t) override { return nullptr; }
  void *get_beam_counts_landing_pointer(size_t) override { return nullptr; }
  void *get_fft_landing_pointer(size_t) override { return (void *)fft_output; }

  void register_beam_data_transfer_complete(size_t) override {}
  void register_visibilities_transfer_complete(size_t) override {}
  void register_arrivals_transfer_complete(size_t) override {}
  void register_eigendecomposition_data_transfer_complete(size_t) override {}
  void register_beam_counts_transfer_complete(size_t) override {}
  void register_fft_transfer_complete(size_t) override {}
};

class FineChannelizedBeamformedSpectraTest : public ::testing::Test {
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
  auto output = std::make_shared<BeamformedFFTHostOutput<ConfigT>>();
  BeamWeightsT<ConfigT> weights = test_support::make_unity_beam_weights<ConfigT>();
  auto pipeline =
      test_support::pipeline_factories::make_beamformed_spectra_pipeline<ConfigT>(
          ConfigT::NR_PACKETS_FOR_CORRELATION, &weights);
  test_support::SyntheticPipelineRun<ConfigT> driver(*pipeline, output);

  driver.run(constant_sample, constant_scale);
  cudaDeviceSynchronize();

  using Element =
      std::remove_all_extents_t<typename BeamformedFFTHostOutput<ConfigT>::FFTOutput>;
  constexpr size_t n =
      sizeof(typename BeamformedFFTHostOutput<ConfigT>::FFTOutput) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(output->fft_output);
  bool any_nonzero = false;
  for (size_t i = 0; i < n; ++i) {
    ASSERT_TRUE(std::isfinite(flat[i])) << "element " << i << " is not finite";
    ASSERT_GE(flat[i], 0.0f) << "element " << i << ": power must be non-negative";
    if (flat[i] > 0.0f)
      any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero) << "constant input should produce some nonzero power somewhere";
}

} // namespace

TEST_F(FineChannelizedBeamformedSpectraTest, SatisfiesPhysicalInvariants) {
  run_and_check_physical_invariants<Config>();
}

// Regression guard: the original whole-band-FFT path (NR_FINE_CHANNELS == 1) must keep working
// unmodified alongside the new channelizer wiring added in the same pipeline.
TEST_F(FineChannelizedBeamformedSpectraTest, DisabledConfigSatisfiesPhysicalInvariants) {
  using UnchannelizedConfig = test_support::SmallBeamformedSpectraConfig;
  static_assert(UnchannelizedConfig::NR_FINE_CHANNELS == 1);
  run_and_check_physical_invariants<UnchannelizedConfig>();
}
