// Self-test for the shared pipeline-test harness (SyntheticPipelineRun et al).
//
// It wires SyntheticPipelineRun<SmallSingleFPGAConfig> to a real
// LambdaGPUPipeline and feeds it the exact same synthetic input
// test_pipeline.cu::Ex1 hand-pokes into DummyFinalPacketData -- but through
// the real ProcessorState ingestion seam instead. Reproducing Ex1's
// hand-derived expected values (beam_data == (8, -8), visibilities == (64, 0))
// end to end is the proof that the harness is wired correctly; it also
// demonstrates both testing styles the harness is meant to support: exact-value
// goldens (where the math is simple enough to hand-derive) and property-based
// invariant checks (the "high level, not implementation-coupled" style for
// everything else).
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/synthetic_packets.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>

namespace {

using Config = test_support::SmallSingleFPGAConfig;

class PipelineHarnessSelfTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// Constant (2, -2) sample / scale-1 input across every channel, FPGA, packet
// slot, time step, receiver and polarization -- the exact values Ex1 fills
// DummyFinalPacketData with by hand. With unity beam weights and
// NR_RECEIVERS == 4, Ex1 hand-derived beam_data == (8, -8) and visibilities ==
// (64, 0) for every index from these inputs.
std::complex<int8_t> constant_sample(size_t /*channel*/, size_t /*fpga*/,
                                     int /*packet*/, int /*time*/,
                                     int /*receiver*/, int /*polarization*/) {
  return std::complex<int8_t>(2, -2);
}

int16_t constant_scale(size_t /*channel*/, size_t /*fpga*/, int /*packet*/,
                       int /*receiver*/, int /*polarization*/) {
  return 1;
}

struct HarnessRun {
  std::shared_ptr<SingleHostMemoryOutput<Config>> output;
  BeamWeightsT<Config> weights;
  std::unique_ptr<LambdaGPUPipeline<Config>> pipeline;
  std::unique_ptr<test_support::SyntheticPipelineRun<Config>> driver;
};

HarnessRun run_ex1_equivalent() {
  HarnessRun run;
  run.output = std::make_shared<SingleHostMemoryOutput<Config>>();
  run.weights = test_support::make_unity_beam_weights<Config>();
  run.pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION, &run.weights);
  run.driver = std::make_unique<test_support::SyntheticPipelineRun<Config>>(
      *run.pipeline, run.output);

  run.driver->run(constant_sample, constant_scale);
  run.pipeline->dump_visibilities();
  cudaDeviceSynchronize();
  return run;
}

} // namespace

TEST_F(PipelineHarnessSelfTest, ReproducesEx1ExactValues) {
  HarnessRun run = run_ex1_equivalent();
  const auto &beam_data = *run.output->beam_data;
  const auto &visibilities = *run.output->visibilities;

  constexpr size_t nr_samples =
      Config::NR_PACKETS_FOR_CORRELATION * Config::NR_TIME_STEPS_PER_PACKET;

  for (size_t c = 0; c < Config::NR_CHANNELS; ++c) {
    for (size_t pol_a = 0; pol_a < Config::NR_POLARIZATIONS; ++pol_a) {
      for (size_t beam = 0; beam < Config::NR_BEAMS; ++beam) {
        for (size_t t = 0; t < nr_samples; ++t) {
          EXPECT_EQ(__half2float(beam_data[c][pol_a][beam][t][0]), 8.0f)
              << "beam_data real mismatch at channel=" << c << " pol=" << pol_a
              << " beam=" << beam << " sample=" << t;
          EXPECT_EQ(__half2float(beam_data[c][pol_a][beam][t][1]), -8.0f)
              << "beam_data imag mismatch at channel=" << c << " pol=" << pol_a
              << " beam=" << beam << " sample=" << t;
        }
      }

      for (size_t pol_b = 0; pol_b < Config::NR_POLARIZATIONS; ++pol_b) {
        for (size_t baseline = 0; baseline < Config::NR_BASELINES_UNPADDED; ++baseline) {
          EXPECT_EQ(visibilities[c][baseline][pol_a][pol_b][0], 64.0f)
              << "visibilities real mismatch at channel=" << c
              << " baseline=" << baseline << " pol_a=" << pol_a
              << " pol_b=" << pol_b;
          EXPECT_EQ(visibilities[c][baseline][pol_a][pol_b][1], 0.0f)
              << "visibilities imag mismatch at channel=" << c
              << " baseline=" << baseline << " pol_a=" << pol_a
              << " pol_b=" << pol_b;
        }
      }
    }
  }
}

TEST_F(PipelineHarnessSelfTest, SatisfiesPhysicalInvariants) {
  HarnessRun run = run_ex1_equivalent();

  test_support::assert_all_finite(*run.output->beam_data, "beam_data");
  test_support::assert_all_finite(*run.output->visibilities, "visibilities");
  test_support::assert_autocorrelation_invariants<Config>(*run.output->visibilities);
}
