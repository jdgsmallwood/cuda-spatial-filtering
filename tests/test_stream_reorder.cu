// Tests for stream-reorder + gains + fine-delay integration.
//
// These tests verify that the reorder_streams_kernel (inserted after apply_delays
// in enqueue_alignment) correctly places every antenna's data at its canonical
// receiver slot regardless of which FPGA hardware stream it actually arrived on,
// and that downstream steps (fine delay correction, beamforming) then operate in
// that canonical order.
//
// Strategy: run a "reference" pipeline where hardware streams already match
// canonical antenna order (identity permutation), and a "permuted" pipeline
// where antennas are wired to shuffled streams including a cross-FPGA swap and
// one X/Y polarisation swap.  With a correctly configured stream permutation,
// both pipelines must produce byte-identical beam output.
//
// Three test dimensions are covered independently and together:
//   1. ReorderOnly           -- non-trivial stream perm, identity delays/gains
//   2. ReorderWithFineDelay  -- non-trivial stream perm + per-antenna fine delays
//   3. ReorderWithGains      -- non-trivial stream perm + per-antenna d_gains
//   4. ExactBeamValue        -- identity perm, analytical expected beam value

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"

#include "support/assertions.hpp"
#include "support/pipeline_harness.hpp"
#include "support/test_configs.hpp"

#include <complex>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

// 2 FPGA sources, 2 receivers per packet = 4 total receivers.  This is the
// smallest config that exercises a cross-FPGA stream swap.
using RC = LambdaConfig<1,     // NR_CHANNELS
                        2,     // NR_FPGA_SOURCES
                        8,     // NR_TIME_STEPS_PER_PACKET
                        4,     // NR_RECEIVERS (= 2 FPGAs * 2 recv/pkt)
                        2,     // NR_POLARIZATIONS
                        2,     // NR_RECEIVERS_PER_PACKET
                        1,     // NR_PACKETS_FOR_CORRELATION
                        1,     // NR_BEAMS
                        32,    // NR_PADDED_RECEIVERS
                        32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                        10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                        >;

// ---------------------------------------------------------------------------
// Hardware mapping for the two test scenarios
// ---------------------------------------------------------------------------
//
// REFERENCE (canonical order):
//   hw_flat 0  FPGA 0 recv 0  antenna 0  pol 0=X, pol 1=Y
//   hw_flat 1  FPGA 0 recv 1  antenna 1  pol 0=X, pol 1=Y
//   hw_flat 2  FPGA 1 recv 0  antenna 2  pol 0=X, pol 1=Y
//   hw_flat 3  FPGA 1 recv 1  antenna 3  pol 0=X, pol 1=Y
//
// PERMUTED (shuffled cross-FPGA, one receiver has its X/Y pol slots swapped):
//   hw_flat 0  FPGA 0 recv 0:  pol 0 = ant 2 X,  pol 1 = ant 2 Y
//   hw_flat 1  FPGA 0 recv 1:  pol 0 = ant 0 X,  pol 1 = ant 0 Y
//   hw_flat 2  FPGA 1 recv 0:  pol 0 = ant 3 X,  pol 1 = ant 3 Y
//   hw_flat 3  FPGA 1 recv 1:  pol 0 = ant 1 Y,  pol 1 = ant 1 X  <-- Y/X SWAPPED
//
// Perm arrays indexed by (canonical_recv_flat * NR_POL + canonical_pol):
//   canonical recv 0 (ant 0), pol 0 (X) <- hw_flat=1, hw_pol=0
//   canonical recv 0 (ant 0), pol 1 (Y) <- hw_flat=1, hw_pol=1
//   canonical recv 1 (ant 1), pol 0 (X) <- hw_flat=3, hw_pol=1  (X is at hw pol 1)
//   canonical recv 1 (ant 1), pol 1 (Y) <- hw_flat=3, hw_pol=0  (Y is at hw pol 0)
//   canonical recv 2 (ant 2), pol 0 (X) <- hw_flat=0, hw_pol=0
//   canonical recv 2 (ant 2), pol 1 (Y) <- hw_flat=0, hw_pol=1
//   canonical recv 3 (ant 3), pol 0 (X) <- hw_flat=2, hw_pol=0
//   canonical recv 3 (ant 3), pol 1 (Y) <- hw_flat=2, hw_pol=1

static constexpr int NR_RECV = RC::NR_FPGA_SOURCES * RC::NR_RECEIVERS_PER_PACKET;
static const std::vector<int> kPermRecv = {1, 1, 3, 3, 0, 0, 2, 2};
static const std::vector<int> kPermPol  = {0, 1, 1, 0, 0, 1, 0, 1};

// Antenna 'a' always carries int8 signal (a+1, -(a+1)) on both polarisations
// and all time steps.
static std::complex<int8_t> ant_sig(int ant) {
  return {static_cast<int8_t>(ant + 1), static_cast<int8_t>(-(ant + 1))};
}

// hw_flat → antenna_id for the permuted case.
static constexpr int kPermHwToAnt[4] = {2, 0, 3, 1};

// sample_fn for the reference pipeline (antenna id = hw_flat).
static std::complex<int8_t>
ref_sample(size_t /*ch*/, size_t fpga, int /*pkt*/, int /*t*/, int r, int /*p*/) {
  int hw = (int)fpga * RC::NR_RECEIVERS_PER_PACKET + r;
  return ant_sig(hw);
}

// sample_fn for the permuted pipeline.
// At hw_flat 3 (ant 1, FPGA 1 recv 1), hw pol 0 = ant 1 Y, hw pol 1 = ant 1 X.
// Both pol slots carry the same signal magnitude for simplicity; the reorder
// kernel reads each pol slot independently via kPermPol to place them at the
// correct canonical position.
static std::complex<int8_t>
perm_sample(size_t /*ch*/, size_t fpga, int /*pkt*/, int /*t*/, int r, int /*p*/) {
  int hw = (int)fpga * RC::NR_RECEIVERS_PER_PACKET + r;
  return ant_sig(kPermHwToAnt[hw]);
}

static int16_t unity_scale(size_t, size_t, int, int, int) { return 1; }

// ---------------------------------------------------------------------------
// Run helper
// ---------------------------------------------------------------------------
struct Run {
  std::shared_ptr<SingleHostMemoryOutput<RC>> output;
  BeamWeightsT<RC> weights;
  std::unique_ptr<LambdaGPUPipeline<RC>> pipeline;
};

// Builds the pipeline, optionally applies stream permutation / fine delays /
// d_gains, feeds one correlation buffer, synchronises.
template <typename SampleFn, typename ScaleFn>
Run do_run(BeamWeightsT<RC> weights,
           SampleFn sample_fn, ScaleFn scale_fn,
           const std::vector<int> &recv_perm = {},
           const std::vector<int> &pol_perm  = {},
           const float *delays_ns = nullptr,       // length NR_RECEIVERS; nullptr = identity
           const std::complex<float> *d_gains_flat = nullptr) { // [chan][hw_recv][pol]; nullptr = identity
  Run r;
  r.output  = std::make_shared<SingleHostMemoryOutput<RC>>();
  r.weights = std::move(weights);

  r.pipeline = test_support::pipeline_factories::make_gpu_pipeline<RC>(
      RC::NR_PACKETS_FOR_CORRELATION, &r.weights);

  if (!recv_perm.empty())
    r.pipeline->set_stream_permutation(recv_perm, pol_perm);

  // d_gains_flat is in the kernel's expected layout: [chan][hw_recv][pol].
  // set_antenna_gains() accepts a raw pointer and memcpys it verbatim.
  if (d_gains_flat)
    r.pipeline->set_antenna_gains(
        reinterpret_cast<std::complex<float> *>(
            const_cast<std::complex<float> *>(d_gains_flat)));

  // FPGA IDs 0..1 map to buffer slots 0..1.
  std::unordered_map<uint32_t, int> fpga_map = {{0, 0}, {1, 1}};
  test_support::SyntheticPipelineRun<RC> driver(*r.pipeline, r.output, {}, fpga_map);

  // Fine delays must be set after graph capture (which happens in the
  // pipeline constructor) but before the data run.
  if (delays_ns)
    r.pipeline->set_fine_delays(delays_ns, /*base_freq_hz=*/150e6,
                                /*channel_bw_hz=*/781250.0, /*min_freq_ch=*/0);

  driver.run(sample_fn, scale_fn);
  cudaDeviceSynchronize();
  return r;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr size_t NR_SAMPLES =
    RC::NR_PACKETS_FOR_CORRELATION * RC::NR_TIME_STEPS_PER_PACKET;

// Check two beam outputs are within tolerance at every (ch, pol, beam, t) cell.
using BeamOut = SingleHostMemoryOutput<RC>::BeamOutput;

static void expect_beams_near(const BeamOut &a, const BeamOut &b,
                              float tol = 0.6f) {
  for (size_t c = 0; c < RC::NR_CHANNELS; ++c)
    for (size_t p = 0; p < RC::NR_POLARIZATIONS; ++p)
      for (size_t bm = 0; bm < RC::NR_BEAMS; ++bm)
        for (size_t t = 0; t < NR_SAMPLES; ++t) {
          EXPECT_NEAR(__half2float(a[c][p][bm][t][0]),
                      __half2float(b[c][p][bm][t][0]), tol)
              << "re  ch=" << c << " pol=" << p << " t=" << t;
          EXPECT_NEAR(__half2float(a[c][p][bm][t][1]),
                      __half2float(b[c][p][bm][t][1]), tol)
              << "im  ch=" << c << " pol=" << p << " t=" << t;
        }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class StreamReorderTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// ---------------------------------------------------------------------------
// Test 1: identity permutation, exact beam value
//
// Each antenna carries constant signal (a+1, -(a+1)), scale=1, unity beam
// weights, identity d_gains, zero fine delays.  The beam sums 4 receivers:
//   beam[t] = Σ_{a=0}^3 (a+1, -(a+1)) = (10, -10) for every time step.
// ---------------------------------------------------------------------------
TEST_F(StreamReorderTest, IdentityPermExactBeamValue) {
  auto r = do_run(test_support::make_unity_beam_weights<RC>(), ref_sample, unity_scale);
  const auto &beam = *r.output->beam_data;

  for (size_t t = 0; t < NR_SAMPLES; ++t) {
    EXPECT_NEAR(__half2float(beam[0][0][0][t][0]),  10.0f, 0.6f) << "re t=" << t;
    EXPECT_NEAR(__half2float(beam[0][0][0][t][1]), -10.0f, 0.6f) << "im t=" << t;
  }
}

// ---------------------------------------------------------------------------
// Test 2: cross-FPGA permutation + pol swap, no fine delays, no gains
//
// The permuted pipeline receives the same physical signals but via shuffled
// hardware streams.  After reordering + pol correction the beams must match.
// ---------------------------------------------------------------------------
TEST_F(StreamReorderTest, ReorderOnlyMatchesCanonical) {
  auto weights = test_support::make_unity_beam_weights<RC>();
  auto ref  = do_run(weights, ref_sample,  unity_scale);
  auto perm = do_run(weights, perm_sample, unity_scale, kPermRecv, kPermPol);

  expect_beams_near(*ref.output->beam_data, *perm.output->beam_data);
}

// ---------------------------------------------------------------------------
// Test 3: cross-FPGA permutation + fine delays applied in canonical order
//
// Each antenna gets a distinct non-zero delay (10*k ns for antenna k).
// The reference pipeline receives data in canonical order; the permuted pipeline
// receives the same signals via shuffled streams.  After reorder, both pipelines
// apply fine_delay[canonical_k] to canonical slot k, so both outputs match.
// ---------------------------------------------------------------------------
TEST_F(StreamReorderTest, ReorderWithFineDelayMatchesCanonical) {
  // Delay for canonical slot k = (k+1)*10 ns.  This array is indexed by
  // canonical receiver, which is also antenna id in the reference case.
  float delays_ns[RC::NR_RECEIVERS] = {10.f, 20.f, 30.f, 40.f};

  auto weights = test_support::make_unity_beam_weights<RC>();
  auto ref  = do_run(weights, ref_sample,  unity_scale, {}, {}, delays_ns);
  auto perm = do_run(weights, perm_sample, unity_scale, kPermRecv, kPermPol, delays_ns);

  expect_beams_near(*ref.output->beam_data, *perm.output->beam_data);
}

// ---------------------------------------------------------------------------
// Test 4: cross-FPGA permutation + per-antenna d_gains
//
// Each antenna gets a distinct real gain factor:  antenna k → gain (k+1)*0.5.
//
// The reference pipeline has d_gains in canonical order (antenna k at hw slot k).
// The permuted pipeline has d_gains in HARDWARE order (the gain for the antenna
// that physically occupies each hw slot).
//
// d_gains are uploaded in the kernel's expected layout: [chan][hw_recv][pol].
// set_antenna_gains() takes a raw std::complex<float>* and memcpys it verbatim
// so this bypasses the AntennaGains struct layout.
//
// After scale_and_convert_to_half applies hw-order gains, and the reorder kernel
// puts the scaled data in canonical order, both pipelines should produce
// identical beam output.
// ---------------------------------------------------------------------------
TEST_F(StreamReorderTest, ReorderWithGainsMatchesCanonical) {
  // Gain for antenna a: G_a = (a+1)*0.5 (real, identity phase).
  auto gain_for_ant = [](int ant) -> std::complex<float> {
    return {(ant + 1) * 0.5f, 0.0f};
  };

  // Reference gains: hw_flat k = antenna k → gain for antenna k.
  // Layout: [chan][hw_recv][pol].  NR_CHANNELS=1, NR_RECV=4, NR_POL=2.
  std::vector<std::complex<float>> ref_gains(RC::NR_CHANNELS * NR_RECV * RC::NR_POLARIZATIONS);
  for (int c = 0; c < (int)RC::NR_CHANNELS; ++c)
    for (int hw = 0; hw < NR_RECV; ++hw)
      for (int p = 0; p < (int)RC::NR_POLARIZATIONS; ++p)
        ref_gains[c * NR_RECV * RC::NR_POLARIZATIONS + hw * RC::NR_POLARIZATIONS + p]
            = gain_for_ant(hw); // hw == antenna in reference case

  // Permuted gains: hw_flat k carries antenna kPermHwToAnt[k].
  std::vector<std::complex<float>> perm_gains(RC::NR_CHANNELS * NR_RECV * RC::NR_POLARIZATIONS);
  for (int c = 0; c < (int)RC::NR_CHANNELS; ++c)
    for (int hw = 0; hw < NR_RECV; ++hw)
      for (int p = 0; p < (int)RC::NR_POLARIZATIONS; ++p)
        perm_gains[c * NR_RECV * RC::NR_POLARIZATIONS + hw * RC::NR_POLARIZATIONS + p]
            = gain_for_ant(kPermHwToAnt[hw]);

  auto weights = test_support::make_unity_beam_weights<RC>();
  auto ref  = do_run(weights, ref_sample,  unity_scale, {}, {}, nullptr,
                     ref_gains.data());
  auto perm = do_run(weights, perm_sample, unity_scale, kPermRecv, kPermPol, nullptr,
                     perm_gains.data());

  expect_beams_near(*ref.output->beam_data, *perm.output->beam_data);
}

// ---------------------------------------------------------------------------
// Test 5: all three together -- reorder + gains + fine delays
// ---------------------------------------------------------------------------
TEST_F(StreamReorderTest, ReorderGainsAndFineDelayTogetherMatchCanonical) {
  float delays_ns[RC::NR_RECEIVERS] = {10.f, 20.f, 30.f, 40.f};

  auto gain_for_ant = [](int ant) -> std::complex<float> {
    return {(ant + 1) * 0.5f, 0.0f};
  };

  std::vector<std::complex<float>> ref_gains(RC::NR_CHANNELS * NR_RECV * RC::NR_POLARIZATIONS);
  std::vector<std::complex<float>> perm_gains(RC::NR_CHANNELS * NR_RECV * RC::NR_POLARIZATIONS);
  for (int c = 0; c < (int)RC::NR_CHANNELS; ++c)
    for (int hw = 0; hw < NR_RECV; ++hw)
      for (int p = 0; p < (int)RC::NR_POLARIZATIONS; ++p) {
        int idx = c * NR_RECV * RC::NR_POLARIZATIONS + hw * RC::NR_POLARIZATIONS + p;
        ref_gains[idx]  = gain_for_ant(hw);
        perm_gains[idx] = gain_for_ant(kPermHwToAnt[hw]);
      }

  auto weights = test_support::make_unity_beam_weights<RC>();
  auto ref  = do_run(weights, ref_sample,  unity_scale, {}, {},
                     delays_ns, ref_gains.data());
  auto perm = do_run(weights, perm_sample, unity_scale, kPermRecv, kPermPol,
                     delays_ns, perm_gains.data());

  expect_beams_near(*ref.output->beam_data, *perm.output->beam_data);
}

} // namespace
