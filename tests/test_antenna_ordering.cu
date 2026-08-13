// End-to-end tests verifying that stream permutation places visibility data at
// the correct canonical baseline indices in the correlator output.
//
// The wiring map assigns antenna IDs in non-hw-flat order:
//   hw_flat 0  FPGA 0 recv 0  antenna 30  (canonical 2)
//   hw_flat 1  FPGA 0 recv 1  antenna 10  (canonical 0)
//   hw_flat 2  FPGA 1 recv 0  antenna 40  (canonical 3)
//   hw_flat 3  FPGA 1 recv 1  antenna 20  (canonical 1)
//
// After build_permutation() the pipeline reorders streams so the correlator
// sees antennas in ascending-ID order: canonical 0=ant10, 1=ant20, 2=ant30,
// 3=ant40.
//
// Each pipeline test injects non-zero signal on exactly two hw-flat receivers
// and asserts that only the three baselines involving those two canonical
// receivers are non-zero and that the non-zero values match the analytically
// expected result (sum of conj(s_a)*s_b over NR_TIME_STEPS time steps).
//
// The HDF5 metadata test additionally writes the pipeline output through
// HDF5VisibilitiesWriter and verifies that antenna_ids, baseline_antenna_ids,
// and baseline_ids reflect the canonical ordering.

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/stream_antenna_map.hpp"
#include "spatial/writers.hpp"

#include "support/pipeline_harness.hpp"

#include <complex>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// 2 FPGAs × 2 receivers/packet = 4 receivers total.  Same sizing as the RC
// config in test_stream_reorder.cu — smallest layout that exercises a
// cross-FPGA stream swap.
using Config = LambdaConfig<1,     // NR_CHANNELS
                            2,     // NR_FPGA_SOURCES
                            8,     // NR_TIME_STEPS_PER_PACKET
                            4,     // NR_RECEIVERS
                            2,     // NR_POLARIZATIONS
                            2,     // NR_RECEIVERS_PER_PACKET
                            1,     // NR_PACKETS_FOR_CORRELATION
                            1,     // NR_BEAMS
                            32,    // NR_PADDED_RECEIVERS
                            32,    // NR_PADDED_RECEIVERS_PER_BLOCK
                            10000  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                            >;

// ---------------------------------------------------------------------------
// Fixed antenna wiring map
// ---------------------------------------------------------------------------
// FPGA 0 streams (stream k: recv = k/2, hw_pol = k%2):
//   stream 0 → antenna 30 pol X      stream 1 → antenna 30 pol Y
//   stream 2 → antenna 10 pol X      stream 3 → antenna 10 pol Y
// FPGA 1 streams:
//   stream 0 → antenna 40 pol X      stream 1 → antenna 40 pol Y
//   stream 2 → antenna 20 pol X      stream 3 → antenna 20 pol Y
//
// build_canonical_antenna_mapping() sorts connected antennas ascending by ID:
//   canonical 0 → antenna 10  (hw_flat 1)
//   canonical 1 → antenna 20  (hw_flat 3)
//   canonical 2 → antenna 30  (hw_flat 0)
//   canonical 3 → antenna 40  (hw_flat 2)
//
// recv_perm / pol_perm indexed by canonical_recv*NR_POL + canonical_pol:
//   canonical 0 pol 0: hw_flat=1, hw_pol=0 → recv_perm[0]=1 pol_perm[0]=0
//   canonical 0 pol 1: hw_flat=1, hw_pol=1 → recv_perm[1]=1 pol_perm[1]=1
//   canonical 1 pol 0: hw_flat=3, hw_pol=0 → recv_perm[2]=3 pol_perm[2]=0
//   canonical 1 pol 1: hw_flat=3, hw_pol=1 → recv_perm[3]=3 pol_perm[3]=1
//   canonical 2 pol 0: hw_flat=0, hw_pol=0 → recv_perm[4]=0 pol_perm[4]=0
//   canonical 2 pol 1: hw_flat=0, hw_pol=1 → recv_perm[5]=0 pol_perm[5]=1
//   canonical 3 pol 0: hw_flat=2, hw_pol=0 → recv_perm[6]=2 pol_perm[6]=0
//   canonical 3 pol 1: hw_flat=2, hw_pol=1 → recv_perm[7]=2 pol_perm[7]=1

static StreamAntennaMap make_test_map() {
  StreamAntennaMap sam;
  sam.entries[0][0] = {30, 0};
  sam.entries[0][1] = {30, 1};
  sam.entries[0][2] = {10, 0};
  sam.entries[0][3] = {10, 1};
  sam.entries[1][0] = {40, 0};
  sam.entries[1][1] = {40, 1};
  sam.entries[1][2] = {20, 0};
  sam.entries[1][3] = {20, 1};
  return sam;
}

// Same wiring as make_test_map() but with antenna 10's X/Y pol slots swapped:
//   FPGA 0 stream 2 → antenna 10 pol Y  (was X)
//   FPGA 0 stream 3 → antenna 10 pol X  (was Y)
//
// build_permutation() therefore produces a different pol_perm for canonical
// receiver 0 (antenna 10):
//   canonical 0 pol 0 (X): stream 3 → hw_flat=1, hw_pol=1 → pol_perm[0]=1
//   canonical 0 pol 1 (Y): stream 2 → hw_flat=1, hw_pol=0 → pol_perm[1]=0
// All other entries are unchanged.
static StreamAntennaMap make_pol_swap_map() {
  StreamAntennaMap sam = make_test_map();
  sam.entries[0][2] = {10, 1};  // stream 2: recv 1, hw_pol 0 → antenna 10 pol Y
  sam.entries[0][3] = {10, 0};  // stream 3: recv 1, hw_pol 1 → antenna 10 pol X
  return sam;
}

static const std::vector<int> kFpgaIds       = {0, 1};
static const int kRecvPerFpga                = static_cast<int>(Config::NR_RECEIVERS_PER_PACKET);
static const int kNrPol                      = static_cast<int>(Config::NR_POLARIZATIONS);
static const std::vector<int> kRecvPerm      = {1, 1, 3, 3, 0, 0, 2, 2};
static const std::vector<int> kPolPerm       = {0, 1, 0, 1, 0, 1, 0, 1};
// kPolPermSwap: same as kPolPerm but indices 0/1 (canonical recv 0, ant 10) swapped.
static const std::vector<int> kPolPermSwap   = {1, 0, 0, 1, 0, 1, 0, 1};

// Packed lower-triangular baseline index: i <= j.
static constexpr size_t bl(size_t i, size_t j) { return j * (j + 1) / 2 + i; }

// Integration count for one pipeline run.
static constexpr float kNT = static_cast<float>(
    Config::NR_PACKETS_FOR_CORRELATION * Config::NR_TIME_STEPS_PER_PACKET);  // = 8

using VisT = SingleHostMemoryOutput<Config>::Visibilities;

static float vis_re(const VisT &v, size_t baseline, size_t p1, size_t p2) {
  return v[0][baseline][p1][p2][0];
}
static float vis_im(const VisT &v, size_t baseline, size_t p1, size_t p2) {
  return v[0][baseline][p1][p2][1];
}

static bool baseline_near_zero(const VisT &v, size_t baseline,
                                float tol = 1.0f) {
  for (int p1 = 0; p1 < 2; ++p1)
    for (int p2 = 0; p2 < 2; ++p2)
      if (std::abs(vis_re(v, baseline, p1, p2)) > tol ||
          std::abs(vis_im(v, baseline, p1, p2)) > tol)
        return false;
  return true;
}

// ---------------------------------------------------------------------------
// Unit tests: CPU-only, no GPU needed.
// ---------------------------------------------------------------------------

TEST(AntennaOrderingMapTest, CanonicalMappingIsAscendingByAntennaID) {
  auto sam = make_test_map();
  auto mapping = sam.build_canonical_antenna_mapping(kFpgaIds, kRecvPerFpga, kNrPol);

  ASSERT_EQ(mapping.size(), 4u);
  EXPECT_EQ(mapping.at(0), 10);
  EXPECT_EQ(mapping.at(1), 20);
  EXPECT_EQ(mapping.at(2), 30);
  EXPECT_EQ(mapping.at(3), 40);
}

TEST(AntennaOrderingMapTest, PermutationArraysMatchExpectedWiring) {
  auto sam = make_test_map();
  auto [recv_perm, pol_perm] = sam.build_permutation(kFpgaIds, kRecvPerFpga, kNrPol);

  EXPECT_EQ(recv_perm, kRecvPerm);
  EXPECT_EQ(pol_perm, kPolPerm);
}

TEST(AntennaOrderingMapTest, PolSwapChangesOnlyPolPermForSwappedAntenna) {
  // The pol-swap map should produce the same recv_perm (antenna 10 is still on
  // hw_flat 1), but flip pol_perm indices 0 and 1 (canonical recv 0 pols).
  auto sam = make_pol_swap_map();
  auto [recv_perm, pol_perm] = sam.build_permutation(kFpgaIds, kRecvPerFpga, kNrPol);

  EXPECT_EQ(recv_perm, kRecvPerm);
  EXPECT_EQ(pol_perm, kPolPermSwap);
}

// ---------------------------------------------------------------------------
// Pipeline test fixture
// ---------------------------------------------------------------------------
class AntennaOrderingPipelineTest : public ::testing::Test {
protected:
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

// Run one correlation buffer with signal on exactly two hw-flat receivers.
// All other receivers carry zeros.  Returns the filled output.
static std::shared_ptr<SingleHostMemoryOutput<Config>>
run_active_pair(int hw_a, std::complex<int8_t> sig_a,
                int hw_b, std::complex<int8_t> sig_b) {
  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  auto pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION, &weights);

  pipeline->set_stream_permutation(kRecvPerm, kPolPerm);

  std::unordered_map<uint32_t, int> fpga_map = {{0, 0}, {1, 1}};
  test_support::SyntheticPipelineRun<Config> driver(*pipeline, output, {}, fpga_map);

  driver.run(
      [&](size_t /*ch*/, size_t fpga, int /*pkt*/, int /*t*/, int r,
          int /*p*/) -> std::complex<int8_t> {
        int hw_flat = static_cast<int>(fpga) * Config::NR_RECEIVERS_PER_PACKET + r;
        if (hw_flat == hw_a) return sig_a;
        if (hw_flat == hw_b) return sig_b;
        return {0, 0};
      },
      [](size_t, size_t, int, int, int) -> int16_t { return 1; });

  // dump_visibilities() fires automatically only after
  // NR_CORRELATED_BLOCKS_TO_ACCUMULATE runs; single-run tests must call it
  // manually (same pattern as test_corr_beam_pipeline.cu and
  // test_harness_selftest.cu).
  pipeline->dump_visibilities();
  cudaDeviceSynchronize();
  return output;
}

// General helper: apply arbitrary perm arrays and a per-(hw_flat, hw_pol) signal.
static std::shared_ptr<SingleHostMemoryOutput<Config>>
run_with_perm_and_fn(const std::vector<int> &recv_perm,
                     const std::vector<int> &pol_perm,
                     std::function<std::complex<int8_t>(int hw_flat, int hw_pol)> sig_fn) {
  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();
  auto weights = test_support::make_unity_beam_weights<Config>();
  auto pipeline = test_support::pipeline_factories::make_gpu_pipeline<Config>(
      Config::NR_PACKETS_FOR_CORRELATION, &weights);

  pipeline->set_stream_permutation(recv_perm, pol_perm);

  std::unordered_map<uint32_t, int> fpga_map = {{0, 0}, {1, 1}};
  test_support::SyntheticPipelineRun<Config> driver(*pipeline, output, {}, fpga_map);

  driver.run(
      [&](size_t, size_t fpga, int, int, int r, int p) -> std::complex<int8_t> {
        int hw_flat = static_cast<int>(fpga) * Config::NR_RECEIVERS_PER_PACKET + r;
        return sig_fn(hw_flat, p);
      },
      [](size_t, size_t, int, int, int) -> int16_t { return 1; });

  pipeline->dump_visibilities();
  cudaDeviceSynchronize();
  return output;
}

// ---------------------------------------------------------------------------
// Test: antennas 10 (canonical 0, hw_flat 1) and 20 (canonical 1, hw_flat 3).
// Expected non-zero baselines: bl(0,0)=0, bl(0,1)=1, bl(1,1)=2.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, ActivePair_Ant10_Ant20) {
  const int8_t sA = 4, sB = 2;  // hw_flat 1 and hw_flat 3 signals
  auto out = run_active_pair(1, {sA, 0}, 3, {sB, 0});
  const auto &vis = *out->visibilities;

  const float Vaa = kNT * sA * sA;  // autocorr ant 10 = 128
  const float Vab = kNT * sA * sB;  // cross = 64
  const float Vbb = kNT * sB * sB;  // autocorr ant 20 = 32

  for (int p1 = 0; p1 < 2; ++p1) {
    for (int p2 = 0; p2 < 2; ++p2) {
      EXPECT_NEAR(vis_re(vis, bl(0, 0), p1, p2), Vaa, 1.0f)
          << "bl(0,0) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_im(vis, bl(0, 0), p1, p2), 0.0f, 1.0f);
      EXPECT_NEAR(vis_re(vis, bl(0, 1), p1, p2), Vab, 1.0f)
          << "bl(0,1) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_im(vis, bl(0, 1), p1, p2), 0.0f, 1.0f);
      EXPECT_NEAR(vis_re(vis, bl(1, 1), p1, p2), Vbb, 1.0f)
          << "bl(1,1) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_im(vis, bl(1, 1), p1, p2), 0.0f, 1.0f);
    }
  }

  // Baselines touching inactive canonical receivers 2 (ant 30) or 3 (ant 40).
  for (size_t b : {bl(0,2), bl(1,2), bl(2,2), bl(0,3), bl(1,3), bl(2,3), bl(3,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// Test: antennas 30 (canonical 2, hw_flat 0) and 40 (canonical 3, hw_flat 2).
// Expected non-zero baselines: bl(2,2)=5, bl(2,3)=8, bl(3,3)=9.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, ActivePair_Ant30_Ant40) {
  const int8_t sA = 3, sB = 2;  // hw_flat 0 and hw_flat 2 signals
  auto out = run_active_pair(0, {sA, 0}, 2, {sB, 0});
  const auto &vis = *out->visibilities;

  const float Vaa = kNT * sA * sA;  // autocorr ant 30 = 72
  const float Vab = kNT * sA * sB;  // cross = 48
  const float Vbb = kNT * sB * sB;  // autocorr ant 40 = 32

  for (int p1 = 0; p1 < 2; ++p1) {
    for (int p2 = 0; p2 < 2; ++p2) {
      EXPECT_NEAR(vis_re(vis, bl(2, 2), p1, p2), Vaa, 1.0f)
          << "bl(2,2) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(2, 3), p1, p2), Vab, 1.0f)
          << "bl(2,3) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(3, 3), p1, p2), Vbb, 1.0f)
          << "bl(3,3) p1=" << p1 << " p2=" << p2;
    }
  }

  for (size_t b : {bl(0,0), bl(0,1), bl(1,1), bl(0,2), bl(1,2), bl(0,3), bl(1,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// Test: antennas 10 (canonical 0, hw_flat 1) and 40 (canonical 3, hw_flat 2).
// Non-adjacent canonical pair — tests the far-separated cross-correlation slot.
// Expected non-zero baselines: bl(0,0)=0, bl(0,3)=6, bl(3,3)=9.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, ActivePair_Ant10_Ant40) {
  const int8_t sA = 5, sB = 3;  // hw_flat 1 and hw_flat 2 signals
  auto out = run_active_pair(1, {sA, 0}, 2, {sB, 0});
  const auto &vis = *out->visibilities;

  const float Vaa = kNT * sA * sA;  // autocorr ant 10 = 200
  const float Vab = kNT * sA * sB;  // cross ant10×ant40 = 120
  const float Vbb = kNT * sB * sB;  // autocorr ant 40 = 72

  for (int p1 = 0; p1 < 2; ++p1) {
    for (int p2 = 0; p2 < 2; ++p2) {
      EXPECT_NEAR(vis_re(vis, bl(0, 0), p1, p2), Vaa, 1.0f)
          << "bl(0,0) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(0, 3), p1, p2), Vab, 1.0f)
          << "bl(0,3) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(3, 3), p1, p2), Vbb, 1.0f)
          << "bl(3,3) p1=" << p1 << " p2=" << p2;
    }
  }

  for (size_t b : {bl(0,1), bl(1,1), bl(0,2), bl(1,2), bl(2,2), bl(1,3), bl(2,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// Test: antennas 20 (canonical 1, hw_flat 3) and 30 (canonical 2, hw_flat 0).
// Expected non-zero baselines: bl(1,1)=2, bl(1,2)=4, bl(2,2)=5.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, ActivePair_Ant20_Ant30) {
  const int8_t sA = 2, sB = 4;  // hw_flat 3 and hw_flat 0 signals
  auto out = run_active_pair(3, {sA, 0}, 0, {sB, 0});
  const auto &vis = *out->visibilities;

  const float Vaa = kNT * sA * sA;  // autocorr ant 20 = 32
  const float Vab = kNT * sA * sB;  // cross ant20×ant30 = 64
  const float Vbb = kNT * sB * sB;  // autocorr ant 30 = 128

  for (int p1 = 0; p1 < 2; ++p1) {
    for (int p2 = 0; p2 < 2; ++p2) {
      EXPECT_NEAR(vis_re(vis, bl(1, 1), p1, p2), Vaa, 1.0f)
          << "bl(1,1) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(1, 2), p1, p2), Vab, 1.0f)
          << "bl(1,2) p1=" << p1 << " p2=" << p2;
      EXPECT_NEAR(vis_re(vis, bl(2, 2), p1, p2), Vbb, 1.0f)
          << "bl(2,2) p1=" << p1 << " p2=" << p2;
    }
  }

  for (size_t b : {bl(0,0), bl(0,1), bl(0,2), bl(0,3), bl(1,3), bl(2,3), bl(3,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// Pol-swap test 1: only antenna 10 active, X/Y wiring swapped.
//
// Antenna 10's physical wiring has canonical X on hw_pol 1 and canonical Y on
// hw_pol 0.  Signal injected: hw_pol 1 (X) = (4,0), hw_pol 0 (Y) = (2,0).
// After reordering, the correlator sees canonical X=(4,0) and canonical Y=(2,0)
// for receiver 0 (ant 10).
//
// The autocorrelation diagonal must reflect this:
//   V[bl(0,0)][0][0]  ∝ |X|² = 128    (XX)
//   V[bl(0,0)][1][1]  ∝ |Y|² = 32     (YY)
//
// If the pol swap were not applied (identity pol_perm), the canonical X slot
// would instead hold the Y signal and these values would be 32 and 128 —
// a clear regression.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, PolSwap_Autocorr_Ant10) {
  const int8_t Sx = 4, Sy = 2;
  auto [rp, pp] = make_pol_swap_map().build_permutation(kFpgaIds, kRecvPerFpga, kNrPol);

  auto out = run_with_perm_and_fn(
      rp, pp, [&](int hw_flat, int hw_pol) -> std::complex<int8_t> {
        if (hw_flat != 1) return {0, 0};
        // hw_pol 1 carries canonical X; hw_pol 0 carries canonical Y.
        return hw_pol == 1 ? std::complex<int8_t>{Sx, 0}
                           : std::complex<int8_t>{Sy, 0};
      });
  const auto &vis = *out->visibilities;

  EXPECT_NEAR(vis_re(vis, bl(0, 0), 0, 0), kNT * Sx * Sx, 1.0f);  // XX = 128
  EXPECT_NEAR(vis_re(vis, bl(0, 0), 0, 1), kNT * Sx * Sy, 1.0f);  // XY = 64
  EXPECT_NEAR(vis_re(vis, bl(0, 0), 1, 0), kNT * Sy * Sx, 1.0f);  // YX = 64
  EXPECT_NEAR(vis_re(vis, bl(0, 0), 1, 1), kNT * Sy * Sy, 1.0f);  // YY = 32
  EXPECT_NEAR(vis_im(vis, bl(0, 0), 0, 0), 0.0f, 1.0f);
  EXPECT_NEAR(vis_im(vis, bl(0, 0), 1, 1), 0.0f, 1.0f);

  for (size_t b : {bl(0,1), bl(1,1), bl(0,2), bl(1,2), bl(2,2),
                   bl(0,3), bl(1,3), bl(2,3), bl(3,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// Pol-swap test 2: antenna 10 (pol swapped) and antenna 20 (normal) active.
//
// Canonical pol assignment after reorder:
//   ant 10: pol 0 = Sx=(4,0), pol 1 = Sy=(2,0)
//   ant 20: pol 0 = S2x=(3,0), pol 1 = S2y=(5,0)  (normal — no swap)
//
// Cross-correlation V[bl(0,1)][p1][p2] = kNT * ant10_pol_p1 * ant20_pol_p2:
//   [0][0] = 8*4*3 = 96    [0][1] = 8*4*5 = 160
//   [1][0] = 8*2*3 = 48    [1][1] = 8*2*5 = 80
//
// With wrong pol_perm (identity), Sx and Sy for ant 10 would be swapped,
// changing [0][0] to 48 and [1][0] to 96 — caught by the EXPECT_NEAR below.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, PolSwap_CrossCorr_Ant10_Ant20) {
  const int8_t Sx = 4, Sy = 2;       // ant 10: canonical X/Y signals
  const int8_t S2x = 3, S2y = 5;    // ant 20: canonical X/Y signals (normal wiring)
  auto [rp, pp] = make_pol_swap_map().build_permutation(kFpgaIds, kRecvPerFpga, kNrPol);

  auto out = run_with_perm_and_fn(
      rp, pp, [&](int hw_flat, int hw_pol) -> std::complex<int8_t> {
        if (hw_flat == 1) {
          // ant 10 pol swap: hw_pol 1 = canonical X, hw_pol 0 = canonical Y
          return hw_pol == 1 ? std::complex<int8_t>{Sx, 0}
                             : std::complex<int8_t>{Sy, 0};
        }
        if (hw_flat == 3) {
          // ant 20 normal: hw_pol 0 = canonical X, hw_pol 1 = canonical Y
          return hw_pol == 0 ? std::complex<int8_t>{S2x, 0}
                             : std::complex<int8_t>{S2y, 0};
        }
        return {0, 0};
      });
  const auto &vis = *out->visibilities;

  // Autocorrelations — diagonal pols reflect canonical X/Y power.
  EXPECT_NEAR(vis_re(vis, bl(0, 0), 0, 0), kNT * Sx * Sx, 1.0f);   // ant10 XX = 128
  EXPECT_NEAR(vis_re(vis, bl(0, 0), 1, 1), kNT * Sy * Sy, 1.0f);   // ant10 YY = 32
  EXPECT_NEAR(vis_re(vis, bl(1, 1), 0, 0), kNT * S2x * S2x, 1.0f); // ant20 XX = 72
  EXPECT_NEAR(vis_re(vis, bl(1, 1), 1, 1), kNT * S2y * S2y, 1.0f); // ant20 YY = 200

  // Cross-correlation bl(0,1): all four pol combinations.
  EXPECT_NEAR(vis_re(vis, bl(0, 1), 0, 0), kNT * Sx * S2x, 1.0f);  // XxX = 96
  EXPECT_NEAR(vis_re(vis, bl(0, 1), 0, 1), kNT * Sx * S2y, 1.0f);  // XxY = 160
  EXPECT_NEAR(vis_re(vis, bl(0, 1), 1, 0), kNT * Sy * S2x, 1.0f);  // YxX = 48
  EXPECT_NEAR(vis_re(vis, bl(0, 1), 1, 1), kNT * Sy * S2y, 1.0f);  // YxY = 80
  EXPECT_NEAR(vis_im(vis, bl(0, 1), 0, 0), 0.0f, 1.0f);
  EXPECT_NEAR(vis_im(vis, bl(0, 1), 1, 0), 0.0f, 1.0f);

  for (size_t b : {bl(0,2), bl(1,2), bl(2,2), bl(0,3), bl(1,3), bl(2,3), bl(3,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "inactive baseline " << b;
}

// ---------------------------------------------------------------------------
// HDF5 metadata test: run the pipeline, write through HDF5VisibilitiesWriter
// with the canonical_antenna_mapping, and verify antenna_ids,
// baseline_antenna_ids, and baseline_ids reflect the correct ascending-ID
// ordering regardless of physical wiring.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, HDF5MetadataMatchesCanonicalOrdering) {
  auto out = run_active_pair(1, {2, 0}, 0, {3, 0});  // any two active

  auto mapping = make_test_map().build_canonical_antenna_mapping(
      kFpgaIds, kRecvPerFpga, kNrPol);

  auto tmp = fs::temp_directory_path() / "test_ant_ord_XXXXXX.h5";
  std::string fname = tmp.string();
  int fd = mkstemps(fname.data(), 3);
  if (fd >= 0) close(fd);

  {
    HighFive::File file(fname, HighFive::File::Truncate);
    HDF5VisibilitiesWriter<Config::VisibilitiesOutputType> writer(
        file, 0, 0, &mapping);

    size_t blk = writer.register_block(0, 8, 0, 8);
    void *land = writer.get_visibilities_landing_pointer(blk);
    std::memcpy(land, out->visibilities, sizeof(Config::VisibilitiesOutputType));
    writer.register_visibilities_transfer_complete(blk);
    writer.drain_ready_blocks();
    writer.flush();
  }

  HighFive::File verify(fname, HighFive::File::ReadOnly);

  // antenna_ids[canonical_idx] must be in ascending antenna-ID order.
  std::vector<int> antenna_ids;
  verify.getDataSet("antenna_ids").read(antenna_ids);
  ASSERT_EQ(antenna_ids.size(), 4u);
  EXPECT_EQ(antenna_ids[0], 10);
  EXPECT_EQ(antenna_ids[1], 20);
  EXPECT_EQ(antenna_ids[2], 30);
  EXPECT_EQ(antenna_ids[3], 40);

  // baseline_antenna_ids[baseline, 0/1] follow packed-triangular baseline order.
  constexpr size_t NR_BL = Config::NR_BASELINES_UNPADDED;  // = 10
  std::vector<int> bl_ants(2 * NR_BL);
  verify.getDataSet("baseline_antenna_ids").read_raw(bl_ants.data());

  // bl(0,0)=0: [ant10, ant10]
  EXPECT_EQ(bl_ants[0], 10);
  EXPECT_EQ(bl_ants[1], 10);
  // bl(0,1)=1: [ant10, ant20]
  EXPECT_EQ(bl_ants[2], 10);
  EXPECT_EQ(bl_ants[3], 20);
  // bl(1,1)=2: [ant20, ant20]
  EXPECT_EQ(bl_ants[4], 20);
  EXPECT_EQ(bl_ants[5], 20);
  // bl(0,2)=3: [ant10, ant30]
  EXPECT_EQ(bl_ants[6], 10);
  EXPECT_EQ(bl_ants[7], 30);
  // bl(1,2)=4: [ant20, ant30]
  EXPECT_EQ(bl_ants[8], 20);
  EXPECT_EQ(bl_ants[9], 30);
  // bl(2,2)=5: [ant30, ant30]
  EXPECT_EQ(bl_ants[10], 30);
  EXPECT_EQ(bl_ants[11], 30);
  // bl(0,3)=6: [ant10, ant40]
  EXPECT_EQ(bl_ants[12], 10);
  EXPECT_EQ(bl_ants[13], 40);
  // bl(1,3)=7: [ant20, ant40]
  EXPECT_EQ(bl_ants[14], 20);
  EXPECT_EQ(bl_ants[15], 40);
  // bl(2,3)=8: [ant30, ant40]
  EXPECT_EQ(bl_ants[16], 30);
  EXPECT_EQ(bl_ants[17], 40);
  // bl(3,3)=9: [ant40, ant40]
  EXPECT_EQ(bl_ants[18], 40);
  EXPECT_EQ(bl_ants[19], 40);

  // baseline_ids encoding: ant1*256 + ant2.
  std::vector<int> baseline_ids;
  verify.getDataSet("baseline_ids").read(baseline_ids);
  ASSERT_EQ(baseline_ids.size(), NR_BL);
  EXPECT_EQ(baseline_ids[0], 10 * 256 + 10);  // bl(0,0)
  EXPECT_EQ(baseline_ids[1], 10 * 256 + 20);  // bl(0,1)
  EXPECT_EQ(baseline_ids[2], 20 * 256 + 20);  // bl(1,1)
  EXPECT_EQ(baseline_ids[3], 10 * 256 + 30);  // bl(0,2)
  EXPECT_EQ(baseline_ids[4], 20 * 256 + 30);  // bl(1,2)
  EXPECT_EQ(baseline_ids[5], 30 * 256 + 30);  // bl(2,2)
  EXPECT_EQ(baseline_ids[6], 10 * 256 + 40);  // bl(0,3)
  EXPECT_EQ(baseline_ids[7], 20 * 256 + 40);  // bl(1,3)
  EXPECT_EQ(baseline_ids[8], 30 * 256 + 40);  // bl(2,3)
  EXPECT_EQ(baseline_ids[9], 40 * 256 + 40);  // bl(3,3)

  fs::remove(fname);
}

// ---------------------------------------------------------------------------
// Real LAMBDA-36 antenna IDs: baseline placement and HDF5 round-trip
//
// Uses actual LAMBDA-36 antenna IDs: 1 and 35 (both on FPGA 0), 15 and 16
// (both on FPGA 1).  The StreamAntennaMap places ant 35 at FPGA 0 recv 0 and
// ant 1 at FPGA 0 recv 1; ant 15 at FPGA 1 recv 0 and ant 16 at FPGA 1
// recv 1.
//
// build_permutation() sorts ascending by antenna ID:
//   canonical 0 → ant  1 (FPGA 0 recv 1, hw_flat 1)
//   canonical 1 → ant 15 (FPGA 1 recv 0, hw_flat 2)
//   canonical 2 → ant 16 (FPGA 1 recv 1, hw_flat 3)
//   canonical 3 → ant 35 (FPGA 0 recv 0, hw_flat 0)
//
// ant 35 ends up at canonical slot 3 even though it is at hw_flat 0 — the
// permutation reverses the hardware order relative to the ID-sorted order.
//
// Signal is injected only on hw_flat 0 (ant 35).  The only non-zero baseline
// must be bl(3,3) (index 9), with the analytically expected autocorrelation
// value.  The HDF5 output must carry the real antenna IDs in ascending order
// and report that visibility value correctly.
// ---------------------------------------------------------------------------
TEST_F(AntennaOrderingPipelineTest, RealLambdaAntennaIDs_CorrectBaselineAndHDF5) {
  StreamAntennaMap sam;
  // FPGA 0: ant 35 at recv 0, ant 1 at recv 1
  sam.entries[0][0] = {35, 0};
  sam.entries[0][1] = {35, 1};
  sam.entries[0][2] = {1, 0};
  sam.entries[0][3] = {1, 1};
  // FPGA 1: ant 15 at recv 0, ant 16 at recv 1
  sam.entries[1][0] = {15, 0};
  sam.entries[1][1] = {15, 1};
  sam.entries[1][2] = {16, 0};
  sam.entries[1][3] = {16, 1};

  auto [recv_perm, pol_perm] =
      sam.build_permutation(kFpgaIds, kRecvPerFpga, kNrPol);

  // Verify canonical ordering is ID-sorted, not hardware-order.
  // canonical 0=ant1  from hw_flat 1, canonical 3=ant35 from hw_flat 0.
  ASSERT_EQ(recv_perm, (std::vector<int>{1, 1, 2, 2, 3, 3, 0, 0}));
  ASSERT_EQ(pol_perm,  (std::vector<int>{0, 1, 0, 1, 0, 1, 0, 1}));

  // Signal only on hw_flat 0 (ant 35, both pols).
  const int8_t s = 3;
  auto out = run_with_perm_and_fn(
      recv_perm, pol_perm,
      [&](int hw_flat, int /*hw_pol*/) -> std::complex<int8_t> {
        return hw_flat == 0 ? std::complex<int8_t>{s, 0}
                            : std::complex<int8_t>{0, 0};
      });

  const auto &vis = *out->visibilities;

  // bl(3,3) = ant35 autocorr = kNT * s^2.  All other baselines touching
  // canonical slots 0/1/2 (ant1/15/16, which carry no signal) must be zero.
  const float Vant35 = kNT * s * s;
  for (int p1 = 0; p1 < 2; ++p1)
    for (int p2 = 0; p2 < 2; ++p2)
      EXPECT_NEAR(vis_re(vis, bl(3, 3), p1, p2), Vant35, 1.0f)
          << "bl(3,3) p1=" << p1 << " p2=" << p2;

  for (size_t b : {bl(0,0), bl(0,1), bl(1,1), bl(0,2), bl(1,2), bl(2,2),
                   bl(0,3), bl(1,3), bl(2,3)})
    EXPECT_TRUE(baseline_near_zero(vis, b)) << "expected zero baseline " << b;

  // HDF5 round-trip: metadata AND visibility values.
  auto mapping =
      sam.build_canonical_antenna_mapping(kFpgaIds, kRecvPerFpga, kNrPol);

  auto tmp = fs::temp_directory_path() / "test_real_lambda_XXXXXX.h5";
  std::string fname = tmp.string();
  int fd = mkstemps(fname.data(), 3);
  if (fd >= 0) close(fd);

  {
    HighFive::File file(fname, HighFive::File::Truncate);
    HDF5VisibilitiesWriter<Config::VisibilitiesOutputType> writer(
        file, 0, 0, &mapping);
    size_t blk = writer.register_block(0, 8, 0, 8);
    void *land = writer.get_visibilities_landing_pointer(blk);
    std::memcpy(land, out->visibilities, sizeof(Config::VisibilitiesOutputType));
    writer.register_visibilities_transfer_complete(blk);
    writer.drain_ready_blocks();
    writer.flush();
  }

  HighFive::File verify(fname, HighFive::File::ReadOnly);

  // Antenna IDs must reflect real LAMBDA IDs in ascending order.
  std::vector<int> antenna_ids;
  verify.getDataSet("antenna_ids").read(antenna_ids);
  ASSERT_EQ(antenna_ids.size(), 4u);
  EXPECT_EQ(antenna_ids[0], 1);
  EXPECT_EQ(antenna_ids[1], 15);
  EXPECT_EQ(antenna_ids[2], 16);
  EXPECT_EQ(antenna_ids[3], 35);

  // baseline_ids encoding: 256*ant1 + ant2 in packed-triangular order.
  constexpr size_t NR_BL = Config::NR_BASELINES_UNPADDED;  // 10
  std::vector<int> baseline_ids;
  verify.getDataSet("baseline_ids").read(baseline_ids);
  ASSERT_EQ(baseline_ids.size(), NR_BL);
  EXPECT_EQ(baseline_ids[bl(0, 0)], 256 * 1  + 1);   //   257
  EXPECT_EQ(baseline_ids[bl(0, 1)], 256 * 1  + 15);  //   271
  EXPECT_EQ(baseline_ids[bl(1, 1)], 256 * 15 + 15);  //  3855
  EXPECT_EQ(baseline_ids[bl(0, 2)], 256 * 1  + 16);  //   272
  EXPECT_EQ(baseline_ids[bl(1, 2)], 256 * 15 + 16);  //  3856
  EXPECT_EQ(baseline_ids[bl(2, 2)], 256 * 16 + 16);  //  4112
  EXPECT_EQ(baseline_ids[bl(0, 3)], 256 * 1  + 35);  //   291
  EXPECT_EQ(baseline_ids[bl(1, 3)], 256 * 15 + 35);  //  3875
  EXPECT_EQ(baseline_ids[bl(2, 3)], 256 * 16 + 35);  //  4131
  EXPECT_EQ(baseline_ids[bl(3, 3)], 256 * 35 + 35);  //  8995

  // Verify the actual visibility value at bl(3,3) survived the HDF5 round-trip.
  // Layout: float[NR_CHANNELS][NR_BASELINES_UNPADDED][NR_POL][NR_POL][COMPLEX]
  //       = float[1][10][2][2][2]   →   80 floats per block.
  constexpr size_t stride_bl = 2 * 2 * 2;  // NR_POL * NR_POL * COMPLEX
  std::vector<float> stored_vis(NR_BL * stride_bl);
  verify.getDataSet("visibilities").read_raw(stored_vis.data());

  // bl(3,3) XX re: offset = bl(3,3) * stride_bl = 9 * 8 = 72
  EXPECT_NEAR(stored_vis[bl(3, 3) * stride_bl], Vant35, 1.0f)
      << "visibility value at bl(3,3) did not survive HDF5 round-trip";
  // All other baselines must be near-zero in the stored file.
  for (size_t b = 0; b < NR_BL; ++b) {
    if (b == bl(3, 3)) continue;
    for (size_t k = 0; k < stride_bl; ++k)
      EXPECT_NEAR(stored_vis[b * stride_bl + k], 0.0f, 1.0f)
          << "unexpected non-zero at baseline " << b << " component " << k;
  }

  fs::remove(fname);
}

} // namespace
