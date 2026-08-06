#include "spatial/spatial.cuh"
#include "support/test_configs.hpp"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <gtest/gtest.h>
#include <vector>

// ---- Helpers ----------------------------------------------------------------

static void check_cuda(cudaError_t e, const char *file, int line) {
  if (e != cudaSuccess)
    FAIL() << file << ":" << line << " CUDA error: " << cudaGetErrorString(e);
}
#define CHECK_CUDA(e) check_cuda(e, __FILE__, __LINE__)

static void check_cufft(cufftResult r, const char *file, int line) {
  if (r != CUFFT_SUCCESS)
    FAIL() << file << ":" << line << " cuFFT error " << r;
}
#define CHECK_CUFFT(r) check_cufft(r, __FILE__, __LINE__)

// Tolerance for fp16 round-trip: __half has ~3 decimal digits of precision.
static constexpr float kHalfTol = 1e-2f;

// ---- Phase-table math (CPU-only) -------------------------------------------

// Replicate the computation in set_fine_delays so we can test it independently.
static std::vector<float2>
compute_phase_table(size_t NR_CHANNELS, size_t NR_RECEIVERS, size_t N_FFT,
                    const std::vector<float> &delays_ns, double base_freq_hz,
                    double channel_bw_hz, int min_freq_ch) {
  std::vector<float2> table(NR_CHANNELS * NR_RECEIVERS * N_FFT);
  for (size_t c = 0; c < NR_CHANNELS; ++c) {
    double f_coarse =
        base_freq_hz + static_cast<double>(min_freq_ch + c) * channel_bw_hz;
    for (size_t r = 0; r < NR_RECEIVERS; ++r) {
      double tau_s = static_cast<double>(delays_ns[r]) * 1e-9;
      for (size_t k = 0; k < N_FFT; ++k) {
        long long k_signed =
            (k < N_FFT / 2) ? (long long)k : (long long)k - (long long)N_FFT;
        double f_fine =
            f_coarse + (static_cast<double>(k_signed) / N_FFT) * channel_bw_hz;
        double phi = -2.0 * M_PI * f_fine * tau_s;
        table[(c * NR_RECEIVERS + r) * N_FFT + k] = {(float)std::cos(phi),
                                                      (float)std::sin(phi)};
      }
    }
  }
  return table;
}

// ---- Test 1: Phase table correctness (CPU only) ----------------------------

TEST(FineDelay, PhaseTableMath) {
  constexpr size_t NC = 2, NR = 3, N = 64;
  const double base_f = 1.5e9;   // 1.5 GHz
  const double bw = 10e6;        // 10 MHz per coarse channel
  const int min_ch = 0;
  // 3 receivers, delays of 0, 10, -5 ns
  std::vector<float> delays = {0.0f, 10.0f, -5.0f};

  auto table = compute_phase_table(NC, NR, N, delays, base_f, bw, min_ch);

  for (size_t c = 0; c < NC; ++c) {
    double f_coarse = base_f + c * bw;
    for (size_t r = 0; r < NR; ++r) {
      double tau_s = delays[r] * 1e-9;
      for (size_t k = 0; k < N; ++k) {
        long long ks = (k < N / 2) ? (long long)k : (long long)k - (long long)N;
        double f_fine = f_coarse + (static_cast<double>(ks) / N) * bw;
        double phi_expected = -2.0 * M_PI * f_fine * tau_s;
        const float2 &v = table[(c * NR + r) * N + k];
        EXPECT_NEAR(v.x, (float)std::cos(phi_expected), 1e-6f)
            << "c=" << c << " r=" << r << " k=" << k;
        EXPECT_NEAR(v.y, (float)std::sin(phi_expected), 1e-6f)
            << "c=" << c << " r=" << r << " k=" << k;
      }
    }
  }
}

// ---- GPU fixture for scatter/gather and kernel tests -----------------------

// Use SmallSingleFPGAConfig (1 ch, 1 FPGA, 4 recv, 8 time/pkt, 1 pkt)
using Cfg = test_support::SmallSingleFPGAConfig;
// N_FFT = NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET = 1 * 8 = 8

class FineDelayGPUTest : public ::testing::Test {
protected:
  // Sizes matching SmallSingleFPGAConfig
  static constexpr size_t NC  = Cfg::NR_CHANNELS;              // 1
  static constexpr size_t NP  = Cfg::NR_PACKETS_FOR_CORRELATION; // 1
  static constexpr size_t NT  = Cfg::NR_TIME_STEPS_PER_PACKET;  // 8
  static constexpr size_t NR  = Cfg::NR_RECEIVERS;              // 4
  static constexpr size_t NPL = Cfg::NR_POLARIZATIONS;          // 2
  static constexpr size_t N   = NP * NT;                        // 8

  // Total elements in samples_aligned
  static constexpr size_t ALIGNED_ELEMS = NC * NP * NT * NR * NPL;
  // Total elements in workspace
  static constexpr size_t WORKSPACE_ELEMS = NC * NR * NPL * N;
  // Phase table elements
  static constexpr size_t PHASE_ELEMS = NC * NR * N;

  __half2 *d_aligned = nullptr;
  float2  *d_ws      = nullptr;
  float2  *d_phases  = nullptr;

  cudaStream_t stream = nullptr;
  cufftHandle fft_plan = 0;

  void SetUp() override {
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(
        cudaMalloc(&d_aligned, ALIGNED_ELEMS * sizeof(__half2)));
    CHECK_CUDA(
        cudaMalloc(&d_ws, WORKSPACE_ELEMS * sizeof(float2)));
    CHECK_CUDA(
        cudaMalloc(&d_phases, PHASE_ELEMS * sizeof(float2)));

    // cuFFT plan: batch of NC*NR*NPL forward+inverse FFTs of length N
    CHECK_CUFFT(cufftCreate(&fft_plan));
    long long fine_n[] = {(long long)N};
    long long batches = NC * NR * NPL;
    size_t ws_size = 0;
    CHECK_CUFFT(cufftXtMakePlanMany(fft_plan, 1, fine_n, nullptr, 1, (long long)N,
                                    CUDA_C_32F, nullptr, 1, (long long)N,
                                    CUDA_C_32F, batches, &ws_size, CUDA_C_32F));
    CHECK_CUFFT(cufftSetStream(fft_plan, stream));
  }

  void TearDown() override {
    if (fft_plan)   cufftDestroy(fft_plan);
    if (d_phases)   cudaFree(d_phases);
    if (d_ws)       cudaFree(d_ws);
    if (d_aligned)  cudaFree(d_aligned);
    if (stream)     cudaStreamDestroy(stream);
  }

  // Upload host __half2 buffer → d_aligned
  void upload_aligned(const std::vector<__half2> &h) {
    CHECK_CUDA(cudaMemcpyAsync(d_aligned, h.data(),
                               h.size() * sizeof(__half2),
                               cudaMemcpyHostToDevice, stream));
  }

  // Download d_aligned → host __half2 buffer
  std::vector<__half2> download_aligned() {
    std::vector<__half2> h(ALIGNED_ELEMS);
    CHECK_CUDA(cudaMemcpyAsync(h.data(), d_aligned,
                               h.size() * sizeof(__half2),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return h;
  }

  // Set d_phases to identity {1, 0}
  void set_identity_phases() {
    std::vector<float2> identity(PHASE_ELEMS, {1.0f, 0.0f});
    CHECK_CUDA(cudaMemcpyAsync(d_phases, identity.data(),
                               identity.size() * sizeof(float2),
                               cudaMemcpyHostToDevice, stream));
  }

  // Index into samples_aligned: [c][p][t][r][pol]
  static size_t aligned_idx(size_t c, size_t p, size_t t, size_t r, size_t pol) {
    return ((c * NP + p) * NT + t) * NR * NPL + r * NPL + pol;
  }
};

// ---- Test 2: scatter/gather round-trip (no FFT) ----------------------------

TEST_F(FineDelayGPUTest, ScatterGatherRoundTrip) {
  // Fill aligned with recognisable constants: real = index, imag = -index
  std::vector<__half2> h_in(ALIGNED_ELEMS);
  for (size_t i = 0; i < ALIGNED_ELEMS; ++i)
    h_in[i] = __float22half2_rn({(float)i, -(float)i});
  upload_aligned(h_in);

  // scatter → gather (skip FFT, use identity phases for no-op multiply)
  fine_delay_scatter_launch<NC, NP, NT, NR, NPL>(d_aligned, d_ws, stream);
  set_identity_phases();
  fine_delay_phase_multiply_launch<NC, NR, NPL, N>(d_ws, d_phases, stream);
  constexpr float inv_n = 1.0f;  // no normalisation (no FFT)
  fine_delay_gather_launch<NC, NP, NT, NR, NPL>(d_ws, d_aligned, inv_n, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto h_out = download_aligned();
  for (size_t i = 0; i < ALIGNED_ELEMS; ++i) {
    EXPECT_NEAR(__half2float(h_out[i].x), (float)i, kHalfTol)
        << "index " << i;
    EXPECT_NEAR(__half2float(h_out[i].y), -(float)i, kHalfTol)
        << "index " << i;
  }
}

// ---- Test 3: zero delay → identity (full scatter+FFT+multiply+IFFT+gather) -

TEST_F(FineDelayGPUTest, ZeroDelayIdentity) {
  // Fill aligned with constant complex value
  std::vector<__half2> h_in(ALIGNED_ELEMS);
  for (size_t i = 0; i < ALIGNED_ELEMS; ++i)
    h_in[i] = __float22half2_rn({3.0f, -1.5f});
  upload_aligned(h_in);

  // Zero delay → identity phasors
  set_identity_phases();

  fine_delay_scatter_launch<NC, NP, NT, NR, NPL>(d_aligned, d_ws, stream);
  CHECK_CUFFT(cufftXtExec(fft_plan, d_ws, d_ws, CUFFT_FORWARD));
  fine_delay_phase_multiply_launch<NC, NR, NPL, N>(d_ws, d_phases, stream);
  CHECK_CUFFT(cufftXtExec(fft_plan, d_ws, d_ws, CUFFT_INVERSE));
  fine_delay_gather_launch<NC, NP, NT, NR, NPL>(d_ws, d_aligned, 1.0f / N,
                                                 stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto h_out = download_aligned();
  for (size_t i = 0; i < ALIGNED_ELEMS; ++i) {
    EXPECT_NEAR(__half2float(h_out[i].x), 3.0f, kHalfTol) << "idx " << i;
    EXPECT_NEAR(__half2float(h_out[i].y), -1.5f, kHalfTol) << "idx " << i;
  }
}

// ---- Test 4: known phase shift for a single complex-tone input -------------
//
// Load one antenna's time series with a pure tone at FFT bin k0.
// After applying a correction phasor {cos(phi), sin(phi)} at bin k0 the
// output tone's phase should be shifted by phi.

TEST_F(FineDelayGPUTest, SingleToneKnownPhase) {
  constexpr size_t r_test = 0;  // test on receiver 0, pol 0
  constexpr size_t pol_test = 0;
  constexpr size_t k0 = 2;      // inject tone at bin 2

  // Build complex tone: x[t] = exp(2*pi*i*k0*t/N)
  std::vector<__half2> h_in(ALIGNED_ELEMS, __float22half2_rn({0.0f, 0.0f}));
  for (size_t p = 0; p < NP; ++p) {
    for (size_t t = 0; t < NT; ++t) {
      const size_t global_t = p * NT + t;
      const double angle = 2.0 * M_PI * k0 * global_t / N;
      h_in[aligned_idx(0, p, t, r_test, pol_test)] =
          __float22half2_rn({(float)std::cos(angle), (float)std::sin(angle)});
    }
  }
  upload_aligned(h_in);

  // Set a known phase correction at bin k0 for receiver r_test only.
  // All other bins / receivers get {1, 0}.
  const float phi_correction = (float)(M_PI / 4.0);  // +45 degrees
  std::vector<float2> h_phases(PHASE_ELEMS, {1.0f, 0.0f});
  h_phases[(0 * NR + r_test) * N + k0] = {std::cos(phi_correction),
                                           std::sin(phi_correction)};
  CHECK_CUDA(cudaMemcpyAsync(d_phases, h_phases.data(),
                             h_phases.size() * sizeof(float2),
                             cudaMemcpyHostToDevice, stream));

  fine_delay_scatter_launch<NC, NP, NT, NR, NPL>(d_aligned, d_ws, stream);
  CHECK_CUFFT(cufftXtExec(fft_plan, d_ws, d_ws, CUFFT_FORWARD));
  fine_delay_phase_multiply_launch<NC, NR, NPL, N>(d_ws, d_phases, stream);
  CHECK_CUFFT(cufftXtExec(fft_plan, d_ws, d_ws, CUFFT_INVERSE));
  fine_delay_gather_launch<NC, NP, NT, NR, NPL>(d_ws, d_aligned, 1.0f / N,
                                                 stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto h_out = download_aligned();

  // The output tone should be exp(2*pi*i*k0*t/N + phi_correction) for r_test.
  // Check the first time step: exp(i*phi_correction) = cos+i*sin.
  const float expected_re = std::cos(phi_correction);
  const float expected_im = std::sin(phi_correction);
  float got_re = __half2float(h_out[aligned_idx(0, 0, 0, r_test, pol_test)].x);
  float got_im = __half2float(h_out[aligned_idx(0, 0, 0, r_test, pol_test)].y);
  EXPECT_NEAR(got_re, expected_re, kHalfTol);
  EXPECT_NEAR(got_im, expected_im, kHalfTol);

  // Other receivers (no tone, no correction) should remain near zero.
  for (size_t r = 0; r < NR; ++r) {
    if (r == r_test) continue;
    for (size_t t = 0; t < NT; ++t) {
      EXPECT_NEAR(__half2float(
                      h_out[aligned_idx(0, 0, t, r, pol_test)].x),
                  0.0f, kHalfTol)
          << "r=" << r << " t=" << t;
      EXPECT_NEAR(__half2float(
                      h_out[aligned_idx(0, 0, t, r, pol_test)].y),
                  0.0f, kHalfTol)
          << "r=" << r << " t=" << t;
    }
  }
}
