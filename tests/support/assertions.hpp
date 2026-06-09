#pragma once

#include "spatial/packet_formats.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <type_traits>

// Property/invariant-based assertions for pipeline output buffers.
//
// These check physical and mathematical truths that must hold regardless of
// *how* the GPU computes them (correlator internals, normalization constants,
// kernel launch shapes, ...) -- the goal is for these checks to keep passing
// across the planned pipeline.hpp refactor, rather than being tied to today's
// exact intermediate values.
//
// Exact-value golden checks (e.g. test_pipeline.cu::Ex1's expected 8.0f/64.0f
// results) are intentionally NOT generalized here: they only make sense for
// LambdaGPUPipeline, the one variant simple enough to hand-derive expected
// numbers for. For the rest (adaptive/eigen-projection/FFT/folding), invariant
// checks like the ones below are the right tool.
namespace test_support {

// The Tensor Core Correlator stores visibilities in packed lower-triangular
// form: baseline_index(i, j) = j*(j+1)/2 + i, valid only for i <= j (see
// storeVisibility in extern/tcc/libtcc/kernel/TCCorrelator.cu). Only that
// triangle (including the diagonal, i.e. autocorrelations) is physically
// present -- there is no separate stored slot for the conjugate pair (j, i)
// when j > i.
inline constexpr size_t baseline_index(size_t i, size_t j) {
  return (j * (j + 1)) / 2 + i;
}

namespace detail {

inline bool is_finite_scalar(float value) { return std::isfinite(value); }
inline bool is_finite_scalar(double value) { return std::isfinite(value); }
inline bool is_finite_scalar(__half value) {
  return std::isfinite(__half2float(value));
}
inline bool is_finite_scalar(const std::complex<float> &value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

template <typename Element>
::testing::AssertionResult all_finite(const Element *data, size_t n,
                                      const std::string &name) {
  for (size_t i = 0; i < n; ++i) {
    if (!is_finite_scalar(data[i])) {
      return ::testing::AssertionFailure()
             << name << "[" << i << "] is not finite";
    }
  }
  return ::testing::AssertionSuccess();
}

} // namespace detail

// Checks that every element of a (possibly multi-dimensional) C-array output
// buffer is finite -- catches uninitialized memory, bad FFT plans,
// divide-by-zero in normalization, NaNs leaking out of eigendecomposition, etc.
template <typename Array>
void assert_all_finite(const Array &array, const std::string &name) {
  using Element = std::remove_all_extents_t<Array>;
  constexpr size_t n = sizeof(Array) / sizeof(Element);
  const Element *flat = reinterpret_cast<const Element *>(&array);
  ASSERT_TRUE(detail::all_finite(flat, n, name));
}

// Checks the physical invariants of a per-receiver autocorrelation that hold
// for *any* correlator: same-polarization autocorrelations are total-power
// measurements (real, non-negative), and the per-receiver polarization
// covariance matrix V[r][r][p][q] is Hermitian (V[p][q] == conj(V[q][p])).
//
// This is the part of "Hermitian visibilities" that survives the correlator's
// packed lower-triangular baseline storage: cross-baseline conjugate pairs
// V[baseline(j, i)] for j > i simply aren't stored, so they can't be compared
// against V[baseline(i, j)] -- but every receiver's own polarization matrix is
// square and must still be Hermitian PSD.
template <typename Config>
void assert_autocorrelation_invariants(
    const typename Config::VisibilitiesOutputType &visibilities,
    float tolerance = 1e-3f) {
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c) {
    for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
      const size_t b = baseline_index(r, r);
      ASSERT_LT(b, Config::NR_BASELINES_UNPADDED);

      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
        const float real = visibilities[c][b][p][p][0];
        const float imag = visibilities[c][b][p][p][1];
        ASSERT_NEAR(imag, 0.0f, tolerance)
            << "channel " << c << " receiver " << r << " pol " << p
            << ": same-polarization autocorrelation must be real (got imag="
            << imag << ")";
        ASSERT_GE(real, -tolerance)
            << "channel " << c << " receiver " << r << " pol " << p
            << ": same-polarization autocorrelation power must be "
               "non-negative (got "
            << real << ")";
      }

      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
        for (size_t q = p + 1; q < Config::NR_POLARIZATIONS; ++q) {
          const std::complex<float> v_pq(visibilities[c][b][p][q][0],
                                         visibilities[c][b][p][q][1]);
          const std::complex<float> v_qp(visibilities[c][b][q][p][0],
                                         visibilities[c][b][q][p][1]);
          ASSERT_NEAR(v_pq.real(), v_qp.real(), tolerance)
              << "channel " << c << " receiver " << r << ": V[" << p << "][" << q
              << "] and conj(V[" << q << "][" << p
              << "]) real parts disagree -- autocorrelation polarization "
                 "matrix isn't Hermitian";
          ASSERT_NEAR(v_pq.imag(), -v_qp.imag(), tolerance)
              << "channel " << c << " receiver " << r << ": V[" << p << "][" << q
              << "] and conj(V[" << q << "][" << p
              << "]) imag parts disagree -- autocorrelation polarization "
                 "matrix isn't Hermitian";
        }
      }
    }
  }
}

// Checks cuSOLVER's CUSOLVER_EIG_MODE_VECTOR contract for
// cusolverDnXsyevBatched: eigenvalues come back in ascending order. They are
// additionally non-negative for the same-polarization (p == q) blocks, since
// those are genuine per-polarization receiver-covariance matrices and
// therefore Hermitian positive-semi-definite; cross-polarization (p != q)
// blocks aren't generally PSD so non-negativity isn't asserted there.
template <typename Config>
void assert_eigenvalues_ascending_nonnegative(
    const typename Config::EigenvalueOutputType &eigenvalues,
    float tolerance = 1e-4f) {
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c) {
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
      for (size_t q = 0; q < Config::NR_POLARIZATIONS; ++q) {
        float previous = -std::numeric_limits<float>::infinity();
        for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
          const float value = eigenvalues[c][p][q][r];
          ASSERT_GE(value, previous - tolerance)
              << "channel " << c << " pol-pair (" << p << "," << q
              << ") eigenvalue[" << r << "]=" << value
              << " breaks ascending order (previous=" << previous << ")";
          if (p == q) {
            ASSERT_GE(value, -tolerance)
                << "channel " << c << " polarization " << p << " eigenvalue["
                << r << "]=" << value << " is negative";
          }
          previous = value;
        }
      }
    }
  }
}

// Finds the dominant frequency bin in `fft_output[channel][polarization][beam]`
// and asserts it lands within `tolerance_bins` of `expected_bin`. Feeding a
// known-frequency tone and checking where the energy lands is robust to
// scaling/normalization-constant changes -- it validates "the FFT pipeline
// does an FFT" without coupling to specific output magnitudes.
template <typename Config>
void assert_tone_detected(const typename Config::FFTOutputType &fft_output,
                          size_t channel, size_t polarization, size_t beam,
                          size_t expected_bin, size_t tolerance_bins = 1) {
  constexpr size_t nr_bins = std::extent<typename Config::FFTOutputType, 3>::value;
  static_assert(nr_bins > 0, "FFTOutputType must have a non-empty bin dimension");

  size_t peak_bin = 0;
  float peak_value = -std::numeric_limits<float>::infinity();
  for (size_t bin = 0; bin < nr_bins; ++bin) {
    const float value = fft_output[channel][polarization][beam][bin];
    if (value > peak_value) {
      peak_value = value;
      peak_bin = bin;
    }
  }

  const size_t distance =
      peak_bin > expected_bin ? peak_bin - expected_bin : expected_bin - peak_bin;
  ASSERT_LE(distance, tolerance_bins)
      << "channel " << channel << " polarization " << polarization << " beam "
      << beam << ": expected FFT peak within " << tolerance_bins
      << " bin(s) of " << expected_bin << ", but the peak (value=" << peak_value
      << ") is at bin " << peak_bin;
}

} // namespace test_support
