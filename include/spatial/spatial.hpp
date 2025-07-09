#pragma once
#include <complex>
#include <cuda.h>
#include <iostream>

#include "spatial/tcc_config.h"
#include <libtcc/Correlator.h>
#define NR_TIMES_PER_BLOCK (128 / NR_BITS)
#define NR_BASELINES (NR_RECEIVERS * (NR_RECEIVERS + 1) / 2)

#include <cuda_fp16.h>

#if NR_BITS == 4
typedef complex_int4_t Sample;
typedef std::complex<int32_t> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::i4;
#define CAST_TO_FLOAT(x) (x)
#elif NR_BITS == 8
typedef std::complex<int8_t> Sample;
typedef std::complex<int32_t> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::i8;
#define CAST_TO_FLOAT(x) (x)
#elif NR_BITS == 16
typedef std::complex<__half> Sample;
typedef std::complex<float> Visibility;
#define CAST_TO_FLOAT(x) __half2float(x)
constexpr tcc::Format inputFormat = tcc::Format::fp16;
#endif

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK]
                      [NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS]
                               [NR_POLARIZATIONS];

typedef std::complex<float> BeamformedData[NR_CHANNELS][NR_POLARIZATIONS]
                                          [NR_SAMPLES_PER_CHANNEL];

typedef int8_t Tin;
typedef int16_t Tscale;
template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A);
template <typename T>
void d_eigendecomposition(float *d_eigenvalues, const int n, T *d_A,
                          cudaStream_t stream);
void correlate(Samples *samples, Visibilities *visibilities);
void ccglib_mma(__half *A, __half *B, float *C, const int n_row,
                const int n_col, const int batch_size, int n_inner = -1);

void ccglib_mma_opt(__half *A, __half *B, float *C, const int n_row,
                    const int n_col, const int batch_size, int n_inner,
                    const int tile_size_x, const int tile_size_y);
template <typename T, typename S>
void beamform(std::complex<T> *data_matrix, std::complex<T> *weights,
              std::complex<S> *output_matrix, const int n_antennas,
              const int n_samples, const int n_beams);

template <typename T>
void rearrange_matrix_to_ccglib_format(const std::complex<T> *input_matrix,
                                       T *output_matrix, const int n_rows,
                                       const int n_cols,
                                       const bool row_major = true);

template <typename T>
void rearrange_ccglib_matrix_to_compact_format(const T *input_matrix,
                                               std::complex<T> *output_matrix,
                                               const int n_rows,
                                               const int n_cols);

inline void checkCudaCall(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}

inline void print_nonzero_visibilities(const Visibilities *vis) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int bl = 0; bl < NR_BASELINES; ++bl) {
      for (int pol1 = 0; pol1 < NR_POLARIZATIONS; ++pol1) {
        for (int pol2 = 0; pol2 < NR_POLARIZATIONS; ++pol2) {
          const Visibility v = (*vis)[ch][bl][pol1][pol2];
          if (v.real() != 0.0f || v.imag() != 0.0f) {
            std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "]["
                      << pol2 << "] = (" << v.real() << ", " << v.imag()
                      << ")\n";
          }
        }
      }
    }
  }
}

inline void print_nonzero_visibilities(const Visibilities *vis,
                                       const Tscale *scales) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int bl = 0; bl < NR_BASELINES; ++bl) {
      for (int pol1 = 0; pol1 < NR_POLARIZATIONS; ++pol1) {
        for (int pol2 = 0; pol2 < NR_POLARIZATIONS; ++pol2) {
          const Visibility v = (*vis)[ch][bl][pol1][pol2];
          if (v.real() != 0.0f || v.imag() != 0.0f) {
            std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "]["
                      << pol2 << "] = (" << v.real() * scales[bl] << ", "
                      << v.imag() * scales[bl] << ") where scale is "
                      << scales[bl] << "\n";
          }
        }
      }
    }
  }
}

inline void print_nonzero_samples(const Samples *samps) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int j = 0; j < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; ++j) {
      for (int k = 0; k < NR_RECEIVERS; k++) {
        for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
          for (int t = 0; t < NR_TIMES_PER_BLOCK; ++t) {
            const Sample s = (*samps)[ch][j][k][pol][t];
            if (CAST_TO_FLOAT(s.real()) != 0.0f ||
                CAST_TO_FLOAT(s.imag()) != 0.0f) {
              std::cout << "samp[" << ch << "][" << j << "][" << k << "]["
                        << pol << "][" << t << "] = ("
                        << static_cast<int>(s.real()) << ", "
                        << static_cast<int>(s.imag()) << ")\n";
            }
          }
        }
      }
    }
  }
}
