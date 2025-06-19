#pragma once
#include <vector>
#include <complex>
#include <cuda.h>
#include <iostream>
#include <pcap/pcap.h>

#include <libtcc/Correlator.h>
#include "spatial/tcc_config.h"
#define NR_TIMES_PER_BLOCK (128 / NR_BITS)
#define NR_BASELINES (NR_RECEIVERS * (NR_RECEIVERS + 1) / 2)

#include <cuda_fp16.h>

#if NR_BITS == 4
typedef complex_int4_t Sample;
typedef std::complex<int32_t> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::i4;
#elif NR_BITS == 8
typedef std::complex<int8_t> Sample;
typedef std::complex<int32_t> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::i8;
#elif NR_BITS == 16
typedef std::complex<__half> Sample;
typedef std::complex<float> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::fp16;
#endif

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

struct VisibilityFrame
{
    Visibilities data;
};
struct SampleFrame
{
    Samples data;
};

template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A);
void correlate(Samples *samples, Visibilities *visibilities);
void ccglib_mma(__half *A, __half *B, float *C, const int n_row, const int n_col, const int batch_size, int n_inner = -1);

inline void checkCudaCall(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cerr << "error " << error << std::endl;
        exit(1);
    }
}

inline void print_nonzero_visibilities(const Visibilities *vis)
{
    for (int ch = 0; ch < NR_CHANNELS; ++ch)
    {
        for (int bl = 0; bl < NR_BASELINES; ++bl)
        {
            for (int pol1 = 0; pol1 < NR_POLARIZATIONS; ++pol1)
            {
                for (int pol2 = 0; pol2 < NR_POLARIZATIONS; ++pol2)
                {
                    const Visibility v = (*vis)[ch][bl][pol1][pol2];
                    if (v.real() != 0.0f || v.imag() != 0.0f)
                    {
                        std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "][" << pol2
                                  << "] = (" << v.real() << ", " << v.imag() << ")\n";
                    }
                }
            }
        }
    }
}

inline void print_nonzero_samples(const Samples *samps)
{
    for (int ch = 0; ch < NR_CHANNELS; ++ch)
    {
        for (int j = 0; j < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; ++j)
        {
            for (int k = 0; k < NR_RECEIVERS; k++)
            {
                for (int pol = 0; pol < NR_POLARIZATIONS; ++pol)
                {
                    for (int t = 0; t < NR_TIMES_PER_BLOCK; ++t)
                    {
                        const Sample s = (*samps)[ch][j][k][pol][t];
                        if (s.real() != 0.0f || s.imag() != 0.0f)
                        {
                            std::cout << "samp[" << ch << "][" << j << "][" << k << "][" << pol << "][" << t << "] = (" << static_cast<int>(s.real()) << ", " << static_cast<int>(s.imag()) << ")\n";
                        }
                    }
                }
            }
        }
    }
}
