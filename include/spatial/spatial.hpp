#pragma once
#include <vector>
#include <complex>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

#include "spatial/tcc_config.h"
#define NR_BITS 16
#define NR_TIMES_PER_BLOCK  (128 / NR_BITS)
#define NR_BASELINES  (NR_RECEIVERS * (NR_RECEIVERS + 1) / 2)

typedef std::complex<__half> Sample;
typedef std::complex<float> Visibility;

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
int add(int a, int b);
void incrementArray(int *data, int size);
template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A);
void correlate(Samples *samples, Visibilities *visibilities);


inline void checkCudaCall(cudaError_t error)
{
    if (error != cudaSuccess)
    {
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
                        std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "][" << pol2
                                  << "] = (" << v.real() << ", " << v.imag() << ")\n";
                    }
                }
            }
        }
    }
}