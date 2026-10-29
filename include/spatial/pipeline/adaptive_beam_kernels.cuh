#pragma once

template <typename Complex>
__global__ void mask_adaptive_matrix(Complex *matrix, const uint8_t *enabled,
                                     int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.y;
  if (i < n * n && (!enabled[batch * n + i % n] || !enabled[batch * n + i / n]))
    matrix[batch * n * n + i] = Complex{};
}

// Beam weights are row vectors; covariance filters are column-major matrices.
template <typename T>
__global__ void apply_coarse_beam_filter(const __half2 *weights,
                                         const __half2 *filters,
                                         __half2 *output) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr size_t N = T::NR_RECEIVERS;
  if (i >= T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_BEAMS * N)
    return;
  const size_t r = i % N;
  const size_t beam = i / N % T::NR_BEAMS;
  const size_t pol = i / N / T::NR_BEAMS % T::NR_POLARIZATIONS;
  const size_t channel = i / N / T::NR_BEAMS / T::NR_POLARIZATIONS;
  const size_t coarse = channel / (T::NR_CHANNELS / T::NR_FPGA_CHANNELS);
  const size_t batch = coarse * T::NR_POLARIZATIONS + pol;
  float2 sum = make_float2(0, 0);
  for (size_t q = 0; q < N; ++q) {
    const auto w = __half22float2(
        weights[((channel * T::NR_POLARIZATIONS + pol) * T::NR_BEAMS + beam) *
                    N +
                q]);
    const auto f = __half22float2(filters[batch * N * N + r * N + q]);
    sum.x += w.x * f.x - w.y * f.y;
    sum.y += w.x * f.y + w.y * f.x;
  }
  output[i] = __floats2half2_rn(sum.x, sum.y);
}
