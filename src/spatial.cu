#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Grid size for grid-stride elementwise kernels: enough blocks to cover n,
// capped at 32 blocks/SM so huge n doesn't oversubscribe the scheduler.  The
// previous fixed cap of 8-16 blocks left these bandwidth-bound kernels running
// on a small fraction of the SMs for multi-megabyte arrays.
static int elementwise_grid_size(int n, int block_size = 1024) {
  static const int max_blocks = [] {
    int device = 0;
    int sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms > 0 ? 32 * sms : 2048;
  }();
  return std::max(1, std::min((n + block_size - 1) / block_size, max_blocks));
}

__global__ void convert_int8_to_half_kernel(const int8_t *d_input,
                                            __half *d_output, const int n) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  while (idx < n) {
    // Promote int8_t to int before conversion
    int val = static_cast<int>(d_input[idx]);
    d_output[idx] = __int2half_rn(val);
    idx += stride;
  }
}

__global__ void
update_weights_kernel(const __half *d_weights, __half *d_weights_output,
                      const int num_beams, const int num_receivers,
                      const int num_channels, const int num_polarizations) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int n =
      num_beams * num_receivers * num_channels * num_polarizations * 2;

  while (idx < n) {
    d_weights_output[idx] = d_weights[idx];
    idx += stride;
  }
}

__global__ void convert_int_to_float_kernel(const int *d_input, float *d_output,
                                            const int n) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  while (idx < n) {
    d_output[idx] = __int2float_rn(d_input[idx]);
    idx += stride;
  }
}

__global__ void
accumulate_visibilities_kernel(const float *d_visibilities,
                               float *d_visibilities_accumulated, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  while (idx < n) {
    atomicAdd(&d_visibilities_accumulated[idx], d_visibilities[idx]);
    idx += stride;
  }
}

__global__ void convert_float_to_half_kernel(const float *input, __half *output,
                                             const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  while (i < n) {
    output[i] = __float2half(input[i]);
    i += stride;
  }
}

void convert_int8_to_half(const int8_t *d_input, __half *d_output, const int n,
                          cudaStream_t stream) {

  const int num_blocks = elementwise_grid_size(n);

  convert_int8_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                               d_output, n);
}

void convert_int_to_float(const int *d_input, float *d_output, const int n,
                          cudaStream_t stream) {

  const int num_blocks = elementwise_grid_size(n);

  convert_int_to_float_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                               d_output, n);
}

void convert_float_to_half(const float *d_input, __half *d_output, const int n,
                           cudaStream_t stream) {

  const int num_blocks = elementwise_grid_size(n);

  convert_float_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                                d_output, n);
}

void update_weights(const __half *d_weights, __half *d_weights_output,
                    const int num_beams, const int num_receivers,
                    const int num_channels, const int num_polarizations,
                    const float *d_eigenvalues, float *d_eigenvectors,
                    cudaStream_t &stream) {

  // The kernel iterates over n * 2 elements (real+imag) internally.
  const int n =
      num_beams * num_receivers * num_channels * num_polarizations * 2;
  const int num_blocks = elementwise_grid_size(n);

  update_weights_kernel<<<num_blocks, 1024, 0, stream>>>(
      d_weights, d_weights_output, num_beams, num_receivers, num_channels,
      num_polarizations);
}

void accumulate_visibilities(const float *d_visibilities,
                             float *d_visibilities_accumulated, const int n,
                             cudaStream_t stream) {

  const int num_blocks = elementwise_grid_size(n);

  accumulate_visibilities_kernel<<<num_blocks, 1024, 0, stream>>>(
      d_visibilities, d_visibilities_accumulated, n);
}

// Fuses visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed +
// accumulate_visibilities into a single kernel.
//
// TCC output:  float[n_ch][n_bl][pol][pol][2]  (CorrelatorOutput)
// Accumulator: float[n_ch][n_up][pol][pol][2]  (TrimmedVisibilities, n_up ≤ n_bl)
//
// For baseline a < n_up: accum[c*n_up*S + a*S + inner] += corr[c*n_bl*S + a*S + inner]
// where S = inner_stride = pol*pol*2.  The full four-step chain collapses to
// this identity mapping because the net permutation of (visCorrToBaseline ∘
// visBaselineTrimmedToTrimmed) is the identity for elements where a < n_up.
__global__ void accumulate_visibilities_from_corr_kernel(
    const float *corr_out, float *accum, const int n_ch, const int n_bl,
    const int n_up, const int inner_stride) {
  const int total = n_ch * n_up * inner_stride;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = blockDim.x * gridDim.x;
  while (idx < total) {
    const int inner = idx % inner_stride;
    const int ca = idx / inner_stride;
    const int a = ca % n_up;
    const int c = ca / n_up;
    atomicAdd(&accum[idx], corr_out[(c * n_bl + a) * inner_stride + inner]);
    idx += grid_stride;
  }
}

void accumulate_visibilities_from_corr(const float *corr_out, float *accum,
                                       int n_channels, int n_baselines,
                                       int n_unpadded, int inner_stride,
                                       cudaStream_t stream) {
  const int total = n_channels * n_unpadded * inner_stride;
  const int num_blocks = elementwise_grid_size(total);
  accumulate_visibilities_from_corr_kernel<<<num_blocks, 1024, 0, stream>>>(
      corr_out, accum, n_channels, n_baselines, n_unpadded, inner_stride);
}
