#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

  const int num_blocks = std::min(8, n / 1024 + 1);

  convert_int8_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                               d_output, n);
}

void convert_int_to_float(const int *d_input, float *d_output, const int n,
                          cudaStream_t stream) {

  const int num_blocks = std::min(8, n / 1024 + 1);

  convert_int_to_float_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                               d_output, n);
}

void convert_float_to_half(const float *d_input, __half *d_output, const int n,
                           cudaStream_t stream) {

  const int num_blocks = std::min(16, n / 1024 + 1);

  convert_float_to_half_kernel<<<num_blocks, 1024, 0, stream>>>(d_input,
                                                                d_output, n);
}

void update_weights(const __half *d_weights, __half *d_weights_output,
                    const int num_beams, const int num_receivers,
                    const int num_channels, const int num_polarizations,
                    const float *d_eigenvalues, float *d_eigenvectors,
                    cudaStream_t &stream) {

  const int n = num_beams * num_receivers * num_channels * num_polarizations;
  const int num_blocks = std::min(8, n / 1024 + 1);

  update_weights_kernel<<<num_blocks, 1024, 0, stream>>>(
      d_weights, d_weights_output, num_beams, num_receivers, num_channels,
      num_polarizations);
}

void accumulate_visibilities(const float *d_visibilities,
                             float *d_visibilities_accumulated, const int n,
                             cudaStream_t stream) {

  const int num_blocks = std::min(8, n / 1024 + 1);

  accumulate_visibilities_kernel<<<num_blocks, 1024, 0, stream>>>(
      d_visibilities, d_visibilities_accumulated, n);
}
