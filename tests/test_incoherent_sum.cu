#include "spatial/spatial.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    FAIL() << "CUDA Error: " << cudaGetErrorString(err);                       \
  }

static void assert_allclose(const std::vector<float> &a,
                            const std::vector<float> &b, float tol = 1e-5f) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_NEAR(a[i], b[i], tol) << "Mismatch at index " << i;
  }
}

void incoherent_sum_cpu(const std::vector<float2> &input,
                        std::vector<float> &output, size_t C, size_t P,
                        size_t R, size_t F, size_t T) {

  for (size_t c = 0; c < C; ++c)
    for (size_t f = 0; f < F; ++f)
      for (size_t p = 0; p < P; ++p)
        for (size_t t = 0; t < T; ++t) {

          float sum = 0.0f;

          for (size_t r = 0; r < R; ++r) {
            size_t idx =
                c * P * F * T * R + f * P * T * R + p * T * R + t * R + r;

            float2 v = input[idx];
            sum += v.x * v.x + v.y * v.y;
          }

          size_t out_idx = c * F * P * T + f * P * T + p * T + t;

          output[out_idx] = sum;
        }
}

class IncoherentSumTest : public ::testing::Test {
protected:
  void run_test(size_t C, size_t P, size_t R, size_t F, size_t T) {

    size_t input_size = C * P * R * F * T;
    size_t output_size = C * P * F * T;

    std::vector<float2> h_input(input_size);
    std::vector<float> h_cpu(output_size, 0.0f);
    std::vector<float> h_gpu(output_size, 0.0f);

    // deterministic fill
    for (size_t i = 0; i < input_size; ++i) {
      h_input[i] = {float(i % 7), float((i * 3) % 11)};
    }

    incoherent_sum_cpu(h_input, h_cpu, C, P, R, F, T);

    float2 *d_input;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float2),
                          cudaMemcpyHostToDevice));

    incoherent_sum_launch(d_input, d_output, C, P, R, F, T, 0);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_output, output_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    assert_allclose(h_cpu, h_gpu);

    cudaFree(d_input);
    cudaFree(d_output);
  }
};

TEST_F(IncoherentSumTest, Minimal) { run_test(1, 1, 1, 1, 1); }

TEST_F(IncoherentSumTest, MultiReceiver) { run_test(1, 1, 8, 1, 16); }

TEST_F(IncoherentSumTest, OddReceivers) { run_test(1, 1, 7, 1, 16); }

TEST_F(IncoherentSumTest, MultiDimensional) { run_test(3, 2, 8, 4, 32); }
