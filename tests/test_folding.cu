
#include "spatial/spatial.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
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
    std::cout << "a was " << a[i] << " and b was " << b[i] << std::endl;
  }
}

void fold_and_accumulate_cpu(const std::vector<float> &incoherent_sum,
                             std::vector<float> &fold_accumulator,
                             std::vector<uint32_t> &hit_counts,
                             const std::vector<int32_t> &dm_delays,
                             int64_t total_samples_elapsed,
                             double period_samples, int n_bins, int nr_channels,
                             int nr_fine_channels, int nr_polarizations,
                             int nr_time_bins) {

  for (int c = 0; c < nr_channels; ++c)
    for (int f = 0; f < nr_fine_channels; ++f)
      for (int p = 0; p < nr_polarizations; ++p)
        for (int t = 0; t < nr_time_bins; ++t) {

          int cf_idx = c * nr_fine_channels + f;
          int32_t delay = dm_delays[cf_idx];

          int64_t abs_sample = total_samples_elapsed + (int64_t)t - delay;

          double phase = fmod((double)abs_sample / period_samples, 1.0);
          if (phase < 0.0)
            phase += 1.0;

          int bin = (int)(phase * n_bins);

          int in_idx = ((c * nr_fine_channels + f) * nr_polarizations + p) *
                           nr_time_bins +
                       t;

          int out_idx =
              ((c * nr_fine_channels + f) * nr_polarizations + p) * n_bins +
              bin;

          fold_accumulator[out_idx] += incoherent_sum[in_idx];
          hit_counts[out_idx] += 1;
        }
}

void normalise_cpu(const std::vector<float> &acc,
                   const std::vector<uint32_t> &hits, std::vector<float> &out) {

  for (size_t i = 0; i < acc.size(); ++i) {
    if (hits[i] > 0) {
      out[i] = acc[i] / (float)hits[i];
    } else {
      out[i] = 0.0f;
    }
  }
}

class FoldKernelTest : public ::testing::Test {
protected:
  void run_test(int C, int F, int P, int T, int n_bins, double period_samples,
                int64_t total_samples_elapsed) {

    int cf = C * F;

    size_t input_size = C * F * P * T;
    size_t output_size = C * F * P * n_bins;

    std::vector<float> h_input(input_size);
    std::vector<float> h_acc_cpu(output_size, 0.0f);
    std::vector<uint32_t> h_hits_cpu(output_size, 0);

    std::vector<float> h_acc_gpu(output_size, 0.0f);
    std::vector<uint32_t> h_hits_gpu(output_size, 0);

    std::vector<float> h_norm_cpu(output_size, 0.0f);
    std::vector<float> h_norm_gpu(output_size, 0.0f);

    std::vector<int32_t> h_dm(cf);

    // deterministic inputs
    for (size_t i = 0; i < input_size; ++i) {
      h_input[i] = float((i % 13) + 1);
    }

    for (int i = 0; i < cf; ++i) {
      h_dm[i] = i % 5; // small delays
    }

    // CPU reference
    fold_and_accumulate_cpu(h_input, h_acc_cpu, h_hits_cpu, h_dm,
                            total_samples_elapsed, period_samples, n_bins, C, F,
                            P, T);

    normalise_cpu(h_acc_cpu, h_hits_cpu, h_norm_cpu);

    // GPU alloc
    float *d_input, *d_acc, *d_out;
    uint32_t *d_hits;
    int32_t *d_dm;

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hits, output_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dm, cf * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_dm, h_dm.data(), cf * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_acc, 0, output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hits, 0, output_size * sizeof(uint32_t)));

    // run fold
    fold_and_accumulate_launch(d_input, d_acc, d_hits, d_dm,
                               total_samples_elapsed, period_samples, n_bins, C,
                               F, P, T, 0);

    CUDA_CHECK(cudaDeviceSynchronize());

    // run normalisation
    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    normalise_fold_kernel<<<blocks, threads>>>(d_acc, d_out, d_hits,
                                               output_size);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_acc_gpu.data(), d_acc, output_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_hits_gpu.data(), d_hits,
                          output_size * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_norm_gpu.data(), d_out, output_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // checks
    assert_allclose(h_acc_cpu, h_acc_gpu);
    ASSERT_EQ(h_hits_cpu, h_hits_gpu);
    assert_allclose(h_norm_cpu, h_norm_gpu);

    cudaFree(d_input);
    cudaFree(d_acc);
    cudaFree(d_out);
    cudaFree(d_hits);
    cudaFree(d_dm);
  }
};

TEST_F(FoldKernelTest, Minimal) { run_test(1, 1, 1, 1, 8, 10.0, 0); }

TEST_F(FoldKernelTest, NoDelays) { run_test(2, 2, 1, 64, 16, 32.0, 0); }

TEST_F(FoldKernelTest, WithDelays) {
  run_test(2, 4, 2, 128, 32, 50.0, 1000);
  EXPECT_TRUE(false);
}
