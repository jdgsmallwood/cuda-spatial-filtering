
#include "spatial/spatial.hpp"
#include <gtest/gtest.h>
#define CONCAT(a, b) a##b
#define TEST_SUITE_NAME(prefix, base) CONCAT(prefix, base)
#if NR_BITS == 8
typedef int8_t TestInputType;
#define NAMESPACE BeamformingTests_8bit
#elif NR_BITS == 16
typedef __half TestInputType;
#define NAMESPACE BeamformingTests_16bit
#endif

TEST(TEST_SUITE_NAME(NAMESPACE, BeamformingTests), SimpleTest) {

  Samples *h_samples;
  Visibilities *h_visibilities;
  BeamWeights *h_weights;
  BeamformedData *h_beamformed_data;

  constexpr int number_of_aggregated_packets = 1;
  cudaMallocHost(&h_samples, number_of_aggregated_packets * sizeof(Samples));
  cudaMallocHost(&h_visibilities,
                 number_of_aggregated_packets * sizeof(Visibilities));
  cudaMallocHost(&h_weights, sizeof(BeamWeights));
  cudaMallocHost(&h_beamformed_data,
                 number_of_aggregated_packets * sizeof(BeamformedData));

  printf("Setting h_samples & h_visibilities memory to zero\n");
  std::memset(h_samples, 0, number_of_aggregated_packets * sizeof(Samples));
  std::memset(h_visibilities, 0,
              number_of_aggregated_packets * sizeof(Visibilities));
  std::memset(h_weights, 0, sizeof(BeamWeights));
  h_samples[0][0][0][0][0][0] = Sample{1, 1};
  h_samples[0][0][0][0][0][1] = Sample{2, 2};
  h_samples[0][0][0][1][0][0] = Sample{2, 2};
  h_samples[0][0][0][1][0][1] = Sample{0, 1};

  h_weights[0][0][0][0][0] =
      std::complex<__half>(__float2half(1.0f), __float2half(1.0f));
  h_weights[0][0][0][0][1] =
      std::complex<__half>(__float2half(1.0f), __float2half(1.0f));

  h_weights[0][0][0][1][0] =
      std::complex<__half>(__float2half(2.0f), __float2half(2.0f));
  h_weights[0][0][0][1][1] =
      std::complex<__half>(__float2half(2.0f), __float2half(2.0f));
  beamform((std::complex<TestInputType> *)h_samples,
           (std::complex<__half> *)h_weights,
           (std::complex<float> *)h_beamformed_data,
           (std::complex<float> *)h_visibilities, number_of_aggregated_packets);

  // To check these figures:
  // import numpy as np

  // data = np.array(
  //[
  //     [1 + 1j, 2 + 2j],
  //     [2 + 2j,     1j]
  //])
  //
  // weights = np.array([
  //     [1 + 1j, 1 + 1j],
  //     [2 + 2j, 2 + 2j],
  //])
  //
  // np.dot(data[0], data[1].conj().T)
  // > 6 - 2j
  //
  // weights @ data
  // > np.array([
  //  [0 +  6j, -1 +  5j],
  //  [0 + 12j, -2 + 10j],
  // ])

  EXPECT_EQ(h_visibilities[0][0][0][0][0], Visibility(10, 0));
  EXPECT_EQ(h_visibilities[0][0][1][0][0], Visibility(6, -2));
  EXPECT_EQ(h_visibilities[0][0][2][0][0], Visibility(9, 0));

  EXPECT_EQ(h_beamformed_data[0][0][0][0][0], std::complex<float>(0, 6));
  EXPECT_EQ(h_beamformed_data[0][0][0][0][1], std::complex<float>(-1, 5));
  EXPECT_EQ(h_beamformed_data[0][0][0][1][0], std::complex<float>(0, 12));
  EXPECT_EQ(h_beamformed_data[0][0][0][1][1], std::complex<float>(-2, 10));
}
