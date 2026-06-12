#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <unordered_set>

struct CudaIsolatedTest : ::testing::Test {
  void TearDown() override {
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};
struct FakeProcessorState : public ProcessorStateBase {
  bool released = false;
  int last_index = -1;

  void release_buffer(const int buffer_index) override {
    released = true;
    last_index = buffer_index;
  }

  void *get_next_write_pointer() override { return nullptr; };
  void *get_current_write_pointer() override { return nullptr; };
  void add_received_packet_metadata(const int length,
                                    const sockaddr_in &client_addr) override {};
  int reserve_write_batch(int, void **, int *) override { return 0; }
  void commit_write_batch(int, const int *, const int *,
                          const sockaddr_in *) override {}
  void set_pipeline(GPUPipeline *pipeline) override {};
  void process_all_available_packets() override {};
  void handle_buffer_completion(bool force_flush = false) override {};
};

// A fake FinalPacketData for tests, minimal stub
template <typename T> struct DummyFinalPacketData : public FinalPacketData {
  using sampleT = typename T::InputPacketSamplesType;
  using scaleT = typename T::PacketScalesType;
  sampleT *samples;
  scaleT *scales;
  typename T::ArrivalsOutputType *arrivals;

  DummyFinalPacketData() {
    CUDA_CHECK(cudaMallocHost((void **)&samples, sizeof(sampleT)));
    CUDA_CHECK(cudaMallocHost((void **)&scales, sizeof(scaleT)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals,
                              sizeof(typename T::ArrivalsOutputType)));
  };

  ~DummyFinalPacketData() {
    CUDA_CHECK(cudaFreeHost(samples));
    CUDA_CHECK(cudaFreeHost(scales));
    CUDA_CHECK(cudaFreeHost(arrivals));
  }

  void *get_samples_ptr() override { return samples; };
  size_t get_samples_elements_size() override { return sizeof(sampleT); };
  void *get_scales_ptr() override { return scales; };
  size_t get_scales_element_size() override { return sizeof(scaleT); };

  bool *get_arrivals_ptr() override { return (bool *)arrivals; };
  size_t get_arrivals_size() override {
    return sizeof(typename T::ArrivalsOutputType);
  }

  void zero_missing_packets() override {};
  int get_num_missing_packets() override { return -1; };
};

constexpr size_t NR_CHANNELS = 1;
constexpr size_t NR_FPGA_SOURCES = 1;
constexpr size_t NR_RECEIVERS = 4;
constexpr size_t NR_RECEIVERS_PER_PACKET = NR_RECEIVERS;
constexpr size_t NR_PACKETS = 1;
constexpr size_t NR_TIME_STEPS_PER_PACKET = 8;
constexpr size_t NR_POLARIZATIONS = 2;
constexpr size_t NR_BEAMS = 1;
constexpr size_t NR_PADDED_RECEIVERS = 32;
constexpr size_t NR_PADDED_RECEIVERS_PER_BLOCK = NR_PADDED_RECEIVERS;
constexpr size_t NR_PACKETS_FOR_CORRELATION = 1;
constexpr size_t NR_VISIBILITIES_BEFORE_DUMP = 10000;
using Config =
    LambdaConfig<NR_CHANNELS, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                 NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                 NR_PACKETS_FOR_CORRELATION, NR_BEAMS, NR_PADDED_RECEIVERS,
                 NR_PADDED_RECEIVERS_PER_BLOCK, NR_VISIBILITIES_BEFORE_DUMP>;

using MultiFPGAConfig =
    LambdaConfig<NR_CHANNELS, 3 /* fpga sources */, NR_TIME_STEPS_PER_PACKET,
                 6 /* receivers */, NR_POLARIZATIONS,
                 2 /* receivers_per_packet */, NR_PACKETS_FOR_CORRELATION,
                 NR_BEAMS, NR_PADDED_RECEIVERS, NR_PADDED_RECEIVERS_PER_BLOCK,
                 NR_VISIBILITIES_BEFORE_DUMP>;

TEST_F(CudaIsolatedTest, Ex1) {
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0) {
              expected = 8.0f;
            } else {
              expected = -8.0f;
            }
            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k=" << k
                << ", l=" << l << ", m=" << m << std::endl;
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          expected_vis = 64.0f;
          EXPECT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          // imag should be zero.
          EXPECT_EQ(output->visibilities[0][i][q][j][p][1], 0.0f);
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, PolarizationBlankTest) {
  // Ensure that polarization is respected. Make the samples in one
  // polarization zero and then check that means everything in that polarization
  // is zero too.

  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS_PER_PACKET; ++l) {
          packet_data.samples[0][i][j + 1][0][k][l][0] =
              std::complex<int8_t>(2, -2);
          packet_data.scales[0][i][j + 1][l][0] = static_cast<int16_t>(1);
          packet_data.samples[0][i][j + 1][0][k][l][1] =
              std::complex<int8_t>(0, 0);
          // Deliberately have the scale non-zero.
          packet_data.scales[0][i][j + 1][l][1] = static_cast<int16_t>(1);
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());

  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (j == 1) {
              expected = 0.0f;
            } else {
              if (m == 0) {
                expected = 8.0f;
              } else {
                expected = -8.0f;
              }
            }
            ASSERT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected);
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          if (j == 1 || p == 1) {
            expected_vis = 0;
          } else {
            expected_vis = 64.0f;
          };
          EXPECT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          ;
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, PolarizationBlankTest2) {
  // Ensure that polarization is respected. Make the samples in one
  // polarization zero and then check that means everything in that polarization
  // is zero too.
  //
  // Also check the visibilities are copied across correctly. I'm choosing a
  // large number of channels to ensure coverage across the whole data
  // structure. If I only had 1 or 2 channels then only a few near the front
  // would be non-zero.

  using Config =
      LambdaConfig<NR_CHANNELS + 5, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                   NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION, NR_BEAMS, NR_PADDED_RECEIVERS,
                   NR_PADDED_RECEIVERS_PER_BLOCK, NR_VISIBILITIES_BEFORE_DUMP>;
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS + 5; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS_PER_PACKET; ++l) {
          packet_data.samples[0][i][j + 1][0][k][l][0] =
              std::complex<int8_t>(0, 0);
          packet_data.scales[0][i][j + 1][l][0] = static_cast<int16_t>(1);
          packet_data.samples[0][i][j + 1][0][k][l][1] =
              std::complex<int8_t>(2, -2);
          // Deliberately have the scale non-zero.
          packet_data.scales[0][i][j + 1][l][1] = static_cast<int16_t>(1);
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS + 5; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS + 5; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (j == 0) {
              expected = 0.0f;
            } else {
              if (m == 0) {
                expected = 8.0f;
              } else {
                expected = -8.0f;
              }
            }
            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k=" << k
                << ", l=" << l << ", m=" << m << std::endl;
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          if (j == 0 || p == 0) {
            expected_vis = 0;
          } else {
            expected_vis = 64.0f;
          };
          EXPECT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          ;
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, BeamBlankTest) {
  // Ensure that beam weights are respected. Make the weights in one
  // beam zero and then check that means everything in that beam
  // is zero too.
  using Config =
      LambdaConfig<NR_CHANNELS, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                   NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION, NR_BEAMS + 1,
                   NR_PADDED_RECEIVERS, NR_PADDED_RECEIVERS_PER_BLOCK,
                   NR_VISIBILITIES_BEFORE_DUMP>;
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          if (l == 0) {
            h_weights.weights[i][k][l][j] =
                std::complex<__half>(__float2half(1.0f), 0);
          } else {
            h_weights.weights[i][k][l][j] = std::complex<__half>(0, 0);
          }
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (k == 1) {
              expected = 0.0f;
            } else {
              if (m == 0) {
                expected = 8.0f;
              } else {
                expected = -8.0f;
              }
            }
            ASSERT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected);
          }
        }
      }
    }
  }
};
TEST_F(CudaIsolatedTest, ChannelWeightBlankTest) {
  // Ensure that channel weights are respected. Make the weights in one
  // channel zero and then check that means everything in that channel output
  // is zero too.
  using Config =
      LambdaConfig<NR_CHANNELS + 1, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                   NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION, NR_BEAMS, NR_PADDED_RECEIVERS,
                   NR_PADDED_RECEIVERS_PER_BLOCK, NR_VISIBILITIES_BEFORE_DUMP>;
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          if (i == 0) {
            h_weights.weights[i][k][l][j] =
                std::complex<__half>(__float2half(1.0f), 0);
          } else {
            h_weights.weights[i][k][l][j] = std::complex<__half>(0, 0);
          }
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (i == 1) {
              expected = 0.0f;
            } else {
              if (m == 0) {
                expected = 8.0f;
              } else {
                expected = -8.0f;
              }
            }
            ASSERT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected);
          }
        }
      }
    }
  }
};
TEST_F(CudaIsolatedTest, ChannelSamplesBlankTest) {
  // Ensure that channel samples are respected. Make the samples in one
  // channel zero and then check that means everything in that channel output
  // is zero too.
  using Config =
      LambdaConfig<NR_CHANNELS + 1, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                   NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION, NR_BEAMS, NR_PADDED_RECEIVERS,
                   NR_PADDED_RECEIVERS_PER_BLOCK, NR_VISIBILITIES_BEFORE_DUMP>;
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            if (i == 0) {
              packet_data.samples[0][i][j + 1][0][k][l][m] =
                  std::complex<int8_t>(2, -2);
              packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
            } else {

              packet_data.samples[0][i][j + 1][0][k][l][m] =
                  std::complex<int8_t>(0, 0);
              packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
            }
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (i == 1) {
              expected = 0.0f;
            } else {
              if (m == 0) {
                expected = 8.0f;
              } else {
                expected = -8.0f;
              }
            }
            ASSERT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected);
          }
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, ScalesTest) {
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS); ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            // 0 is on FPGA_ID
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(2);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET; ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0) {
              expected = 16.0f;
            } else {
              expected = -16.0f;
            }
            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k=" << k
                << ", l=" << l << ", m=" << m << " → actual="
                << __half2float(output->beam_data[0][i][j][k][l][m])
                << ", expected=" << expected;
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          expected_vis = 256.0f;
          EXPECT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          ;
        }
      }
    }
  }
};
TEST_F(CudaIsolatedTest, ScalesMultiplePacketsTest) {
  FakeProcessorState state;

  using Config =
      LambdaConfig<NR_CHANNELS, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                   NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION + 1, NR_BEAMS,
                   NR_PADDED_RECEIVERS, NR_PADDED_RECEIVERS_PER_BLOCK,
                   NR_VISIBILITIES_BEFORE_DUMP>;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1;
         j < static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(j);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      2, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           /*num_buffers=*/2));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < Config::NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;
             ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (l / NR_TIME_STEPS_PER_PACKET == 0) {
              expected = 0;
            } else if (m == 0) {
              expected = 8.0f;
            } else {
              expected = -8.0f;
            }
            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k=" << k
                << ", l=" << l << ", m=" << m << " → actual="
                << __half2float(output->beam_data[0][i][j][k][l][m])
                << ", expected=" << expected;
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          expected_vis = 64.0f;
          EXPECT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          ;
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, ScalesPerReceiverTest) {
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1;
         j < static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(l, -l);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(l);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      2, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           /*num_buffers=*/2));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_BEAMS; ++k) {
        for (auto l = 0;
             l < Config::NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;
             ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0) {
              expected = 14.0f;
            } else {
              expected = -14.0f;
            }
            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k=" << k
                << ", l=" << l << ", m=" << m << " → actual="
                << __half2float(output->beam_data[0][i][j][k][l][m])
                << ", expected=" << expected;
          }
        }
      }
      // Now visibilities
      for (auto p = 0; p < NR_POLARIZATIONS; ++p) {
        for (auto q = 0; q < Config::NR_BASELINES_UNPADDED; ++q) {
          float expected_vis;
          if (q == 0 || q == 1 || q == 3 || q == 6) {
            expected_vis = 0;
          } else if (q == 2) {
            expected_vis = 16.0f;
          } else if (q == 4) {
            expected_vis = 64.0f;
          } else if (q == 5) {
            expected_vis = 256.0f;
          } else if (q == 7) {
            expected_vis = 144.0f;
          } else if (q == 8) {
            expected_vis = 576.0f;
          } else if (q == 9) {
            expected_vis = 1296.0f;
          } else {
            expected_vis = 0;
          };
          EXPECT_FLOAT_EQ(output->visibilities[0][i][q][j][p][0], expected_vis)
              << "Mismatch at i=" << i << ", q=" << q << ", j=" << j
              << ", p=" << p
              << " → actual=" << output->visibilities[0][i][q][j][p][0]
              << ", expected=" << expected_vis;
          ;
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, EigenvalueBasic) {
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = -1; j < static_cast<int>(NR_PACKETS) + 1; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j + 1][0][k][l][m] =
                std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j + 1][l][m] = static_cast<int16_t>(1);
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_RECEIVERS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  for (auto i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    subpacket_delays[i] = 0;
  }

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_POLARIZATIONS; ++k) {
        for (auto n = 0; n < NR_RECEIVERS; ++n) {
          float expected_val = 0.0f;
          if (n == NR_RECEIVERS - 1) {
            expected_val = 256.0f;
          }
          EXPECT_NEAR(output->eigenvalues[0][i][j][k][n], expected_val, 1e-4f);

          for (auto l = 0; l < NR_RECEIVERS; ++l) {
            // The eigenvector is extracted by keeping n constant and moving
            // along l.
            float expected_vec = 0.0f;
            if (n == NR_RECEIVERS - 1) {
              // principal eigenvector.
              expected_vec = 0.5f;
            } else if (n == 0 && (l == 1 || l == 0)) {
              expected_vec = -0.0599f;
            } else if (n == 1 && (l == 0 || l == 1)) {
              expected_vec = -0.4964f;
            } else if (n == 2 && (l == 0)) {
              expected_vec = 0.7071f;
            } else if (n == 2 && l == 1) {
              expected_vec = -0.7071f;
            } else if (n == 0 && l == 2) {
              expected_vec = -0.6421f;
            } else if (n == 0 && l == 3) {
              expected_vec = 0.7619f;
            } else if (n == 1 && l == 2) {
              expected_vec = 0.5811f;
            } else if (n == 1 && l == 3) {
              expected_vec = 0.4116f;
            }
            EXPECT_NEAR(output->eigenvectors[0][i][j][k][n][l].real(),
                        expected_vec, 1e-4f)
                << "i: " << i << ", j: " << j << ", k:" << k << ", l: " << l
                << ", n:" << n;
            EXPECT_NEAR(output->eigenvectors[0][i][j][k][n][l].imag(), 0.0f,
                        1e-4f);
          }
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, DelayBeamTest) {
  // Ensure that beam weights are respected. Make the weights in one
  // beam zero and then check that means everything in that beam
  // is zero too.
  using Config = MultiFPGAConfig;
  FakeProcessorState state;

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  subpacket_delays[0] = 0;
  subpacket_delays[1] = 3;
  subpacket_delays[2] = -4;

  DummyFinalPacketData<Config> packet_data;
  for (auto f = 0; f < Config::NR_FPGA_SOURCES; ++f) {
    for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
      for (auto j = -1;
           j < static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION) + 1; ++j) {
        for (auto l = 0; l < Config::NR_RECEIVERS_PER_PACKET; ++l) {
          for (auto m = 0; m < Config::NR_POLARIZATIONS; ++m) {
            packet_data.scales[0][i][j + 1]
                              [f * Config::NR_RECEIVERS_PER_PACKET + l][m] =
                static_cast<int16_t>(1);
            for (int k = 0; k < Config::NR_TIME_STEPS_PER_PACKET; ++k) {

              std::complex<int8_t> val;
              if ((f == 0 &&
                   (j != -1 && j != Config::NR_PACKETS_FOR_CORRELATION)) ||
                  (f == 1 && ((j == Config::NR_PACKETS_FOR_CORRELATION &&
                               k < subpacket_delays[1]) ||
                              (j == 0 && k >= subpacket_delays[1]))

                       ) ||
                  (f == 2 &&
                   ((j == 0 &&
                     k < -1 * static_cast<int>(subpacket_delays[2])) ||
                    (j == -1 && k >= Config::NR_TIME_STEPS_PER_PACKET +
                                         subpacket_delays[2])))

              ) {
                val = {2, -2};
              } else {
                val = {0, 0};
              }
              packet_data.samples[0][i][j + 1][f][k][l][m] = val;
            }
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_RECEIVERS; ++j) {
      for (auto k = 0; k < Config::NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < Config::NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      Config::NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           Config::NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < Config::NR_BEAMS; ++k) {
        for (auto l = 0; l < Config::NR_PACKETS_FOR_CORRELATION *
                                 Config::NR_TIME_STEPS_PER_PACKET;
             ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0) {
              expected = 12.0f;
            } else {
              expected = -12.0f;
            }

            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k =" << k
                << ", l=" << l << ", m=" << m;
          }
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, GainsTest) {
  // Ensure that beam weights are respected. Make the weights in one
  // beam zero and then check that means everything in that beam
  // is zero too.
  using Config = MultiFPGAConfig;
  FakeProcessorState state;

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  subpacket_delays[0] = 0;
  subpacket_delays[1] = 0;
  subpacket_delays[2] = 0;

  DummyFinalPacketData<Config> packet_data;
  for (auto f = 0; f < Config::NR_FPGA_SOURCES; ++f) {
    for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
      for (auto j = -1;
           j < static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION) + 1; ++j) {
        for (auto l = 0; l < Config::NR_RECEIVERS_PER_PACKET; ++l) {
          for (auto m = 0; m < Config::NR_POLARIZATIONS; ++m) {
            packet_data.scales[0][i][j + 1]
                              [f * Config::NR_RECEIVERS_PER_PACKET + l][m] =
                static_cast<int16_t>(1);
            for (int k = 0; k < Config::NR_TIME_STEPS_PER_PACKET; ++k) {

              std::complex<int8_t> val = {2, -2};
              packet_data.samples[0][i][j + 1][f][k][l][m] = val;
            }
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_RECEIVERS; ++j) {
      for (auto k = 0; k < Config::NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < Config::NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  std::array<std::complex<float>, Config::NR_CHANNELS * Config::NR_RECEIVERS *
                                      Config::NR_POLARIZATIONS>
      gains;

  for (auto i = 0; i < Config::NR_CHANNELS * Config::NR_RECEIVERS *
                           Config::NR_POLARIZATIONS;
       ++i) {
    gains[i] = {static_cast<float>(i), static_cast<float>(i)};
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      Config::NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           Config::NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.set_antenna_gains(gains.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < Config::NR_BEAMS; ++k) {
        for (auto l = 0; l < Config::NR_PACKETS_FOR_CORRELATION *
                                 Config::NR_TIME_STEPS_PER_PACKET;
             ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0 && j == 0) {
              expected = 120.0f;
            } else if (m == 0 && j == 1) {
              expected = 144.0f;
            } else {
              expected = 0.0f;
            }

            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k =" << k
                << ", l=" << l << ", m=" << m;
          }
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, GainsOnlyOneFPGAPresentTest) {
  // Ensure that beam weights are respected. Make the weights in one
  // beam zero and then check that means everything in that beam
  // is zero too.
  using Config = MultiFPGAConfig;
  FakeProcessorState state;

  std::array<int, Config::NR_FPGA_SOURCES> subpacket_delays;
  subpacket_delays[0] = 0;
  subpacket_delays[1] = 0;
  subpacket_delays[2] = 0;

  DummyFinalPacketData<Config> packet_data;
  for (auto f = 0; f < Config::NR_FPGA_SOURCES; ++f) {
    for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
      for (auto j = -1;
           j < static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION) + 1; ++j) {
        for (auto l = 0; l < Config::NR_RECEIVERS_PER_PACKET; ++l) {
          for (auto m = 0; m < Config::NR_POLARIZATIONS; ++m) {
            packet_data.scales[0][i][j + 1]
                              [f * Config::NR_RECEIVERS_PER_PACKET + l][m] =
                static_cast<int16_t>(1);
            for (int k = 0; k < Config::NR_TIME_STEPS_PER_PACKET; ++k) {

              std::complex<int8_t> val = {0, 0};
              if (f == 2) {
                val = {2, -2};
              }
              packet_data.samples[0][i][j + 1][f][k][l][m] = val;
            }
          }
        }
      }
    }
  }

  BeamWeightsT<Config> h_weights;
  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_RECEIVERS; ++j) {
      for (auto k = 0; k < Config::NR_POLARIZATIONS; ++k) {
        for (auto l = 0; l < Config::NR_BEAMS; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  std::array<std::complex<float>, Config::NR_CHANNELS * Config::NR_RECEIVERS *
                                      Config::NR_POLARIZATIONS>
      gains;

  for (auto i = 0; i < Config::NR_CHANNELS * Config::NR_RECEIVERS *
                           Config::NR_POLARIZATIONS;
       ++i) {
    gains[i] = {static_cast<float>(i), static_cast<float>(i)};
  }

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(
      Config::NR_PACKETS_FOR_CORRELATION, &h_weights,
      BeamSteering<Config>({}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0,
                           ArrayLocation{}, /*update_interval_seconds=*/180.0,
                           Config::NR_PACKETS_FOR_CORRELATION));

  pipeline.set_state(&state);
  pipeline.set_output(output);

  pipeline.set_subpacket_delays(subpacket_delays.data());
  pipeline.set_antenna_gains(gains.data());
  pipeline.execute_pipeline(&packet_data);
  cudaDeviceSynchronize();

  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < Config::NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < Config::NR_BEAMS; ++k) {
        for (auto l = 0; l < Config::NR_PACKETS_FOR_CORRELATION *
                                 Config::NR_TIME_STEPS_PER_PACKET;
             ++l) {
          for (auto m = 0; m < 2; ++m) {
            float expected;
            if (m == 0 && j == 0) {
              expected = 72.0f;
            } else if (m == 0 && j == 1) {
              expected = 80.0f;
            } else {
              expected = 0.0f;
            }

            EXPECT_EQ(__half2float(output->beam_data[0][i][j][k][l][m]),
                      expected)
                << "Mismatch at i=" << i << ", j=" << j << ", k =" << k
                << ", l=" << l << ", m=" << m;
          }
        }
      }
    }
  }
};

TEST_F(CudaIsolatedTest, ComputeSteeringWeightsMatchesGeometricPhaseFormula) {
  // Cross-checks compute_steering_weights() end-to-end for a "radec" target:
  // calls topocentric_direction() directly with the same inputs and
  // re-derives the expected per-receiver weight from the documented formula
  // (phase = -2*pi*f/c * (l*east + m*north + n*up), normalized by
  // 1/NR_RECEIVERS, unit calibration gain), exercising the channel ->
  // frequency mapping and the antenna_mapping -> antenna_positions lookup
  // chain along the way.
  using namespace std::chrono;

  const std::vector<BeamTarget> targets{
      BeamTarget{"radec", /*ra_deg=*/83.633, /*dec_deg=*/22.014}};
  const std::unordered_map<int, ENUPosition> antenna_positions{
      {100, ENUPosition{0.0, 0.0, 0.0}},
      {101, ENUPosition{25.0, -10.0, 1.5}},
      {102, ENUPosition{-12.5, 30.0, -2.0}},
      {103, ENUPosition{8.0, 8.0, 0.5}},
  };
  // Receiver index -> absolute antenna ID (deliberately not the identity
  // mapping, so the lookup chain is actually exercised).
  const std::unordered_map<int, int> antenna_mapping{
      {0, 100}, {1, 101}, {2, 102}, {3, 103}};
  const FrequencyPlan frequency_plan{/*base_frequency_hz=*/1.4e9,
                                     /*channel_bandwidth_hz=*/1.0e5};
  const int min_freq_channel = 64;
  const ArrayLocation array_location{/*latitude_deg=*/52.91,
                                     /*longitude_deg=*/6.87,
                                     /*height_m=*/30.0};
  const auto utc_time = system_clock::now();

  BeamWeightsT<Config> result = compute_steering_weights<Config>(
      targets, antenna_positions, antenna_mapping, frequency_plan,
      min_freq_channel, array_location, utc_time,
      /*calibration_gains=*/nullptr);

  DirectionCosines dc = topocentric_direction(
      targets[0].ra_deg, targets[0].dec_deg, utc_time,
      array_location.latitude_deg, array_location.longitude_deg,
      array_location.height_m);

  constexpr double kHalfPrecisionTolerance = 2e-3;
  for (size_t chan = 0; chan < Config::NR_CHANNELS; ++chan) {
    double frequency_hz = channel_to_frequency_hz(
        min_freq_channel + static_cast<int>(chan), frequency_plan);
    double phase_scale =
        -2.0 * M_PI * frequency_hz / kSpeedOfLightMetresPerSecond;

    for (size_t receiver_idx = 0; receiver_idx < Config::NR_RECEIVERS;
         ++receiver_idx) {
      const ENUPosition &enu =
          antenna_positions.at(antenna_mapping.at(receiver_idx));
      double phase =
          phase_scale * (dc.l * enu.east + dc.m * enu.north + dc.n * enu.up);
      std::complex<double> expected =
          (1.0 / static_cast<double>(Config::NR_RECEIVERS)) *
          std::complex<double>(std::cos(phase), std::sin(phase));

      for (size_t pol = 0; pol < Config::NR_POLARIZATIONS; ++pol) {
        std::complex<__half> actual =
            result.weights[chan][pol][/*beam=*/0][receiver_idx];
        EXPECT_NEAR(__half2float(actual.real()), expected.real(),
                    kHalfPrecisionTolerance)
            << "chan=" << chan << " pol=" << pol
            << " receiver=" << receiver_idx;
        EXPECT_NEAR(__half2float(actual.imag()), expected.imag(),
                    kHalfPrecisionTolerance)
            << "chan=" << chan << " pol=" << pol
            << " receiver=" << receiver_idx;
      }
    }
  }
};

TEST_F(CudaIsolatedTest, BeamSteeringInactiveWithoutTargetsIsNoOp) {
  // An empty target list (no --targets-filename) makes BeamSteering
  // permanently inert: active() is false and maybe_refresh() never touches
  // device_weights, so a pre-existing sentinel value must survive untouched.
  BeamSteering<Config> steering(
      {}, {}, {}, FrequencyPlan{}, /*min_freq_channel=*/0, ArrayLocation{},
      /*update_interval_seconds=*/180.0, /*num_buffers=*/2);
  EXPECT_FALSE(steering.active());

  auto device_weights = make_device_ptr<BeamWeightsT<Config>>();
  BeamWeightsT<Config> sentinel{};
  sentinel.weights[0][0][0][0] =
      std::complex<__half>(__float2half(0.5f), __float2half(-0.25f));
  CUDA_CHECK(cudaMemcpy(device_weights.get(), &sentinel, sizeof(sentinel),
                        cudaMemcpyDefault));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  EXPECT_FALSE(steering.maybe_refresh(device_weights.get(), stream, 0));
  EXPECT_FALSE(steering.maybe_refresh(device_weights.get(), stream, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  BeamWeightsT<Config> after{};
  CUDA_CHECK(cudaMemcpy(&after, device_weights.get(), sizeof(after),
                        cudaMemcpyDefault));
  EXPECT_EQ(__half2float(after.weights[0][0][0][0].real()), 0.5f);
  EXPECT_EQ(__half2float(after.weights[0][0][0][0].imag()), -0.25f);

  CUDA_CHECK(cudaStreamDestroy(stream));
};

TEST_F(CudaIsolatedTest, BeamSteeringSpreadsRefreshAcrossBuffersRoundRobin) {
  // BeamSteering starts "always overdue", so the first maybe_refresh() call
  // (buffer_index 0) synthesizes fresh weights and arms
  // buffers_pending_refresh_ = num_buffers; the following num_buffers-1 calls
  // (one per buffer, round-robin) copy them down, then the cycle goes quiet
  // until the next due tick. A "zenith" target is time-invariant
  // (zenith_direction() ignores `now`), so a second compute_steering_weights()
  // call is guaranteed to produce bit-identical output to compare against.
  using namespace std::chrono;
  constexpr int kNumBuffers = 3;

  const std::vector<BeamTarget> targets{BeamTarget{"zenith"}};
  const FrequencyPlan frequency_plan{/*base_frequency_hz=*/1.4e9,
                                     /*channel_bandwidth_hz=*/1.0e5};
  const ArrayLocation array_location{/*latitude_deg=*/52.91,
                                     /*longitude_deg=*/6.87,
                                     /*height_m=*/30.0};
  const std::unordered_map<int, ENUPosition> antenna_positions{
      {0, ENUPosition{0.0, 0.0, 0.0}},
      {1, ENUPosition{25.0, -10.0, 1.5}},
      {2, ENUPosition{-12.5, 30.0, -2.0}},
      {3, ENUPosition{8.0, 8.0, 0.5}},
  };
  const std::unordered_map<int, int> antenna_mapping{
      {0, 0}, {1, 1}, {2, 2}, {3, 3}};

  BeamSteering<Config> steering(targets, antenna_positions, antenna_mapping,
                                frequency_plan, /*min_freq_channel=*/0,
                                array_location,
                                /*update_interval_seconds=*/180.0, kNumBuffers);
  ASSERT_TRUE(steering.active());

  std::vector<DevicePtr<BeamWeightsT<Config>>> device_weights;
  std::vector<cudaStream_t> streams(kNumBuffers);
  for (int i = 0; i < kNumBuffers; ++i) {
    device_weights.push_back(make_device_ptr<BeamWeightsT<Config>>());
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    BeamWeightsT<Config> sentinel{};
    sentinel.weights[0][0][0][0] = std::complex<__half>(
        __float2half(100.0f + static_cast<float>(i)), __float2half(0.0f));
    CUDA_CHECK(cudaMemcpy(device_weights[i].get(), &sentinel, sizeof(sentinel),
                          cudaMemcpyDefault));
  }

  // buffer 0: overdue at startup -> recompute + consume the first refresh.
  EXPECT_TRUE(steering.maybe_refresh(device_weights[0].get(), streams[0], 0));
  // buffers 1..kNumBuffers-1 pick theirs up over the next calls.
  for (int i = 1; i < kNumBuffers; ++i) {
    EXPECT_TRUE(steering.maybe_refresh(device_weights[i].get(), streams[i], i));
  }
  // cycle complete; nothing due again until the next update_interval.
  EXPECT_FALSE(steering.maybe_refresh(device_weights[0].get(), streams[0], 0));

  for (auto stream : streams) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  BeamWeightsT<Config> expected = compute_steering_weights<Config>(
      targets, antenna_positions, antenna_mapping, frequency_plan,
      /*min_freq_channel=*/0, array_location, system_clock::now(),
      /*calibration_gains=*/nullptr);

  for (int i = 0; i < kNumBuffers; ++i) {
    BeamWeightsT<Config> actual{};
    CUDA_CHECK(cudaMemcpy(&actual, device_weights[i].get(), sizeof(actual),
                          cudaMemcpyDefault));
    for (size_t chan = 0; chan < Config::NR_CHANNELS; ++chan) {
      for (size_t pol = 0; pol < Config::NR_POLARIZATIONS; ++pol) {
        for (size_t receiver_idx = 0; receiver_idx < Config::NR_RECEIVERS;
             ++receiver_idx) {
          auto a = actual.weights[chan][pol][0][receiver_idx];
          auto e = expected.weights[chan][pol][0][receiver_idx];
          EXPECT_EQ(__half2float(a.real()), __half2float(e.real()))
              << "buffer=" << i << " chan=" << chan << " pol=" << pol
              << " receiver=" << receiver_idx;
          EXPECT_EQ(__half2float(a.imag()), __half2float(e.imag()))
              << "buffer=" << i << " chan=" << chan << " pol=" << pol
              << " receiver=" << receiver_idx;
        }
      }
    }
  }

  for (auto stream : streams) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
};
