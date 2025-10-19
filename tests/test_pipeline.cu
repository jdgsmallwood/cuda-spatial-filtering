#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

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
  void set_pipeline(GPUPipeline *pipeline) override {};
};

// A fake FinalPacketData for tests, minimal stub
template <typename T> struct DummyFinalPacketData : public FinalPacketData {
  using sampleT = typename T::PacketSamplesType;
  using scaleT = typename T::PacketScalesType;
  sampleT *samples;
  scaleT *scales;

  DummyFinalPacketData() {
    CUDA_CHECK(cudaMallocHost((void **)&samples, sizeof(sampleT)));
    CUDA_CHECK(cudaMallocHost((void **)&scales, sizeof(scaleT)));
  };

  ~DummyFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(scales);
  }

  void *get_samples_ptr() override { return samples; };
  size_t get_samples_elements_size() override { return sizeof(sampleT); };
  void *get_scales_ptr() override { return scales; };
  size_t get_scales_element_size() override { return sizeof(scaleT); };

  bool *get_arrivals_ptr() override {
    bool out = true;
    return &out;
  };

  void zero_missing_packets() override {};
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
TEST(LambdaGPUPipelineTest, Ex1) {
  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_PACKETS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j][k][l][m] = std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j][l][m] = static_cast<int16_t>(1);
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

  SingleHostMemoryOutput<Config> output;

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      10000 /* blocks to integrate - set high for now. */
  );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

  pipeline.execute_pipeline(&packet_data);
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
            ASSERT_EQ(output.beam_data[0][i][j][k][l][m], expected);
          }
        }
      }
    }
  }
};

TEST(LambdaGPUPipelineTest, PolarizationBlankTest) {
  // Ensure that polarization is respected. Make the samples in one
  // polarization zero and then check that means everything in that polarization
  // is zero too.

  FakeProcessorState state;

  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_PACKETS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          packet_data.samples[0][i][j][k][l][0] = std::complex<int8_t>(2, -2);
          packet_data.scales[0][i][j][l][0] = static_cast<int16_t>(1);
          packet_data.samples[0][i][j][k][l][1] = std::complex<int8_t>(0, 0);
          // Deliberately have the scale non-zero.
          packet_data.scales[0][i][j][l][1] = static_cast<int16_t>(1);
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

  SingleHostMemoryOutput<Config> output;

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      10000 /* blocks to integrate - set high for now. */
  );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

  pipeline.execute_pipeline(&packet_data);
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
            ASSERT_EQ(output.beam_data[0][i][j][k][l][m], expected);
          }
        }
      }
    }
  }
};

TEST(LambdaGPUPipelineTest, BeamBlankTest) {
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
    for (auto j = 0; j < NR_PACKETS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j][k][l][m] = std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j][l][m] = static_cast<int16_t>(1);
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

  SingleHostMemoryOutput<Config> output;

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      10000 /* blocks to integrate - set high for now. */
  );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

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
            ASSERT_EQ(output.beam_data[0][i][j][k][l][m], expected);
          }
        }
      }
    }
  }
};
TEST(LambdaGPUPipelineTest, ChannelWeightBlankTest) {
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
    for (auto j = 0; j < NR_PACKETS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j][k][l][m] = std::complex<int8_t>(2, -2);
            packet_data.scales[0][i][j][l][m] = static_cast<int16_t>(1);
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

  SingleHostMemoryOutput<Config> output;

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      10000 /* blocks to integrate - set high for now. */
  );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

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
            ASSERT_EQ(output.beam_data[0][i][j][k][l][m], expected);
          }
        }
      }
    }
  }
};
TEST(LambdaGPUPipelineTest, ChannelSamplesBlankTest) {
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
    for (auto j = 0; j < NR_PACKETS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            if (i == 0) {
              packet_data.samples[0][i][j][k][l][m] =
                  std::complex<int8_t>(2, -2);
              packet_data.scales[0][i][j][l][m] = static_cast<int16_t>(1);
            } else {

              packet_data.samples[0][i][j][k][l][m] =
                  std::complex<int8_t>(0, 0);
              packet_data.scales[0][i][j][l][m] = static_cast<int16_t>(1);
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

  SingleHostMemoryOutput<Config> output;

  LambdaGPUPipeline<Config> pipeline(
      NR_PACKETS_FOR_CORRELATION, &h_weights,
      10000 /* blocks to integrate - set high for now. */
  );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

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
            ASSERT_EQ(output.beam_data[0][i][j][k][l][m], expected);
          }
        }
      }
    }
  }
};
//  TEST(LambdaGPUPipelineTest, StateNotSetThrows) {
//    // Use a small instantiation: e.g., 1 buffer, some dimension args
//    // You’ll need a valid BeamWeights pointer; you can allocate dummy
//    using MyPipe = LambdaGPUPipeline<1, 1, 1, 1, 1, 1, 1, 1, 1>;
//    BeamWeights<1, 1, 1, 1> weights;
//
//    MyPipe pipe(1, &weights, 1);
//    // Force state_ to null
//    pipe.set_state(nullptr);
//
//    // expect logic_error
//    EXPECT_THROW(
//        {
//          FinalPacketData data;
//          pipe.execute_pipeline(&data);
//        },
//        std::logic_error);
//  }
//
//// You may subclass pipeline to override GPU parts so no real GPU calls
// template <int A, int B, int C, int D, int E, int F, int G, int H, int I>
// class TestablePipeline : public LambdaGPUPipeline<A, B, C, D, E, F, G, H, I>
// { public:
//   using Base = LambdaGPUPipeline<A, B, C, D, E, F, G, H, I>;
//   using Base::num_buffers;
//   using Base::streams;
//
//   TestablePipeline(int num_buffers, BeamWeights<A, B, C, D> *w, size_t ncb)
//       : Base(num_buffers, w, ncb) {}
//
//   // Override the actual GPU steps to no-op or minimal
//   void execute_pipeline(FinalPacketData *packet_data) override {
//     // skip copying, kernel launches, etc.
//     // Directly call release buffer logic
//     BufferReleaseContext *ctx = new BufferReleaseContext{
//         .state = this->state_, .buffer_index = packet_data->buffer_index};
//     release_buffer_host_func(ctx);
//   }
// };
//
// TEST(LambdaGPUPipelineTest, ExecutePipelineReleasesBuffer) {
//   using MyPipe = TestablePipeline<1, 1, 1, 1, 1, 1, 1, 1, 1>;
//   BeamWeights<1, 1, 1, 1> weights;
//   MyPipe pipe(1, &weights, 1);
//   FakeProcessorState state;
//   pipe.state_ = &state;
//
//   DummyFinalPacketData data(7, /*n_samples=*/10, /*n_scales=*/5);
//
//   pipe.execute_pipeline(&data);
//
//   EXPECT_TRUE(state.released);
//   EXPECT_EQ(state.last_index, 7);
// }
