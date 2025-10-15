#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

struct FakeProcessorState : public ProcessorStateBase {
  bool released = false;
  size_t last_index = SIZE_MAX;

  void release_buffer(size_t idx) override {
    released = true;
    last_index = idx;
  }
};

// A fake FinalPacketData for tests, minimal stub
template <typename sampleT, typename scaleT>
struct DummyFinalPacketData : public FinalPacketData {
  sampleT *samples;
  scaleT *scales;

  DummyFinalPacketData() {
    CUDA_CHECK(cudaMallocHost((void **)&samples, sizeof(sampleT)));
    CUDA_CHECK(cudaMallocHost((void **)&scales, sizeof(scalesTT)));
  };

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

TEST(LambdaGPUPipelineTest, Ex1) {
  constexpr size_t nr_channels = 1;
  constexpr size_t nr_receivers = 4;
  constexpr size_t nr_packets = 1;
  constexpr size_t nr_time_per_packet = 8;
  constexpr size_t nr_polarizations = 2;
  constexpr size_t nr_beams = 1;
  using sampleT =
      std::complex<int8_t>[nr_channels][nr_packets][nr_time_per_packet]
                          [nr_receivers][nr_polarizations];
  using scaleT =
      int16_t[nr_channels][nr_packets][nr_receivers][nr_polarizations];

  FakeProcessorState state;

  DummyFinalPacketData<sampleT, scaleT> packet_data =
      new DummyFinalPacketData<sampleT, scaleT>();

  BeamWeights<nr_channels, nr_receivers, nr_polarizations, nr_beams> h_weights;
  for (auto i = 0; i < nr_channels; ++i) {
    for (auto j = 0; j < nr_receivers; ++j) {
      for (auto k = 0; k < nr_polarizations; ++k) {
        for (auto l = 0; l < nr_beams; ++l) {
          h_weights.weights[i][k][l][j] =
              std::complex<__half>(__float2half(1.0f), 0);
        }
      }
    }
  }

  SingleHostMemoryOutput<nr_channels, nr_polarizations, nr_beams> output;

  GPUPipeline<sizeof(int8_t), nr_channels, nr_time_per_packet, nr_packets,
              nr_receivers, 32, nr_polarizations, nr_beams, nr_receivers>
      pipeline(1, &h_weights,
               10000 /* blocks to integrate - set high for now. */
      );

  pipeline.set_state(&state);
  pipeline.set_output(&output);

  pipeline.execute_pipeline(&packet_data);
};

TEST(LambdaGPUPipelineTest, StateNotSetThrows) {
  // Use a small instantiation: e.g., 1 buffer, some dimension args
  // You’ll need a valid BeamWeights pointer; you can allocate dummy
  using MyPipe = LambdaGPUPipeline<1, 1, 1, 1, 1, 1, 1, 1, 1>;
  BeamWeights<1, 1, 1, 1> weights;

  MyPipe pipe(1, &weights, 1);
  // Force state_ to null
  pipe.state_ = nullptr;

  // expect logic_error
  EXPECT_THROW(
      {
        FinalPacketData data;
        pipe.execute_pipeline(&data);
      },
      std::logic_error);
}

// You may subclass pipeline to override GPU parts so no real GPU calls
template <int A, int B, int C, int D, int E, int F, int G, int H, int I>
class TestablePipeline : public LambdaGPUPipeline<A, B, C, D, E, F, G, H, I> {
public:
  using Base = LambdaGPUPipeline<A, B, C, D, E, F, G, H, I>;
  using Base::num_buffers;
  using Base::streams;

  TestablePipeline(int num_buffers, BeamWeights<A, B, C, D> *w, size_t ncb)
      : Base(num_buffers, w, ncb) {}

  // Override the actual GPU steps to no-op or minimal
  void execute_pipeline(FinalPacketData *packet_data) override {
    // skip copying, kernel launches, etc.
    // Directly call release buffer logic
    BufferReleaseContext *ctx = new BufferReleaseContext{
        .state = this->state_, .buffer_index = packet_data->buffer_index};
    release_buffer_host_func(ctx);
  }
};

TEST(LambdaGPUPipelineTest, ExecutePipelineReleasesBuffer) {
  using MyPipe = TestablePipeline<1, 1, 1, 1, 1, 1, 1, 1, 1>;
  BeamWeights<1, 1, 1, 1> weights;
  MyPipe pipe(1, &weights, 1);
  FakeProcessorState state;
  pipe.state_ = &state;

  DummyFinalPacketData data(7, /*n_samples=*/10, /*n_scales=*/5);

  pipe.execute_pipeline(&data);

  EXPECT_TRUE(state.released);
  EXPECT_EQ(state.last_index, 7);
}
