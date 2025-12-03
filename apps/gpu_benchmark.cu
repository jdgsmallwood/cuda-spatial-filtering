
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include "spatial/writers.hpp"
#include <chrono>
#include <iostream>
#include <thread>

constexpr size_t NR_CHANNELS = 1;
constexpr size_t NR_FPGA_SOURCES = 1;
constexpr size_t NR_RECEIVERS = 32;
constexpr size_t NR_RECEIVERS_PER_PACKET = NR_RECEIVERS;
constexpr size_t NR_TIME_STEPS_PER_PACKET = 64;
constexpr size_t NR_POLARIZATIONS = 2;
constexpr size_t NR_BEAMS = 1;
constexpr size_t NR_PADDED_RECEIVERS = 32;
constexpr size_t NR_PADDED_RECEIVERS_PER_BLOCK = NR_PADDED_RECEIVERS;
constexpr size_t NR_PACKETS_FOR_CORRELATION = 256;
constexpr size_t NR_VISIBILITIES_BEFORE_DUMP = 1000000000;
using Config =
    LambdaConfig<NR_CHANNELS, NR_FPGA_SOURCES, NR_TIME_STEPS_PER_PACKET,
                 NR_RECEIVERS, NR_POLARIZATIONS, NR_RECEIVERS_PER_PACKET,
                 NR_PACKETS_FOR_CORRELATION, NR_BEAMS, NR_PADDED_RECEIVERS,
                 NR_PADDED_RECEIVERS_PER_BLOCK, NR_VISIBILITIES_BEFORE_DUMP>;

template <typename T> struct DummyFinalPacketData : public FinalPacketData {
  using sampleT = typename T::InputPacketSamplesType;
  using scaleT = typename T::PacketScalesType;
  sampleT *samples;
  scaleT *scales;
  typename T::ArrivalsOutputType arrivals;

  DummyFinalPacketData() {
    CUDA_CHECK(cudaMallocHost((void **)&samples, sizeof(sampleT)));
    CUDA_CHECK(cudaMallocHost((void **)&scales, sizeof(scaleT)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals,
                              sizeof(typename T::ArrivalsOutputType)));
  };

  ~DummyFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(scales);
    cudaFreeHost(arrivals);
  }

  void *get_samples_ptr() override { return samples; };
  size_t get_samples_elements_size() override { return sizeof(sampleT); };
  void *get_scales_ptr() override { return scales; };
  size_t get_scales_element_size() override { return sizeof(scaleT); };

  bool *get_arrivals_ptr() override { return (bool *)&arrivals; };
  size_t get_arrivals_size() override {
    return sizeof(typename T::ArrivalsOutputType);
  }

  void zero_missing_packets() override {};
  int get_num_missing_packets() override { return -1; };
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
  void set_pipeline(GPUPipeline *pipeline) override {};
  void process_all_available_packets() override {};
  void handle_buffer_completion(bool force_flush = false) override {};
};

int main() {
  FakeProcessorState state;
  DummyFinalPacketData<Config> packet_data;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_RECEIVERS; ++l) {
          for (auto m = 0; m < NR_POLARIZATIONS; ++m) {
            packet_data.samples[0][i][j][0][k][l][m] =
                std::complex<int8_t>(2, -2);
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

  auto output = std::make_shared<SingleHostMemoryOutput<Config>>();

  LambdaGPUPipeline<Config> pipeline(NR_PACKETS_FOR_CORRELATION, &h_weights);

  pipeline.set_state(&state);
  pipeline.set_output(output);
  auto start_time = std::chrono::steady_clock::now();
  auto run_duration = std::chrono::seconds(60); // Run for 60 seconds

  while (std::chrono::steady_clock::now() - start_time < run_duration) {
    pipeline.execute_pipeline(&packet_data);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }

  cudaDeviceSynchronize();
  std::cout << "Finished running for 60 seconds." << std::endl;
  return 0;
}
