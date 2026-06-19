
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include "spatial/writers.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <thread>

#ifndef NUMBER_BEAMS
#define NUMBER_BEAMS 1
#endif
constexpr size_t NR_CHANNELS = NR_OBSERVING_CHANNELS;
constexpr size_t NR_FPGA_SOURCES = NR_OBSERVING_FPGA_SOURCES;
constexpr size_t NR_RECEIVERS_PER_PACKET = NR_OBSERVING_RECEIVERS_PER_PACKET;
constexpr size_t NR_RECEIVERS = NR_RECEIVERS_PER_PACKET * NR_FPGA_SOURCES;
constexpr size_t NR_TIME_STEPS_PER_PACKET = 64;
constexpr size_t NR_POLARIZATIONS = 2;
constexpr size_t NR_BEAMS = NUMBER_BEAMS;
constexpr size_t NR_PADDED_RECEIVERS = NR_OBSERVING_PADDED_RECEIVERS;
constexpr size_t NR_PADDED_RECEIVERS_PER_BLOCK =
    NR_OBSERVING_PADDED_RECEIVERS_PER_BLOCK;
constexpr size_t NR_PACKETS_FOR_CORRELATION =
    NR_OBSERVING_PACKETS_FOR_CORRELATION;
constexpr size_t NR_VISIBILITIES_BEFORE_DUMP = 10000000;
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
  int reserve_write_batch(int, void **, int *) override { return 0; }
  void commit_write_batch(int, const int *, const int *,
                          const sockaddr_in *) override {}
  void set_pipeline(GPUPipeline *pipeline) override {};
  void process_all_available_packets() override {};
  void handle_buffer_completion(bool force_flush = false) override {};
};

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("gpu_benchmark");
  program.add_argument("--duration")
      .help("Duration in seconds to run the GPU pipeline")
      .default_value(60.0)
      .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");

  static auto tp = std::make_shared<spdlog::details::thread_pool>(4 * 8192, 2);
  auto app_logger = std::make_shared<spdlog::async_logger>(
      "async_logger",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", true), tp,
      spdlog::async_overflow_policy::overrun_oldest);

  // auto app_logger = spdlog::basic_logger_mt<spdlog::async_factory>(
  //   "async_logger", "app.log", true);

  // auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
  //                                         "app.log", /*truncate*/ true);
  app_logger->set_level(spdlog::level::info);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  std::cout << "NUMBER_BEAMS is " << NUMBER_BEAMS << std::endl;

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

  // No CommonArgs/config.json here, so no target to steer toward -- an empty
  // target list makes BeamSteering permanently inert, leaving the static
  // `h_weights` behaviour below unchanged.
  BeamSteering<Config> beam_steering({}, {}, {}, FrequencyPlan{}, 0,
                                     ArrayLocation{}, 180.0, 5);

  LambdaCorrBeamOnlyGPUPipeline<Config> pipeline(5, &h_weights,
                                                 std::move(beam_steering));

  pipeline.set_state(&state);
  pipeline.set_output(output);
  unsigned long long pipeline_runs = 0;
  const auto start_time = std::chrono::steady_clock::now();

  while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start_time)
             .count() < duration_s) {
    pipeline.execute_pipeline(&packet_data);
    pipeline_runs++;
  }

  cudaDeviceSynchronize();
  const auto end_time = std::chrono::steady_clock::now();
  const double elapsed_seconds =
      std::chrono::duration<double>(end_time - start_time).count();

  constexpr size_t size_input_bytes = sizeof(Config::InputPacketSamplesType);
  constexpr size_t size_output_bytes = sizeof(Config::BeamOutputType);
  const double input_GB_sec =
      static_cast<double>(size_input_bytes) * pipeline_runs /
      elapsed_seconds / 1e9;
  const double output_GB_sec =
      static_cast<double>(size_output_bytes) * pipeline_runs /
      elapsed_seconds / 1e9;
  const double GB_sec = static_cast<double>(size_input_bytes + size_output_bytes) *
                         pipeline_runs / elapsed_seconds / 1e9;
  const double runs_per_sec = pipeline_runs / elapsed_seconds;

  std::printf(
      "[GPU Pipeline] config=ch%zu_fpga%zu_rx%zu "
      "elapsed=%.3f runs=%llu "
      "runs/sec=%.4f "
      "input_bytes=%zu output_bytes=%zu "
      "input_GB/sec=%.6f output_GB/sec=%.6f GB/sec=%.6f\n",
      NR_CHANNELS, NR_FPGA_SOURCES, NR_RECEIVERS,
      elapsed_seconds, (unsigned long long)pipeline_runs,
      runs_per_sec,
      size_input_bytes, size_output_bytes,
      input_GB_sec, output_GB_sec, GB_sec);
  return 0;
}
