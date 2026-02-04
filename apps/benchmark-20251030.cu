#include "hdf5.h"
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include "spatial/writers.hpp"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <complex>
#include <csignal>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <highfive/highfive.hpp>
#include <iostream>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#ifndef NUMBER_BEAMS
#define NUMBER_BEAMS 1
#endif

#ifndef NUMBER_PACKETS_TO_CORRELATE
#define NUMBER_PACKETS_TO_CORRELATE 16
#endif

std::atomic<bool> running{true};

void signal_handler(int signal) {
  LOG_INFO("Caught CTRL+C, shutting down...");
  running = false;
}

void writeVectorToCSV(const std::vector<float> &times,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << "\n";
    return;
  }

  // Write CSV header
  file << "index,time\n";

  // Write data
  for (size_t i = 0; i < times.size(); ++i) {
    file << i << "," << times[i] << "\n";
  }

  file.close();
  std::cout << "Data successfully written to " << filename << "\n";
}

std::string
make_default_visibilities_filename(const int min_freq_channel,
                                   const int num_channels,
                                   const std::vector<int> fpga_ids) {
  // timestamp
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d-%H%M") << "_" << min_freq_channel << "_"
      << min_freq_channel + num_channels - 1;
  for (auto id : fpga_ids) {
    oss << "_ALVEO" << id;
  }
  oss << ".hdf5";
  return oss.str();
}

std::vector<std::string> split_ifnames(const std::string &ifname) {
  std::vector<std::string> result;
  std::stringstream ss(ifname);
  std::string token;

  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
}

class AntennaMapRegistry {
public:
  AntennaMapRegistry() {
    // Initialize your 4 base maps here
    // FPGA 0
    base_maps[0] = {{0, -100}, {1, 35},   {2, -100}, {3, 1},  {4, -100},
                    {5, 14},   {6, -100}, {7, 36},   {8, 18}, {9, 25}};

    base_maps[1] = {{0, 15}, {1, 16}, {2, 23}, {3, 24}, {4, 26},
                    {5, 32}, {6, 17}, {7, 33}, {8, 11}, {9, 13}};

    base_maps[2] = {

        {0, 4}, {1, 6}, {2, 5}, {3, 29}, {4, 10}, {5, 20},
        {6, 7}, {7, 9}, {8, 2}, {9, 3}

    };

    base_maps[3] = {

        {0, 19}, {1, 28}, {2, 31}, {3, 34}, {4, 27},
        {5, 30}, {6, 12}, {7, 22}, {8, 8},  {9, 21}};
  }

  std::unordered_map<int, int>
  get_combined_map(const std::vector<int> &selected_fpgas) {
    std::unordered_map<int, int> combined;

    for (size_t i = 0; i < selected_fpgas.size(); ++i) {
      int fpga_id = selected_fpgas[i];
      int stream_offset = i * 10; // 0 for 1st, 10 for 2nd, 20 for 3rd...

      if (base_maps.find(fpga_id) == base_maps.end()) {
        throw std::runtime_error("Unknown FPGA ID: " + std::to_string(fpga_id));
      }

      const auto &source_map = base_maps[fpga_id];
      for (const auto &[stream, antenna] : source_map) {
        // Key: New Offset Stream, Value: Antenna ID
        combined[stream + stream_offset] = antenna;
      }
    }
    return combined;
  }

private:
  std::unordered_map<int, std::unordered_map<int, int>> base_maps;
};

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");
  std::string pcap_filename;
  std::string vis_filename;
  std::string ifname;
  bool loop_pcap, debug_logging;
  int min_freq_channel;
  int port;
  int packets_to_receive;
  program.add_argument("-p", "--pcap_file")
      .help("specify a PCAP file to replay")
      .store_into(pcap_filename);

  program.add_argument("-l", "--loop")
      .help("loop the specified PCAP file")
      .default_value(false)
      .implicit_value(true)
      .store_into(loop_pcap);
  program.add_argument("-v", "--vis_output_file")
      .help("specify a file name for the output visibilities")
      .store_into(vis_filename);

  program.add_argument("-f", "--min_freq_channel")
      .help("specify the lowest frequency channel.")
      .store_into(min_freq_channel);

  program.add_argument("-i", "--network-interface")
      .help("Network interface to bind on")
      .default_value("enp216s0np0")
      .store_into(ifname);

  program.add_argument("-L", "--port")
      .help("Port to bind on")
      .default_value(36001)
      .store_into(port);

  program.add_argument("-d", "--debug-logging")
      .help("Enable debug logging")
      .default_value(false)
      .implicit_value(true)
      .store_into(debug_logging);

  program.add_argument("-n", "--num-packets")
      .help("How many packets to receive before exiting.")
      .default_value(0)
      .store_into(packets_to_receive);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  std::signal(SIGINT, signal_handler);
  static auto tp = std::make_shared<spdlog::details::thread_pool>(4 * 8192, 2);
  auto app_logger = std::make_shared<spdlog::async_logger>(
      "async_logger",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", true), tp,
      spdlog::async_overflow_policy::overrun_oldest);

  // auto app_logger = spdlog::basic_logger_mt<spdlog::async_factory>(
  //   "async_logger", "app.log", true);

  // auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
  //                                         "app.log", /*truncate*/ true);
  if (debug_logging) {
    app_logger->set_level(spdlog::level::debug);
  } else {
    app_logger->set_level(spdlog::level::info);
  }
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  constexpr int num_buffers = NR_OBSERVING_BUFFERS;
  constexpr int nr_fpga_sources = NR_OBSERVING_FPGA_SOURCES;
  constexpr size_t num_packet_buffers = 24;
  constexpr int num_lambda_channels = NR_OBSERVING_CHANNELS;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers_per_packet =
      NR_OBSERVING_RECEIVERS_PER_PACKET;
  constexpr int nr_lambda_receivers =
      nr_lambda_receivers_per_packet * nr_fpga_sources;
  constexpr int nr_lambda_padded_receivers = NR_OBSERVING_PADDED_RECEIVERS;
  constexpr int nr_lambda_beams = NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr size_t PACKET_RING_BUFFER_SIZE = 50000;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers_per_packet,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers, nr_lambda_padded_receivers,
                   nr_correlation_blocks_to_integrate, true>;

  const std::unordered_map<std::string, int> ifname_to_fpga{
      {"enp216s0np0", 3}, {"enp175s0np0", 2}, {"enp134s0np0", 1}};

  using MapType = std::unordered_map<uint32_t, int>;
  auto fpga_ids = std::make_unique<MapType>();
  std::vector<int> fpga_id_vec;

  {
    // use scope here to deallocate i at the end.
    int i = 0;
    for (const auto &name : split_ifnames(ifname)) {
      int fpga_id = 0;

      auto it = ifname_to_fpga.find(name);
      if (it != ifname_to_fpga.end()) {
        fpga_id = it->second;
      }
      (*fpga_ids)[fpga_id] = i;
      fpga_id_vec.push_back(fpga_id);
      i++;
    }
  }

  if (fpga_id_vec.size() != nr_fpga_sources ||
      fpga_ids->size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  ProcessorState<Config, num_packet_buffers, PACKET_RING_BUFFER_SIZE> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      min_freq_channel, &fpga_ids);

  // const char *beam_filename = "hdf5_trial.hdf5";
  // std::string beam_filename = "/tmp/hdf5_trial.hdf5";
  // std::string vis_filename = "hdf5_trial_vis.hdf5";
  // hid_t beam_file =
  //    H5Fcreate(beam_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);

  if (!program.is_used("-v")) {
    vis_filename = make_default_visibilities_filename(
        min_freq_channel, num_lambda_channels, fpga_id_vec);
  }
  HighFive::File vis_file(vis_filename, HighFive::File::Truncate);
  // auto beam_writer = std::make_unique<
  //     HDF5RawBeamWriter<Config::BeamOutputType, Config::ArrivalsOutputType>>(
  //    beam_file);
  //  auto beam_writer = std::make_unique<BatchedHDF5BeamWriter<
  //    Config::BeamOutputType, Config::ArrivalsOutputType>>(beam_file, 100);
  auto beam_writer = std::make_unique<
      InMemoryBeamWriter<Config::BeamOutputType, Config::ArrivalsOutputType>>(
      100);
  // auto beam_writer = std::make_unique<
  //     BinaryRawBeamWriter<Config::BeamOutputType,
  //     Config::ArrivalsOutputType>>(
  //    beam_filename);
  //  auto vis_writer = std::make_unique<
  //      UVFITSVisibilitiesWriter<Config::VisibilitiesOutputType>>(
  //      vis_filename, Config::NR_CHANNELS, Config::NR_POLARIZATIONS,
  //      Config::NR_PADDED_RECEIVERS, 1.0, 1.0, 1.0, 1.0);

  AntennaMapRegistry registry;

  std::unordered_map<int, int> antenna_mapping =
      registry.get_combined_map(fpga_id_vec);
  std::cout << "Antenna mapping is:\n";
  for (const auto &[key, val] : antenna_mapping) {
    std::cout << "Key: " << key << ", Val: " << val << std::endl;
  };

  auto vis_writer = std::make_unique<
      HDF5AndRedisVisibilitiesWriter<Config::VisibilitiesOutputType>>(
      vis_file, 55 /* nr baselines */, min_freq_channel,
      min_freq_channel + num_lambda_channels - 1, &antenna_mapping);
  auto eigen_writer =
      std::make_unique<RedisEigendataWriter<Config::EigenvalueOutputType,
                                            Config::EigenvectorOutputType>>();

  auto fft_writer = std::make_unique<RedisFFTWriter<Config::FFTOutputType>>(
      num_lambda_channels, nr_lambda_receivers, nr_lambda_polarizations);

  auto output = std::make_shared<BufferedOutput<Config>>(
      std::move(beam_writer), std::move(vis_writer), std::move(eigen_writer),
      std::move(fft_writer), 100, 100, 100, 100);

  BeamWeightsT<Config> h_weights;

  for (auto i = 0; i < num_lambda_channels; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] =
              std::complex<__half>(__float2half(1.0f), __float2half(0.0f));
        }
      }
    }
  }

  LambdaGPUPipeline<Config> pipeline(num_buffers, &h_weights);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);

  std::unique_ptr<PacketInput> capture;

  if (!pcap_filename.empty()) {
    capture = std::make_unique<PCAPPacketCapture>(pcap_filename, loop_pcap);
  } else {
    capture = std::make_unique<KernelSocketPacketCapture>(
        ifname, port, BUFFER_SIZE, 256 * 1024 * 1024);
  }
  LOG_INFO("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  LOG_INFO("Starting threads....");
  std::thread receiver([&capture, &state]() { capture->get_packets(state); });

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

  // Start writer thread
  std::thread writer_thread_([&output] { output->writer_loop(); });
  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  int packets_received = 0;
  int timeout = 0;
  while (state.running) {
    sleep(5);
    // This is nice to see outside of log files.
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed
              << ", Missing=" << state.packets_missing
              << ", Discarded=" << state.packets_discarded << std::endl;
    std::cout << "Pipeline Runs Queued = " << state.pipeline_runs_queued
              << std::endl;
    state.running.store((int)running, std::memory_order_release);
    // This is my attempt at a rudimentary shutdown procedure
    // when there are no more packets running through in a 20sec period.
    if (packets_received != 0) {
      if (packets_received == state.packets_received) {
        std::cout
            << "Packets received is same as state... adding to timeout.\n";
        timeout += 1;
      } else {
        std::cout << "Packets received is " << packets_received
                  << " and state.packets_received is " << state.packets_received
                  << ".\n";
        timeout = 0;
      }
      if (timeout > 4) {
        std::cout << "Timeout reached...shutting down\n";
        state.running.store(0, std::memory_order_release);
        running = false;
      }
    }
    packets_received = state.packets_received;

    if (packets_to_receive > 0 && packets_received >= packets_to_receive) {
      std::cout << "Number of packets to observe reached...shutting down\n";
      state.running.store(0, std::memory_order_release);
      running = false;
    }
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
  std::cout << "Shutting down...\n";
  state.running.store(0, std::memory_order_release);
  state.shutdown();

  std::cout << "Waiting for receiver to finish...\n";
  receiver.join();
  std::cout << "Waiting for processor to finish...\n";
  processor.join();
  std::cout << "Waiting for pipeline feeder to finish...\n";
  pipeline_feeder.join();
  std::cout << "Dumping visibilities....\n";
  cudaDeviceSynchronize();
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  std::cout << "Synchronizing GPU...\n";
  cudaDeviceSynchronize();

  output->running_ = false;
  std::cout << "Waiting for writer thread to finish...\n";
  writer_thread_.join();
  std::vector<float> run_timings;
  run_timings.reserve(pipeline.NR_BENCHMARKING_RUNS);
  for (auto i = 0; i < pipeline.NR_BENCHMARKING_RUNS; ++i) {
    float ms;
    cudaEventElapsedTime(&ms, pipeline.start_run[i], pipeline.stop_run[i]);
    if (ms != 0.0f) {
      run_timings.push_back(ms);
    };
  }
  writeVectorToCSV(run_timings, "output_timings.csv");
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
