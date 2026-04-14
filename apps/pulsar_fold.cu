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

std::atomic<bool> running{true};

void signal_handler(int signal) {
  LOG_INFO("Caught CTRL+C, shutting down...");
  running = false;
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
  argparse::ArgumentParser program("pulsar_fold");
  std::string pcap_filename;
  std::string output_filename;
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
      .store_into(output_filename);

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
  constexpr int nr_lambda_beams = 1; // NUMBER_BEAMS placeholder;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr int fft_downsample_factor = 1;
  constexpr size_t PACKET_RING_BUFFER_SIZE = 50000;
  using Config = LambdaConfig<
      num_lambda_channels, nr_fpga_sources, nr_lambda_time_steps_per_packet,
      nr_lambda_receivers, nr_lambda_polarizations,
      nr_lambda_receivers_per_packet, nr_lambda_packets_for_correlation,
      nr_lambda_beams, nr_lambda_padded_receivers, nr_lambda_padded_receivers,
      nr_correlation_blocks_to_integrate, true, fft_downsample_factor>;

  // 2x as there will be original & RFI mitigated beams.
  const std::unordered_map<std::string, int> ifname_to_fpga{
      {"enp216s0np0", 3}, {"enp175s0np0", 2}, {"enp134s0np0", 1}};

  using MapType = std::unordered_map<uint32_t, int>;
  auto fpga_ids = std::make_unique<MapType>();
  std::vector<int> fpga_id_vec;
  auto fpga_names = split_ifnames(ifname);

  {
    // use scope here to deallocate i at the end.
    int i = 0;
    for (const auto &name : fpga_names) {
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

  AntennaMapRegistry registry;

  std::unordered_map<int, int> antenna_mapping =
      registry.get_combined_map(fpga_id_vec);
  std::cout << "Antenna mapping is:\n";
  for (const auto &[key, val] : antenna_mapping) {
    std::cout << "Key: " << key << ", Val: " << val << std::endl;
  };

  std::cout << "Creating Output Handler\n";
  constexpr int n_bins = 512;
  using PulsarType =
      float[num_lambda_channels][16][nr_lambda_polarizations][n_bins];
  auto pulsar_writer = std::make_unique<RedisPulsarFoldWriter<PulsarType>>(
      num_lambda_channels, 16, nr_lambda_polarizations, n_bins);

  auto output = std::make_shared<BufferedOutput<Config>>(
      nullptr, nullptr, nullptr, nullptr, std::move(pulsar_writer), 100, 100,
      100, 100, 100);

  std::cout << "Loading weights...\n";
  // BeamWeightsT<Config> h_weights;
  //
  // for (auto i = 0; i < num_lambda_channels; ++i) {
  //   for (auto j = 0; j < nr_lambda_receivers; ++j) {
  //     for (auto k = 0; k < nr_lambda_beams; ++k) {
  //       for (auto l = 0; l < nr_lambda_polarizations; ++l) {
  //         h_weights.weights[i][l][k][j] =
  //             std::complex<__half>(__float2half(1.0f), __float2half(0.0f));
  //       }
  //     }
  //   }
  // }

  std::cout << "Initializing pipeline...\n";
  PulsarFoldParameters pulsar;
  pulsar.period_samples = 5175.4575;
  pulsar.n_bins = n_bins;
  pulsar.dm = 67.771;
  pulsar.ref_freq_mhz = 149.5 * 781.25 / 1000;
  pulsar.chan_bw_mhz = 781.25 * 32 / 27 / 1000;
  pulsar.lowest_chan_freq_mhz = (146 * 781.25 - 0.5 * 781.25 * 32 / 27) / 1000;

  LambdaPulsarFoldPipeline<Config> pipeline(num_buffers, pulsar, 100);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  std::cout << "Initializing packet capture...\n";
  std::vector<std::unique_ptr<PacketInput>> capture;

  if (!pcap_filename.empty()) {
    capture.push_back(
        std::make_unique<PCAPPacketCapture>(pcap_filename, loop_pcap));
  } else {
    for (auto nic : fpga_names) {
      capture.push_back(std::make_unique<KernelSocketPacketCapture>(
          nic, port, BUFFER_SIZE, 256 * 1024 * 1024));
    }
  }
  LOG_INFO("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  std::cout << "Starting threads...\n";
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

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

  std::cout << "Waiting for receivers to finish...\n";
  for (auto &t : receiver_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
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
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
