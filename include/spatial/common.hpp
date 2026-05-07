#pragma once

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
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
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
#include <vector>

#ifndef NUMBER_BEAMS
#define NUMBER_BEAMS 1
#endif

using json = nlohmann::json;
inline std::atomic<bool> running{true};

inline void signal_handler(int signal) {
  LOG_INFO("Caught CTRL+C, shutting down...");
  running = false;
}

inline std::string make_default_filename(const std::string prefix,
                                         const int min_freq_channel,
                                         const int num_channels,
                                         const std::vector<int> fpga_ids) {
  // timestamp
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << prefix << "_" << std::put_time(&tm, "%Y%m%d-%H%M") << "_"
      << min_freq_channel << "_" << min_freq_channel + num_channels - 1;
  for (auto id : fpga_ids) {
    oss << "_ALVEO" << id;
  }
  oss << ".hdf5";
  return oss.str();
}

inline std::vector<std::string> split_ifnames(const std::string &ifname) {
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
    // FPGA 0
    base_maps[0] = {{0, -100}, {1, 35},   {2, -100}, {3, 1},  {4, -100},
                    {5, 14},   {6, -100}, {7, 36},   {8, 18}, {9, 25}};

    base_maps[1] = {{0, 15}, {1, 16}, {2, 23}, {3, 24}, {4, 26},
                    {5, 32}, {6, 17}, {7, 33}, {8, 11}, {9, 13}};

    base_maps[2] = {{0, 4},  {1, 6}, {2, 5}, {3, 29}, {4, 10},
                    {5, 20}, {6, 7}, {7, 9}, {8, 2},  {9, 3}};

    base_maps[3] = {{0, 19}, {1, 28}, {2, 31}, {3, 34}, {4, 27},
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

struct CommonArgs {
  std::string pcap_filename;
  std::string output_filename; // may be empty → caller picks a default
  std::string config_filename;
  std::string gains_filename;
  std::string ifname;
  bool loop_pcap = false;
  bool debug_logging = false;
  int min_freq_channel = 0;
  int port = 36001;
  int packets_to_receive = 0;
  int fpga_delay = 0;
  json config;
  json gains;
  std::vector<int> fpga_id_vec;
  std::unordered_map<uint32_t, int> fpga_ids;
  std::unordered_map<int, int> antenna_mapping;
  std::vector<std::string> fpga_names;
};

// Registers and parses the arguments that are common to every pipeline
// binary. Extra arguments (e.g. --pulsar-period-samples) can be added to
// `program` before calling this function; they will be parsed in the same
// pass.
//
// Returns true on success; on parse failure the function prints usage to
// stderr and calls std::exit(1).
inline CommonArgs parse_common_args(argparse::ArgumentParser &program, int argc,
                                    char *argv[]) {
  CommonArgs args;

  program.add_argument("-c", "--config-file")
      .help("specify a configuration file")
      .default_value(std::string("config.json"))
      .store_into(args.config_filename);

  program.add_argument("-p", "--pcap_file")
      .help("specify a PCAP file to replay")
      .store_into(args.pcap_filename);

  program.add_argument("-l", "--loop")
      .help("loop the specified PCAP file")
      .default_value(false)
      .implicit_value(true)
      .store_into(args.loop_pcap);

  program.add_argument("-v", "--vis_output_file")
      .help("specify a file name for the output")
      .store_into(args.output_filename);

  program.add_argument("-f", "--min_freq_channel")
      .help("specify the lowest frequency channel")
      .store_into(args.min_freq_channel);

  program.add_argument("-i", "--network-interface")
      .help("Network interface to bind on (comma-separated for multiple)")
      .default_value(std::string("enp216s0np0"))
      .store_into(args.ifname);

  program.add_argument("-L", "--port")
      .help("Port to bind on")
      .default_value(36001)
      .store_into(args.port);

  program.add_argument("-d", "--debug-logging")
      .help("Enable debug logging")
      .default_value(false)
      .implicit_value(true)
      .store_into(args.debug_logging);

  program.add_argument("-n", "--num-packets")
      .help("How many packets to receive before exiting (0 = unlimited)")
      .default_value(0)
      .store_into(args.packets_to_receive);

  program.add_argument("-y", "--delay")
      .help("Delay from FPGA 1 to 0")
      .default_value(0)
      .store_into(args.fpga_delay);

  program.add_argument("-g", "--gains")
      .help("JSON file with weights")
      .default_value("weights.json")
      .store_into(args.gains_filename);

  try {
    program.parse_args(argc, argv);
    std::ifstream f(args.config_filename);
    args.config = json::parse(f);

    std::ifstream g(args.gains_filename);
    args.gains = json::parse(g);

    std::cout << args.config.dump(4) << std::endl;

    const std::unordered_map<std::string, int> ifname_to_fpga{
        {"enp216s0np0", 3}, {"enp175s0np0", 2}, {"enp134s0np0", 1}};

    args.fpga_names = split_ifnames(args.ifname);

    {
      // use scope here to deallocate i at the end.
      int i = 0;
      for (const auto &name : args.fpga_names) {
        int fpga_id = 0;

        auto it = ifname_to_fpga.find(name);
        if (it != ifname_to_fpga.end()) {
          fpga_id = it->second;
        }
        args.fpga_ids[fpga_id] = i;
        args.fpga_id_vec.push_back(fpga_id);
        i++;
      }
    }

    AntennaMapRegistry registry;

    args.antenna_mapping = registry.get_combined_map(args.fpga_id_vec);
    std::cout << "Antenna mapping is:\n";
    for (const auto &[key, val] : args.antenna_mapping) {
      std::cout << "Key: " << key << ", Val: " << val << std::endl;
    };

  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    std::exit(1);
  }

  return args;
}

inline std::shared_ptr<spdlog::async_logger> setup_logger(bool debug_logging) {

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
  return app_logger;
}

template <typename T>
inline typename T::AntennaGains get_gains_structure(CommonArgs &args) {
  // AntennaGains objects should be [Channel][Pol][Antenna]
  typename T::AntennaGains output{};
  for (auto i = 0; i < T::NR_CHANNELS; ++i) {
    for (auto j = 0; j < T::NR_POLARIZATIONS; ++j) {
      for (auto f = 0; f < T::NR_FPGA_SOURCES; ++f) {
        int fpga_id = args.fpga_id_vec[f];
        for (auto k = 0; k < T::NR_RECEIVERS_PER_PACKET; ++k) {
          std::string pol_string;
          if (j == 0) {
            pol_string = "XX";
          } else {
            pol_string = "YY";
          }
          int receiver_idx = f * T::NR_RECEIVERS_PER_PACKET + k;

          std::complex<float> val = {
              args.gains["weights"][std::to_string(args.min_freq_channel + i)]
                        [pol_string][std::to_string(
                            args.antenna_mapping[receiver_idx])]["real"],

              args.gains["weights"][std::to_string(args.min_freq_channel + i)]
                        [pol_string][std::to_string(
                            args.antenna_mapping[receiver_idx])]["imag"]};

          float mag = val.real() * val.real() + val.imag() * val.imag();
          // we take the conjugate and divide by the magnitude to
          // correct for both the phase and the amplitude.
          output[i][j][receiver_idx] = {val.real() / mag, -val.imag() / mag};
          std::cout << "Gain for channel " << args.min_freq_channel + i
                    << ", pol " << pol_string << " FPGA " << f << " receiver "
                    << k << " is " << val.real() << " + " << val.imag()
                    << "j.\n";
        }
      }
    }
  }
  return output;
};
