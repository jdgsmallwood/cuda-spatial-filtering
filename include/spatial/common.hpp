#pragma once

#include "hdf5.h"
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pointing.hpp"
#include "spatial/spatial.hpp"
// Self-guarded: expands to nothing unless the build defined HAVE_IBVERBS.
#include "spatial/libibverbs.hpp"
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
#include <sched.h>
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
  INFO_LOG("Caught CTRL+C, shutting down...");
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

// Returns the NUMA node the given CUDA device's PCI device is attached to, or
// -1 if it couldn't be determined (no such sysfs entry, device not present).
inline int gpu_numa_node(int device = 0) {
  char pci_bus_id[16];
  if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device) !=
      cudaSuccess)
    return -1;
  std::string bus(pci_bus_id);
  std::transform(bus.begin(), bus.end(), bus.begin(), ::tolower);
  std::ifstream f("/sys/bus/pci/devices/" + bus + "/numa_node");
  int node = -1;
  if (f >> node && node >= 0)
    return node;
  return -1;
}

// Returns the NUMA node the given network interface's PCI device is attached
// to, or -1 if it couldn't be determined (interface missing, no sysfs entry).
inline int nic_numa_node(const std::string &ifname) {
  std::ifstream f("/sys/class/net/" + ifname + "/device/numa_node");
  int node = -1;
  if (f >> node && node >= 0)
    return node;
  return -1;
}

// Pins the calling thread to the CPUs of `node`. Since CPU affinity masks are
// inherited across clone(), calling this on the main thread before any other
// threads are spawned pins the whole process. This box is a dual-socket Xeon
// with a *split* L3 (one slice per socket) and separate memory controllers,
// so letting the scheduler scatter ring-buffer/pipeline threads across both
// sockets means every ring-slot read/write and every pinned-buffer access can
// cross the inter-socket link -- confining everything to the GPU's node keeps
// it node-local. node < 0 disables pinning (e.g. topology couldn't be
// determined).
inline void pin_current_thread_to_numa_node(int node) {
  if (node < 0)
    return;
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                  "/cpulist");
  std::string cpulist;
  if (!std::getline(f, cpulist) || cpulist.empty())
    return;

  cpu_set_t set;
  CPU_ZERO(&set);
  // cpulist is a comma-separated list of ranges, e.g. "0-9" or "0-4,10-14".
  size_t i = 0;
  while (i < cpulist.size()) {
    size_t comma = cpulist.find(',', i);
    if (comma == std::string::npos)
      comma = cpulist.size();
    std::string token = cpulist.substr(i, comma - i);
    size_t dash = token.find('-');
    int lo = std::stoi(token.substr(0, dash));
    int hi = dash == std::string::npos ? lo : std::stoi(token.substr(dash + 1));
    for (int c = lo; c <= hi; ++c)
      CPU_SET(c, &set);
    i = comma + 1;
  }
  sched_setaffinity(0, sizeof(set), &set);
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

// ENUPosition, ArrayLocation, FrequencyPlan, and BeamTarget come from
// pointing.hpp (via pipeline.hpp above) -- shared geometry/target types also
// consumed by compute_steering_weights().

struct CommonArgs {
  std::string pcap_filename;
  std::string output_filename; // may be empty → caller picks a default
  std::string beam_output_filename;
  std::string config_filename;
  std::string gains_filename;
  std::string beam_weights_filename;
  std::string nr_signal_eigenvectors_filename;
  std::string targets_filename;
  std::string ifname;
  std::string fpga_delay_file;
  bool loop_pcap = false;
  bool debug_logging = false;
  bool apply_gains = false;
  int min_freq_channel = 0;
  int port = 36001;
  int packets_to_receive = 0;
  double steering_update_interval_seconds = 180.0;
  std::string capture_backend = "kernel";
  int busy_poll_us = 0;
  json config;
  json gains;
  json beam_weights;
  json targets;
  std::vector<int> fpga_id_vec;
  std::unordered_map<uint32_t, int> fpga_ids;
  std::unordered_map<int, int> antenna_mapping;
  std::unordered_map<int, int> nr_signal_eigenvectors;
  bool shrink_eigenvalues = false;
  std::vector<std::string> fpga_names;
  std::unordered_map<int, int64_t> fpga_delays;
  // Antenna ENU positions (metres), keyed by absolute antenna ID; from
  // config.json's "antenna_positions" (defaults to zeros if absent).
  std::unordered_map<int, ENUPosition> antenna_positions;
  // Array reference point; from config.json's "array_location".
  ArrayLocation array_location;
  // Channel-to-frequency mapping; from config.json's "frequency_plan".
  FrequencyPlan frequency_plan;
  // One pointing target per beam, from --targets-filename. Empty when no
  // targets file is supplied, in which case callers fall back to static
  // beam_weights/uniform-default behaviour (steering is opt-in).
  std::vector<BeamTarget> beam_targets;
  // How many channels to write to Redis per output block. 0 = all channels.
  // Use a value < NR_CHANNELS to cap per-block payload (round-robin rotation).
  int redis_channels_per_write = 8;
  // Number of correlation blocks to integrate per visibility dump.
  // 0 means "use the compile-time default baked into LambdaConfig".
  int nr_integration_blocks = 0;
  // Maximum wall-clock duration in seconds before the app shuts down.
  // 0 means run indefinitely.
  double run_duration_seconds = 0.0;
};

// Builds the per-NIC packet-capture objects for an app from the parsed args,
// centralizing the pcap-vs-live and kernel-vs-ibverbs backend choice so every
// binary selects backends identically: offline pcap replay when --pcap-filename
// is set, otherwise one live capture per NIC using the --capture-backend the
// user asked for. `kernel_recv_buffer_size` tunes SO_RCVBUF for the kernel
// backend (pulsar-fold historically uses a larger value than the others).
inline std::vector<std::unique_ptr<PacketInput>>
make_packet_captures(const CommonArgs &args,
                     int kernel_recv_buffer_size = 256 * 1024 * 1024) {
  std::vector<std::unique_ptr<PacketInput>> capture;

  if (!args.pcap_filename.empty()) {
    capture.push_back(std::make_unique<PCAPPacketCapture>(args.pcap_filename,
                                                          args.loop_pcap));
    return capture;
  }

  const bool use_ibverbs = args.capture_backend == "ibverbs";
  if (!use_ibverbs && args.capture_backend != "kernel") {
    throw std::runtime_error("Unknown --capture-backend '" +
                             args.capture_backend +
                             "' (expected 'kernel' or 'ibverbs')");
  }
#ifndef HAVE_IBVERBS
  if (use_ibverbs) {
    throw std::runtime_error(
        "--capture-backend=ibverbs requested but this binary was built without "
        "libibverbs (install libibverbs-dev and rebuild on an RDMA host)");
  }
#endif

  const int nr_nics = static_cast<int>(args.fpga_names.size());
  for (int i = 0; i < nr_nics; ++i) {
    auto nic = args.fpga_names[i];
#ifdef HAVE_IBVERBS
    if (use_ibverbs) {
      capture.push_back(std::make_unique<LibibverbsPacketCapture>(
          nic, args.port, BUFFER_SIZE));
      continue;
    }
#endif
    capture.push_back(std::make_unique<KernelSocketPacketCapture>(
        nic, args.port, BUFFER_SIZE, kernel_recv_buffer_size,
        args.busy_poll_us, /*thread_id=*/i, /*nr_threads=*/nr_nics));
  }
  return capture;
}

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

  program.add_argument("--beam-output-file")
      .help("fully qualified or relative filename for beam HDF5 output")
      .default_value(std::string(""))
      .store_into(args.beam_output_filename);

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

  program.add_argument("--capture-backend")
      .help("Live packet-capture backend: 'kernel' (SOCK_DGRAM/recvmmsg) or "
            "'ibverbs' (libibverbs raw-packet QP, requires an RDMA NIC and a "
            "build with libibverbs)")
      .default_value(std::string("kernel"))
      .store_into(args.capture_backend);

  program.add_argument("--busy-poll")
      .help("Enable SO_BUSY_POLL on the kernel receive socket: number of "
            "microseconds to busy-spin waiting for packets before sleeping "
            "(0 = disabled, typical useful range 10-200).  Only effective "
            "with --capture-backend=kernel on a dedicated NIC IRQ core.")
      .default_value(0)
      .store_into(args.busy_poll_us);

  program.add_argument("-d", "--debug-logging")
      .help("Enable debug logging")
      .default_value(false)
      .implicit_value(true)
      .store_into(args.debug_logging);

  program.add_argument("-n", "--num-packets")
      .help("How many packets to receive before exiting (0 = unlimited)")
      .default_value(0)
      .store_into(args.packets_to_receive);

  program.add_argument("-y", "--delay-file")
      .help("JSON file with delays between each FPGA.")
      .default_value("alveo_delays.json")
      .store_into(args.fpga_delay_file);

  program.add_argument("-g", "--gains")
      .help("JSON file with weights")
      .default_value("weights.json")
      .store_into(args.gains_filename);

  program.add_argument("-e", "--eigenvalue-num-filename")
      .help("JSON file with number of eigenvalues to num per channel")
      .default_value("nr-signal-eigenvalues.json")
      .store_into(args.nr_signal_eigenvectors_filename);

  program.add_argument("--shrink-eigenvalues")
      .help("Shrink RFI eigenvalues to the mean of the non-RFI eigenvalues "
            "instead of nulling them")
      .default_value(false)
      .implicit_value(true)
      .store_into(args.shrink_eigenvalues);

  program.add_argument("-a", "--apply-gains-to-vis")
      .help("Apply the inverse of the gains to the raw data")
      .default_value(false)
      .implicit_value(true)
      .store_into(args.apply_gains);

  program.add_argument("-b", "--beam-weights-filename")
      .help("Filename of json file with beam weights")
      .default_value("")
      .store_into(args.beam_weights_filename);

  program.add_argument("-T", "--targets-filename")
      .help("JSON file describing per-beam pointing targets (RA/Dec or "
            "zenith). When supplied, beam weights are computed from these "
            "targets and the array/antenna geometry in the config file "
            "instead of from --beam-weights-filename, and re-steered "
            "periodically to track sidereal motion.")
      .default_value("")
      .store_into(args.targets_filename);

  program.add_argument("--steering-update-interval-seconds")
      .help("How often (in seconds) to recompute beam-steering weights so "
            "tracked sources stay pointed at as the sky rotates")
      .default_value(180.0)
      .store_into(args.steering_update_interval_seconds);

  program.add_argument("--redis-channels-per-write")
      .help("Channels written to Redis per output block (0 = all). Smaller "
            "values cap the TS.MADD payload via round-robin rotation, keeping "
            "Redis write time constant regardless of total channel count.")
      .default_value(8)
      .scan<'i', int>()
      .store_into(args.redis_channels_per_write);

  program.add_argument("--accumulation-length")
      .help("Number of correlation blocks to integrate per visibility dump "
            "(0 = use compile-time default)")
      .default_value(0)
      .scan<'i', int>()
      .store_into(args.nr_integration_blocks);

  program.add_argument("--obs-length")
      .help("Maximum wall-clock run time in seconds before the app shuts down "
            "(0 = run indefinitely)")
      .default_value(0.0)
      .scan<'g', double>()
      .store_into(args.run_duration_seconds);

  try {
    program.parse_args(argc, argv);
    std::ifstream f(args.config_filename);
    args.config = json::parse(f);

    if (args.config.contains("antenna_positions")) {
      for (const auto &[fpga_key, antennas] :
           args.config["antenna_positions"].items()) {
        for (const auto &[antenna_key, pos] : antennas.items()) {
          int absolute_antenna_id = std::stoi(antenna_key);
          ENUPosition enu;
          enu.east = pos.value("east", 0.0);
          enu.north = pos.value("north", 0.0);
          enu.up = pos.value("up", 0.0);
          args.antenna_positions[absolute_antenna_id] = enu;
        }
      }
      std::cout << "Loaded " << args.antenna_positions.size()
                << " antenna position(s) from config\n";
    }

    if (args.config.contains("array_location")) {
      const auto &loc = args.config["array_location"];
      args.array_location.latitude_deg = loc.value("latitude_deg", 0.0);
      args.array_location.longitude_deg = loc.value("longitude_deg", 0.0);
      args.array_location.height_m = loc.value("height_m", 0.0);
      std::cout << "Array location: lat=" << args.array_location.latitude_deg
                << " lon=" << args.array_location.longitude_deg
                << " height=" << args.array_location.height_m << "m\n";
    }

    if (args.config.contains("frequency_plan")) {
      const auto &plan = args.config["frequency_plan"];
      args.frequency_plan.base_frequency_hz =
          plan.value("base_frequency_hz", 0.0);
      args.frequency_plan.channel_bandwidth_hz =
          plan.value("channel_bandwidth_hz", 0.0);
      std::cout << "Frequency plan: base="
                << args.frequency_plan.base_frequency_hz
                << "Hz channel_bandwidth="
                << args.frequency_plan.channel_bandwidth_hz << "Hz\n";
    }

    if (args.targets_filename != "") {
      std::ifstream tf(args.targets_filename);
      args.targets = json::parse(tf);

      for (const auto &target : args.targets.at("targets")) {
        BeamTarget bt;
        bt.mode = target.value("mode", std::string("zenith"));
        bt.ra_deg = target.value("ra_deg", 0.0);
        bt.dec_deg = target.value("dec_deg", 0.0);

        if (bt.mode != "zenith" && bt.mode != "radec") {
          throw std::runtime_error("Unknown beam target mode '" + bt.mode +
                                   "' (expected 'radec' or 'zenith')");
        }

        args.beam_targets.push_back(bt);
        std::cout << "Beam target " << args.beam_targets.size() - 1
                  << ": mode=" << bt.mode;
        if (bt.mode == "radec") {
          std::cout << " ra_deg=" << bt.ra_deg << " dec_deg=" << bt.dec_deg;
        }
        std::cout << std::endl;
      }
    }

    std::ifstream g(args.gains_filename);
    args.gains = json::parse(g);

    if (args.beam_weights_filename != "") {
      std::ifstream h(args.beam_weights_filename);
      args.beam_weights = json::parse(h);
    }

    // default initialize the map
    for (auto i = 0; i < 500; ++i) {
      args.nr_signal_eigenvectors[i] = 1;
    }

    std::ifstream e(args.nr_signal_eigenvectors_filename);
    json sig = json::parse(e);

    for (const auto &[key, value] : sig.items()) {
      args.nr_signal_eigenvectors[std::stoi(key)] = value;
      std::cout << "setting nr_signal_eigenvectors for channel " << key
                << " to " << value << std::endl;
    }

    std::cout << args.config.dump(4) << std::endl;

    std::ifstream j(args.fpga_delay_file);
    json delays = json::parse(j);

    for (const auto &[key, value] : delays.items()) {
      args.fpga_delays[std::stoi(key)] = value;
      std::cout << "setting FPGA delays for Alveo " << key << " to " << value
                << std::endl;
    }

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

  // Pin this thread (and, via clone() inheritance, every thread the app
  // later spawns that doesn't self-override -- capture threads can via
  // SPATIAL_CAPTURE_CPUS) to the GPU's NUMA node, so the ring buffer /
  // pipeline stay node-local to the GPU DMA target. SPATIAL_NUMA_NODE
  // overrides the auto-detected node; set it to a negative number to
  // disable pinning entirely.
  int numa_node;
  if (const char *node_env = std::getenv("SPATIAL_NUMA_NODE")) {
    numa_node = std::atoi(node_env);
  } else {
    numa_node = gpu_numa_node();
  }

  if (numa_node >= 0) {
    pin_current_thread_to_numa_node(numa_node);
    INFO_LOG("Pinned main thread to NUMA node {} (GPU's node)", numa_node);

    if (std::getenv("SPATIAL_CAPTURE_CPUS") == nullptr) {
      for (const auto &nic : args.fpga_names) {
        int nic_node = nic_numa_node(nic);
        if (nic_node >= 0 && nic_node != numa_node) {
          WARN_LOG("NIC {} is on NUMA node {} but the pipeline is pinned to "
                    "node {} (the GPU's node); consider setting "
                    "SPATIAL_CAPTURE_CPUS to keep this NIC's capture thread "
                    "node-local",
                    nic, nic_node, numa_node);
        }
      }
    }
  } else {
    INFO_LOG("Could not determine GPU NUMA node; not pinning to a NUMA node");
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

          std::complex<float> val;
          try {
            val = {
                args.gains["weights"][std::to_string(args.min_freq_channel + i)]
                          [pol_string][std::to_string(
                              args.antenna_mapping[receiver_idx])]["real"],

                args.gains["weights"][std::to_string(args.min_freq_channel + i)]
                          [pol_string][std::to_string(
                              args.antenna_mapping[receiver_idx])]["imag"]};
          } catch (const std::exception &err) {
            std::cout << "Gain not found for channel "
                      << std::to_string(args.min_freq_channel + i) << " pol "
                      << pol_string << " receiver_idx " << receiver_idx
                      << std::endl;
            val = {1.0f, 0.0f};
          }

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
