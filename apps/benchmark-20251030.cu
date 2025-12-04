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
#include <complex>
#include <csignal>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <highfive/highfive.hpp>
#include <iostream>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

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

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");
  std::string pcap_filename;
  std::string vis_filename;
  bool loop_pcap;
  int min_freq_channel;
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
  app_logger->set_level(spdlog::level::info);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  constexpr int num_buffers = 3;
  constexpr int nr_fpga_sources = 4;
  constexpr size_t num_packet_buffers = 24;
  constexpr int num_lambda_channels = 8;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers_per_packet = 10;
  constexpr int nr_lambda_receivers =
      nr_lambda_receivers_per_packet * nr_fpga_sources;
  constexpr int nr_lambda_padded_receivers = 64;
  constexpr int nr_lambda_beams = NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_receivers_per_block = 64;
  constexpr int nr_lambda_packets_for_correlation =
      256; // NUMBER_PACKETS_TO_CORRELATE;
  constexpr int nr_correlation_blocks_to_integrate = 100000000;
  constexpr size_t PACKET_RING_BUFFER_SIZE = 50000;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers_per_packet,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers, nr_lambda_padded_receivers,
                   nr_correlation_blocks_to_integrate>;

  ProcessorState<Config, num_packet_buffers, PACKET_RING_BUFFER_SIZE> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      min_freq_channel);

  // const char *beam_filename = "hdf5_trial.hdf5";
  // std::string beam_filename = "/tmp/hdf5_trial.hdf5";
  // std::string vis_filename = "hdf5_trial_vis.hdf5";
  // hid_t beam_file =
  //    H5Fcreate(beam_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);
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
  auto vis_writer =
      std::make_unique<HDF5VisibilitiesWriter<Config::VisibilitiesOutputType>>(
          vis_file);

  auto output = std::make_shared<BufferedOutput<Config>>(
      std::move(beam_writer), std::move(vis_writer), 100, 100);

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
  int port = 36001;
  std::string ifname = "enp216s0np0";
  //  KernelSocketIP6PacketCapture capture(ifname, port, BUFFER_SIZE);
  // LibpcapIP6PacketCapture capture(ifname, port, BUFFER_SIZE);
  PCAPMultiFPGAPacketCapture capture(pcap_filename, loop_pcap, nr_fpga_sources);
  LOG_INFO("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  LOG_INFO("Starting threads....");
  std::thread receiver([&capture, &state]() { capture.get_packets(state); });

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
