#include "hdf5.h"
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include "spatial/writers.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <complex>
#include <csignal>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <highfive/highfive.hpp>
#include <iostream>
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

int main() {
  std::cout << "Starting....\n";
  std::signal(SIGINT, signal_handler);
  auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
                                            "app.log", /*truncate*/ true);
  app_logger->set_level(spdlog::level::info);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  int num_buffers = 2;
  constexpr size_t num_packet_buffers = 12;
  constexpr int num_lambda_channels = 8;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers = 10;
  constexpr int nr_lambda_padded_receivers = 32;
  constexpr int nr_lambda_beams = NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_receivers_per_block = 32;
  constexpr int nr_lambda_packets_for_correlation = NUMBER_PACKETS_TO_CORRELATE;
  constexpr int nr_fpga_sources = 1;
  constexpr int min_freq_channel = 252;
  constexpr int nr_correlation_blocks_to_integrate = 1000;
  constexpr size_t PACKET_RING_BUFFER_SIZE = 50000;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers, nr_lambda_padded_receivers,
                   nr_correlation_blocks_to_integrate>;

  ProcessorState<Config, num_packet_buffers, PACKET_RING_BUFFER_SIZE> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      min_freq_channel);

  // const char *beam_filename = "hdf5_trial.hdf5";
  // std::string beam_filename = "/tmp/hdf5_trial.hdf5";
  std::string vis_filename = "hdf5_trial_vis.uvfits";
  // hid_t beam_file =
  //    H5Fcreate(beam_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);
  // HighFive::File vis_file(vis_filename, HighFive::File::Truncate);
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
  auto vis_writer = std::make_unique<
      UVFITSVisibilitiesWriter<Config::VisibilitiesOutputType>>(vis_filename);

  auto output = std::make_shared<BufferedOutput<Config>>(
      std::move(beam_writer), std::move(vis_writer), 100, 100);

  BeamWeightsT<Config> h_weights;

  for (auto i = 0; i < num_lambda_channels; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] = 1 / nr_lambda_receivers;
        }
      }
    }
  }

  LambdaGPUPipeline<Config> pipeline(num_buffers, &h_weights);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  int port = 12345;
  // KernelSocketPacketCapture socket_capture(port, BUFFER_SIZE);
  PCAPPacketCapture capture(
      "/tmp/cuda-spatial-filtering/cap_13Dec2024_0.pcapng", true);
  LOG_INFO("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  LOG_INFO("Starting threads....");
  std::thread receiver([&capture, &state]() { capture.get_packets(state); });

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  int packets_received = 0;
  int timeout = 0;
  while (state.running) {
    sleep(5);
    // This is nice to see outside of log files.
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed
              << ", Discarded=" << state.packets_discarded << std::endl;
    std::cout << "Pipeline Runs Queued = " << state.pipeline_runs_queued
              << std::endl;
    state.running = (int)running;
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
        state.running = 0;
        running = false;
      }
    }
    packets_received = state.packets_received;
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
  std::cout << "Shutting down...\n";
  state.running = 0;
  std::cout << "Waiting for receiver to finish...\n";
  receiver.join();
  std::cout << "Waiting for processor to finish...\n";
  processor.join();
  std::cout << "Waiting for pipeline feeder to finish...\n";
  pipeline_feeder.join();

  std::cout << "Synchronizing GPU...\n";
  cudaDeviceSynchronize();

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
  return 0;
}
