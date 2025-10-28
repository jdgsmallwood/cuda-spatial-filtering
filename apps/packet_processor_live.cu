#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include <csignal>
#include "spatial/writers.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <complex>
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

std::atomic<bool> running{true};

void signal_handler(int signal) {
    LOG_INFO("Caught CTRL+C, shutting down...");
    running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);
  auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
                                            "app.log", /*truncate*/ true);
  app_logger->set_level(spdlog::level::info);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  int num_buffers = 2;
  constexpr size_t num_packet_buffers = 10;
  constexpr int num_lambda_channels = 8;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers = 10;
  constexpr int nr_lambda_padded_receivers = 32;
  constexpr int nr_lambda_beams = 8;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_receivers_per_block = 32;
  constexpr int nr_lambda_packets_for_correlation = 16;
  constexpr int nr_fpga_sources = 1;
  constexpr int min_freq_channel = 252;
  constexpr int nr_correlation_blocks_to_integrate = 10000000;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers, nr_lambda_padded_receivers,
                   nr_correlation_blocks_to_integrate>;

  ProcessorState<Config, num_packet_buffers> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      min_freq_channel);

  std::string beam_filename =
      "/tmp/cuda-spatial-filtering/build/apps/hdf5_trial.hdf5";
  std::string vis_filename =
      "/tmp/cuda-spatial-filtering/build/apps/hdf5_trial_vis.hdf5";
  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);
  HighFive::File vis_file(vis_filename, HighFive::File::Truncate);
  auto beam_writer = std::make_unique<
      HDF5BeamWriter<Config::BeamOutputType, Config::ArrivalsOutputType>>(
      beam_file);
  auto vis_writer =
      std::make_unique<HDF5VisibilitiesWriter<Config::VisibilitiesOutputType>>(
          vis_file);

  auto output = std::make_shared<BufferedOutput<Config>>(
      std::move(beam_writer), std::move(vis_writer), 1000, 1000);

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
  KernelSocketPacketCapture socket_capture(port, BUFFER_SIZE);

  LOG_INFO("Ring buffer size: {} packets\n", RING_BUFFER_SIZE);
  LOG_INFO("Starting threads....");
  std::thread receiver(
      [&socket_capture, &state]() { socket_capture.get_packets(state); });

  std::thread processor([&state]() { state.process_packets(); });

  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  while (running) {
    sleep(5);
    // This is nice to see outside of log files.
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed << std::endl;
    state.running = (int)running;
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
    std::cout << "Shutting down...\n";
  state.running = 0;
    std::cout << "Waiting for receiver to finish...\n";
  receiver.join();
    std::cout << "Waiting for processor to finish...\n";
  processor.join();
  FLUSH_LOG();
  spdlog::shutdown();
  return 0;
}
