#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
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

int main() {
  auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
                                            "app.log", /*truncate*/ true);
  app_logger->set_level(spdlog::level::debug);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  spatial::Logger::set(app_logger);

  ProcessorState<LambdaPacketStructure> state;

  int num_buffers = 2;
  constexpr int num_lambda_channels = 8;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers = 10;
  constexpr int nr_lambda_padded_receivers = 32;
  constexpr int nr_lambda_beams = 8;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_receivers_per_block = 32;
  constexpr int nr_lambda_packets_for_correlation = 16;
  BeamWeights<num_lambda_channels, nr_lambda_receivers, nr_lambda_polarizations,
              nr_lambda_beams>
      h_weights;

  for (auto i = 0; i < num_lambda_channels; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] = 1 / nr_lambda_receivers;
        }
      }
    }
  }

  LambdaGPUPipeline<
      sizeof(int8_t), num_lambda_channels, nr_lambda_time_steps_per_packet,
      nr_lambda_packets_for_correlation, nr_lambda_receivers,
      /* padded receivers (round up to * of 32) */ nr_lambda_padded_receivers,
      /* nr_polarizations */ 2, nr_lambda_beams, nr_lambda_receivers_per_block>
      pipeline(num_buffers, &h_weights);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  int port = 12345;
  KernelSocketPacketCapture socket_capture(port, BUFFER_SIZE);
  LOG_INFO("Ring buffer size: {} packets\n", RING_BUFFER_SIZE);
  LOG_INFO("Starting threads....");
  std::thread receiver(
      [&socket_capture, &state]() { socket_capture.get_packets(state); });

  std::thread processor([&state]() { state.process_packets(); });

  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  while (state.running) {
    sleep(5);
    // This is nice to see outside of log files.
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed << std::endl;
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
  state.running = 0;
  receiver.join();
  processor.join();
  app_logger->flush();
  return 0;
}
