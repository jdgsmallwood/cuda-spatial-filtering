#include "spatial/ethernet.hpp"
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

// Processor thread - continuously processes packets

void print_startup_info() {

  // Startup debug info
  LOG_INFO("NR_CHANNELS: {}", NR_CHANNELS);
  LOG_INFO("NUM_FRAMES_PER_ITERATION: {}", NUM_FRAMES_PER_ITERATION);
  LOG_INFO("NR_TOTAL_FRAMES_PER_CHANNEL: {}", NR_TOTAL_FRAMES_PER_CHANNEL);
  LOG_INFO("NR_FPGA_SOURCES: {}", NR_FPGA_SOURCES);
  LOG_INFO("NR_RECEIVERS: {}", NR_RECEIVERS);
  LOG_INFO("NR_RECEIVERS_PER_PACKET: {}", NR_RECEIVERS_PER_PACKET);
  LOG_INFO("NR_PACKETS_FOR_CORRELATION: {}", NR_PACKETS_FOR_CORRELATION);
  LOG_INFO("NR_INPUT_BUFFERS: {}", NR_INPUT_BUFFERS);
  LOG_INFO("PacketDataStructure size is {}", sizeof(PacketDataStructure));
  LOG_INFO("PacketScaleStructure size is {}", sizeof(PacketScaleStructure));
}

int main() {
  auto app_logger = spdlog::basic_logger_mt("packet_processor_live_logger",
                                            "app.log", /*truncate*/ true);
  app_logger->set_level(spdlog::level::debug);
  app_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  // Provide it to the library
  spatial::Logger::set(app_logger);
  print_startup_info();

  ProcessorState<LambdaPacketStructure> state;
  LambdaGPUPipeline pipeline;

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  int port = 12345;
  KernelSocketPacketCapture socket_capture(port, BUFFER_SIZE);
  LOG_INFO("Ring buffer size: {} packets\n", RING_BUFFER_SIZE);

  // Start receiver thread
  std::thread receiver(
      [&socket_capture, &state]() { socket_capture.get_packets(state); });

  // Start processor thread
  std::thread processor([&state]() { state.process_packets(); });

  // Print statistics periodically
  while (state.running) {
    sleep(5);
    LOG_INFO("Stats: Received={}, Processed={}", state.packets_received,
             state.packets_processed);
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
  state.running = 0;
  receiver.join();
  processor.join();
  app_logger->flush();
  return 0;
}
