#include "spatial/ethernet.hpp"
#include "spatial/packet_formats.hpp"
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
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

// Processor thread - continuously processes packets

void print_startup_info() {

  // Startup debug info
  std::cout << "NR_CHANNELS: " << NR_CHANNELS << std::endl;
  std::cout << "NUM_FRAMES_PER_ITERATION: " << NUM_FRAMES_PER_ITERATION
            << std::endl;
  std::cout << "NR_TOTAL_FRAMES_PER_CHANNEL: " << NR_TOTAL_FRAMES_PER_CHANNEL
            << std::endl;
  std::cout << "NR_FPGA_SOURCES: " << NR_FPGA_SOURCES << std::endl;
  std::cout << "NR_RECEIVERS: " << NR_RECEIVERS << std::endl;
  std::cout << "NR_RECEIVERS_PER_PACKET: " << NR_RECEIVERS_PER_PACKET
            << std::endl;
  std::cout << "NR_PACKETS_FOR_CORRELATION: " << NR_PACKETS_FOR_CORRELATION
            << std::endl;
  std::cout << "NR_INPUT_BUFFERS: " << NR_INPUT_BUFFERS << std::endl;
  std::cout << "PacketDataStructure size is " << sizeof(PacketDataStructure)
            << std::endl;
  std::cout << "PacketScaleStructure size is " << sizeof(PacketScaleStructure)
            << std::endl;
}

int main() {
  print_startup_info();

  ProcessorState<LambdaPacketStructure> state;
  int port = 12345;
  KernelSocketPacketCapture socket_capture(port, BUFFER_SIZE);
  printf("Ring buffer size: %d packets\n\n", RING_BUFFER_SIZE);

  // Start receiver thread
  std::thread receiver(
      [&socket_capture, &state]() { socket_capture.get_packets(state); });

  // Start processor thread
  std::thread processor([&state]() { state.process_packets(); });

  // Print statistics periodically
  while (state.running) {
    sleep(5);
    printf("Stats: Received=%llu, Processed=%llu\n", state.packets_received,
           state.packets_processed);
  }

  // Cleanup
  printf("\nShutting down...\n");
  state.running = 0;
  receiver.join();
  processor.join();
  return 0;
}
