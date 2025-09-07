#include "spatial/spatial.hpp"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include <random>

constexpr int NUM_FRAMES = 100;
constexpr int NR_FPGA_SOURCES = 4;
constexpr int NR_RECEIVERS_PER_PACKET = NR_RECEIVERS / NR_FPGA_SOURCES;

int randomChannel() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, NR_CHANNELS - 1);
  return dist(gen);
}
int randomSource() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, NR_FPGA_SOURCES - 1);
  return dist(gen);
}

typedef Sample Packet[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS]
                     [NR_TIME_STEPS_PER_PACKET];

typedef Packet Packets[NUM_FRAMES];
typedef bool SampleOccupancy[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION]
                            [NR_RECEIVERS];

struct PacketInfo {
  int source;
  int packet_seq_number;
  int channel;
};

int main() {

  Packet example_packet;
  for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        example_packet[i][j][k] = Sample(i, j);
      }
    }
  }

  Samples *d_samples[NR_BUFFERS];
  Packets *d_packet_data[NR_CHANNELS];

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    d_packet_data[i] = (Packets *)calloc(1, sizeof(Packets));
  }

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    d_samples[i] = (Samples *)calloc(1, sizeof(Samples));
  }

  PacketInfo h_packet_info[NR_CHANNELS][NUM_FRAMES];
  int h_input_buffer_start_seq[NR_BUFFERS];
  int h_input_buffer_end_seq[NR_BUFFERS];

  bool h_is_buffer_ready[NR_BUFFERS];
  bool h_is_input_buffer_populated[NR_BUFFERS];
  int latest_packet_received[NR_CHANNELS];
  // Receive n packets

  int last_generated_seq_num_for_channel_and_source[NR_CHANNELS]
                                                   [NR_FPGA_SOURCES] = {};

  for (auto i = 0; i < NUM_FRAMES; ++i) {
    // Receive packets
    int channel = randomChannel();
    int source = randomSource();
    last_generated_seq_num_for_channel_and_source[channel][source]++;
    int seq = last_generated_seq_num_for_channel_and_source[channel][source];
    PacketInfo info;
    info.source = source;
    info.channel = channel;
    info.packet_seq_number = seq;
    h_packet_info[channel][source] = info;
    std::cout << "About to copy...\n";
    std::memcpy(&((*d_packet_data[channel])[i]), &example_packet,
                sizeof(Packet));
  }

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_FPGA_SOURCES; ++j) {
      PacketInfo info = h_packet_info[i][j];
      std::cout << i << "," << j << ": " << info.packet_seq_number << ", "
                << info.channel << ", " << info.source << std::endl;
    }
  }

  std::cout << "h_packet_data" << std::endl;

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NUM_FRAMES; ++j) {
      for (auto k = 0; k < NR_RECEIVERS_PER_PACKET; ++k) {
        for (auto l = 0; l < NR_POLARIZATIONS; ++l) {
          for (auto m = 0; m < NR_TIME_STEPS_PER_PACKET; ++m) {

            std::cout << "i,j,k,l,m" << i << j << k << l << m << ": "
                      << __half2float((*d_packet_data[i])[j][k][l][m].real())
                      << __half2float((*d_packet_data[i])[j][k][l][m].imag())
                      << std::endl;
          }
        }
      }
    }
  }
  return 0;
}
