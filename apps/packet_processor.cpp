#include "spatial/spatial.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

constexpr int NUM_CHANNELS = 2;
constexpr int NUM_FRAMES = 100;
constexpr int NUM_BUFFERS = 2;
constexpr int NR_RECEIVERS_PER_PACKET = 32;

typedef Sample Packets[NUM_FRAMES][NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS]
                      [NR_TIME_STEPS_PER_PACKET];

struct PacketInfo {
  int source;
  int packet_seq_number;
};

int main() {

  Samples *d_samples[NUM_BUFFERS];
  Packets *d_packet_data[NUM_CHANNELS];

  PacketInfo *h_packet_info[NUM_CHANNELS][NUM_FRAMES];

  return 0;
}
