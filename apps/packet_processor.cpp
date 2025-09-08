#include "spatial/spatial.hpp"
#include <algorithm>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

constexpr int NUM_ITERATIONS = 10;
constexpr int NUM_FRAMES_PER_ITERATION = 100;
constexpr int NR_TOTAL_FRAMES_PER_CHANNEL = 5 * NUM_FRAMES_PER_ITERATION;
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
typedef Sample PacketSamples[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                            [NR_RECEIVERS][NR_POLARIZATIONS]
                            [NR_TIME_STEPS_PER_PACKET];
typedef Packet Packets[NR_TOTAL_FRAMES_PER_CHANNEL];
typedef bool SampleOccupancy[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION]
                            [NR_RECEIVERS];

struct PacketInfo {
  int source;
  int packet_seq_number;
  int channel;
};

int main() {

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

  Packet example_packet;
  for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        example_packet[i][j][k] = Sample(i, j);
      }
    }
  }

  PacketSamples *d_samples[NR_BUFFERS];
  Packets *d_packet_data[NR_CHANNELS];

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    d_packet_data[i] = (Packets *)calloc(1, sizeof(Packets));
  }

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    d_samples[i] = (PacketSamples *)calloc(1, sizeof(PacketSamples));
  }

  int current_buffer{0};
  PacketInfo h_packet_info[NR_CHANNELS][NR_TOTAL_FRAMES_PER_CHANNEL];
  bool h_packet_info_filled[NR_CHANNELS][NR_TOTAL_FRAMES_PER_CHANNEL] = {false};
  int h_input_buffer_start_seq[NR_BUFFERS] = {};
  int h_input_buffer_end_seq[NR_BUFFERS] = {NR_PACKETS_FOR_CORRELATION - 1};

  bool h_is_buffer_ready[NR_BUFFERS];
  bool h_is_input_buffer_populated[NR_BUFFERS][NR_CHANNELS] = {false};
  int latest_packet_received[NR_CHANNELS][NR_FPGA_SOURCES] = {};
  // Receive n packets

  int last_generated_seq_num_for_channel_and_source[NR_CHANNELS]
                                                   [NR_FPGA_SOURCES] = {};

  bool h_capture_buffer_slot_available[NR_CHANNELS]
                                      [NR_TOTAL_FRAMES_PER_CHANNEL]{true};

  int next_frame_for_channel[NR_CHANNELS] = {0};

  // make all buffers ready
  std::fill(std::begin(h_is_buffer_ready), std::end(h_is_buffer_ready), true);

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    h_input_buffer_start_seq[i] = i * NR_PACKETS_FOR_CORRELATION;
    h_input_buffer_end_seq[i] = (i + 1) * NR_PACKETS_FOR_CORRELATION - 1;
  }

  for (auto iter = 0; iter < NUM_ITERATIONS; ++iter) {
    std::cout << "Starting iteration " << iter << std::endl;
    // Generate some packets and put it in the buffer.
    std::cout << "Randomly generate data..." << std::endl;
    for (auto i = 0; i < NUM_FRAMES_PER_ITERATION; ++i) {

      // Receive packets
      int channel = randomChannel();
      int source = randomSource();
      last_generated_seq_num_for_channel_and_source[channel][source]++;
      int seq = last_generated_seq_num_for_channel_and_source[channel][source];
      std::cout << "packet " << i << ", channel: " << channel
                << ", source: " << source << ", seq: " << seq << std::endl;
      PacketInfo info;
      info.source = source;
      info.channel = channel;
      info.packet_seq_number = seq;
      h_packet_info[channel][next_frame_for_channel[channel]] = info;
      std::memcpy(&((*d_packet_data[channel])[next_frame_for_channel[channel]]),
                  &example_packet, sizeof(Packet));
      h_packet_info_filled[channel][next_frame_for_channel[channel]] = true;
      int gate_check{0};
      while (h_packet_info_filled[channel][next_frame_for_channel[channel]]) {
        next_frame_for_channel[channel] =
            (next_frame_for_channel[channel] + 1) % NR_TOTAL_FRAMES_PER_CHANNEL;
        gate_check++;
        if (gate_check > NR_TOTAL_FRAMES_PER_CHANNEL) {
          throw std::runtime_error(
              "We ate our own tail - no frames left for this channel.");
        }
      }
    }

    // Debug: print packet info
    std::cout << "printing packet info..." << std::endl;
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_TOTAL_FRAMES_PER_CHANNEL; ++j) {
        if (h_packet_info_filled[i][j]) {
          PacketInfo info = h_packet_info[i][j];
          std::cout << i << "," << j << ": " << info.packet_seq_number << ", "
                    << info.channel << ", " << info.source << std::endl;
        }
      }
    }
    /*
    // Debug: print packet data
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
  */
    // Copy in data to correct place
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      std::cout << "Update latest numbers for channel " << i << std::endl;
      for (auto j = 0; j < NR_TOTAL_FRAMES_PER_CHANNEL; ++j) {
        if (!h_packet_info_filled[i][j])
          continue;

        //        std::cout << "i: " << i << ", j: " << j << std::endl;
        PacketInfo packet_info = h_packet_info[i][j];
        latest_packet_received[i][packet_info.source] =
            std::max(latest_packet_received[i][packet_info.source],
                     packet_info.packet_seq_number);

        // copy to correct place or leave it.
        for (int buffer = 0; buffer < NR_BUFFERS; ++buffer) {
          int buffer_index = (current_buffer + buffer) % NR_BUFFERS;
          int packet_index = packet_info.packet_seq_number -
                             h_input_buffer_start_seq[buffer_index];

          if (buffer == 0 && packet_index < 0) {
            // This means that this packet is less than the lowest possible
            // start token. Maybe an out-of-order packet that's coming in?
            // Regardless we can't do anything with this.
            std::cout << "Discarding packet as it is before current buffer "
                         "with begin_seq "
                      << h_input_buffer_start_seq[current_buffer]
                      << " actually has packet_index "
                      << packet_info.packet_seq_number << std::endl;
            h_packet_info_filled[i][j] = false;
            break;
          }

          if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
            int receiver_index = packet_info.source * NR_RECEIVERS_PER_PACKET;
            std::cout << "Copying data from frame " << j << " of channel " << i
                      << " to packet_index " << packet_index
                      << " and receiver index " << receiver_index
                      << " of buffer "
                      << (current_buffer + buffer_index) % NR_BUFFERS
                      << std::endl;
            std::memcpy(
                &(*d_samples[buffer_index])[i][packet_index][receiver_index],
                &(*d_packet_data[i])[j], sizeof(Packet));
            h_packet_info_filled[i][j] = false;
            break;
          }
        }
      }

      std::cout << "Check if buffers are complete for channel " << i
                << std::endl;
      if (std::all_of(std::begin(latest_packet_received[i]),
                      std::end(latest_packet_received[i]),
                      [current_buffer, &h_input_buffer_end_seq](int x) {
                        std::cout << x << ", "
                                  << h_input_buffer_end_seq[current_buffer]
                                  << std::endl;
                        return x >= h_input_buffer_end_seq[current_buffer];
                      })) {
        h_is_input_buffer_populated[current_buffer][i] = true;
        std::cout << "Buffer is complete for channel " << i << ".\n";
      } else {
        std::cout << "Buffer is not complete for channel " << i
                  << " as end_seq is " << h_input_buffer_end_seq[current_buffer]
                  << " and latest_packet_receives are ";
        for (int check = 0; check < NR_FPGA_SOURCES; ++check) {
          std::cout << latest_packet_received[i][check] << ", ";
        }
        std::cout << std::endl;
      }
    }
    if (std::all_of(std::begin(h_is_input_buffer_populated[current_buffer]),
                    std::end(h_is_input_buffer_populated[current_buffer]),
                    [](bool i) { return i; })) {
      h_is_buffer_ready[current_buffer] = false;
      // Call onward processing.
      h_is_buffer_ready[current_buffer] = true;
      int old_buffer = current_buffer;
      int end_current_buffer_seq = h_input_buffer_end_seq[old_buffer];
      current_buffer = (current_buffer + 1) % NR_BUFFERS;
      std::cout << "current_buffer is " << current_buffer
                << " and is it ready? " << h_is_buffer_ready[current_buffer]
                << std::endl;
      while (!h_is_buffer_ready[current_buffer]) {
        std::cout << "Waiting for buffer to be ready..." << std::endl;
      }
      std::memset(std::begin(h_is_input_buffer_populated[current_buffer]),
                  (int)false, NR_CHANNELS);
      h_input_buffer_start_seq[current_buffer] = end_current_buffer_seq + 1;
      h_input_buffer_end_seq[current_buffer] =
          h_input_buffer_start_seq[current_buffer] +
          NR_PACKETS_FOR_CORRELATION - 1;

      // This bit probably happens somewhere else but is included here for now -
      // initializes other input buffer to have new packet bracket.
      int max_end_seq_in_buffers = 0;
      for (auto i = 0; i < NR_BUFFERS; ++i) {
        max_end_seq_in_buffers =
            std::max(max_end_seq_in_buffers, h_input_buffer_end_seq[i]);
      }
      h_input_buffer_start_seq[old_buffer] = max_end_seq_in_buffers + 1;
      h_input_buffer_end_seq[old_buffer] =
          h_input_buffer_start_seq[old_buffer] + NR_PACKETS_FOR_CORRELATION - 1;

      std::cout
          << "Current buffer is all complete. Moving to next buffer which is #"
          << current_buffer << std::endl;
      std::cout << "New buffer starts at packet "
                << h_input_buffer_start_seq[current_buffer] << " and ends at "
                << h_input_buffer_end_seq[current_buffer] << "\n";
    }
  }
  return 0;
}
