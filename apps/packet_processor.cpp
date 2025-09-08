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
struct PacketData {
  Packet data;
};
struct PacketInfo {
  int source;
  int packet_seq_number;
  int channel;
};

struct BufferState {
  bool is_ready;
  int start_seq;
  int end_seq;
  std::array<bool, NR_CHANNELS> is_populated{};
};

struct ProcessorState {
  PacketSamples *d_samples[NR_BUFFERS];
  Packets *d_packet_data[NR_CHANNELS];
  PacketInfo h_packet_info[NR_CHANNELS][NR_TOTAL_FRAMES_PER_CHANNEL];
  bool h_packet_info_filled[NR_CHANNELS][NR_TOTAL_FRAMES_PER_CHANNEL] = {false};
  std::array<BufferState, NR_BUFFERS> buffers;
  int latest_packet_received[NR_CHANNELS][NR_FPGA_SOURCES] = {};
  int next_frame_for_channel[NR_CHANNELS] = {0};
  int current_buffer = 0;
};

struct GeneratorState {

  int last_generated_seq_num_for_channel_and_source[NR_CHANNELS]
                                                   [NR_FPGA_SOURCES] = {};
};

PacketData create_packet(const int channel, const int seq, const int source) {

  PacketData packet;

  for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
    for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
      for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
        packet.data[i][j][k] = Sample(channel + seq, source + j);
      }
    }
  }
  return packet;
}

void print_packet_sample(const PacketSamples &packet_sample) {
  std::cout << "Packet Data:" << std::endl;
  for (auto i = 0; i < NR_CHANNELS; ++i) {
    for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
      for (auto k = 0; k < NR_RECEIVERS; ++k) {
        for (auto l = 0; l < NR_POLARIZATIONS; ++l) {
          for (auto m = 0; m < NR_TIME_STEPS_PER_PACKET; ++m) {
            std::cout << "channel: " << i << ", packet: " << j
                      << ", receiver: " << k << ", polarization: " << l
                      << ", time: " << m << ", val: "
                      << __half2float(packet_sample[i][j][k][l][m].real())
                      << " + "
                      << __half2float(packet_sample[i][j][k][l][m].imag())
                      << "j" << std::endl;
          }
        }
      }
    }
  }
}

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
}

void initialize_memory(ProcessorState &state) {

  for (auto i = 0; i < NR_CHANNELS; ++i) {
    state.d_packet_data[i] = (Packets *)calloc(1, sizeof(Packets));
  }

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    state.d_samples[i] = (PacketSamples *)calloc(1, sizeof(PacketSamples));
  }
}

void initialize_buffers(ProcessorState &state) {

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    state.buffers[i].start_seq = i * NR_PACKETS_FOR_CORRELATION;
    state.buffers[i].end_seq = (i + 1) * NR_PACKETS_FOR_CORRELATION - 1;
    state.buffers[i].is_ready = true;
  }
}

void check_buffer_completion(ProcessorState &state) {
  for (auto channel = 0; channel < NR_CHANNELS; ++channel) {
    std::cout << "Check if buffers are complete for channel " << channel
              << std::endl;

    if (std::all_of(std::begin(state.latest_packet_received[channel]),
                    std::end(state.latest_packet_received[channel]),
                    [&state](int x) {
                      return x >= state.buffers[state.current_buffer].end_seq;
                    })) {
      state.buffers[state.current_buffer].is_populated[channel] = true;
      std::cout << "Buffer is complete for channel " << channel << ".\n";
    } else {
      std::cout << "Buffer is not complete for channel " << channel
                << " as end_seq is "
                << state.buffers[state.current_buffer].end_seq
                << " and latest_packet_receives are ";
      for (int check = 0; check < NR_FPGA_SOURCES; ++check) {
        std::cout << state.latest_packet_received[channel][check] << ", ";
      }
      std::cout << std::endl;
    }
  }
}

void advance_to_next_buffer(ProcessorState &state) {
  state.buffers[state.current_buffer].is_ready = false;

  // Process current buffer
  print_packet_sample(*state.d_samples[state.current_buffer]);

  state.buffers[state.current_buffer].is_ready = true;
  int old_buffer = state.current_buffer;
  int end_current_buffer_seq = state.buffers[old_buffer].end_seq;

  // Move to next buffer
  state.current_buffer = (state.current_buffer + 1) % NR_BUFFERS;

  std::cout << "current_buffer is " << state.current_buffer
            << " and is it ready? "
            << state.buffers[state.current_buffer].is_ready << std::endl;

  while (!state.buffers[state.current_buffer].is_ready) {
    std::cout << "Waiting for buffer to be ready..." << std::endl;
  }

  // Reset new current buffer
  std::memset(std::begin(state.buffers[state.current_buffer].is_populated),
              (int)false, NR_CHANNELS);
  state.buffers[state.current_buffer].start_seq = end_current_buffer_seq + 1;
  state.buffers[state.current_buffer].end_seq =
      state.buffers[state.current_buffer].start_seq +
      NR_PACKETS_FOR_CORRELATION - 1;

  // Update old buffer for future use
  int max_end_seq_in_buffers = 0;
  for (auto i = 0; i < NR_BUFFERS; ++i) {
    max_end_seq_in_buffers =
        std::max(max_end_seq_in_buffers, state.buffers[i].end_seq);
  }
  state.buffers[old_buffer].start_seq = max_end_seq_in_buffers + 1;
  state.buffers[old_buffer].end_seq =
      state.buffers[old_buffer].start_seq + NR_PACKETS_FOR_CORRELATION - 1;

  std::cout
      << "Current buffer is all complete. Moving to next buffer which is #"
      << state.current_buffer << std::endl;
  std::cout << "New buffer starts at packet "
            << state.buffers[state.current_buffer].start_seq << " and ends at "
            << state.buffers[state.current_buffer].end_seq << "\n";
}

void generate_packet(ProcessorState &state, GeneratorState &gen_state) {

  // Receive packets
  int channel = randomChannel();
  int source = randomSource();
  gen_state.last_generated_seq_num_for_channel_and_source[channel][source]++;
  int seq =
      gen_state.last_generated_seq_num_for_channel_and_source[channel][source];
  PacketInfo info;
  info.source = source;
  info.channel = channel;
  info.packet_seq_number = seq;
  PacketData constructed_packet = create_packet(channel, seq, source);
  state.h_packet_info[channel][state.next_frame_for_channel[channel]] = info;
  std::memcpy(
      &((*state.d_packet_data[channel])[state.next_frame_for_channel[channel]]),
      &constructed_packet.data, sizeof(Packet));
  state.h_packet_info_filled[channel][state.next_frame_for_channel[channel]] =
      true;
  int gate_check{0};
  while (state.h_packet_info_filled[channel]
                                   [state.next_frame_for_channel[channel]]) {
    state.next_frame_for_channel[channel] =
        (state.next_frame_for_channel[channel] + 1) %
        NR_TOTAL_FRAMES_PER_CHANNEL;
    gate_check++;
    if (gate_check > NR_TOTAL_FRAMES_PER_CHANNEL) {
      throw std::runtime_error(
          "We ate our own tail - no frames left for this channel.");
    }
  }
}

void copy_data_to_input_buffer_if_able(ProcessorState &state,
                                       const int channel) {

  std::cout << "Update latest numbers for channel " << channel << std::endl;
  for (auto j = 0; j < NR_TOTAL_FRAMES_PER_CHANNEL; ++j) {
    if (!state.h_packet_info_filled[channel][j])
      continue;

    PacketInfo packet_info = state.h_packet_info[channel][j];
    state.latest_packet_received[channel][packet_info.source] =
        std::max(state.latest_packet_received[channel][packet_info.source],
                 packet_info.packet_seq_number);

    // copy to correct place or leave it.
    for (int buffer = 0; buffer < NR_BUFFERS; ++buffer) {
      int buffer_index = (state.current_buffer + buffer) % NR_BUFFERS;
      int packet_index =
          packet_info.packet_seq_number - state.buffers[buffer_index].start_seq;

      if (buffer == 0 && packet_index < 0) {
        // This means that this packet is less than the lowest possible
        // start token. Maybe an out-of-order packet that's coming in?
        // Regardless we can't do anything with this.
        std::cout << "Discarding packet as it is before current buffer "
                     "with begin_seq "
                  << state.buffers[state.current_buffer].start_seq
                  << " actually has packet_index "
                  << packet_info.packet_seq_number << std::endl;
        state.h_packet_info_filled[channel][j] = false;
        break;
      }

      if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
        int receiver_index = packet_info.source * NR_RECEIVERS_PER_PACKET;
        std::cout << "Copying data from frame " << j << " of channel "
                  << channel << " to packet_index " << packet_index
                  << " and receiver index " << receiver_index << " of buffer "
                  << (state.current_buffer + buffer_index) % NR_BUFFERS
                  << std::endl;
        std::memcpy(&(*state.d_samples[buffer_index])[channel][packet_index]
                                                     [receiver_index],
                    &(*state.d_packet_data[channel])[j], sizeof(Packet));
        state.h_packet_info_filled[channel][j] = false;
        break;
      }
    }
  }
}

int main() {
  print_startup_info();

  ProcessorState state;
  GeneratorState gen_state;
  initialize_memory(state);

  // Receive n packets

  initialize_buffers(state);

  for (auto iter = 0; iter < NUM_ITERATIONS; ++iter) {
    std::cout << "Starting iteration " << iter << std::endl;
    // Generate some packets and put it in the buffer.
    std::cout << "Randomly generate data..." << std::endl;
    for (auto i = 0; i < NUM_FRAMES_PER_ITERATION; ++i) {
      generate_packet(state, gen_state);
    }

    // Copy in data to correct place
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      copy_data_to_input_buffer_if_able(state, i);
      check_buffer_completion(state);
    }
    if (std::all_of(
            std::begin(state.buffers[state.current_buffer].is_populated),
            std::end(state.buffers[state.current_buffer].is_populated),
            [](bool i) { return i; })) {
      advance_to_next_buffer(state);
    }
  }
  return 0;
}
