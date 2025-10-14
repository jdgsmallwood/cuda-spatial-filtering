#pragma once
#include "spatial/spatial.hpp"
#include <cstdint>
#include <pcap/pcap.h>
#include <sys/types.h>
#include <vector>

struct PacketInfo {
  uint64_t sample_count;
  uint16_t freq_channel;
};

struct PCAPInfo {
  int start_freq;
  int end_freq;
  int start_seq;
  int end_seq;
  int num_packets_captured;
};

PacketInfo get_packet_info(const u_char *packet, const int size);
void process_packet(const u_char *packet, const int size, Samples *agg_samples,
                    std::vector<Tscale> &scales_output, const int start_seq_id,
                    const int start_freq, const int nr_time_steps_per_packet,
                    const int nr_blocks_for_correlation,
                    const int nr_times_per_block,
                    const int nr_actual_receivers);
